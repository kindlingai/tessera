#!/bin/sh
# Prints the rank of each open PR on MAIN, one line each:
# "exempt <n>" or "<tier> <labelled at> <n> <head> <draft> <red> <label>".
# Tier 1 is priority and tier 2 is landing.
# A label counts when the last change to it by a setter added it.
# Changes by others count for nothing, added or removed.
# Setters are the logins in .github/tier-setters.txt, one a line. An app
# login ends in [bot], so a user with the same name does not match.
# Red means a CI job other than success failed or timed out.
# Exits 2 on any API error. Needs GH_TOKEN, GITHUB_REPOSITORY and MAIN.
set -e
retry=tools/scripts/gh-retry.sh
tmp=$(mktemp -d)

fail() {
    echo "tier-ranks: $1" >&2
    exit 2
}

[ -f .github/tier-setters.txt ] || fail "Could not read .github/tier-setters.txt."
grep -v -e '^#' -e '^$' .github/tier-setters.txt | jq -R . | jq -s . >"$tmp/setters" ||
    fail "Could not parse .github/tier-setters.txt."

# One read of every open PR: the last 100 label events and the CI checks.
$retry graphql gh api graphql --paginate -F owner="${GITHUB_REPOSITORY%/*}" \
    -F name="${GITHUB_REPOSITORY#*/}" -F base="${MAIN#refs/heads/}" -f query='
query($owner: String!, $name: String!, $base: String!, $endCursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: OPEN, baseRefName: $base, first: 100, after: $endCursor) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number isDraft headRefOid
        timelineItems(itemTypes: [LABELED_EVENT, UNLABELED_EVENT], last: 100) {
          pageInfo { hasPreviousPage }
          nodes {
            __typename
            ... on LabeledEvent { createdAt actor { __typename login } label { name } }
            ... on UnlabeledEvent { createdAt actor { __typename login } label { name } }
          }
        }
        commits(last: 1) { nodes { commit { statusCheckRollup { contexts(first: 100) { nodes {
          ... on CheckRun { name conclusion checkSuite { workflowRun { workflow { name } } } }
        } } } } } }
      }
    }
  }
}' >"$tmp/prs" || fail "Could not read the open PRs."

# A PR with more label events than one page reads them all over REST.
jq -rs '.[].data.repository.pullRequests.nodes[]
    | select(.timelineItems.pageInfo.hasPreviousPage) | .number' "$tmp/prs" >"$tmp/more" ||
    fail "Could not parse the open PRs."
echo '{}' >"$tmp/rest"
while read -r n; do
    $retry core gh api --paginate "repos/$GITHUB_REPOSITORY/issues/$n/events?per_page=100" \
        --jq '[.[] | select(.event == "labeled" or .event == "unlabeled")
            | {kind: .event, at: .created_at, who: (.actor.login // ""), label: .label.name}]' \
        >"$tmp/events" </dev/null || fail "Could not read the label events of #$n."
    jq -s --arg n "$n" --slurpfile rest "$tmp/rest" '$rest[0] + {($n): add}' "$tmp/events" >"$tmp/next" ||
        fail "Could not parse the label events of #$n."
    mv "$tmp/next" "$tmp/rest"
done <"$tmp/more"

jq -rs --slurpfile who "$tmp/setters" --slurpfile rest "$tmp/rest" '
    def events: $rest[0][.number | tostring] // [.timelineItems.nodes[]
        | {kind: (if .__typename == "LabeledEvent" then "labeled" else "unlabeled" end),
           at: .createdAt, label: .label.name,
           who: ((.actor.login // "") + (if .actor.__typename == "Bot" then "[bot]" else "" end))}];
    def since($l): [events[] | select(.label == $l and (.who | IN($who[0][])))]
        | last | if . != null and .kind == "labeled" then .at else null end;
    def red: [.commits.nodes[0].commit.statusCheckRollup.contexts.nodes[]?
        | select(.checkSuite.workflowRun.workflow.name? == "CI" and .name != "success")
        | select(.conclusion == "FAILURE" or .conclusion == "TIMED_OUT")] | length > 0;
    def line($t; $l): "\($t) \(since($l)) \(.number) \(.headRefOid) \(.isDraft) \(red) \($l)";
    [.[].data.repository.pullRequests.nodes[]][]
    | (if since("conflict-exempt") then "exempt \(.number)" else empty end),
      (if since("priority") then line(1; "priority")
       elif since("landing") then line(2; "landing")
       else empty end)' "$tmp/prs" || fail "Could not parse the open PRs."
