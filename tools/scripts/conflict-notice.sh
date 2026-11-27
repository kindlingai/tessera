#!/bin/sh
# Test-merges each pair of open PRs that target MAIN, drafts included.
# Also lists files both PRs change that merge cleanly.
# Keeps one notice comment on each PR with those lists. Deletes the
# notice when both lists are empty, or when the PR conflicts with
# main. Writes only when the text changes. Fails on
# errors only.
# Each list line reads "- conflict #<n>: <files>" or "- shared #<n>: <files>".
# Needs GH_TOKEN, GITHUB_REPOSITORY and MAIN.
set -e
main=origin/$MAIN
repo=repos/$GITHUB_REPOSITORY
marker="<!-- conflict-notice -->"
tmp=$(mktemp -d)

fail() {
    echo "conflict-notice: $1" >&2
    exit 2
}

# One read of every open PR on main and its first 100 comments.
gh api graphql --paginate -F owner="${GITHUB_REPOSITORY%/*}" -F name="${GITHUB_REPOSITORY#*/}" \
    -F base="$MAIN" -f query='
query($owner: String!, $name: String!, $base: String!, $endCursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: OPEN, baseRefName: $base, first: 100, after: $endCursor) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number headRefOid
        comments(first: 100) {
          pageInfo { hasNextPage }
          nodes { databaseId body author { login } }
        }
      }
    }
  }
}' >"$tmp/prs" || fail "Could not read the open PRs."
# Lines: "<n> <head> <notice id or -> <more comments>".
jq -rs --arg marker "$marker" '[.[].data.repository.pullRequests.nodes[]][]
    | ([.comments.nodes[] | select(.author.login == "github-actions"
        and (.body | startswith($marker)))][0]) as $notice
    | "\(.number) \(.headRefOid) \($notice.databaseId // "-") \(.comments.pageInfo.hasNextPage)"' \
    "$tmp/prs" >"$tmp/open" || fail "Could not parse the open PRs."

# Sets nid to the notice comment id of PR $1, or to nothing.
# Writes the notice text to $tmp/old.
notice_id() {
    grep "^$1 " "$tmp/open" >"$tmp/line"
    read -r _ _ nid more <"$tmp/line"
    if [ "$nid" = - ] && [ "$more" = true ]; then
        gh api --paginate "$repo/issues/$1/comments" --jq ".[] | select(.user.login == \"github-actions[bot]\"
            and (.body | startswith(\"$marker\"))) | .id" >"$tmp/id" </dev/null ||
            fail "Could not read the comments of #$1."
        nid=$(head -n 1 "$tmp/id")
        [ -n "$nid" ] || nid=-
        if [ "$nid" != - ]; then
            gh api "$repo/issues/comments/$nid" --jq .body >"$tmp/old" </dev/null ||
                fail "Could not read the notice on #$1."
        fi
    elif [ "$nid" != - ]; then
        jq -rs --argjson id "$nid" '.[].data.repository.pullRequests.nodes[].comments.nodes[]
            | select(.databaseId == $id) | .body' "$tmp/prs" >"$tmp/old" ||
            fail "Could not parse the notice on #$1."
    fi
    [ "$nid" != - ] || nid=
}

# Deletes the notice of PR $1 if it has one.
clear_notice() {
    notice_id "$1"
    [ -n "$nid" ] || return 0
    gh api -X DELETE "$repo/issues/comments/$nid" >/dev/null </dev/null ||
        fail "Could not delete the notice on #$1."
    echo "#$1: notice deleted."
}

# Posts report $2 on PR $1, or edits its notice if the text differs.
post_notice() {
    notice_id "$1"
    if [ -z "$nid" ]; then
        gh api "$repo/issues/$1/comments" -F body=@"$2" >/dev/null </dev/null || fail "Could not post on #$1."
        echo "#$1: notice posted."
        return
    fi
    if diff -q -B --strip-trailing-cr "$2" "$tmp/old" >/dev/null; then
        echo "#$1: notice unchanged."
        return
    fi
    gh api -X PATCH "$repo/issues/comments/$nid" -F body=@"$2" >/dev/null </dev/null ||
        fail "Could not edit the notice on #$1."
    echo "#$1: notice edited."
}

# GitHub already marks a PR that conflicts with main.
: >"$tmp/ready"
while read -r n sha _; do
    git fetch -q origin "$MAIN" "$sha" || fail "Could not fetch #$n."
    code=0
    git merge-tree --write-tree "$main" "$sha" >/dev/null || code=$?
    [ "$code" -le 1 ] || fail "git merge-tree failed on #$n."
    if [ "$code" -eq 1 ]; then
        clear_notice "$n"
        continue
    fi
    echo "$n $sha" >>"$tmp/ready"
    git diff --name-only "$main...$sha" >"$tmp/diff" || fail "git diff failed on #$n."
    sort "$tmp/diff" >"$tmp/files-$n"
    : >"$tmp/lines-$n"
done <"$tmp/open"

# Prints the paths on stdin as code spans, on one line. The fence is
# one backtick longer than the longest backtick run in the path, so
# a path cannot end its span or add markup such as a mention.
quote() {
    awk '{
        fence = "`"
        while (index($0, fence)) fence = fence "`"
        pad = (index($0, "`")) ? " " : ""
        printf "%s%s%s%s%s%s", (NR > 1 ? " " : ""), fence, pad, $0, pad, fence
    } END { print "" }'
}

# Each pair once. A line goes in the reports of both PRs.
while read -r a asha; do
    while read -r b bsha; do
        [ "$a" -lt "$b" ] || continue
        code=0
        tools/scripts/merge-conflicts.sh "$main" "$asha" "$bsha" >"$tmp/pair" || code=$?
        [ "$code" -le 1 ] || fail "merge-conflicts.sh failed on #$a and #$b."
        sed "s|^$bsha ||" "$tmp/pair" | sort >"$tmp/conflicts"
        if [ -s "$tmp/conflicts" ]; then
            files=$(quote <"$tmp/conflicts")
            echo "- conflict #$b: $files" >>"$tmp/lines-$a"
            echo "- conflict #$a: $files" >>"$tmp/lines-$b"
        fi
        comm -12 "$tmp/files-$a" "$tmp/files-$b" | comm -23 - "$tmp/conflicts" >"$tmp/both"
        if [ -s "$tmp/both" ]; then
            files=$(quote <"$tmp/both")
            echo "- shared #$b: $files" >>"$tmp/lines-$a"
            echo "- shared #$a: $files" >>"$tmp/lines-$b"
        fi
    done <"$tmp/ready"
done <"$tmp/ready"

while read -r n _; do
    if [ ! -s "$tmp/lines-$n" ]; then
        clear_notice "$n"
        continue
    fi
    report=$tmp/report-$n.md
    {
        echo "$marker"
        echo "### Overlap with other open pull requests"
        echo "A conflict line names files that conflict with that PR."
        echo "The PR that merges second must rebase. A shared line names"
        echo "files both PRs change that merge cleanly. This notice fails no check."
        echo
        sort -k2,2 -k3,3 "$tmp/lines-$n"
    } >"$report"
    sed 1,2d "$report" | sed -n 's/^- //p' | sed "s/^/#$n: /"
    post_notice "$n" "$report"
done <"$tmp/ready"
