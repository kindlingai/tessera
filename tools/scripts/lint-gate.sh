#!/bin/sh
# Runs stickler lint and writes its JSON report to <report.json>.
# Fails on any failed rule except the size rules. A size breach that
# the base commit does not have, or that grows a count past the base,
# needs a human review instead. Prints only "review=<labels>" on
# stdout, with one lint-<rule> label for each such rule, for
# $GITHUB_OUTPUT.
# Without BASE, as on main and in the merge queue, size breaches pass:
# their pull request was reviewed.
# Usage: lint-gate.sh <report.json>
set -e
report=$1
sizes='["file-lines", "crate-files", "crate-lines"]'
tmp=$(mktemp -d)

code=0
tools/scripts/stickler.sh lint --json "$report" >&2 || code=$?
jq -e '.findings' "$report" >/dev/null || {
    echo "lint-gate: stickler wrote no report." >&2
    exit 1
}
other=$(jq --argjson s "$sizes" '[.findings[] | select(.level == "fail" and (.rule | IN($s[]) | not))] | length' "$report")
[ "$other" -eq 0 ] || exit 1

# "<rule> <path> <count>" for each size breach. The count is the
# first number of the message, as in "994 lines (max 900)".
breaches() {
    jq -r --argjson s "$sizes" '.findings[] | select(.level == "fail" and (.rule | IN($s[])))
        | "\(.rule) \(.path) \(.message | capture("(?<n>[0-9]+)").n)"' "$1" | sort
}
breaches "$report" >"$tmp/head"
if [ ! -s "$tmp/head" ]; then
    [ "$code" -eq 0 ] || {
        echo "lint-gate: stickler failed with no failed rule." >&2
        exit 1
    }
    echo "review="
    exit 0
fi
if [ -z "$BASE" ]; then
    echo "Size limits breached. Their pull requests had a human review." >&2
    echo "review="
    exit 0
fi

# Lint the base with the same rules.
git fetch -q --depth=1 origin "$BASE"
git worktree add -q --detach "$tmp/tree" "$BASE"
cp lint.toml "$tmp/tree/lint.toml"
(cd "$tmp/tree" && tools/scripts/stickler.sh lint --json "$tmp/base.json" >/dev/null) || true
jq -e '.findings' "$tmp/base.json" >/dev/null || {
    echo "lint-gate: stickler wrote no report for $BASE." >&2
    exit 1
}
breaches "$tmp/base.json" >"$tmp/base"
git worktree remove --force "$tmp/tree"

: >"$tmp/labels"
while read -r rule path n; do
    was=$(awk -v r="$rule" -v p="$path" '$1 == r && $2 == p { print $3 }' "$tmp/base")
    if [ -z "$was" ] || [ "$n" -gt "$was" ]; then
        echo "New size breach needs a human review: $rule $path ($n, base ${was:-under the limit})" >&2
        echo "lint-$rule" >>"$tmp/labels"
    fi
done <"$tmp/head"
printf 'review=%s\n' "$(sort -u "$tmp/labels" | tr '\n' ' ' | sed 's/ $//')"
