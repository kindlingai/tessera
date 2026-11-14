#!/bin/sh
# Creates or edits the pull request comment a report owns.
# The report's first line is a marker that finds its comment.
# Usage: pr-comment.sh <report.md>
# Needs GH_TOKEN, PR and GITHUB_REPOSITORY in the environment.
set -e
report=$1
[ -f "$report" ] || exit 0
marker=$(head -n 1 "$report")
comments="repos/$GITHUB_REPOSITORY/issues/$PR/comments"
id=$(gh api --paginate "$comments" \
    --jq ".[] | select(.body | startswith(\"$marker\")) | .id" | head -n 1)
if [ -n "$id" ]; then
    gh api -X PATCH "repos/$GITHUB_REPOSITORY/issues/comments/$id" -F body=@"$report" >/dev/null
elif ! grep -q '^None found' "$report" || grep -q '^### New files' "$report"; then
    # A clean run with no earlier comment says nothing.
    gh api "$comments" -F body=@"$report" >/dev/null
fi
