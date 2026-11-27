#!/bin/sh
# Prints each label on the PR whose last label event was the CI bot
# adding it. A label that a person added or removed last is left out.
# Needs GH_TOKEN, PR and GITHUB_REPOSITORY in the environment.
set -e
events=$(mktemp)
gh api --paginate "repos/$GITHUB_REPOSITORY/issues/$PR/events" \
    --jq '.[] | select(.event == "labeled" or .event == "unlabeled")
        | "\(.event) \(.actor.login) \(.label.name)"' >"$events"
awk '{ last[$3] = $1 " " $2 } END {
    for (l in last) if (last[l] == "labeled github-actions[bot]") print l }' "$events"
rm -f "$events"
