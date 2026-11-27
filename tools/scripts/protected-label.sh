#!/bin/sh
# Reads protected.sh output on stdin and labels the PR with each kind of
# change it finds. Adds needs-human-review unless the PR already holds
# the label of each kind found, added by this workflow. So a maintainer
# who removes needs-human-review is asked again only for a new kind.
# A label that a person added does not count, so nobody can skip the
# review by adding one first.
# Needs GH_TOKEN, PR and GITHUB_REPOSITORY in the environment.
set -e
kinds=$(cut -d' ' -f1 | sort -u)
[ -n "$kinds" ] || exit 0
# The labels on the PR whose last label event was this workflow adding it.
known=$(gh api --paginate "repos/$GITHUB_REPOSITORY/issues/$PR/events" \
    --jq '.[] | select(.event == "labeled" or .event == "unlabeled")
        | "\(.event) \(.actor.login) \(.label.name)"' |
    awk '{ last[$3] = $1 " " $2 } END {
        for (l in last) if (last[l] == "labeled github-actions[bot]") print l }')
review=false
for kind in $kinds; do
    if ! printf '%s\n' "$known" | grep -qxF "$kind"; then
        echo "New kind of protected change: $kind"
        review=true
    fi
    set -- "$@" -f "labels[]=$kind"
done
if [ "$review" = true ]; then
    set -- "$@" -f 'labels[]=needs-human-review'
fi
gh api "repos/$GITHUB_REPOSITORY/issues/$PR/labels" "$@" >/dev/null
