#!/bin/sh
# Reads "<kind> ..." lines on stdin, as protected.sh and lint-gate.sh
# give, and labels the PR with each kind. Adds needs-human-review
# unless the PR already holds the label of each kind found, added by
# CI. So a maintainer who removes needs-human-review is asked again
# only for a new kind.
# A label that a person added does not count, so nobody can skip the
# review by adding one first.
# Needs GH_TOKEN, PR and GITHUB_REPOSITORY in the environment.
set -e
kinds=$(cut -d' ' -f1 | sort -u)
[ -n "$kinds" ] || exit 0
known=$(tools/scripts/bot-labels.sh)
review=false
for kind in $kinds; do
    if ! printf '%s\n' "$known" | grep -qxF "$kind"; then
        echo "New kind of change for review: $kind"
        review=true
    fi
    set -- "$@" -f "labels[]=$kind"
done
if [ "$review" = true ]; then
    set -- "$@" -f 'labels[]=needs-human-review'
fi
gh api "repos/$GITHUB_REPOSITORY/issues/$PR/labels" "$@" >/dev/null
