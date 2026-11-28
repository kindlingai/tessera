#!/bin/sh
# Fails a pull request that waits on a human, carries a held-* label or
# is stacked on a branch other than the default branch. The other CI jobs still run on it.
# Reads the PR live, so a rerun sees label, setting and base changes.
# REVIEW holds the lint-<rule> labels that lint-gate.sh asks for. Each
# must be on the PR, added by CI, so that its review was asked for.
# Needs GH_TOKEN, PR and GITHUB_REPOSITORY in the environment.
set -e
# Five words: labeled for review, a fork maintainers cannot push to,
# the first held-* label or "-", the base branch and the default branch.
state=$(gh api "repos/$GITHUB_REPOSITORY/pulls/$PR" --jq '[
    (.labels | any(.name == "needs-human-review")),
    (.head.repo.full_name != .base.repo.full_name and (.maintainer_can_modify | not)),
    ([.labels[].name | select(startswith("held-"))] | first // "-"),
    .base.ref, .base.repo.default_branch
] | map(tostring) | join(" ")')
# Branch names cannot hold spaces.
read -r review locked held base main <<EOF
$state
EOF
status=0
if [ "$base" != "$main" ]; then
    echo "This PR is stacked on $base. It can land only on $main."
    echo "After $base merges, rebase onto $main and change the base to $main."
    status=1
fi
if [ "$review" = true ]; then
    echo "Protected files changed or a size limit breached. A maintainer reviews, then removes needs-human-review."
    status=1
fi
if [ "$held" != - ]; then
    echo "This PR is on hold: $held. Remove the label to land it."
    status=1
fi
if [ "$locked" = true ]; then
    echo "Tick \"Allow edits by maintainers\" on the PR, then rerun this check."
    status=1
fi
if [ -n "$REVIEW" ]; then
    known=$(tools/scripts/bot-labels.sh)
    for label in $REVIEW; do
        printf '%s\n' "$known" | grep -qxF "$label" && continue
        echo "A new size limit breach needs the $label label from CI and a review."
        echo "CI cannot label fork PRs. Push the branch to this repository."
        status=1
    done
fi
exit $status
