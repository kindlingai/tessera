#!/bin/sh
# Fails a pull request that waits on a human.
# Reads the PR live, so a rerun sees label and setting changes.
# Needs GH_TOKEN, PR and GITHUB_REPOSITORY in the environment.
set -e
# Two words: labeled for review, and a fork maintainers cannot push to.
state=$(gh api "repos/$GITHUB_REPOSITORY/pulls/$PR" --jq '[
    (.labels | any(.name == "needs-human-review")),
    (.head.repo.full_name != .base.repo.full_name and (.maintainer_can_modify | not))
] | map(tostring) | join(" ")')
status=0
if [ "${state% *}" = true ]; then
    echo "Protected files changed. A maintainer reviews, then removes needs-human-review."
    status=1
fi
if [ "${state#* }" = true ]; then
    echo "Tick \"Allow edits by maintainers\" on the PR, then rerun this check."
    status=1
fi
exit $status
