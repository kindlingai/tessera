#!/bin/sh
# Cancels this run attempt as soon as one of its other jobs fails. The
# success job still runs after a cancel and fails on that job, so the
# merge queue drops the group without waiting for the slowest jobs.
# Ends when the success job starts, because then every job it needs
# has finished.
# Polls each 30 seconds: the run token allows 1000 calls an hour for
# the whole repository, and several groups can run at once.
# Needs GH_TOKEN, GITHUB_REPOSITORY, GITHUB_RUN_ID, GITHUB_RUN_ATTEMPT
# and SELF, the name of the job that runs this script.
set -e
jobs="repos/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID/attempts/$GITHUB_RUN_ATTEMPT/jobs?per_page=100"
while :; do
    state=$(gh api --paginate "$jobs" --jq ".jobs[] | select(.name != \"$SELF\")
        | \"\(.name)\t\(.status)\t\(.conclusion // \"\")\"")
    failed=$(printf '%s\n' "$state" | awk -F '\t' '$1 != "success" && $3 == "failure" { print $1 }')
    if [ -n "$failed" ]; then
        echo "Failed: $failed"
        echo "Cancelling the rest of this run."
        gh run cancel "$GITHUB_RUN_ID" -R "$GITHUB_REPOSITORY"
        exit 0
    fi
    if printf '%s\n' "$state" | awk -F '\t' '$1 == "success" && $2 != "queued" && $2 != "waiting" && $2 != "pending" { found = 1 } END { exit !found }'; then
        exit 0
    fi
    sleep 30
done
