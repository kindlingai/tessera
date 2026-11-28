#!/bin/sh
# Fails unless the newest Protected files run on this commit passed.
# A run that GitHub blocks or skips must not let a PR through.
# Needs GH_TOKEN, SHA and GITHUB_REPOSITORY in the environment.
set -e
runs="repos/$GITHUB_REPOSITORY/actions/workflows/protected.yml/runs"
# Removing the label reruns this job while that run is still going.
# Other added labels skip the job. Skipped runs do not count.
for _ in $(seq 30); do
    run=$(gh api "$runs?head_sha=$SHA&event=pull_request_target&per_page=20" \
        --jq '[.workflow_runs[] | select(.conclusion != "skipped")][0] // empty
            | "\(.status) \(.conclusion) \(.html_url)"')
    case "$run" in
    "completed success "*) exit 0 ;;
    completed*)
        echo "Protected files did not pass: ${run#completed }"
        exit 1
        ;;
    esac
    sleep 10
done
echo "No finished Protected files run for $SHA."
exit 1
