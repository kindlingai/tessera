#!/bin/sh
# Passes when each cancelled job in this run attempt was cancelled
# because a hosted runner never took it. Any other cancel fails, for
# example a run that a newer run replaced.
# Needs GH_TOKEN, GITHUB_REPOSITORY, GITHUB_RUN_ID and GITHUB_RUN_ATTEMPT.
set -e
repo="repos/$GITHUB_REPOSITORY"
jobs="$repo/actions/runs/$GITHUB_RUN_ID/attempts/$GITHUB_RUN_ATTEMPT/jobs"
tab=$(printf '\t')
gh api --paginate "$jobs" --jq '.jobs[] | select(.conclusion == "cancelled")
    | [.id, ((.runner_name // "") == "" and (.steps | length) == 0), .name]
    | @tsv' |
    while IFS="$tab" read -r id idle name; do
        # The job ID is also its check run ID.
        if [ "$idle" = true ] &&
            gh api "$repo/check-runs/$id/annotations" --jq '.[].message' |
            grep -q 'not acquired by Runner'; then
            echo "::warning::$name did not run: no hosted runner took it."
        else
            echo "$name was cancelled."
            exit 1
        fi
    done
