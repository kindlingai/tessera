#!/bin/sh
# Reruns the CI success job of each head SHA given.
# A run whose success job has not started reads the new state when
# it starts, so it is left alone. A success job that started or
# finished may have read the old state. GitHub reruns a job only in
# a finished run, so wait for the run, then rerun.
# Needs GH_TOKEN and GITHUB_REPOSITORY.
tmp=$(mktemp -d)
status=0
for sha in "$@"; do
    if ! gh run list -R "$GITHUB_REPOSITORY" --workflow ci.yml --commit "$sha" \
        --json databaseId,status --jq '.[0] // empty | "\(.databaseId) \(.status)"' >"$tmp/run"; then
        echo "Could not list the CI runs of $sha."
        status=1
        continue
    fi
    [ -s "$tmp/run" ] || continue
    read -r run state <"$tmp/run"
    if ! gh run view -R "$GITHUB_REPOSITORY" "$run" --json jobs \
        --jq '.jobs[] | select(.name == "success") | "\(.databaseId) \(.status)"' >"$tmp/job"; then
        echo "Could not read the jobs of run $run."
        status=1
        continue
    fi
    [ -s "$tmp/job" ] || continue
    read -r job jobstate <"$tmp/job"
    if [ "$state" != completed ]; then
        case "$jobstate" in
        in_progress | completed) gh run watch -R "$GITHUB_REPOSITORY" "$run" >/dev/null || true ;;
        *) continue ;;
        esac
    fi
    gh run rerun -R "$GITHUB_REPOSITORY" --job "$job" || status=1
done
exit $status
