#!/bin/sh
# Prints true when any attempt of a check run passed on a commit, else
# false. A rerun of one job leaves the earlier result in place.
# Usage: check-passed.sh <sha> <check name>
# Needs GH_TOKEN and GITHUB_REPOSITORY.
set -e
passed=$(gh api -X GET "repos/$GITHUB_REPOSITORY/commits/$1/check-runs" \
    -f check_name="$2" -f filter=all \
    --jq '[.check_runs[] | select(.conclusion == "success")] | length')
if [ "$passed" -gt 0 ]; then echo true; else echo false; fi
