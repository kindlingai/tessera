#!/bin/sh
# Prints which success jobs a tier change makes stale:
# "rerun=all", "rerun=<head sha>" or "rerun=".
# The tier labels are priority and landing. A PR is ranked as
# tier-ranks.sh ranks it, from the label changes by setters.
# tiers.yml skips workflow_run events that cannot change a result.
# Exits 2 on any API error.
# Needs GH_TOKEN, GITHUB_REPOSITORY, MAIN, GITHUB_EVENT_NAME and
# GITHUB_EVENT_PATH.
set -e
ev=$GITHUB_EVENT_PATH
tmp=$(mktemp -d)
rerun=
# A change to a tier label by anyone reruns all. A change by a
# non-setter leaves every result as it was, and the reruns are cheap.
if [ "$GITHUB_EVENT_NAME" = pull_request_target ]; then
    jq -r 'if .action == "labeled" or .action == "unlabeled" then
        if .label.name == "priority" or .label.name == "landing" then "all"
        elif .label.name == "conflict-exempt" then .pull_request.head.sha
        else "" end
    else "" end' "$ev" >"$tmp/rerun"
    read -r rerun <"$tmp/rerun" || rerun=
fi
if [ -z "$rerun" ]; then
    tools/scripts/tier-ranks.sh >"$tmp/ranks"
    case "$GITHUB_EVENT_NAME" in
    pull_request_target)
        # A closed PR has left the open list. It counts when a setter
        # ever added a tier label to it.
        n=$(jq -r .pull_request.number "$ev")
        if awk -v n="$n" '$1 != "exempt" && $3 == n { found = 1 } END { exit !found }' "$tmp/ranks"; then
            rerun=all
        elif [ "$(jq -r .action "$ev")" = closed ]; then
            tools/scripts/gh-retry.sh core gh api --paginate "repos/$GITHUB_REPOSITORY/issues/$n/events?per_page=100" \
                --jq '.[] | select(.event == "labeled" and (.label.name == "priority" or .label.name == "landing"))
                    | .actor.login' >"$tmp/adders" || exit 2
            grep -v -e '^#' -e '^$' .github/tier-setters.txt >"$tmp/setters"
            if grep -qxFf "$tmp/setters" "$tmp/adders"; then
                rerun=all
            fi
        fi
        ;;
    push)
        # Main moved. Results change only while a ranked PR is open.
        if grep -qv '^exempt ' "$tmp/ranks"; then
            rerun=all
        fi
        ;;
    workflow_run)
        sha=$(jq -r .workflow_run.head_sha "$ev")
        if awk -v s="$sha" '$1 != "exempt" && $4 == s { found = 1 } END { exit !found }' "$tmp/ranks"; then
            rerun=all
        fi
        ;;
    esac
fi
echo "rerun=$rerun"
