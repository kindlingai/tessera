#!/bin/sh
# Runs a gh command and retries it after a transient failure: an
# HTTP 5xx, a rate limit, a timeout or a lost connection. Other
# failures, such as a 404, fail at once.
# The bound is the reset time of the API rate-limit window that was
# open at the first failure, from the rate_limit response. When the
# window has used all its calls, wait for its reset and try once more.
# Prints the command's output. Exits 1 when the command fails.
# Usage: gh-retry.sh <core|graphql|search> gh <args>...
# Needs GH_TOKEN.
resource=$1
shift
tmp=$(mktemp -d)
deadline=
pause=1
while :; do
    if "$@" >"$tmp/out" 2>"$tmp/err"; then
        cat "$tmp/out"
        exit 0
    fi
    cat "$tmp/err" >&2
    grep -qiE 'HTTP 5[0-9][0-9]|rate limit|timeout|timed out|connection|EOF' "$tmp/err" || exit 1
    now=$(date +%s)
    if [ -z "$deadline" ]; then
        gh api rate_limit --jq ".resources.$resource | \"\(.remaining) \(.reset)\"" >"$tmp/limit" ||
            exit 1
        read -r remaining deadline <"$tmp/limit"
        if [ "$remaining" -eq 0 ]; then
            echo "gh-retry: no $resource calls left until $deadline. Waiting." >&2
            sleep $((deadline - now + 1))
            deadline=$((deadline + 1))
            continue
        fi
    fi
    [ "$now" -lt "$deadline" ] || exit 1
    echo "gh-retry: retrying in $pause s." >&2
    sleep "$pause"
    pause=$((pause * 2))
    [ $((now + pause)) -le "$deadline" ] || pause=$((deadline - now))
done
