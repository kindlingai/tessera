#!/bin/sh
# Fails on Co-Authored-By trailers that credit an AI tool or bot.
# Usage: check-coauthors.sh <rev-range>
#        check-coauthors.sh --message <file>
set -e

bots='anthropic|claude|openai|chatgpt|codex|z\.ai|zhipu|glm|copilot|gemini|cursor|devin|aider|\[bot\]'
pattern="^[[:space:]]*co-authored-by:.*($bots)"

# Prints offending lines from stdin. True if any.
offending() {
    grep -v '^#' | grep -Ei "$pattern"
}

if [ "$1" = "--message" ]; then
    if offending <"$2"; then
        echo "Remove Co-Authored-By lines that credit AI tools." >&2
        exit 1
    fi
    exit 0
fi

if [ -z "$1" ]; then
    echo "usage: $0 <rev-range> | --message <file>" >&2
    exit 2
fi

failed=0
for c in $(git rev-list "$1"); do
    lines=$(git log -1 --format=%B "$c" | offending) || continue
    echo "$(git log -1 --format='%h %s' "$c")" >&2
    echo "$lines" | sed 's/^/    /' >&2
    failed=1
done
if [ "$failed" = 1 ]; then
    echo "Remove Co-Authored-By lines that credit AI tools." >&2
fi
exit "$failed"
