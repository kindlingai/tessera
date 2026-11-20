#!/bin/sh
# Prints true when a merge group holds the same tree as its pull
# request's head, and that head passed `success`. Else prints false.
# Needs GH_TOKEN, GITHUB_REPOSITORY and HEAD_REF, the merge group's
# `gh-readonly-queue/<base>/pr-<n>-<sha>` branch. A group of several
# pull requests never matches one head's tree.
set -e
pr=$(printf '%s\n' "$HEAD_REF" | sed -nE 's|^gh-readonly-queue/.+/pr-([0-9]+)-[0-9a-f]+$|\1|p')
head=$(gh api "repos/$GITHUB_REPOSITORY/pulls/$pr" --jq .head.sha 2>/dev/null) || head=
reuse=false
if [ -n "$pr" ] && [ -n "$head" ] && git fetch -q origin "$head" 2>/dev/null &&
    [ "$(git rev-parse "HEAD^{tree}")" = "$(git rev-parse "$head^{tree}")" ]; then
    passed=$(gh api "repos/$GITHUB_REPOSITORY/commits/$head/check-runs?check_name=success" \
        --jq '[.check_runs[] | select(.conclusion == "success")] | length')
    if [ "$passed" -gt 0 ]; then
        reuse=true
    fi
fi
echo "pull request: ${pr:-none}, head: ${head:-none}, reuse: $reuse" >&2
echo "$reuse"
