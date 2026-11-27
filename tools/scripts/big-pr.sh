#!/bin/sh
# Prints "big-pr <lines>" when the PR adds and deletes more than
# BIG_PR_LINES lines in all. protected-label.sh reads the output.
# Needs GH_TOKEN, PR, BIG_PR_LINES and GITHUB_REPOSITORY.
set -e
lines=$(gh api "repos/$GITHUB_REPOSITORY/pulls/$PR" --jq '.additions + .deletions')
[ "$lines" -le "$BIG_PR_LINES" ] || echo "big-pr $lines"
