#!/bin/sh
# Reads a PR's files as JSON on stdin, from the GitHub files API.
# Prints "big-pr <lines>" when the PR adds and deletes more than
# BIG_PR_LINES lines in all, without the files that match
# .github/generated-paths.txt. protected-label.sh reads the output.
# protected.yml runs this on main, so main's list applies.
# The files API gives 0 lines for files past its diff size limit, so
# the total comes from the PR and only generated files are subtracted.
# Needs GH_TOKEN, PR, BIG_PR_LINES and GITHUB_REPOSITORY.
set -e
re=$(grep -v '^#' .github/generated-paths.txt | paste -sd'|' -)
total=$(gh api "repos/$GITHUB_REPOSITORY/pulls/$PR" --jq '.additions + .deletions')
generated=$(jq -s --arg re "$re" '[.[][] | select(.filename | test($re)) | .additions + .deletions] | add // 0')
lines=$((total - generated))
[ "$lines" -le "$BIG_PR_LINES" ] || echo "big-pr $lines"
