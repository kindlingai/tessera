#!/bin/sh
# Points git at the repository's hooks in .githooks.
set -e
git config core.hooksPath .githooks
echo "hooks installed: pre-commit, commit-msg (.githooks)"
