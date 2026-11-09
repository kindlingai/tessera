#!/bin/sh
# Points git at the repository's hooks in .githooks.
set -e
git config core.hooksPath .githooks
echo "pre-commit hook installed (.githooks/pre-commit)"
