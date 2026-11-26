#!/bin/sh
# Test-merges <head> into <base> merged with each <other>.
# Prints "<other> <path>" for each conflicting path.
# Exits 1 when any path conflicts and 2 when a rev is not a commit.
# Needs only git objects.
# Usage: merge-conflicts.sh <base> <head> <other>...
set -e
base=$1
head=$2
shift 2
# commit-tree needs a name. The commits stay unreferenced.
export GIT_AUTHOR_NAME=merge-conflicts GIT_AUTHOR_EMAIL=none
export GIT_COMMITTER_NAME=merge-conflicts GIT_COMMITTER_EMAIL=none
# merge-tree also exits 1 on a rev it cannot read.
for rev in "$base" "$head" "$@"; do
    if ! git rev-parse -q --verify "$rev^{commit}" >/dev/null; then
        echo "$rev is not a commit in this repository." >&2
        exit 2
    fi
done
status=0
for other in "$@"; do
    # merge-tree exits 1 on a conflict and above 1 on an error.
    code=0
    tree=$(git merge-tree --write-tree --no-messages "$base" "$other") || code=$?
    [ "$code" -le 1 ] || exit "$code"
    if [ "$code" -eq 1 ]; then
        echo "$other does not merge with $base. Skipped." >&2
        continue
    fi
    both=$(git commit-tree "$tree" -p "$base" -p "$other" -m test-merge)
    code=0
    out=$(git merge-tree --write-tree --name-only --no-messages "$both" "$head") || code=$?
    [ "$code" -le 1 ] || exit "$code"
    if [ "$code" -eq 1 ]; then
        printf '%s\n' "$out" | sed -n "2,\$s|^\(..*\)|$other \1|p"
        status=1
    fi
done
exit $status
