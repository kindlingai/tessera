#!/bin/sh
# Runs `cargo fmt --check` with the given arguments. When files that this
# branch does not change fail, and the base branch changed them after this
# branch forked, it says to rebase. It fails in both cases.
# FMT_BASE names the base branch. Default: origin/main.
base=${FMT_BASE:-origin/main}
cd "$(git rev-parse --show-toplevel)" || exit 1
cargo fmt "$@" --check && exit 0
fork=$(git merge-base "$base" HEAD 2>/dev/null) || exit 1
mine=$(git diff --name-only "$fork" && git ls-files --others --exclude-standard)
stale=$(cargo fmt "$@" --check -- -l 2>/dev/null | while IFS= read -r path; do
    path=${path#"$PWD"/}
    printf '%s\n' "$mine" | grep -qxF "$path" && continue
    [ -n "$(git log -1 --format=%h "$fork..$base" -- "$path")" ] && echo "$path"
done)
[ -n "$stale" ] || exit 1
echo
echo "These files fail formatting, but this branch does not change them:"
printf '%s\n' "$stale" | sed 's/^/  /'
echo "$base changed them after this branch forked:"
printf '%s\n' "$stale" | xargs git log --format='  %h %s' "$fork..$base" --
echo "Rebase on $base to get these changes."
if [ -n "$GITHUB_ACTIONS" ]; then
    echo "::error title=Stale base::Formatting fails in files this branch does not change. $base changed them after this branch forked. Rebase on $base."
fi
exit 1
