# Sourced by each hook. Git runs hooks from the worktree root.
# When core.hooksPath names another checkout, this runs the
# worktree's own copy of the hook instead.
own="$PWD/.githooks/$(basename "$0")"
here=$(cd "$(dirname "$0")" && pwd -P)
mine=$(cd "$PWD/.githooks" 2>/dev/null && pwd -P)
if [ "$here" != "$mine" ] && [ -x "$own" ]; then
    exec "$own" "$@"
fi
