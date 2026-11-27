# Sourced by a hook after worktree.sh, with hook_limit set to a number of seconds.
# Runs the hook again under limit.pl, which stops it at the limit.
# TESSERA_HOOK_LIMIT overrides the seconds. 0 turns the limit off.
hook_limit=${TESSERA_HOOK_LIMIT:-$hook_limit}
if [ -z "${TESSERA_HOOK_LIMITED:-}" ] && [ "$hook_limit" -gt 0 ]; then
    TESSERA_HOOK_LIMITED=1 exec perl "$(dirname "$0")/limit.pl" hook "$hook_limit" "$0" "$@"
fi
