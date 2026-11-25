# Sourced by a hook after worktree.sh, with hook_limit set to a number of seconds.
# Runs the hook again in its own process group and stops the group at the limit.
# TESSERA_HOOK_LIMIT overrides the seconds. 0 turns the limit off.
hook_limit=${TESSERA_HOOK_LIMIT:-$hook_limit}
if [ -z "${TESSERA_HOOK_LIMITED:-}" ] && [ "$hook_limit" -gt 0 ]; then
    TESSERA_HOOK_LIMITED=1 exec perl -e '
        my $limit = shift;
        my $name = $ARGV[0] =~ s{.*/}{}r;
        my $pid = fork // die "$name: fork: $!\n";
        if (!$pid) {
            setpgrp(0, 0);
            exec @ARGV or die "$name: exec: $!\n";
        }
        $SIG{INT} = $SIG{TERM} = sub { kill "TERM", -$pid; exit 130 };
        $SIG{ALRM} = sub {
            kill "TERM", -$pid;
            print STDERR "$name: stopped after $limit s. Set TESSERA_HOOK_LIMIT for more seconds.\n";
            exit 124;
        };
        alarm $limit;
        waitpid $pid, 0;
        exit($? & 127 ? 128 + ($? & 127) : $? >> 8);
    ' "$hook_limit" "$0" "$@"
fi
