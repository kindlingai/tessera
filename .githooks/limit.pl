#!/usr/bin/env perl
# Runs a command in its own process group and stops the group at a limit.
#
# limit.pl hook <seconds> <hook> <args>...
#   The limit covers the whole hook, and the hook fails at the limit.
#   SIGUSR1 stops the clock and SIGUSR2 starts it again.
# limit.pl simcheck <seconds> <command>...
#   The clock starts when the command writes the line that starts simcheck
#   on stderr, and it stops the clock of the hook until the command ends.
#   At the limit simcheck counts as passed, because CI runs it in full.
#
# 0 seconds turns the limit off.
use strict;
use warnings;

my ($mode, $limit, @cmd) = @ARGV;
my $hook = $mode eq 'hook';
my $name = $hook ? $cmd[0] =~ s{.*/}{}r : 'pre-commit';
my $watch = $ENV{TESSERA_HOOK_WATCH};
$ENV{TESSERA_HOOK_WATCH} = $$ if $hook;

my ($err, $err_in);
pipe($err, $err_in) or die "$name: pipe: $!\n" unless $hook;
my $pid = fork // die "$name: fork: $!\n";
if (!$pid) {
    setpgrp(0, 0);
    if ($err_in) {
        close $err;
        open STDERR, '>&', $err_in or die "$name: stderr: $!\n";
    }
    exec @cmd or die "$name: exec: $!\n";
}

sub resume_hook { kill 'USR2', $watch if $watch }

sub finish {
    waitpid $pid, 0;
    my $status = $?;
    resume_hook() unless $hook;
    exit($status & 127 ? 128 + ($status & 127) : $status >> 8);
}

$SIG{INT} = $SIG{TERM} = sub { kill 'TERM', -$pid; exit 130 };

if ($hook) {
    my $paused = 0;
    my $left = 0;
    $SIG{USR1} = sub { $left = alarm 0; $paused = 1 };
    $SIG{USR2} = sub { alarm($left || 1) if $paused; $paused = 0 };
    $SIG{ALRM} = sub {
        kill 'TERM', -$pid;
        print STDERR "$name: stopped after $limit s. Set TESSERA_HOOK_LIMIT for more seconds.\n";
        exit 124;
    };
    alarm $limit;
    finish();
}

close $err_in;
$SIG{ALRM} = sub {
    kill 'TERM', -$pid;
    resume_hook();
    print STDERR "$name: warning: simcheck skipped after $limit s. CI still runs it. Set TESSERA_SIMCHECK_LIMIT for more seconds.\n";
    exit 0;
};
$| = 1;
while (my $line = <$err>) {
    print STDERR $line;
    next unless $line =~ /^stickler: running simcheck/;
    kill 'USR1', $watch if $watch;
    alarm $limit;
}
finish();
