#!/bin/sh
# Cargo runs every binary through this script (.cargo/config.toml).
# Test binaries must run under nextest. nextest kills a test after 60 s.
# Plain `cargo test` has no per-test timeout, so this script stops it.
case "$1" in
*/deps/*)
	if [ -z "$NEXTEST" ]; then
		for arg in "$@"; do
			[ "$arg" = "--bench" ] && exec "$@"
		done
		echo "error: run tests with \`cargo nextest run\`, not \`cargo test\`." >&2
		echo "nextest kills any test that runs longer than 60 s." >&2
		exit 1
	fi
	;;
esac
exec "$@"
