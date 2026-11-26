#!/bin/sh
# Formats, refreshes the lock and builds the tree in the current
# directory. rebase-build.sh runs this file, in a container on CI.
# PINS names a file of "<package> <version>" lines that the lock
# must keep. FEATURES holds cargo feature flags.
# The exit code names the step that failed.
cargo fmt --all || exit 10
cargo metadata --format-version 1 >/dev/null || exit 11
while read -r name version; do
    # The lock lists a version on the line after its package name.
    if ! grep -A 1 -x "name = \"$name\"" Cargo.lock | grep -q -x "version = \"$version\""; then
        cargo update --package "$name" --precise "$version" || exit 11
    fi
done <"$PINS"
# shellcheck disable=SC2086
cargo check --workspace --all-targets $FEATURES || exit 12
# shellcheck disable=SC2086
cargo clippy --workspace --all-targets $FEATURES || exit 13
