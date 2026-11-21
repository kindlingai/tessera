#!/bin/sh
# Runs clippy, as CI does, on each crate that holds a staged file.
# Markdown and files outside a crate do not count.
# tools/simcheck is its own workspace, so it runs alone.
set -e
cd "$(git rev-parse --show-toplevel)"
dirs=$(git diff --cached --name-only | grep -v '\.md$' | while IFS= read -r path; do
    dir=$(dirname "$path")
    while [ "$dir" != . ] && [ ! -f "$dir/Cargo.toml" ]; do
        dir=$(dirname "$dir")
    done
    [ "$dir" != . ] && echo "$dir"
done | sort -u)
packages=
for dir in $dirs; do
    if grep -q '^\[workspace\]' "$dir/Cargo.toml"; then
        cargo clippy -q --manifest-path "$dir/Cargo.toml" --all-targets -- -D warnings
    else
        name=$(sed -n 's/^name = "\(.*\)"$/\1/p' "$dir/Cargo.toml" | head -n 1)
        packages="$packages -p $name"
    fi
done
[ -z "$packages" ] && exit 0
# Unlike RUSTFLAGS, `-D warnings` here does not rebuild dependencies.
cargo clippy -q $packages --all-targets -- -D warnings
