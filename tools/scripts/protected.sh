#!/bin/sh
# Prints "<label> <path>" for each path on stdin that needs a human
# review. The label names the kind of change: rules and checkers, CI,
# hooks and scripts, the toolchain, or markdown.
while IFS= read -r path; do
    case "$path" in
    stickler/* | lint.toml | clippy.toml | */clippy.toml | rustfmt.toml | \
        */rustfmt.toml | *allow.toml) label=changes-lint-rules ;;
    .github/* | .githooks/* | tools/scripts/*) label=changes-ci ;;
    rust-toolchain*) label=changes-toolchain ;;
    *.md) label=changes-docs ;;
    *) continue ;;
    esac
    echo "$label $path"
done
