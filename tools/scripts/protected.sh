#!/bin/sh
# Prints each path on stdin that needs a human review.
# Covers rules, checkers, CI, hooks, allow lists and markdown.
while IFS= read -r path; do
    case "$path" in
    *.md | .github/* | .githooks/* | tools/scripts/* | \
        stickler/* | \
        lint.toml | clippy.toml | */clippy.toml | rustfmt.toml | */rustfmt.toml | \
        rust-toolchain* | *allow.toml)
        echo "$path"
        ;;
    esac
done
