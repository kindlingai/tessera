#!/bin/sh
# Runs non-fail-fast nextest ci gates with isolated target dir and sccache.
set -eu

root=$(git rev-parse --show-toplevel)
worktree=$(cd "$root" && pwd -P)

hash_stdin() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum | awk '{print $1}'
    else
        shasum -a 256 | awk '{print $1}'
    fi
}

wt_name=$(basename "$worktree")
wt_hash=$(printf '%s' "$worktree" | hash_stdin | cut -c1-10)
wt_tag="${wt_name}-${wt_hash}"

if [ -n "${TESSERA_TARGET_DIR:-}" ]; then
    mkdir -p "$TESSERA_TARGET_DIR"
    target_dir=$(cd "$TESSERA_TARGET_DIR" && pwd -P)
elif [ -n "${CARGO_TARGET_DIR:-}" ]; then
    mkdir -p "$CARGO_TARGET_DIR"
    target_in=$(cd "$CARGO_TARGET_DIR" && pwd -P)
    case "$target_in" in
        "$worktree"/*|*"/worktrees/$wt_tag")
            target_dir="$target_in"
            ;;
        *)
            target_dir="$target_in/worktrees/$wt_tag"
            ;;
    esac
elif [ -n "${TESSERA_TARGET_BASE:-}" ]; then
    mkdir -p "$TESSERA_TARGET_BASE"
    target_base=$(cd "$TESSERA_TARGET_BASE" && pwd -P)
    target_dir="$target_base/$wt_tag"
else
    target_dir="$worktree/target"
fi

mkdir -p "$target_dir"
target_dir=$(cd "$target_dir" && pwd -P)
export CARGO_TARGET_DIR="$target_dir"

if [ -z "${RUSTC_WRAPPER:-}" ] && command -v sccache >/dev/null 2>&1; then
    export RUSTC_WRAPPER=sccache
fi

if ! cargo nextest --version >/dev/null 2>&1; then
    echo "error: cargo-nextest is required for gpu-gate.sh." >&2
    echo "install: cargo install cargo-nextest --locked" >&2
    exit 127
fi

has_scope=0
has_profile=0
has_fail_fast=0
profile=ci
expect_profile=0
for arg in "$@"; do
    if [ "$expect_profile" -eq 1 ]; then
        profile="$arg"
        expect_profile=0
        continue
    fi
    case "$arg" in
        --)
            break
            ;;
        -p|--package|--package=*|--workspace|--manifest-path|--manifest-path=*)
            has_scope=1
            ;;
        -P|--profile)
            has_profile=1
            expect_profile=1
            ;;
        --profile=*)
            has_profile=1
            profile="${arg#--profile=}"
            ;;
        --fail-fast|--no-fail-fast)
            has_fail_fast=1
            ;;
    esac
done

git_head=$(git -C "$worktree" rev-parse --short=12 HEAD)
printf '=== tessera gpu-gate ===\n'
printf 'worktree: %s\n' "$worktree"
printf 'git_head: %s\n' "$git_head"
printf 'target:   %s\n' "$CARGO_TARGET_DIR"
printf 'wrapper:  %s\n' "${RUSTC_WRAPPER:-none}"

set -- "$@"
if [ "$has_scope" -eq 0 ]; then
    set -- --workspace "$@"
fi
if [ "$has_fail_fast" -eq 0 ]; then
    set -- --no-fail-fast "$@"
fi
if [ "$has_profile" -eq 0 ]; then
    set -- --profile ci "$@"
fi

cd "$worktree"
junit="$CARGO_TARGET_DIR/nextest/$profile/junit.xml"
rm -f "$junit"

set +e
cargo nextest run "$@"
status=$?
set -e

if [ -f "$junit" ]; then
    printf 'junit:    %s\n' "$junit"
else
    printf 'junit:    (none)\n'
fi
if [ "$status" -eq 0 ]; then
    printf 'result:   PASS\n'
else
    printf 'result:   FAIL (exit %s)\n' "$status"
fi
printf '========================\n'
exit "$status"
