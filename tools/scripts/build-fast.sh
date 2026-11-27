#!/bin/sh
# Builds targets with sccache, per-worktree target isolation, and a SHA banner.
set -eu

root=$(git rev-parse --show-toplevel)
worktree=$(cd "$root" && pwd -P)

hash_stdin() {
    if command -v sha256sum >/dev/null 2>&1; then
        raw=$(sha256sum) || return 2
    else
        raw=$(shasum -a 256) || return 2
    fi
    digest="${raw%% *}"
    [ -n "$digest" ] || return 2
    printf '%s\n' "$digest"
}

hash_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        raw=$(sha256sum < "$1") || return 2
    else
        raw=$(shasum -a 256 < "$1") || return 2
    fi
    digest="${raw%% *}"
    [ -n "$digest" ] || return 2
    printf '%s\n' "$digest"
}

wt_name=$(basename "$worktree")
wt_digest=$(printf '%s' "$worktree" | hash_stdin)
wt_hash=$(printf '%s' "$wt_digest" | cut -c1-10)
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
target_abs=$(cd "$target_dir" && pwd -P)
export CARGO_TARGET_DIR="$target_abs"

if [ -z "${RUSTC_WRAPPER:-}" ] && command -v sccache >/dev/null 2>&1; then
    export RUSTC_WRAPPER=sccache
fi

if [ "$#" -eq 0 ]; then
    set -- --release -p tessera-cli --bin tesserax
fi

for arg in "$@"; do
    case "$arg" in
        --)
            break
            ;;
        --message-format|--message-format=*)
            echo "error: build-fast.sh manages --message-format to attest binaries; omit it." >&2
            exit 2
            ;;
    esac
done

target_rel=""
case "$target_abs" in
    "$worktree"/*)
        target_rel="${target_abs#"$worktree"/}"
        ;;
esac

tree_identity() {
    head_sha=$(git rev-parse HEAD)
    untracked_z=$(mktemp "${TMPDIR:-/tmp}/tessera-build-untracked.XXXXXX")
    untracked_manifest=$(mktemp "${TMPDIR:-/tmp}/tessera-build-umanifest.XXXXXX")
    manifest=$(mktemp "${TMPDIR:-/tmp}/tessera-build-manifest.XXXXXX")

    git -c core.quotePath=false ls-files --others --exclude-standard -z >"$untracked_z"

    # Hash untracked files outside target_rel safely using NUL-delimited paths.
    if [ -s "$untracked_z" ]; then
        if ! xargs -0 sh -eu -c '
            target_rel="$1"
            shift
            for rel in "$@"; do
                if [ -n "$target_rel" ]; then
                    case "$rel" in
                        "$target_rel"|"$target_rel"/*) continue ;;
                    esac
                fi
                path="./$rel"
                if [ -L "$path" ]; then
                    printf "L %s\0" "$rel"
                    readlink "$path" || exit 2
                    printf "\0"
                elif [ -f "$path" ]; then
                    if [ ! -r "$path" ]; then
                        echo "error: build-fast.sh cannot read untracked file: $rel" >&2
                        exit 2
                    fi
                    if command -v sha256sum >/dev/null 2>&1; then
                        raw=$(sha256sum < "$path") || exit 2
                    else
                        raw=$(shasum -a 256 < "$path") || exit 2
                    fi
                    digest="${raw%% *}"
                    [ -n "$digest" ] || exit 2
                    printf "F %s\0%s\0" "$rel" "$digest"
                else
                    echo "error: build-fast.sh cannot hash untracked path: $rel" >&2
                    exit 2
                fi
            done
        ' _ "$target_rel" <"$untracked_z" >"$untracked_manifest"; then
            rm -f "$untracked_z" "$untracked_manifest" "$manifest"
            return 2
        fi
    fi

    has_tracked=0
    if ! git diff --quiet HEAD -- 2>/dev/null || ! git diff --cached --quiet -- 2>/dev/null; then
        has_tracked=1
    fi
    status_tracked=$(git -c core.quotePath=false status --porcelain --untracked-files=no)

    if [ "$has_tracked" -eq 1 ] || [ -n "$status_tracked" ] || [ -s "$untracked_manifest" ]; then
        {
            git diff --no-ext-diff HEAD
            printf '\n--staged--\n'
            git diff --no-ext-diff --cached
            printf '\n--status--\n%s\n--untracked--\n' "$status_tracked"
            cat "$untracked_manifest"
        } >"$manifest"
        if ! digest=$(hash_file "$manifest") || [ -z "$digest" ]; then
            rm -f "$untracked_z" "$untracked_manifest" "$manifest"
            return 2
        fi
        diff_hash=$(printf '%s' "$digest" | cut -c1-12)
        rm -f "$untracked_z" "$untracked_manifest" "$manifest"
        printf '%s dirty diff_sha256=%s' "$head_sha" "$diff_hash"
    else
        rm -f "$untracked_z" "$untracked_manifest" "$manifest"
        printf '%s clean' "$head_sha"
    fi
}

msg_log=$(mktemp "${TMPDIR:-/tmp}/tessera-build-fast.XXXXXX")
trap 'rm -f "$msg_log"' EXIT

cd "$worktree"
pre_id=$(tree_identity)
cargo build --message-format=json-render-diagnostics "$@" >"$msg_log"
post_id=$(tree_identity)

if [ "$pre_id" != "$post_id" ]; then
    echo "error: worktree changed during build; refusing to attest provenance." >&2
    echo "  pre:  $pre_id" >&2
    echo "  post: $post_id" >&2
    exit 2
fi

git_head=$(printf '%s' "${pre_id%% *}" | cut -c1-12)
tree_state="${pre_id#* }"

printf '\n=== tessera build-fast ===\n'
printf 'worktree: %s\n' "$worktree"
printf 'git_head: %s (%s)\n' "$git_head" "$tree_state"
printf 'target:   %s\n' "$CARGO_TARGET_DIR"
printf 'wrapper:  %s\n' "${RUSTC_WRAPPER:-none}"

bins=""
if [ -s "$msg_log" ]; then
    if command -v jq >/dev/null 2>&1; then
        bins=$(jq -r 'select(.reason == "compiler-artifact" and .executable != null) | .executable' "$msg_log" | sort -u)
    else
        bins=$(grep '"reason":"compiler-artifact"' "$msg_log" \
            | sed -n 's/.*"executable":"\([^"]*\)".*/\1/p' \
            | sort -u)
    fi
fi

if [ -n "$bins" ]; then
    printf '%s\n' "$bins" | while IFS= read -r bin; do
        if [ -f "$bin" ]; then
            sha=$(hash_file "$bin")
            printf 'binary:   %s\n' "$bin"
            printf 'sha256:   %s\n' "$sha"
        fi
    done
else
    printf 'binary:   (none)\n'
fi
printf '==========================\n'
