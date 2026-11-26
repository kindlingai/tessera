#!/bin/sh
# Builds the list rebase of HEAD on BASE, see tools/rebase.
# Leaves in OUT either "tree" and the files that had a conflict, or
# "failed" with the step that failed.
# This is the one script that runs pull request code: build scripts
# and proc macros. SANDBOX=docker runs them in a container that sees
# the work tree and nothing else of the runner.
# Needs TOOL, BASE, HEAD, WORK and OUT. FEATURES holds cargo flags.
set -e
tmp=$(mktemp -d)
rm -rf "$OUT"
mkdir -p "$OUT/files"
"$TOOL" open "$BASE" "$HEAD" "$WORK" >"$tmp/plan"
gitdir=$(git -C "$WORK" rev-parse --absolute-git-dir)
root=$(cd "$WORK" && pwd -P)
sed -n 's/^path //p' "$tmp/plan" >"$tmp/paths"
sed -n 's/^pin //p' "$tmp/plan" >"$tmp/pins"
steps=$(pwd)/tools/scripts/rebase-cargo.sh

code=0
if [ "$SANDBOX" = docker ]; then
    channel=$(sed -n 's/^channel = "\(.*\)"$/\1/p' rust-toolchain.toml)
    docker run --rm -v "$root:/work" -w /work \
        -v "$steps:/steps.sh:ro" -v "$tmp/pins:/pins:ro" -e PINS=/pins \
        -e FEATURES -e RUSTFLAGS -e CARGO_INCREMENTAL \
        "rust:$channel" sh /steps.sh || code=$?
else
    (cd "$root" && PINS=$tmp/pins sh "$steps") || code=$?
fi

# The work tree is not trusted from here. git reads it through the
# git directory found above, not through the tree's own .git file.
fail() {
    echo "$1" >"$OUT/failed"
    echo "The build failed at: $1" >&2
    exit 1
}
case $code in
0) ;;
10) fail fmt ;;
11) fail lock ;;
12) fail check ;;
13) fail clippy ;;
*) fail build ;;
esac
git --git-dir="$gitdir" --work-tree="$root" -c core.hooksPath=/dev/null -c core.fsmonitor=false \
    status --porcelain --no-renames | cut -c 4- >"$tmp/changed"
if grep -v -x -F -f "$tmp/paths" "$tmp/changed" >&2; then
    fail stray
fi
while IFS= read -r path; do
    file=$root/$path
    if [ -f "$file" ] && [ ! -L "$file" ]; then
        dir=$(cd "$(dirname "$file")" && pwd -P)
        case $dir/ in
        "$root"/*)
            mkdir -p "$(dirname "$OUT/files/$path")"
            cp "$file" "$OUT/files/$path"
            ;;
        esac
    fi
done <"$tmp/paths"
sed -n 's/^tree //p' "$tmp/plan" >"$OUT/tree"
