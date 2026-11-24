#!/usr/bin/env bash
# Runs the stickler release that lint.toml names, with the given arguments.
# The first run downloads it to ~/.cache/tessera/stickler and checks its hash.
set -euo pipefail

root=$(git rev-parse --show-toplevel)
key() {
    sed -n "s/^stickler\.$1 = \"\(.*\)\"$/\1/p" "$root/lint.toml"
}
version=$(key version)
releases=$(key releases)
[ -n "$version" ] && [ -n "$releases" ] || {
    echo "stickler.sh: lint.toml needs stickler.version and stickler.releases" >&2
    exit 2
}

case "$(uname -s)-$(uname -m)" in
Linux-x86_64) platform=linux-x64 ;;
Linux-aarch64) platform=linux-arm64 ;;
Darwin-arm64) platform=darwin-arm64 ;;
*)
    echo "stickler.sh: no stickler release for $(uname -s) $(uname -m)" >&2
    exit 2
    ;;
esac

dir="${XDG_CACHE_HOME:-$HOME/.cache}/tessera/stickler/$version"
bin="$dir/stickler"
if [ ! -x "$bin" ]; then
    part="$dir.part.$$"
    rm -rf "$part"
    mkdir -p "$part"
    trap 'rm -rf "$part"' EXIT
    zip="stickler-$platform.zip"
    base="$releases/v$version"
    curl -fsSL --max-time 300 -o "$part/$zip" "$base/$zip"
    curl -fsSL --max-time 60 -o "$part/SHA256SUMS" "$base/SHA256SUMS"
    if command -v sha256sum >/dev/null; then
        sum=(sha256sum)
    else
        sum=(shasum -a 256)
    fi
    (cd "$part" && grep " $zip\$" SHA256SUMS | "${sum[@]}" -c --status) || {
        echo "stickler.sh: $base/$zip does not match SHA256SUMS" >&2
        exit 2
    }
    unzip -q "$part/$zip" -d "$part"
    rm -rf "$dir"
    mv "$part" "$dir"
    trap - EXIT
fi
exec "$bin" "$@"
