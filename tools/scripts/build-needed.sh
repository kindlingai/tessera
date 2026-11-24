#!/bin/sh
# Reads changed paths on stdin. Prints true if any path needs the
# build jobs: anything other than Markdown, release-please metadata,
# lint code or lint.toml.
# No paths prints true.
need=true
while IFS= read -r path || [ -n "$path" ]; do
    case "$path" in
    *.md | .release-please-manifest.json | release-please-config.json | tools/lint/* | lint.toml) need=false ;;
    *)
        echo true
        exit 0
        ;;
    esac
done
echo "$need"
