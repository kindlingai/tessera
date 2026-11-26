#!/bin/sh
# Prints true when the registry holds $CI_IMAGE:<backend>-<tree> for each
# backend in the arguments, where <tree> is the tree of HEAD. Else
# prints false. The docker job pushes that tag only after the image
# passes its checks. Needs CI_IMAGE (ghcr.io/<owner>/<name>) and GH_TOKEN
# with packages: read.
set -e
tree=$(git rev-parse 'HEAD^{tree}')
repo=${CI_IMAGE#ghcr.io/}
token=$(curl -fsS --max-time 30 -u "token:$GH_TOKEN" \
    "https://ghcr.io/token?scope=repository:$repo:pull" |
    sed -nE 's/.*"token":"([^"]+)".*/\1/p') || token=
for backend in "$@"; do
    if [ -z "$token" ] || ! curl -fsS -I -o /dev/null --max-time 30 \
        -H "Authorization: Bearer $token" \
        -H 'Accept: application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.index.v1+json' \
        "https://ghcr.io/v2/$repo/manifests/$backend-$tree"; then
        echo "$CI_IMAGE:$backend-$tree: not found" >&2
        echo false
        exit 0
    fi
    echo "$CI_IMAGE:$backend-$tree: found" >&2
done
echo true
