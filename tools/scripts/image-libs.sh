#!/bin/sh
# Runs inside the tesserax image. Fails unless the dynamic loader finds
# each library named in the arguments, with all of its dependencies.
# The names are the sonames that tesserax opens at runtime.
set -e
for lib in "$@"; do
    path=$(ldconfig -p | awk -v lib="$lib" '$1 == lib { print $NF; exit }')
    if [ -z "$path" ]; then
        echo "$lib: not found" >&2
        exit 1
    fi
    echo "$lib => $path"
    if ldd "$path" | grep 'not found'; then
        exit 1
    fi
done
