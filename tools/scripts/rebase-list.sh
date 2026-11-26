#!/bin/sh
# Finds the open pull requests that conflict with MAIN and plans a
# rebase of each, see tools/rebase.
# Prints "matrix=<json>" with each pull request the plan resolves.
# Comments on a pull request that the plan refuses.
# Reads pull requests with the priority or landing label, and skips
# drafts and the no-auto-rebase label. ANY_TIER=true drops the label
# check. Each skip writes its reason to stderr.
# A fork must allow edits by maintainers. A conflicted fork that does
# not gets a comment.
# Skips a build that failed before for the same head and conflicts.
# Does not run pull request code.
# Needs GH_TOKEN, GITHUB_REPOSITORY, MAIN, ANY_TIER, TOOL and
# RUNNER_TEMP.
# PR limits the run to one pull request.
set -e
repo=repos/$GITHUB_REPOSITORY
base=$(git rev-parse "origin/$MAIN")
tmp=$(mktemp -d)
# jq reads ANY_TIER as code, so it must be a plain boolean.
[ "$ANY_TIER" = true ] || ANY_TIER=false
pick='"\(.number) \(.head.sha) \(.head.repo.full_name != .base.repo.full_name) \(if .state != "open" then "is not open"
    elif .draft then "is a draft"
    elif .base.ref != "'$MAIN'" then "does not target '$MAIN'"
    elif .head.repo == null then "has a deleted head repository"
    elif any(.labels[]; .name == "no-auto-rebase") then "has the no-auto-rebase label"
    elif '$ANY_TIER' then ""
    elif any(.labels[]; .name == "priority" or .name == "landing") then ""
    else "has no priority or landing label" end)"'
if [ -n "$PR" ]; then
    gh api "$repo/pulls/$PR" --jq "$pick" >"$tmp/pulls"
else
    gh api --paginate "$repo/pulls?state=open&per_page=100" --jq ".[] | $pick" >"$tmp/pulls"
fi

status=0
found=
while read -r n sha fork why; do
    if [ -n "$why" ]; then
        echo "#$n $why. Skipped." >&2
        continue
    fi
    # A fork head is in this repository under refs/pull.
    git fetch -q origin "refs/pull/$n/head"
    if ! git cat-file -e "$sha^{commit}"; then
        echo "#$n moved while this run read it. Skipped." >&2
        continue
    fi
    # merge-tree exits 1 on a conflict and above 1 on an error.
    code=0
    git merge-tree --write-tree --name-only --no-messages "$base" "$sha" >"$tmp/merge" || code=$?
    report=$tmp/report-$n.md
    echo '<!-- list-rebase -->' >"$report"
    if [ "$code" -eq 0 ]; then
        echo "None found. This pull request merges with \`$MAIN\`." >>"$report"
    elif [ "$code" -eq 1 ] && [ "$fork" = true ] &&
        [ "$(gh api "$repo/pulls/$n" --jq .maintainer_can_modify)" != true ]; then
        echo "This pull request conflicts with \`$MAIN\`. It comes from a fork that does not allow edits by maintainers, so the rebase bot cannot push to it. The rebase bot changed nothing." >>"$report"
        code=0
    elif [ "$code" -eq 1 ]; then
        # tessera-rebase exits 2 when it refuses.
        code=0
        out=$("$TOOL" plan "$base" "$sha" "$RUNNER_TEMP/rebase-$n") || code=$?
        if [ "$code" -eq 0 ]; then
            # The key names the conflicting files and main's copy of each.
            key=$(sed 1d "$tmp/merge" | while IFS= read -r path; do
                git rev-parse -q --verify "$base:$path" || true
                printf '%s\n' "$path"
            done | git hash-object --stdin)
            if gh api --paginate "$repo/issues/$n/comments" \
                --jq '.[] | select(.body | startswith("<!-- list-rebase -->")) | .body' |
                grep -q -F "<!-- failed $sha $key -->"; then
                echo "#$n: this head failed to build on the same conflicts. Skipped." >&2
            else
                found="$found{\"pr\":$n,\"head\":\"$sha\",\"base\":\"$base\",\"key\":\"$key\"},"
            fi
            continue
        fi
        echo "This pull request conflicts with \`$MAIN\`. The rebase bot changed nothing:" >>"$report"
        printf '%s\n' "$out" | sed -n 's/^refused //p' >>"$report"
    fi
    if [ "$code" -ne 0 ] && [ "$code" -ne 2 ]; then
        echo "#$n: the plan failed." >&2
        status=1
        continue
    fi
    PR=$n tools/scripts/pr-comment.sh "$report"
done <"$tmp/pulls"
echo "matrix=[${found%,}]"
exit $status
