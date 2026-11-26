#!/bin/sh
# Finds the open pull requests that conflict with MAIN and plans a
# list rebase of each, see tools/rebase.
# Prints "matrix=<json>" with each pull request the plan resolves.
# Comments on a pull request that the plan refuses.
# Skips forks, drafts and the no-auto-rebase label.
# Does not run pull request code.
# Needs GH_TOKEN, GITHUB_REPOSITORY, MAIN, TOOL and RUNNER_TEMP.
# PR limits the run to one pull request.
set -e
repo=repos/$GITHUB_REPOSITORY
base=$(git rev-parse "origin/$MAIN")
tmp=$(mktemp -d)
pick='select(.state == "open" and .draft == false
    and .base.ref == "'$MAIN'"
    and .head.repo.full_name == .base.repo.full_name
    and (.labels | all(.name != "no-auto-rebase")))
    | "\(.number) \(.head.sha)"'
if [ -n "$PR" ]; then
    gh api "$repo/pulls/$PR" --jq "$pick" >"$tmp/pulls"
else
    gh api --paginate "$repo/pulls?state=open&per_page=100" --jq ".[] | $pick" >"$tmp/pulls"
fi

status=0
found=
while read -r n sha; do
    git fetch -q origin "$sha"
    # merge-tree exits 1 on a conflict and above 1 on an error.
    code=0
    git merge-tree --write-tree --no-messages "$base" "$sha" >/dev/null || code=$?
    report=$tmp/report-$n.md
    echo '<!-- list-rebase -->' >"$report"
    if [ "$code" -eq 0 ]; then
        echo "None found. This pull request merges with \`$MAIN\`." >>"$report"
    elif [ "$code" -eq 1 ]; then
        # tessera-rebase exits 2 when it refuses.
        code=0
        out=$("$TOOL" plan "$base" "$sha" "$RUNNER_TEMP/rebase-$n") || code=$?
        if [ "$code" -eq 0 ]; then
            found="$found{\"pr\":$n,\"head\":\"$sha\",\"base\":\"$base\"},"
            continue
        fi
        echo "This pull request conflicts with \`$MAIN\`. The list rebase changed nothing:" >>"$report"
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
