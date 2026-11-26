#!/bin/sh
# Pushes the rebase of one pull request, see tools/rebase.
# Plans the rebase again and commits the files that the build job
# left in BUILT. Does not run pull request code.
# The push needs the branch to still be at HEAD. A fork branch is
# pushed in the fork. The fork must allow edits by maintainers. A
# push that GitHub rejects gets a comment.
# Each push leaves a new comment with the old head, the new head and
# the files. A failed build leaves one comment, with a line that
# stops a repeat of the same build.
# Needs GH_TOKEN, GITHUB_REPOSITORY, MAIN, ANY_TIER, TOOL, RUNNER_TEMP,
# PR, HEAD, BASE, KEY, BUILT and RUN_URL. BOT is the login that
# commits, without "[bot]". git pushes with the checkout's token.
set -e
repo=repos/$GITHUB_REPOSITORY
tmp=$(mktemp -d)
say() {
    printf '%s\n' '<!-- list-rebase -->' "$@" >"$tmp/report.md"
    tools/scripts/pr-comment.sh "$tmp/report.md"
}

# The pull request can change after the plan job reads it.
gh api "$repo/pulls/$PR" --jq '[.state, .draft, .head.sha, .head.repo.full_name // "", .base.ref,
    (.labels | any(.name == "no-auto-rebase")),
    (.labels | any(.name == "priority" or .name == "landing")),
    .maintainer_can_modify, (.head.ref | @uri), .head.ref] | @tsv' >"$tmp/pull"
IFS='	' read -r state draft sha owner target skip tier edits encoded branch <"$tmp/pull"
fork=false
[ "$owner" = "$GITHUB_REPOSITORY" ] || fork=true
if [ "$state" != open ] || [ "$draft" != false ] || [ "$sha" != "$HEAD" ] ||
    [ -z "$owner" ] || [ "$target" != "$MAIN" ] || [ "$skip" != false ] ||
    { [ "$fork" = true ] && [ "$edits" != true ]; } ||
    { [ "$tier" != true ] && [ "$ANY_TIER" != true ]; }; then
    echo "#$PR changed after the plan. Skipped."
    exit 0
fi
# The branch name is the author's text. git reads it as a ref only.
if ! git check-ref-format --branch "$branch" >/dev/null || [ "${branch#-}" != "$branch" ]; then
    echo "#$PR has a branch name that git does not accept. Skipped."
    exit 0
fi
if { [ "$fork" = false ] && [ "$branch" = "$MAIN" ]; } ||
    [ "$(gh api "repos/$owner/branches/$encoded" --jq .protected)" != false ]; then
    echo "$branch in $owner is protected. Skipped."
    exit 0
fi

if [ -f "$BUILT/failed" ]; then
    # The build job runs pull request code, so its text is not shown.
    case $(cat "$BUILT/failed") in
    fmt) why='cargo fmt failed' ;;
    lock) why='the lock refresh failed, or it lost a version that the branch set' ;;
    check) why='cargo check failed' ;;
    clippy) why='cargo clippy failed' ;;
    stray) why='cargo fmt or the build changed a file that had no conflict' ;;
    *) why='the build failed' ;;
    esac
    say "The rebase of this pull request on \`$MAIN\` did not pass its build: $why. The rebase bot changed nothing. Build log: $RUN_URL" \
        "<!-- failed $HEAD $KEY -->"
    exit 0
fi
if [ ! -f "$BUILT/tree" ]; then
    echo "The build job left no result. It was cancelled or its runner stopped."
    exit 0
fi

id=$(gh api "users/$BOT%5Bbot%5D" --jq .id)
export GIT_COMMITTER_NAME="${BOT}[bot]"
export GIT_COMMITTER_EMAIL="$id+${BOT}[bot]@users.noreply.github.com"
export GIT_AUTHOR_NAME="$GIT_COMMITTER_NAME" GIT_AUTHOR_EMAIL="$GIT_COMMITTER_EMAIL"
git fetch -q origin "refs/pull/$PR/head"
out=$("$TOOL" seal "$BASE" "$HEAD" "$RUNNER_TEMP/rebase" "$BUILT")
tip=$(printf '%s\n' "$out" | sed -n 's/^tip //p')
# The checkout sends its token to each github.com URL.
code=0
git push --force-with-lease="refs/heads/$branch:$HEAD" "https://github.com/$owner.git" \
    "$tip:refs/heads/$branch" 2>"$tmp/push" || code=$?
cat "$tmp/push" >&2
if [ "$code" -ne 0 ]; then
    if grep -q 'stale info' "$tmp/push"; then
        echo "The branch moved after the plan. It stays as it was."
    else
        say "GitHub rejected the push of the rebase to \`$owner\`. The rebase bot changed nothing. Log: $RUN_URL"
    fi
    exit 0
fi
{
    echo "The rebase bot rebased this pull request on \`$MAIN\` at $BASE."
    echo "The head moved from $HEAD to $tip."
    echo "To update a local copy of the branch, fetch it and run \`git reset --hard $tip\`."
    echo "Files with joined lists, and the refreshed lock file:"
    printf '%s\n' "$out" | sed -n 's/^path //p' | sed -e 's/[^A-Za-z0-9_.\/-]/?/g' -e "s/.*/- \`&\`/"
} >"$tmp/pushed.md"
gh api "$repo/issues/$PR/comments" -F body=@"$tmp/pushed.md" >/dev/null
say "None found. The rebase bot pushed $tip."
