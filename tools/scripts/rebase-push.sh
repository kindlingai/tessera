#!/bin/sh
# Pushes the list rebase of one pull request, see tools/rebase.
# Plans the rebase again and commits only the files that the build
# job left in BUILT. Does not run pull request code.
# The push needs the branch to still be at HEAD.
# Needs GH_TOKEN, GITHUB_REPOSITORY, MAIN, TOOL, RUNNER_TEMP, PR, HEAD,
# BASE, BUILT and RUN_URL. BOT is the login that commits, without
# "[bot]". git pushes with the checkout's token.
set -e
repo=repos/$GITHUB_REPOSITORY
tmp=$(mktemp -d)
say() {
    printf '%s\n%s\n' '<!-- list-rebase -->' "$1" >"$tmp/report.md"
    tools/scripts/pr-comment.sh "$tmp/report.md"
}

# The pull request can change after the plan job reads it.
gh api "$repo/pulls/$PR" --jq '[.state, .draft, .head.sha, .head.repo.full_name, .base.ref,
    (.labels | any(.name == "no-auto-rebase")), (.head.ref | @uri), .head.ref] | @tsv' >"$tmp/pull"
IFS='	' read -r state draft sha owner target skip encoded branch <"$tmp/pull"
if [ "$state" != open ] || [ "$draft" != false ] || [ "$sha" != "$HEAD" ] ||
    [ "$owner" != "$GITHUB_REPOSITORY" ] || [ "$target" != "$MAIN" ] || [ "$skip" != false ]; then
    echo "#$PR changed after the plan. Skipped."
    exit 0
fi
if [ "$branch" = "$MAIN" ] || [ "$(gh api "$repo/branches/$encoded" --jq .protected)" != false ]; then
    echo "$branch is protected. Skipped."
    exit 0
fi

if [ ! -f "$BUILT/tree" ]; then
    say "This pull request conflicts with \`$MAIN\` only in lists, but the merged lists do not build. The list rebase changed nothing. Build log: $RUN_URL"
    exit 0
fi
id=$(gh api "users/$BOT%5Bbot%5D" --jq .id)
export GIT_COMMITTER_NAME="${BOT}[bot]"
export GIT_COMMITTER_EMAIL="$id+${BOT}[bot]@users.noreply.github.com"
export GIT_AUTHOR_NAME="$GIT_COMMITTER_NAME" GIT_AUTHOR_EMAIL="$GIT_COMMITTER_EMAIL"
git fetch -q origin "$HEAD"
out=$("$TOOL" seal "$BASE" "$HEAD" "$RUNNER_TEMP/rebase" "$BUILT")
tip=$(printf '%s\n' "$out" | sed -n 's/^tip //p')
if git push --force-with-lease="refs/heads/$branch:$HEAD" origin "$tip:refs/heads/$branch"; then
    say "None found. The list rebase pushed $tip, on \`$MAIN\` at $BASE."
else
    echo "The push failed. The branch stays as it was."
fi
