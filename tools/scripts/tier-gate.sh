#!/bin/sh
# Fails when HEAD conflicts with a labelled PR of a higher tier.
# Tiers, highest first: priority, landing, then PRs with neither label.
# Inside a tier, the PR labelled first ranks first.
# Reads labels and CI live, so a rerun sees changes.
# Needs GH_TOKEN, GITHUB_REPOSITORY, HEAD, BASE, MAIN, and PR or
# QUEUE_REF. BASE is the commit HEAD adds its changes to.
# tier-ranks.sh reads the ranks. Any API or git error fails the gate.
set -e
main=${MAIN#refs/heads/}
tmp=$(mktemp -d)
# commit-tree needs a name. The commits stay unreferenced.
export GIT_AUTHOR_NAME=tier-gate GIT_AUTHOR_EMAIL=none
export GIT_COMMITTER_NAME=tier-gate GIT_COMMITTER_EMAIL=none

fail() {
    echo "tier-gate: $1 The gate fails closed." >&2
    exit 2
}

# A merge group head ref ends in gh-readonly-queue/<base>/pr-<n>-<sha>.
if [ -z "$PR" ]; then
    printf '%s\n' "$QUEUE_REF" >"$tmp/ref"
    PR=$(sed -nE 's|^(refs/heads/)?gh-readonly-queue/.+/pr-([0-9]+)-[0-9a-f]+$|\2|p' "$tmp/ref")
fi

tools/scripts/tier-ranks.sh >"$tmp/listed" || fail "Could not read the ranks."

if [ -n "$PR" ] && grep -qx "exempt $PR" "$tmp/listed"; then
    echo "#$PR has conflict-exempt. Skipped."
    exit 0
fi

# Other PRs that are drafts or red block nobody.
# This PR stays listed whatever its state, so it knows its rank.
: >"$tmp/ranked"
grep -v '^exempt ' "$tmp/listed" | sort -k1,1n -k2,2 >"$tmp/sorted"
while read -r tier at n sha draft red label; do
    if [ "$n" != "$PR" ] && [ "$draft" = true ]; then
        echo "#$n ($label) is a draft. Skipped."
    elif [ "$n" != "$PR" ] && [ "$red" = true ]; then
        echo "#$n ($label) has failed CI jobs. Skipped."
    else
        echo "$tier $at $n $sha $label" >>"$tmp/ranked"
    fi
done <"$tmp/sorted"
[ -s "$tmp/ranked" ] || exit 0
awk -v pr="$PR" '$3 != pr { print $4 }' "$tmp/ranked" >"$tmp/heads"
xargs git fetch -q origin "$main" <"$tmp/heads" || fail "Could not fetch main and the ranked PR heads."
# merge-tree exits 1 on a rev it cannot read, as on a conflict.
for rev in "origin/$main" "$BASE" "$HEAD" $(cat "$tmp/heads"); do
    git rev-parse -q --verify "$rev^{commit}" >/dev/null || fail "$rev is not a commit here."
done

# Prints the commit of $1 merged with $2.
# Exits 1 on a conflict and above 1 on an error.
stack() {
    tree=$(git merge-tree --write-tree --no-messages "$1" "$2") || return $?
    git commit-tree "$tree" -p "$1" -p "$2" -m tier-merge
}

# Rank order. A PR that conflicts with main and the PRs ranked
# before it loses. It drops out, so it does not break the base
# for lower ranks.
chain=origin/$main
active=
while read -r _ _ n sha label; do
    [ "$n" != "$PR" ] || break
    code=0
    next=$(stack "$chain" "$sha") || code=$?
    [ "$code" -le 1 ] || fail "git merge-tree failed on #$n."
    if [ "$code" -eq 1 ]; then
        echo "#$n ($label) conflicts with $main and the PRs ranked before it. Skipped."
        continue
    fi
    chain=$next
    active="$active $sha"
    echo "$sha $n $label" >>"$tmp/names"
done <"$tmp/ranked"
[ -n "$active" ] || exit 0

# HEAD adds its changes to BASE. Stack the ranked PRs on BASE too.
chain=$BASE
for sha in $active; do
    code=0
    next=$(stack "$chain" "$sha") || code=$?
    [ "$code" -le 1 ] || fail "git merge-tree failed on $sha."
    if [ "$code" -eq 1 ]; then
        echo "$sha conflicts with $BASE and the PRs ranked before it. Skipped."
    else
        chain=$next
    fi
done
code=0
git merge-tree --write-tree --name-only --no-messages "$chain" "$HEAD" >"$tmp/out" || code=$?
[ "$code" -le 1 ] || fail "git merge-tree failed on HEAD."
[ "$code" -eq 1 ] || exit 0

# Name each PR that conflicts alone, else the whole stack.
code=0
tools/scripts/merge-conflicts.sh "$BASE" "$HEAD" $active >"$tmp/hits" || code=$?
[ "$code" -le 1 ] || fail "merge-conflicts.sh failed."
if [ "$code" -eq 1 ]; then
    while read -r sha n label; do
        grep -q "^$sha " "$tmp/hits" || continue
        echo "Conflicts with $label PR #$n in these files:"
        sed -n "s|^$sha |  |p" "$tmp/hits"
    done <"$tmp/names"
else
    echo "Conflicts with these ranked PRs merged together:"
    awk '{ printf "  #%s (%s)\n", $2, $3 }' "$tmp/names"
    echo "In these files:"
    sed -n '2,$s|^\(..*\)|  \1|p' "$tmp/out"
fi
echo "Rebase onto those PRs or wait until they merge."
echo "A maintainer can add the conflict-exempt label to skip this check."
exit 1
