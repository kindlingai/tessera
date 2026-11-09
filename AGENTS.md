# Agent rules

Agents may not touch any file named README.md.

No design docs, plans or history in the repo, outside an explicit changelog.

No `Co-Authored-By` lines that credit an AI tool or bot, in commits or PR bodies. CI and the commit-msg hook
refuse them.

## Code shape

`cargo run -p tessera-lint` checks these, and CI fails on any breach. The limits live in `lint.toml`.

Run `scripts/install-hooks.sh` once to check formatting, these rules and commit messages before each commit.

## Code

Pure Rust, apart from GPU kernels. Backends: Metal, ROCm, CUDA and CPU.
