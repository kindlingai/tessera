# Agent rules

Agents may not touch any file named `README.md`. They may, however, call scripts
that rewrite `README.md` as part of their required functionality to pass CI
checks.

No design docs, plans or history in the repo, outside an explicit changelog.

No `Co-Authored-By` lines that credit an AI tool or bot, in commits or PR
bodies. CI and the commit-msg hook refuse them.

No mannered prose in comments, error messages or text. Avoid flamboyant speech
and prefer ASD-STE100 where possible. No historical commentary. Comments are
intentionally limited - code should be clearly written.

Excessive code duplication is strictly checked. Use macros if duplicate
detectors fire. Never change lint rules or add exceptions without a project
maintainer's _human_ approval.

Do not use strings for composing large chunks of code - use structured macros
instead.

Prefer procedural loops over functional ones when heavily nested (eg `for` and
`if` over `iter` and `map`/`filter`).

Do not over-test. If lint flags excessive tests, assume you are over-testing and
remove tests or refactor. Do not over-index on "byte identical".

## Process Self-Improvement

If your PR was stuck for a long time on a process or code failure that was
preventable in an automated way, mention this in your PRs for self-improvement,
tagging @mmastrac.

While implementing your feature, analyze the code you are working on, code it
interacts with and other related code for:

 - opportunities to simplify and refactor to reduce the overall code volume.
 - any hardcoded constants, sites, model names, etc in the code where there shouldn't be. Is there anywhere we have accidentally tied the code to a specific model where it should be generic (ie: hoisting names into recipes).

Tag @mmastrac with any of these or other self-improvement ideas in any PR.

## Code shape

`cargo run -p tessera-lint` checks these, and CI fails on any breach. The limits
live in `lint.toml`.

Run `tools/scripts/install-hooks.sh` once to check formatting, these rules and
commit messages before each commit. `cargo run -p tessera-check -- fetch` adds
the style and near-copy checks to the hook.

## Code

Pure Rust, apart from GPU kernels. Backends: Metal, ROCm, CUDA and CPU.
