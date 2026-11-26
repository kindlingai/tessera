# Agent rules

Agents may not touch any file named `README.md`. They may, however, call scripts
that rewrite `README.md` as part of their required functionality to pass CI
checks.

## Docs

No design docs, plans or history in the repo, outside an explicit changelog.

No mannered prose in comments, error messages or text. Avoid flamboyant speech
and prefer ASD-STE100 where possible. No historical commentary. Comments are
intentionally limited - code should be clearly written.

## Code

Excessive code duplication is strictly checked. Use macros if duplicate
detectors fire. Never change lint rules or add exceptions without a project
maintainer's _human_ approval.

Do not use strings for composing large chunks of code - use structured macros
instead.

Prefer procedural loops over functional ones when heavily nested (eg `for` and
`if` over `iter` and `map`/`filter`).

Do not over-test. If lint flags excessive tests, assume you are over-testing and
remove tests or refactor. Do not over-index on "byte identical".

Do not ignore performance improvements because they are "within the noise floor"
or "too small to be worth it". The project lives on improvements as small as 0.5%
and human must always judge whether to drop an experiment that has a likelihood
of success.

No arbitrary limits. A size limit must come from a real constraint: an index
width, a hardware limit, or a fixed device structure. Do not use "maximum size
tested" as a limit. Compute a limit that depends on the machine or the model
from the plan and the device. Crashing beyond a memory limit is not a reason
to add a hardcoded limit - fix the memory allocation strategy instead. Remove
any arbitrary limits you find that do not represent physical constraints.

## Pull Requests

Pull request titles must follow Conventional Commits (`feat:`, `fix:`, `perf:`,
`docs:`, `ci:`, `refactor:`, `test:`, `chore:`, with optional scope such as
`feat(serve):`). `release-please` derives version bumps and `CHANGELOG.md` from
squash-merged titles on `main`.

No `Co-Authored-By` lines that credit an AI tool or bot, in commits or PR
bodies. CI and the commit-msg hook reject those lines.

https://huggingface.co/datasets/kindlingai/tessera-references contains the
golden references for a number of models. Use these to validate PRs against
golden runs when you work near code that may break them.

This project moves quickly. When submitting PRs, you must watch them for
conflicts and actively rebase. If you have permission to do so, mark your
PRs to auto-merge.

### Process Self-Improvement

If your PR was stuck for a long time on a process or code failure that was
preventable in an automated way, mention this in your PRs for self-improvement,
tagging @mmastrac.

While implementing your feature, analyze the code you are working on, code it
interacts with and other related code for:

 - opportunities to simplify and refactor to reduce the overall code volume. Can code
   be combined or migrated to macros?
 - any hardcoded constants, sites, model names, etc in the code where there shouldn't
   be. Is there anywhere we have accidentally tied the code to a specific model where it
   should be generic?

Tag @mmastrac with any of these or other self-improvement ideas in any PR.

## Code shape

`tools/scripts/stickler.sh lint` checks these, and CI fails on any breach. The limits
live in `lint.toml`.

Run `tools/scripts/install-hooks.sh` once to check formatting, these rules and
commit messages before each commit. `tools/scripts/stickler.sh fetch` adds
the style and near-copy checks to the hook.

## Code

Pure Rust, apart from GPU kernels. Backends: Metal, ROCm, CUDA and CPU.
