# Agent rules

Agents may not touch any file named README.md.

## Code shape

`cargo run -p tessera-lint` checks these, and CI fails on any breach:

- No comment longer than 10 words. Adjacent comment lines count as one comment.
- No function name with more than 3 snake_case segments, or 4 when another function in the crate shares its first
  or last segment (`rms_norm_backward_cpu` beside `gelu_cpu`). Kernel entry points count. Methods in a trait impl and
  functions in an `extern` block are exempt: their names come from elsewhere.
- No source file longer than 1000 lines.
- No crate with more than 30 source files or 15,000 source lines. Kernel sources (.metal, .cu, ...) count.
- A crate may keep up to 5 files per language in one `macros/` directory. They obey the file limits but do not count
  toward the crate's. Rust ones hold only `macro_rules!` (and `mod` lines), and each macro needs at least two
  callsites in the crate.
- Inline tests may fill at most 1% of a file. Put tests in a file whose name ends in `test` (`foo_test.rs`), declared
  with `#[cfg(test)] mod foo_test;`.

Run `scripts/install-hooks.sh` once to check formatting and these rules before each commit.

## Code

Pure Rust, apart from GPU kernels. Backends: Metal, ROCm, CUDA and CPU.
