# History

Remove comments that describe history or past state: "changed X to Y",
"previously...", "now uses Z", "removed the old...". A comment describes the
code as it is. Version control holds how it got there.

## Regression tests

These are the exception. State the defect behind a regression test, or the
next reader deletes it as arbitrary. Keep it to the prior behaviour and
the consequence, in the test's own doc comment.

    /// The route half went unguarded, so a forced profile leaked into `/all/`.

The Jev re-check drops test docstrings in Python test files, so this exception
holds there without a stamp.
