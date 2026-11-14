# Tests

Flag superfluous tests: duplicates, tests that restate the implementation, or
cases already covered at a higher level. Assume coverage at the right level.
Don't test the same thing at every level.

Layer tests. Integration tests catch as much as they can. Lower-level tests
exercise details. Fold repeated patterns into a shared-setup suite, a macro or
a loop over input/output tuples.
