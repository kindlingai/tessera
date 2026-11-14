# Restated code

Default is no comment. Write one only when the code cannot say it: a why, a
constraint from outside the file, a hazard, or a non-obvious cost. If the
answer is in the code, delete the comment. If a better name would delete it,
rename instead. See [naming](../naming/rule.md).

Public items invert this default. See
[Public items](../../COMMENTS.md#public-items) in `COMMENTS.md`.

An LLM can regenerate a comment from the code. The context window is limited,
and every unnecessary comment uses space the code needs.

A vague comment can hide that it restates the code. Make the claim concrete
first, then check it against the code. See
[category-not-value](../category-not-value/rule.md).
