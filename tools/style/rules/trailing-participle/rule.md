# Trailing participle

A clause after a comma that claims a benefit is an unchecked claim:
", ensuring thread safety", ", allowing faster startup". Cut the clause. If
nothing remains, cut the comment.

A past participle can end a sentence the same way: "The rung-0 seam, closed."
Cut it too.

The detector for the mid-sentence form (", sorted so a ...") ran 224 hits in
a large codebase, mixed with plain description, so review that form by hand. When the
sentence has no main verb, it is a
[causal-fragment](../causal-fragment/rule.md).

A clause after ", so" that states a consequence the reader needs is fine:
"so a containerised daemon needs `network_mode: host`". The defect is a
claimed benefit with nothing behind it.
