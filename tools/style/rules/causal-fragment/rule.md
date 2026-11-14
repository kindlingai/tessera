# Causal fragment

Fragments are fine for labels and section markers. A causal claim needs a full
sentence. The moment a comment says "because" or "so", it needs a subject and
a verb. `COMMENTS.md` explains the conflict with terseness.

    /// Routes that share a prefix, merged so a lookup reads one node per segment.
    /// Merged routes sharing a prefix so lookups read one node per segment
    /// A lookup reads one node per segment, because routes that share a prefix
    /// are merged.

The first two attach the reason to a participle, and neither has a main verb,
so the reader supplies what merged the routes and why that helps. The third
gives the reason a subject and a verb.

A type's doc comment may open with a noun phrase, but only for a label with
no reason in it: "Routes merged into a tree by shared prefix." Jev reads a
noun phrase followed by a sentence that explains it as one causal fragment.

A participle clause after a comma that claims a benefit is a
[trailing-participle](../trailing-participle/rule.md). This rule covers the
reason that has no sentence to trail.
