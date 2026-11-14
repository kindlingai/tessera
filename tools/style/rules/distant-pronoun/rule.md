# Distant pronoun

A pronoun whose noun sits in the previous sentence makes the reader search
back. When a closer noun of the same number sits in the way, the reader has to
choose:

    // A local name must be unique within its routine. Other routines may reuse it.
    // A local name must be unique within its routine. Other routines may reuse
    // the name.

"it" means the name, but the nearest singular noun is "routine". Repeat the
noun. [Terms of art](../terms-of-art/rule.md) repeat the exact term every time,
so the repeat costs nothing.

The parse flags a pronoun whose candidate nouns all sit in the previous
sentence, when there are two or more and the nearest is not that sentence's
subject. As an object or possessive, the pronoun fails. As the subject of its
sentence, it continues the previous subject. Readers expect that, so the
parse reports a lead: "The daemon runs on each node. It holds the cluster
state." A pronoun with its noun in the same sentence ("reads the file and
parses it") and a dummy "it" ("It is safe to call this twice") pass.
