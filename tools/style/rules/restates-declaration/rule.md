# Restated declaration

A comment that defines an item by restating its declaration says what the
reader can already see. `/// A Fold is a container for routes.` above
`struct Fold` is one. Delete it, or replace it with a fact beyond the
declaration.

The `is a` shape is only a hint, since ordinary prose uses it constantly ("the
row is the dependent"). Check whether the comment's subject names the item it
sits on. If it does, the comment is a restatement.

The bare `is a` or `is the` pattern ran 562 hits in a large codebase, most of them
ordinary prose, so this rule has no detector. Its Jev question is a lead.
