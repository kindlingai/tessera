# Category where a value belongs

Describing what the code does has one shape: "If <condition>, <this> returns
<value>". Name the value the code returns.

    // An output the graph does not hold gets an answer, not a panic.
    // If the route is missing, this returns an empty fanout.

The second names the condition, the verb and the value. The first gives a
category ("an answer") where the code gives a value, and its second half
describes an event that does not occur. Any comment that leaves the reader
knowing only what did not happen has said nothing.

Negation and abstraction produce the first line. Negation supplies the
contrast a writer reaches for when the actual value feels too plain to state,
and abstraction supplies "an answer" when "an empty fanout" is right there in
the assertion. Naming the value avoids both.

## Vagueness hides redundancy

The existence check runs twice: make the claim concrete, then ask whether the
code already says it.

The order is forced. "gets an answer" does not visibly duplicate
`Vec::<Node>::new()`, and "returns an empty fanout" does. A vague comment
survives the existence check because nobody can tell what it repeats, so
naming the value is what exposes the deletion.

Then finish the job. The rewrite sits directly above
`assert_eq!(g.pull(&out("/nowhere/")), Vec::<Node>::new())`. That assertion
already says the route is missing and the result is empty, so the comment
goes. A
concrete fact that the surrounding code makes obvious is a deletion, and the
concrete version is the only one you can check that against.
