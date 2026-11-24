# Passive with an agent

Passive with an explicit agent is the reliable defect. The agent is already in
the sentence, so putting it after the verb costs the reader a reordering.

    /// Which slot a citation takes is decided by the address table.
    /// The address table decides which slot a citation takes.

Passive with no agent is often right. "The row is cached" describes a state.
An active rewrite invents an actor that is not there.

Passive with an agent is right when the agent is an action or an event. The
sentence below is about the slots, and the redo is an operation that happens
to them.

    // Drafts 8 and 9 overwrite the slots of positions 4 and 5, which are read
    // by the redo of 7.
    // Drafts 8 and 9 overwrite the slots of positions 4 and 5. The redo of 7
    // reads those slots.

The active version makes the operation the subject and splits the thought in
two. Lead with the agent when it is something that exists and does the work: a
component, a table, a thread, a process or a person.

The pattern skips an agent that names an action ("read by the redo of 7"), an
operand ("multiplied by 1e10") or code in backticks. The `Action`, `Arith` and
`Identifier` classes in `src/pattern.rs` hold those lists.
