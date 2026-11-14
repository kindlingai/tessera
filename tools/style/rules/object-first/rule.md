# Object first

A clause that puts the object before its subject and verb makes the reader
reorder it: "the slot which the redo of 7 reads", "a term that prose invented".

    // This is the slot which the redo of 7 reads.
    // This slot is read by the redo of 7.

The redo is an action. Passive with an action as the agent is fine. See
[agent-passive](../agent-passive/rule.md). When the agent is a thing that does
the work, lead with it: "The parser fills the buffer."

The same shape without "which" or "that" is harder to read the heavier its
subject: "the pool a relation's walk names read", "the lines `explain`
prints". A "what" clause as an object does it too: "the French twins of what
the pool holds". With a light subject it reads easily ("the box it connected
from", "every actor the daemon knows"), so jev_parse.py reports that case as a
lead and fails the heavy ones.

The Jev question, used with `--no-parse`, is a `choice` between the two word
orders, since a yes or no question also flagged plain sentences.
