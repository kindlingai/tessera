# Rare words

Every word that is not a term of art takes the commonest exact form. Exactness
gates first and frequency breaks the tie, so this never argues for a vaguer
word. Among words that are equally exact, take the one a reader meets most
often.

    use / utilize          let / facilitate       start / initiate
    show / surface         end / terminate        cut / excise
    build / construct      is / serves as         has / features

Rare words slow down the reader. They stop on "utilize", look for whatever
distinguishes it from "use", and find nothing. The same defect takes these
forms:

- Buzzwords. A rare word implying a meaning it does not have.
- LLM vocabulary. A rare word chosen for texture.
- Puffery. An evaluative word implying a property it does not name. See
  [puffery](../puffery/rule.md).
- Elegant variation. A synonym implying a distinction that does not exist.

Prefer "reference" to "golden". Drop "load-bearing".

The detectors hold instances of this rule. A word's absence from the
detectors clears nothing, so apply the rule.

## Repetition

The second-commonest word is available for one purpose: to break repetition
that reads as clumsy within a sentence or across adjacent ones. Use the
second-commonest word there and nowhere else.

Deliberate repetition is fine. A word repeated in the same
grammatical role is emphasis, so keep it. A word appearing twice in different
roles is an accident, so change one.

    // The fold owns the route. The fold owns the slot. The fold owns the shell.
    // Build the index first, then build the fanout from the index it built.

Line 1 repeats on purpose. Keep all three. Line 2 says "build" three times by
accident, so the middle one becomes "derive". "index" stays both times because
it names a thing. The ordinary word is the one to change.
