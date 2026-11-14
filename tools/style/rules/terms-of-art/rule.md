# Terms of art

These never vary. One word, one meaning. One meaning, one word.

The project's `GLOSSARY.md` is the approved dictionary. Read it before
rewriting any comment. A term listed there keeps its exact spelling however
often it repeats. A domain-sounding verb with no entry is a coinage, so
replace it with the commonest exact word.

Without a glossary the identifiers are the dictionary, which is weaker: it
cannot tell a term the code is built on from a term invented in prose and later
copied into a single identifier.

- Call the thing what the code calls it. If the type is `Axis`, write "axis",
  however many times it has already appeared.
- No word with two senses in one repo. If "render" means both "produce
  HTML" and "evaluate a template", rename one.
- Jargon standard to the domain (HTTP, UTF-8, monomorphization) is plain to
  its readers. Invented jargon needs an explanation, unless the file defines
  it.
