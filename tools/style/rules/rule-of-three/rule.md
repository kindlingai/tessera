# Rule of three

Three items read as complete, so a writer pads a list to three for rhythm:
"fast, safe, and simple", "clarity, consistency, and readability". The items
overlap, and none of them says what the code does.

Length comes from the domain. A list of three is fine when the code has three
things: "Retries on 502, 503 and 504." Name what the code does in place of the
qualities it has. If the list holds one real item, write that item alone.

`COMMENTS.md` covers the conflict with STE's vertical lists. The same defect in
a commit subject line is part of [commit-message](../commit-message/rule.md).

The detectors flag a list of three whose first and last items are abstract
qualities. The Jev question compares a padded list with one the domain
supplies.
