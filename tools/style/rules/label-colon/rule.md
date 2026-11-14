# Label colon

A fragment introducing a list or a definition takes a colon. "Stage 3.
Era-specific" becomes "Stage 3: era-specific".

Lead with the label, then a colon, then the contents. Drop any sentence whose
only job is to explain the label.

    // Bold standing in for a heading, and bolded lead-ins on list items. Both
    // are on the Wikipedia signs-of-AI-writing list. Use a real heading.
    // Wikipedia signs-of-AI-writing list: bold used as a heading, bolded
    // lead-ins on list items.

The second drops the instruction because the surrounding context implies it,
and drops "Both are on" because the colon already says it.

The detector matches a label at the start of a line, so "resolved in Stage 5."
stays legal.
