#!/usr/bin/env python3
"""Writes a PR comment of the style rules broken by a diff's added lines.

Usage: report.py <base> <head> <report.md> [link_base]

Runs jev_style.py on `git diff base...head` with detectors only: no Jev,
no dependency parse. Each broken rule shows its fix and one fixed example.
The report never fails the job.
"""

import json
import os
import subprocess
import sys

import jev_style as js

HERE = os.path.dirname(os.path.realpath(__file__))
MARKER = "<!-- tessera-style -->"
# Keeps the comment under GitHub's 65,536-character limit.
MAX_PER_RULE = 15


def hits(base, head):
    """[(path, line, rule, match, source line)] for the diff's added lines."""
    diff = subprocess.run(["git", "diff", "--no-color", f"{base}...{head}"],
                          capture_output=True, text=True, check=True).stdout
    if not diff.strip():
        return []
    run = subprocess.run([sys.executable, os.path.join(HERE, "jev_style.py"), "--no-jev", "--no-parse", "--json"],
                         input=diff, capture_output=True, text=True)
    out = []
    for line in run.stdout.splitlines():
        hunk = json.loads(line)
        for lineno, rule, _source, match, src in hunk["detectors"]:
            out.append((hunk["path"], lineno, rule, match, src))
    return out


def guidance(rid):
    """The first paragraph of the rule's rule.md, which says how to fix it."""
    path = os.path.join(HERE, "rules", rid, "rule.md")
    if not os.path.isfile(path):
        return ""
    with open(path) as f:
        paras = [p.strip() for p in f.read().split("\n\n") if p.strip() and not p.startswith("#")]
    return " ".join(paras[0].split()) if paras else ""


def code(text):
    """Inline code that survives backticks inside it."""
    return f"`` {text} ``" if "`" in text else f"`{text}`"


def place(path, line, links):
    text = f"`{path}:{line}`"
    return f"[{text}]({links}/{path}#L{line})" if links else text


def markdown(found, rules, links):
    out = [MARKER, "### Comment style", ""]
    if not found:
        return "\n".join(out + ["None found.", ""])
    out.append(f"{len(found)} added lines break the style rules in `tools/style/rules`. "
               "This check does not fail CI.")
    by_rule = {}
    for f in found:
        by_rule.setdefault(f[2], []).append(f)
    for rid in sorted(by_rule, key=lambda r: list(rules).index(r) if r in rules else 99):
        rule = rules.get(rid, {})
        rows = by_rule[rid]
        doc = f"tools/style/rules/{rid}/rule.md"
        name = rule.get("name", rid)
        out += ["", f"<details open><summary>{name} ({len(rows)})</summary>", ""]
        see = f"[`{doc}`]({links}/{doc})" if links else f"`{doc}`"
        out.append(f"{guidance(rid) or rule.get('description', '')} More in {see}.")
        fixes = js.read_examples(os.path.join(HERE, "rules", rid, "fixes.txt"), pairs=True)
        if fixes:
            bad, good = fixes[0]
            out += ["", "````text", bad, "````", "becomes", "````text", good, "````"]
        out.append("")
        for path, line, _rule, match, src in rows[:MAX_PER_RULE]:
            text = src.strip()
            seen = "" if match.strip() == text else f" (matched {code(match)})"
            out.append(f"- {place(path, line, links)}: {code(text)}{seen}")
        if len(rows) > MAX_PER_RULE:
            out.append(f"- {len(rows) - MAX_PER_RULE} more not shown.")
        out += ["", "</details>"]
    return "\n".join(out + [""])


def main():
    if len(sys.argv) not in (4, 5):
        sys.exit(__doc__)
    base, head, report = sys.argv[1:4]
    links = sys.argv[4] if len(sys.argv) == 5 else None
    rules = js.load_rules(os.path.join(HERE, "rules"))
    found = hits(base, head)
    with open(report, "w") as f:
        f.write(markdown(found, rules, links))
    for path, line, rule, match, _ in found:
        print(f"{path}:{line}: {rule}: {match}")


if __name__ == "__main__":
    main()
