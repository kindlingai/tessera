#!/usr/bin/env python3
"""Checks code hunks against the staffeng-review style guide.

Each rule lives in `rules/<id>/`: `rule.yaml` holds its name, description,
detectors and Jev question, `rule.md` explains it, and `good.txt`, `bad.txt`
and `fixes.txt` hold examples that `--test` checks. Passes on each hunk's
added lines:

1. Detectors, the regexes and literals in each `rule.yaml`. A hit fails the
   review unless a `style-ok: <reason>` stamp sits on the same line or the line
   above. A rule with `stampable: false` ignores stamps.

   `prose: ignore [N]` in a comment (`<!-- -->`, `#` or `//`) takes lines out
   of both passes. On its own line it covers the next N lines, and after code
   it covers its own line and the N-1 after it. N defaults to 1 and counts
   blank lines. Unstampable hits still fire on covered lines.
2. Jev requests, one question per rule that has one. A rule fires when its
   probability reaches `--threshold`.
3. Each Jev hit is asked again on a filtered hunk (see `confirm_text`), and
   stands only if it still reaches the threshold. `--no-confirm` skips this.
4. Rules with `parse: true` go to jev_parse.py, which checks them with a
   dependency parse in place of their detectors and Jev question. A parse hit
   either fails the review or, marked `[?]`, is a lead. `--no-parse`, or a
   missing `uv`, falls back to the detectors and Jev.

    git diff | jev_style.py
    jev_style.py --scan src/ README.md
    jev_style.py --self
    jev_style.py --test [RULE ...]
"""

import argparse
import json
import shutil
import subprocess
import os
import re
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SKILL_DIR = os.path.dirname(os.path.realpath(__file__))
RULES_DIR = os.path.join(SKILL_DIR, "rules")
PARSE_SCRIPT = os.path.join(SKILL_DIR, "jev_parse.py")
URL = os.environ.get("JEV_URL", "")
MODEL = "dgemma"

PREAMBLE = (
    "The text is a hunk of source code under review. Lines starting with '+' "
    "are new, and only they are judged. Other lines are context. "
)
PROSE_PREAMBLE = (
    "The text is part of a prose document under review. Lines starting with '+' "
    "are new, and only they are judged. Other lines are context. Treat every "
    "sentence as a comment. Quoted examples and code are replaced with '…'. "
    "The document may name or describe a writing defect, and that is fine. "
    "Judge only whether a sentence commits the defect itself. "
)

YAML_KEY_RE = re.compile(r"^('(?:[^']|'')*'|[A-Za-z0-9_.-]+):(?:\s+(.*))?$")


def load_yaml(text, path):
    """Parses the YAML subset rule files use: block mappings, block lists of
    scalars, and plain, quoted, boolean or numeric scalars. Comments take a
    whole line. Anything else raises ValueError."""
    lines = []
    for n, raw in enumerate(text.splitlines(), 1):
        if raw.strip() and not raw.lstrip().startswith("#"):
            lines.append((n, len(raw) - len(raw.lstrip(" ")), raw.strip()))
    if not lines:
        return {}
    value, i = _yaml_block(lines, 0, lines[0][1], path)
    if i < len(lines):
        raise ValueError(f"{path}:{lines[i][0]}: unexpected indent")
    return value


def _yaml_block(lines, i, indent, path):
    if lines[i][2].startswith("-"):
        items = []
        while i < len(lines) and lines[i][1] == indent and lines[i][2].startswith("-"):
            items.append(_yaml_scalar(lines[i][2][1:].strip(), path, lines[i][0]))
            i += 1
        return items, i
    out = {}
    while i < len(lines) and lines[i][1] == indent:
        n, _, content = lines[i]
        m = YAML_KEY_RE.match(content)
        if not m:
            raise ValueError(f"{path}:{n}: expected 'key: value'")
        key = _yaml_scalar(m.group(1), path, n) if m.group(1).startswith("'") else m.group(1)
        i += 1
        if m.group(2):
            out[key] = _yaml_scalar(m.group(2), path, n)
        elif i < len(lines) and (lines[i][1] > indent or (lines[i][1] == indent and lines[i][2].startswith("-"))):
            out[key], i = _yaml_block(lines, i, lines[i][1], path)
        else:
            out[key] = None
    return out, i


def _yaml_scalar(text, path, n):
    if text.startswith("'"):
        body = text[1:-1]
        if len(text) < 2 or not text.endswith("'") or "'" in body.replace("''", ""):
            raise ValueError(f"{path}:{n}: bad single-quoted string")
        return body.replace("''", "'")
    if text.startswith('"'):
        return json.loads(text)
    if text[0] in "[{|>&*!%@`" or " #" in text:
        raise ValueError(f"{path}:{n}: unsupported YAML, quote the value: {text[:40]}")
    if text in ("true", "false"):
        return text == "true"
    if text in ("null", "~"):
        return None
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    return text


def load_rules(root):
    """Returns {id: rule} for each folder under `root` with a `rule.yaml`,
    ordered by stage and then id. Jev batches follow this order."""
    rules = {}
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name, "rule.yaml")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            rule = load_yaml(f.read(), path)
        rules[name] = rule | {"id": name, "dir": os.path.dirname(path)}
    return dict(sorted(rules.items(), key=lambda kv: (kv[1].get("stage", 99), kv[0])))


# Filled by main() from --rules-dir.
RULES = {}
# Rules with `parse: true`, when jev_parse.py can run. The parse replaces their
# detectors and Jev question.
PARSED = set()


def enable_parse(no_parse):
    """Fills PARSED unless --no-parse is set or uv is missing."""
    PARSED.clear()
    wanted = {rid for rid, r in RULES.items() if r.get("parse")}
    if no_parse or not wanted:
        return
    if not shutil.which("uv"):
        print("uv not found, so parse rules fall back to detectors and Jev", file=sys.stderr)
        return
    PARSED.update(wanted)


def parse_hits(hunks, rules):
    """Returns, for each (path, first, text) hunk, [(line, rule_id, level,
    marker, sentence)] from jev_parse.py. Returns None when it fails."""
    if not hunks or not rules:
        return [[] for _ in hunks]
    request = json.dumps({"rules": sorted(rules), "hunks": [list(h) for h in hunks]})
    try:
        done = subprocess.run([PARSE_SCRIPT, "--json"], input=request, capture_output=True, text=True, timeout=1800)
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"jev_parse.py failed: {e}", file=sys.stderr)
        return None
    if done.returncode:
        print(f"jev_parse.py failed: {done.stderr.strip()[-500:]}", file=sys.stderr)
        return None
    return [[tuple(h) for h in hunk] for hunk in json.loads(done.stdout)]


def jev(key):
    return RULES[key].get("jev") or {}

HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def parse_diff(text):
    """Yields (path, first_line, last_line, hunk_text) for hunks that add lines."""
    path, start, lines, added = None, 0, [], False

    def flush():
        if path and lines and added:
            yield path, start, start + sum(1 for l in lines if not l.startswith("-")) - 1, "\n".join(lines)

    all_lines = text.splitlines()
    for i, line in enumerate(all_lines):
        # The next file's "--- a/..." header would read as a removed line.
        if line.startswith("--- ") and i + 1 < len(all_lines) and all_lines[i + 1].startswith("+++ "):
            continue
        if line.startswith("+++ "):
            yield from flush()
            path, lines, added = line[4:].removeprefix("b/"), [], False
        elif m := HUNK_RE.match(line):
            yield from flush()
            start, lines, added = int(m.group(1)), [], False
        elif path and line[:1] in "+- ":
            lines.append(line)
            added |= line.startswith("+")
    yield from flush()


# Scan hunks by kind: "comments" hunks go to the prose rules, "code" hunks to
# the rules with `judges: code`. Keyed by (path, first, last).
HUNK_KIND = {}
COMMENT_ONLY_RE = re.compile(r"^\s*(//|/\*|\*)")
# Code a comment block documents, when that code is a test.
TEST_ITEM_RE = re.compile(r"^\s*#\[(\w+::)*test\b|^\s*(pub\s+)?(async\s+)?fn\s+test_|^\s*(async\s+)?def\s+test")
CONTEXT_LINES = 6


def file_hunks(path, window):
    """Splits a file into hunks, all marked as new. Markdown splits into
    paragraphs, since a window of prose always holds something for a rule to
    fire on. Code with `//` comments yields each comment block, with the code
    after it as context, and windows of `window` lines with the comments
    blanked. Other files split into windows."""
    with open(path) as f:
        src = apply_ignore(["+" + l for l in f.read().splitlines()])
    ext = os.path.splitext(path)[1]
    if not is_markdown(path) and ext not in PROSE_EXTS and ext not in HASH_EXTS:
        yield from code_hunks(path, src, window)
        return
    if is_markdown(path):
        masked = mask_markdown("\n".join(src)).splitlines()
        start = None
        for i, line in enumerate([*masked, "+"]):
            if line[1:].strip() and start is None:
                start = i
            elif not line[1:].strip() and start is not None:
                yield path, start + 1, i, "\n".join(masked[start:i])
                start = None
        return
    for i in range(0, len(src), window):
        chunk = src[i : i + window]
        yield path, i + 1, i + len(chunk), "\n".join(chunk)


def code_hunks(path, src, window):
    """Yields comment blocks, each with up to `CONTEXT_LINES` lines of the code
    after it as context, then code windows with comment-only lines blanked."""
    is_comment = [bool(COMMENT_ONLY_RE.match(l[1:])) for l in src]
    i = 0
    while i < len(src):
        if not is_comment[i]:
            i += 1
            continue
        start = i
        while i < len(src) and is_comment[i]:
            i += 1
        end = i
        while end < len(src) and end - i < CONTEXT_LINES and src[end][1:].strip() and not is_comment[end]:
            end += 1
        lines = src[start:i] + [" " + l[1:] for l in src[i:end]]
        HUNK_KIND[path, start + 1, end] = "comments"
        yield path, start + 1, end, "\n".join(lines)
    code = [("+" if l[0] == "+" else l[0]) if c else l for l, c in zip(src, is_comment)]
    for i in range(0, len(code), window):
        chunk = code[i : i + window]
        if any(l[1:].strip() for l in chunk):
            HUNK_KIND[path, i + 1, i + len(chunk)] = "code"
            yield path, i + 1, i + len(chunk), "\n".join(chunk)


def documents_test(hunk_text):
    """True when the first line of context code after the added lines is a
    test, so the added comment documents it."""
    seen_added = False
    for l in hunk_text.splitlines():
        seen_added |= l.startswith("+")
        if seen_added and l.startswith(" ") and l[1:].strip():
            return bool(TEST_ITEM_RE.match(l[1:]))
    return False


def hunk_keys(args, hunk):
    """Returns the Jev rules to ask about a scan hunk, or None for all of them."""
    path, first, last, _ = hunk
    kind = HUNK_KIND.get((path, first, last))
    if kind is None:
        return None
    keys = jev_keys(args.rules, is_prose_path(path))
    return [k for k in keys if (jev(k).get("judges") == "code") == (kind == "code")]


IGNORE_RE = re.compile(r"(?:<!--|//+|#+)\s*prose:\s*ignore(?:\s+(\d+))?\s*(?:-->)?\s*$")


def apply_ignore(lines):
    """Changes the mark of each line a `prose: ignore` marker covers to '!'.
    Removed lines keep their mark and do not count toward N."""
    out, left = list(lines), 0
    for i, raw in enumerate(lines):
        if raw.startswith("-"):
            continue
        if left:
            out[i], left = "!" + raw[1:], left - 1
            continue
        if m := IGNORE_RE.search(raw[1:]):
            n = int(m.group(1) or 1)
            own_line = not raw[1:][: m.start()].strip()
            out[i], left = "!" + raw[1:], n if own_line else n - 1
    return out


def jev_view(hunk_text):
    """Returns the hunk without the lines a `prose: ignore` marker covers."""
    return "\n".join(l for l in hunk_text.splitlines() if not l.startswith("!"))


SKIP_DIRS = {"__pycache__", "node_modules", "target", "build", "dist", "venv"}


def scan_paths(paths):
    """Expands directories to the text files under them. Named files pass as given."""
    for path in paths:
        if not os.path.isdir(path):
            yield path
            continue
        for root, dirs, files in os.walk(path):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS)
            for name in sorted(files):
                if os.path.splitext(name)[1] in SCAN_EXTS:
                    yield os.path.join(root, name)


def is_markdown(path):
    return path.endswith(".md")


# Opening quote -> closing quote.
QUOTES = {"`": "`", '"': '"', "“": "”"}


def mask_markdown(hunk_text):
    """Blanks code blocks and replaces quoted and backticked spans with '"…"'.

    A style guide quotes the defects it describes, and these spans hold them.
    A quote can wrap to the next line and ends at a blank line. A hunk that
    starts inside a code block or a quote reads it as prose.
    """
    out, fence, indented, prev_blank, close = [], False, False, True, None
    for raw in hunk_text.splitlines():
        mark, line = raw[:1], raw[1:]
        if mark == "-":
            out.append(raw)
            continue
        if not line.strip():
            close = None
        if line.lstrip().startswith(("```", "~~~")):
            fence = not fence
            out.append(mark)
        elif fence:
            out.append(mark)
        elif line.startswith(("    ", "\t")) and (prev_blank or indented):
            indented = True
            out.append(mark)
        else:
            indented = indented and not line.strip()
            kept = []
            for ch in line:
                if close:
                    close = None if ch == close else close
                elif ch in QUOTES:
                    close = QUOTES[ch]
                    kept.append('"…"')
                else:
                    kept.append(ch)
            out.append(mark + "".join(kept))
        prev_blank = not line.strip()
    return "\n".join(out)

# Extensions whose line comments start with '#'. Everything else uses '//'.
HASH_EXTS = {".py", ".sh", ".bash", ".zsh", ".rb", ".pl", ".toml", ".yaml", ".yml", ".nix", ".cfg", ".ini"}
# Files where every line is prose.
PROSE_EXTS = {".md", ".txt", ".rst", ""}
# Files a directory scan picks up.
SCAN_EXTS = HASH_EXTS | {".md", ".rst", ".rs", ".c", ".h", ".cc", ".cpp", ".go", ".java", ".js", ".ts", ".tsx", ".swift", ".kt"}
STAMP_RE = re.compile(r"style-ok:\s*(\S)?")
# A stamp on the line above counts only when it is that line's whole comment.
STAMP_LINE_RE = re.compile(r"^\s*(//+|#+|\*)\s*style-ok:")


def compile_detector(entry):
    """Compiles one detector. `re:` marks a regex. Anything else is a
    case-insensitive literal."""
    if entry.startswith("re:"):
        return re.compile(entry[3:])
    # Word boundaries stop "ping" matching "mapping". Single words also match
    # "-s", "-ed" and "-ing", so "leverages" hits.
    if entry.isalpha() and entry.endswith("e"):
        pat = rf"\b{entry[:-1]}(?:e|es|ed|ing)\b"
    elif entry.isalpha():
        pat = rf"\b{entry}(?:s|es|ed|ing)?\b"
    else:
        pat = re.escape(entry)
        pat = (r"\b" if entry[0].isalnum() else "") + pat + (r"\b" if entry[-1].isalnum() else "")
    return re.compile(pat, re.IGNORECASE)


def load_detectors(rules):
    """Returns [(rule_id, source, compiled, prose_only)] for `rules`."""
    out = []
    for rid, rule in rules.items():
        for key, prose_only in (("detectors", False), ("prose_detectors", True)):
            for entry in rule.get(key) or []:
                try:
                    out.append((rid, entry, compile_detector(entry), prose_only))
                except re.error as e:
                    sys.exit(f"{rule['dir']}/rule.yaml: {entry}: {e}")
    return out


def prose(path, line):
    """Returns the comment part of a code line with its marker, or the whole line for prose files."""
    ext = path[path.rfind(".") :] if "." in path.rsplit("/", 1)[-1] else ""
    if ext in PROSE_EXTS:
        return line
    if line.lstrip().startswith(("*", "/*")):
        return line
    marker = r"#" if ext in HASH_EXTS else r"//"
    # The marker must follow whitespace, so "http://" and "a#b" are code.
    m = re.search(rf"(^|\s){marker}", line)
    return line[m.start() :].lstrip() if m else None


def is_framework(path):
    """A diff names the framework by its repo path, a scan by its file path."""
    inside = os.path.exists(path) and os.path.realpath(path).startswith(SKILL_DIR + os.sep)
    return "staffeng-review/" in path or inside


def unstampable(rid):
    return RULES.get(rid, {}).get("stampable") is False


def check_detectors(detectors, path, first, hunk_text, exempt=True):
    """Returns [(line_number, rule_id, detector, matched_text)] for added lines."""
    if exempt and is_framework(path):
        return []
    hits, lineno, prev = [], first, ""
    prose_file = os.path.splitext(path)[1] in {".md", ".txt", ".rst"}
    for raw in hunk_text.splitlines():
        if raw.startswith("-"):
            continue
        line, ignored = raw[1:], raw.startswith("!")
        text = prose(path, line) if raw[:1] in "+!" else None
        if text and ignored:
            hits += [(lineno, r, src, m.group(0)) for r, src, rx, _ in detectors if unstampable(r) and (m := rx.search(text))]
        elif text:
            # Anchored detectors expect the marker, others read the words after it.
            body = re.sub(r"^\s*(//+!?|#+|/?\*+)\s?", "", text)
            if prose_file:
                # A line of prose reads as a whole comment.
                text, body = "// " + line, line
            stamp = STAMP_RE.search(line) or (STAMP_LINE_RE.match(prev) and STAMP_RE.search(prev))
            if stamp and not stamp.group(1):
                hits.append((lineno, "stamp", "style-ok without a reason", stamp.group(0)))
            for rid, source, rx, prose_only in detectors:
                if prose_only and not prose_file:
                    continue
                m = rx.search(text) or rx.search(body)
                if m and (unstampable(rid) or not (stamp and stamp.group(1))):
                    hits.append((lineno, rid, source, m.group(0)))
        prev, lineno = line, lineno + 1
    return hits


TEST_DEF_RE = re.compile(r"\s*(async\s+)?def\s+test")
TRIPLE = ('"""', "'''")


def is_test_path(path):
    name = path.rsplit("/", 1)[-1]
    return "/tests/" in f"/{path}" or name.startswith("test_") or name.endswith("_test.py")


def test_docstring_lines(lines):
    """Returns the indexes of lines in the docstrings of test functions."""
    found, i = set(), 0
    while i < len(lines):
        if TEST_DEF_RE.match(lines[i][1:]) and i + 1 < len(lines) and lines[i + 1][1:].lstrip().startswith(TRIPLE):
            end = i + 1
            if sum(lines[end].count(q) for q in TRIPLE) < 2:
                end += 1
                while end < len(lines) and not any(q in lines[end] for q in TRIPLE):
                    end += 1
            found.update(range(i + 1, end + 1))
            i = end
        i += 1
    return found


def without_comment(path, line):
    """Returns `line` with its comment removed, or None if it is only a comment."""
    comment = prose(path, line)
    if comment is None:
        return line
    return line[: line.rfind(comment)].rstrip() or None


def confirm_text(path, hunk_text, rule):
    """Returns the hunk for re-checking a hit on `rule`. Removed lines go.
    Context lines go for rules with `recheck: added`, and keep only their
    code for the rest. With `recheck_skips_test_docstrings`, test docstrings in
    a test file go too."""
    lines = [l for l in jev_view(hunk_text).splitlines() if not l.startswith("-")]
    skip_docs = jev(rule).get("recheck_skips_test_docstrings") and is_test_path(path)
    skip = test_docstring_lines(lines) if skip_docs else set()
    out = []
    for i, raw in enumerate(lines):
        if i in skip:
            continue
        if raw.startswith("+"):
            out.append(raw)
        elif jev(rule).get("recheck") != "added" and (code := without_comment(path, raw[1:])) is not None:
            out.append(" " + code)
    return "\n".join(out)


def adds_prose(path, hunk_text):
    """Returns False only when the added lines are plainly code: no comment
    marker, no docstring quote anywhere in the hunk, and, for Python, the
    added lines compile. A hunk inside a long docstring has no quotes, so
    only Python can show that its lines are code."""
    lines = hunk_text.splitlines()
    added = [l[1:] for l in lines if l.startswith("+")]
    if any(prose(path, l) is not None for l in added) or any(q in l for l in lines for q in TRIPLE):
        return True
    if not path.endswith(".py"):
        return True
    try:
        compile(textwrap.dedent("\n".join(added)), path, "exec")
    except SyntaxError:
        return True
    return False


def confirm(args, path, hunk_text, scores):
    """Re-checks each hit on its confirm text. Returns {rule: (first, second)}
    for the hits the re-check puts below the threshold. The re-check asks the
    same batches as the first pass: every question in a batch shares one
    prompt, so asking a hit alone moves its score as much as the text does."""
    groups, filtered = {}, {}
    code_only = not adds_prose(path, hunk_text)
    for k, v in scores.items():
        if v < threshold(args, k) or jev(k).get("recheck") == "none":
            continue
        if code_only and jev(k).get("judges") != "code":
            filtered[k] = (v, 0.0)
            continue
        groups.setdefault(confirm_text(path, hunk_text, k), []).append(k)
    for text, keys in groups.items():
        # A whole-file scan has no context lines to filter out.
        if text == jev_view(hunk_text):
            continue
        again = decide(args, path, text) if "\n+" in "\n" + text else dict.fromkeys(keys, 0.0)
        filtered |= {k: (scores[k], again[k]) for k in keys if again[k] < threshold(args, k)}
    return filtered


# The server rejects requests with more than 10 questions.
BATCH = 10


class Stats:
    """Counts Jev requests and tokens across worker threads."""

    def __init__(self):
        self.lock = threading.Lock()
        self.requests = self.input_tokens = self.output_tokens = 0

    def add(self, usage):
        with self.lock:
            self.requests += 1
            self.input_tokens += usage.get("input_tokens", 0)
            self.output_tokens += usage.get("output_tokens", 0)


STATS = Stats()


def question(key, preamble):
    q = jev(key)
    if q["type"] == "choice":
        return {"type": "choice", "instructions": preamble + q["question"], "criteria": q["labels"]}
    return {"type": "noul", "instructions": preamble + q["question"], "criteria": {"true": q["true"], "false": q["false"]}}


def score(key, answer):
    """Returns the probability that the hunk breaks rule `key`."""
    if jev(key)["type"] == "choice":
        return answer["probabilities"][jev(key)["violation"]]
    return answer["noul"]


def threshold(args, key):
    """Returns the Jev score at which rule `key` fires: its own `threshold`,
    or --threshold."""
    return jev(key).get("threshold", args.threshold)


def jev_keys(only, prose_file):
    """Returns the rules to ask Jev about, in batch order. `only` limits them
    to a list of rule ids."""
    return [
        k for k, rule in RULES.items()
        if rule.get("jev") and (not only or k in only) and k not in PARSED
        and not (prose_file and jev(k).get("applies_to") == "code")
    ]


def decide(args, path, hunk_text, keys=None):
    """Returns {rule: probability} for one hunk, one request per batch of rules.
    Prose files skip rules with `applies_to: code`."""
    prose_file = os.path.splitext(path)[1] in {".md", ".txt", ".rst"}
    keys = jev_keys(args.rules, prose_file) if keys is None else keys
    preamble = PROSE_PREAMBLE if prose_file else PREAMBLE
    hunk_text = jev_view(hunk_text)
    if is_markdown(path):
        # A table is data, and every prose question misreads it.
        hunk_text = "\n".join(l for l in hunk_text.splitlines() if not TABLE_ROW_RE.match(l[1:]))
    scores = {}
    if "\n+" not in "\n" + hunk_text:
        return scores
    size = getattr(args, "batch", BATCH)
    for i in range(0, len(keys), size):
        questions = {k: question(k, preamble) for k in keys[i : i + size]}
        body = json.dumps({"model": args.model, "state": hunk_text, "questions": questions}).encode()
        req = urllib.request.Request(args.url, body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                res = json.load(resp)
        except urllib.error.HTTPError as e:
            sys.exit(f"{args.url}: HTTP {e.code}: {e.read().decode()}")
        STATS.add(res.get("usage", {}))
        scores |= {k: score(k, a) for k, a in res["answers"].items()}
    ext = os.path.splitext(path)[1]
    added = [l[1:] for l in hunk_text.splitlines() if l.startswith("+") and l[1:].strip()]
    only_headings = prose_file and added and all(HEADING_RE.match(l) for l in added)
    test_doc = documents_test(hunk_text)
    return {
        k: v for k, v in scores.items()
        if ext not in (jev(k).get("ignore_score_for") or [])
        and not (only_headings and jev(k).get("ignore_headings"))
        and not (test_doc and jev(k).get("ignore_test_docs"))
    }


def source_lines(hunks, scan):
    """Returns {(path, line_number): text} for display, before any masking."""
    if scan:
        out = {}
        for path in dict.fromkeys(h[0] for h in hunks):
            with open(path) as f:
                out |= {(path, n): l for n, l in enumerate(f.read().splitlines(), 1)}
        return out
    out = {}
    for path, first, _, text in hunks:
        n = first
        for raw in text.splitlines():
            if not raw.startswith("-"):
                out[path, n], n = raw[1:], n + 1
    return out


def added_lines(first, hunk_text):
    """Returns the line numbers of a hunk's non-blank added lines."""
    out, n = [], first
    for raw in hunk_text.splitlines():
        if raw.startswith("-"):
            continue
        if raw.startswith("+") and raw[1:].strip():
            out.append(n)
        n += 1
    return out


def excerpt(line, matches, color, width=120, strip=True):
    """Returns `line` trimmed to `width` around the first match, with matches highlighted."""
    line = line.strip() if strip else line.rstrip()
    at = line.find(matches[0]) if matches and matches[0] in line else 0
    if len(line) > width:
        start = max(0, min(at - width // 3, len(line) - width))
        line = ("…" if start else "") + line[start : start + width] + ("…" if start + width < len(line) else "")
    if color:
        for m in dict.fromkeys(m.strip() for m in matches if m.strip()):
            line = line.replace(m, f"\033[1;31m{m}\033[0m", 1)
    return line


def describe(key):
    q = jev(key)
    return q["labels"][q["violation"]] if q["type"] == "choice" else q["true"]


STAGES = {0: "existence", 1: "hazards first", 2: "structure", 3: "words", 4: "clutter", 5: "humanity", 6: "provenance"}
GROUPS = {"comments": "Comments and prose", "markdown": "Markdown", "commits": "Commit messages", "code": "Code"}
INDEX_START, INDEX_END = "<!-- rules-index -->", "<!-- /rules-index -->"


def rules_index():
    """Returns the markdown list of rules that SKILL.md holds between its index markers."""
    lines = []
    for group in [*GROUPS, *sorted({r.get("group", "other") for r in RULES.values()} - set(GROUPS))]:
        members = [r for r in RULES.values() if r.get("group", "other") == group]
        if not members:
            continue
        lines += [f"### {GROUPS.get(group, group.capitalize())}", ""]
        stage = None
        for r in members:
            if r.get("stage") is not None and r["stage"] != stage:
                stage = r["stage"]
                if lines[-1]:
                    lines.append("")
                lines += [f"Stage {stage}: {STAGES.get(stage, '')}", ""]
            lines.append(f"- [{r['id']}](rules/{r['id']}/rule.md): {r['description']}")
        lines.append("")
    return "\n".join(lines).rstrip("\n")


def skill_index():
    """Returns the rules index as SKILL.md holds it, or None without markers."""
    with open(os.path.join(SKILL_DIR, "SKILL.md")) as f:
        text = f.read()
    if INDEX_START not in text or INDEX_END not in text:
        return None
    return text.split(INDEX_START, 1)[1].split(INDEX_END, 1)[0].strip("\n")


EXAMPLE_SEP_RE = re.compile(r"^---[ \t]*$", re.M)
FIX_SEP_RE = re.compile(r"^=>[ \t]*$", re.M)


def read_examples(path, pairs=False):
    """Returns the examples in a good, bad or fixes file. A `---` line
    separates examples. In a fixes file, a `=>` line splits each example into
    (bad, good)."""
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        blocks = [b.strip("\n") for b in EXAMPLE_SEP_RE.split(f.read()) if b.strip()]
    if not pairs:
        return blocks
    out = []
    for i, block in enumerate(blocks, 1):
        halves = FIX_SEP_RE.split(block)
        if len(halves) != 2:
            sys.exit(f"{path}: example {i} needs one '=>' line")
        out.append((halves[0].strip("\n"), halves[1].strip("\n")))
    return out


CODE_HINT_RE = re.compile(r"^\s*(//|/\*|\*\s)|\{\s*$|^\s*\}|;\s*$|^\s*(fn|def|let|const|return|pub|impl|struct)\b|^\s*#\[", re.M)


HEADING_RE = re.compile(r"^#{1,6}\s")
TABLE_ROW_RE = re.compile(r"^\s*\|")
MARKDOWN_HINT_RE = re.compile(r"^#{1,6}\s|^\s*[-*+]\s|^\s*\|.*\|\s*$|\*\*[^*]+\*\*", re.M)


def guess_ext(text):
    """Returns ".rs" for text with a comment marker or code in it, ".md" for
    prose with a heading, list item or bold span, and ".txt" for plain prose."""
    if CODE_HINT_RE.search(text):
        return ".rs"
    return ".md" if MARKDOWN_HINT_RE.search(text) else ".txt"


def example_hunk(rid, text):
    """Returns (path, hunk_text) for an example. The rule's `examples` type
    wins, and otherwise `guess_ext` picks one per example."""
    path = "example" + (RULES[rid].get("examples") or guess_ext(text))
    hunk = "\n".join(apply_ignore(["+" + l for l in text.splitlines()]))
    return path, mask_markdown(hunk) if is_markdown(path) else hunk


def run_tests(args):
    """Checks each rule's examples. A bad example, and the bad half of a fix,
    must fire the rule. A good example, and the good half of a fix, must not.
    Returns the exit status."""
    ids = args.paths or list(RULES)
    if unknown := [i for i in ids if i not in RULES]:
        sys.exit(f"unknown rules: {', '.join(unknown)}")
    cases, problems = [], {}
    for rid in ids:
        folder = RULES[rid]["dir"]
        problems[rid] = [f"missing {n}" for n in ("rule.md", "good.txt", "bad.txt", "fixes.txt")
                         if not os.path.isfile(os.path.join(folder, n))]
        for name, expect in (("bad.txt", True), ("good.txt", False)):
            cases += [(rid, f"{name} #{i}", text, expect)
                      for i, text in enumerate(read_examples(os.path.join(folder, name)), 1)]
        for i, (bad, good) in enumerate(read_examples(os.path.join(folder, "fixes.txt"), pairs=True), 1):
            cases += [(rid, f"fixes.txt #{i} bad", bad, True), (rid, f"fixes.txt #{i} good", good, False)]

    detectors = load_detectors(RULES)
    started = time.monotonic()
    wanted = sorted({example_hunk(rid, text) for rid, _, text, _ in cases
                     if jev(rid) and rid not in PARSED and not args.no_jev})
    parse_cases = sorted({example_hunk(rid, text) for rid, _, text, _ in cases if rid in PARSED})
    parsed = parse_hits([(p, 1, h) for p, h in parse_cases], PARSED) or [[] for _ in parse_cases]
    parsed = dict(zip(parse_cases, parsed))
    # Every example asks the full batch, as a real run does.
    with ThreadPoolExecutor(args.jobs) as pool:
        scores = dict(zip(wanted, pool.map(lambda h: decide(args, h[0], h[1], jev_keys(None, is_prose_path(h[0]))), wanted)))
    elapsed = time.monotonic() - started

    results = {rid: [] for rid in ids}
    for rid, where, text, expect in cases:
        path, hunk = example_hunk(rid, text)
        if rid in PARSED:
            found = [(h[0], rid, "parse", h[3]) for h in parsed[path, hunk] if h[1] == rid and h[2] == "fail"]
            score = None
        else:
            found = check_detectors([d for d in detectors if d[0] == rid], path, 1, hunk, exempt=False)
            score = scores.get((path, hunk), {}).get(rid)
        fired = bool(found) or (score is not None and score >= threshold(args, rid))
        results[rid].append((where, text, expect, fired, found, score))

    failed = manual = 0
    for rid in ids:
        rule, rows = RULES[rid], results[rid]
        checked = bool(rule.get("detectors") or rule.get("prose_detectors") or (jev(rid) and not args.no_jev)
                       or rid in PARSED)
        misses = [r for r in rows if r[2] and not r[3]]
        false_hits = [r for r in rows if not r[2] and r[3]]
        bad_n, good_n = sum(r[2] for r in rows), sum(not r[2] for r in rows)
        # Without Jev, a miss on a rule that has a question is not a failure.
        soft = args.no_jev and bool(jev(rid)) and rid not in PARSED
        bad_rows = false_hits + ([] if soft else misses)
        if not checked:
            status, manual = "--  ", manual + 1
        elif bad_rows or problems[rid]:
            status, failed = "FAIL", failed + 1
        else:
            status = "ok  "
        caught = f"{bad_n - len(misses)}/{bad_n} bad caught, {good_n - len(false_hits)}/{good_n} good passed"
        note = "review by hand" if not checked else caught
        print(f"{status} {rid}: {note}")
        for problem in problems[rid]:
            print(f"       {problem}")
        if not checked:
            continue
        for where, text, expect, _, found, score in (false_hits + misses if not soft else false_hits + misses):
            why = []
            if found:
                why.append("detector " + ", ".join(repr(h[3]) for h in found[:3]))
            if score is not None:
                why.append(f"jev {score:.2f}")
            verdict = "flagged" if not expect else "missed"
            first = excerpt(text.splitlines()[0] if text.strip() else "", [], False, width=90)
            print(f"       {where} {verdict} ({'; '.join(why) or 'no detector hit'}): {first}")

    stale = skill_index() != rules_index()
    if stale:
        print("FAIL SKILL.md rules index differs from `jev_style.py --index`")
    print(f"{len(ids)} rules: {len(ids) - failed - manual} ok, {failed} failed, {manual} by hand")
    if STATS.requests and not args.no_request_summary:
        print(f"jev: {STATS.requests} requests for {len(wanted)} examples in {elapsed:.1f}s,"
              f" {STATS.input_tokens:,} input tokens", file=sys.stderr)
    return 1 if failed or stale else 0


def is_prose_path(path):
    return os.path.splitext(path)[1] in {".md", ".txt", ".rst"}


def main():
    global RULES
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("paths", nargs="*", metavar="PATH",
                   help="a unified diff file (default: stdin), with --scan files and directories, "
                        "or with --test rule ids")
    p.add_argument("--scan", "--file", action="store_true",
                   help="check PATHs as they are, instead of a diff. Directories are walked.")
    p.add_argument("--self", action="store_true",
                   help="scan this skill's markdown files. Implies --scan --no-exempt.")
    p.add_argument("--test", action="store_true",
                   help="check each rule's good.txt, bad.txt and fixes.txt, and the SKILL.md index")
    p.add_argument("--index", action="store_true", help="print the rules index for SKILL.md")
    p.add_argument("--no-exempt", action="store_true", help="check staffeng-review's own files too")
    p.add_argument("--window", type=int, default=60, help="lines per request in --scan mode")
    p.add_argument("--rules", type=lambda s: s.split(","), help="comma-separated rule ids to check")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--all", action="store_true", help="print every rule's probability")
    p.add_argument("--show", type=int, default=6, metavar="N",
                   help="print up to N lines of a hunk with a Jev hit (default 6, 0 for none)")
    p.add_argument("--json", action="store_true", help="print scores as JSON lines")
    p.add_argument("--rules-dir", default=RULES_DIR)
    p.add_argument("--no-detectors", "--no-wordlist", action="store_true", help="skip the detector pass")
    p.add_argument("--no-jev", action="store_true", help="skip the Jev pass")
    p.add_argument("--no-parse", action="store_true",
                   help="check rules with `parse: true` with their detectors and Jev question")
    p.add_argument("--no-confirm", action="store_true", help="keep Jev hits without re-checking them")
    p.add_argument("--url", default=URL)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--timeout", type=float, default=60)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--batch", type=int, default=BATCH, choices=range(1, BATCH + 1), metavar="N",
                   help=f"questions per Jev request, 1 to {BATCH} (default {BATCH})")
    p.add_argument("--no-request-summary", action="store_true", help="skip the Jev request summary on stderr")
    args = p.parse_args()
    try:
        RULES = load_rules(args.rules_dir)
    except (OSError, ValueError) as e:
        sys.exit(str(e))
    enable_parse(args.no_parse)
    if args.rules and (unknown := [r for r in args.rules if r not in RULES]):
        p.error(f"unknown rules: {', '.join(unknown)}")
    if args.index:
        print(rules_index())
        return
    if args.test:
        sys.exit(run_tests(args))
    if args.self:
        args.scan, args.no_exempt = True, True
        args.paths += [os.path.join(SKILL_DIR, n) for n in ("SKILL.md", "COMMENTS.md")]
        args.paths += [os.path.join(args.rules_dir, "README.md")] * os.path.isfile(os.path.join(args.rules_dir, "README.md"))
        args.paths += [os.path.join(r["dir"], "rule.md") for r in RULES.values() if os.path.isfile(os.path.join(r["dir"], "rule.md"))]
    if not args.scan and len(args.paths) > 1:
        p.error("a diff takes one PATH. Use --scan to check files.")
    if missing := [path for path in args.paths if not os.path.exists(path)]:
        p.error(f"no such file or directory: {', '.join(missing)}")

    if args.scan:
        if not args.paths:
            p.error("--scan needs at least one PATH")
        hunks = [h for path in scan_paths(args.paths) for h in file_hunks(path, args.window)]
    else:
        hunks = list(parse_diff(open(args.paths[0]).read() if args.paths else sys.stdin.read()))
    if not hunks:
        sys.exit("no hunks with added lines")
    sources = source_lines(hunks, args.scan)
    hunks = [(path, first, last, "\n".join(apply_ignore(text.splitlines()))) for path, first, last, text in hunks]
    hunks = [(path, first, last, mask_markdown(text) if is_markdown(path) else text) for path, first, last, text in hunks]

    chosen = {k: r for k, r in RULES.items() if (not args.rules or k in args.rules) and k not in PARSED}
    detectors = [] if args.no_detectors else load_detectors(chosen)
    found_all = [
        [h for h in check_detectors(detectors, path, first, text, not args.no_exempt)
         if not (documents_test(text) and jev(h[1]).get("ignore_test_docs"))]
        for path, first, _, text in hunks
    ]
    parse_rules = {k for k in PARSED if not args.rules or k in args.rules}
    prose_hunks = [i for i, (path, first, last, _) in enumerate(hunks) if HUNK_KIND.get((path, first, last)) != "code"]
    parsed = parse_hits([(hunks[i][0], hunks[i][1], hunks[i][3]) for i in prose_hunks], parse_rules) or []
    for i, hits in zip(prose_hunks, parsed):
        found_all[i] += [(line, rid, "parse" if level == "fail" else "parse lead", marker)
                         for line, rid, level, marker, _ in hits
                         if not (documents_test(hunks[i][3]) and jev(rid).get("ignore_test_docs"))]
        found_all[i].sort(key=lambda h: h[0])
    started = time.monotonic()
    if args.no_jev:
        results = [{} for _ in hunks]
    else:
        with ThreadPoolExecutor(args.jobs) as pool:
            results = list(pool.map(lambda h: decide(args, h[0], h[3], hunk_keys(args, h)), hunks))
    first_pass = STATS.requests
    filtered = [{} for _ in hunks]
    if not (args.no_jev or args.no_confirm):
        with ThreadPoolExecutor(args.jobs) as pool:
            filtered = list(pool.map(lambda hs: confirm(args, hs[0][0], hs[0][3], hs[1]), zip(hunks, results)))
    elapsed = time.monotonic() - started

    color = sys.stdout.isatty()
    failed = 0
    for (path, first, last, text), found, scores, dropped in zip(hunks, found_all, results, filtered):
        if args.json:
            found = [[*h, sources.get((path, h[0]), "")] for h in found]
            print(json.dumps({"path": path, "lines": [first, last], "detectors": found, "scores": scores,
                              "filtered": dropped}))
            continue
        hits = {k: v for k, v in scores.items() if v >= threshold(args, k) and k not in dropped}
        failed += bool(hits or [h for h in found if h[2] != "parse lead"])
        where = f"{path}:{first}-{last}"
        shown = {k: v for k, v in scores.items() if k not in dropped} if args.all else hits
        if not (shown or found or (args.all and dropped)):
            print(f"{where} - clean")
            continue
        print(where)
        by_line = {}
        for hit in found:
            by_line.setdefault(hit[0], []).append(hit)
        for lineno, line_hits in by_line.items():
            print(f"  L{lineno} | {excerpt(sources.get((path, lineno), ''), [h[3] for h in line_hits], color)}")
            for _, rid, source, match in line_hits:
                mark = "?" if source == "parse lead" else "x"
                source = source if len(source) <= 60 else source[:57] + "..."
                print(f"      [{mark}] {rid} - {match!r} ({source})")
        for k, v in sorted(shown.items(), key=lambda kv: -kv[1]):
            mark = "x" if v >= threshold(args, k) else " "
            print(f"  [{mark}] {v:.2f} {k} (jev) - {describe(k)}")
        if args.all:
            for k, (v, again) in dropped.items():
                print(f"  [-] {v:.2f} -> {again:.2f} {k} (jev) - filtered on re-check")
        if hits and args.show:
            numbers = added_lines(first, text)
            pad = len(str(max(numbers[: args.show], default=0)))
            for n in numbers[: args.show]:
                print(f"      L{n:<{pad}} | {excerpt(sources.get((path, n), ''), [], color, strip=False)}")
            if len(numbers) > args.show:
                print(f"      … {len(numbers) - args.show} more lines")
    if STATS.requests and not args.no_request_summary:
        rechecks = STATS.requests - first_pass
        print(
            f"jev: {STATS.requests} requests ({rechecks} re-checks) for {len(hunks)} hunks"
            f" in {elapsed:.1f}s, {STATS.requests / elapsed:.1f} req/s,"
            f" {STATS.input_tokens:,} input tokens, {STATS.output_tokens:,} output tokens",
            file=sys.stderr,
        )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
