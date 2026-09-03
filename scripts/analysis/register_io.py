#!/usr/bin/env python3
"""register_io.py -- programmatic read/write for the livo_recon Register and Archive.

The Register and the Archive are hand-written HTML documents published as claude.ai
artifacts.  Two parties edit them: the coding agent on h2 writes round reports and
``[BATCH-*]`` data blocks INTO the Register (the inbox), and Claude reads them out,
files them into the Archive (the record), and cuts them from the Register leaving a
pointer.  That contract is rule 31-0, and until now both halves were done by hand --
ad-hoc regexes on one side and one-off Python splices on the other.

This module is that contract as code.

DESIGN: BYTE-PRESERVING SPLICES
-------------------------------
Rule 12 says nothing is edited on the way into the Archive.  A parse-then-serialize
round trip would silently reformat every untouched line, which violates that rule in
a way no diff review would catch.  So this module never re-serializes.  It builds an
*index* of element byte-spans with an offset-tracking parser, and every mutation is a
slice on the original string.  ``Doc.roundtrip_ok()`` asserts the invariant directly:
indexing a document and reassembling it from its own spans must be byte-identical.

Edits are STAGED and applied once, in descending offset order, with overlap detection.
Applying edits in ascending order invalidates every offset after the first one -- that
is the specific bug this API exists to make unrepresentable.

WHAT IT DOES NOT DO
-------------------
It does not talk to claude.ai.  Rule 19: never handle credentials.  It reads and
writes local HTML files; publishing stays a human (or Claude-with-the-Artifact-tool)
step.  ``body`` recovers a publishable body from a saved artifact file so the two
halves meet cleanly.

RULE 21 NOTE
------------
Rule 21's "I verified this means I read the code" applies to livo_recon's C++ and to
anything needing ROS/CUDA/bags.  It does not apply here: this is stdlib-only Python
over HTML files, so it can be, and has been, actually executed and checked against the
live documents.  ``--self-check`` is that check, and it is worth re-running after any
edit to this file.

CLI
---
    register_io.py report    FILE
    register_io.py validate  FILE [--strict] [--no-compaction]
    register_io.py blocks    FILE [--out DIR] [--json] [--name Q1-SERIES]
    register_io.py show      FILE (--card ID | --section TITLE | --row TABLE:ID | --block NAME)
    register_io.py cut       FILE (--card ID | --section TITLE)... [--pointer HTML] -o OUT
    register_io.py file      --register R --archive A --title T (--card ID | --section T)...
                             -o-register OUT -o-archive OUT [--anchor "Closed defects"]
    register_io.py actual    FILE --id PRED_ID (--html H | --html-file F) -o OUT
    register_io.py addrow    FILE --table queue (--html H | --html-file F)
                             [--at top|bottom] [--before ID] [--after ID] -o OUT
    register_io.py rmrow     FILE --table queue --id ID -o OUT
    register_io.py body      SAVED_ARTIFACT -o OUT
    register_io.py diff      OLD NEW
    register_io.py emit      --name Q1-VERIFY --version 1 --csv FILE
    register_io.py inject    FILE --after-card ID --caption TEXT --block FILE... -o OUT
    register_io.py self-check FILE [FILE...]

Every subcommand that writes takes an explicit ``-o``; nothing is edited in place.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import html as _html
import json
import os
import re
import sys
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Callable, Iterable, Iterator, Optional, Sequence

__all__ = [
    "Doc", "Node", "Block", "Finding",
    "norm_id", "esc_id", "strip_tags", "body_of",
    "validate", "structural_diff", "emit_block",
]

# --------------------------------------------------------------------------------------
# 0.  Text normalisation
# --------------------------------------------------------------------------------------

# The documents write ids with U+2011 NON-BREAKING HYPHEN (&#8209;) so that "DX-5" never
# line-breaks.  A naive grep for "C-6" therefore returns nothing -- which happened, and
# cost a round of confusion.  Every id comparison in this module goes through norm_id.
_DASHES = {
    "‐": "-",  # hyphen
    "‑": "-",  # non-breaking hyphen        <- the one that bites
    "‒": "-",  # figure dash
    "–": "-",  # en dash
    "—": "-",  # em dash
    "−": "-",  # minus sign
    " ": " ",  # nbsp
    " ": " ",  # thin space
    " ": " ",  # hair space
}

_TAG_RE = re.compile(r"<[^>]*>")
_WS_RE = re.compile(r"\s+")


def strip_tags(s: str) -> str:
    """Tags out, entities resolved, whitespace collapsed.  For comparison, not display."""
    s = _TAG_RE.sub(" ", s)
    s = _html.unescape(s)
    for k, v in _DASHES.items():
        s = s.replace(k, v)
    return _WS_RE.sub(" ", s).strip()


def norm_id(s: str) -> str:
    """Canonical form of a row/card id: 'DX&#8209;5' -> 'DX-5', 'T7&#8209;s1 / SW&#8209;5'
    -> 'T7-s1 / SW-5'.  Case is preserved; ids in these documents are case-meaningful
    (``Q-L5`` vs ``q-l5`` would be a different cell)."""
    return strip_tags(s)


def esc_id(s: str) -> str:
    """Inverse of norm_id for writing: '-' becomes the non-breaking hyphen entity, so a
    new row's id renders like every existing one."""
    return s.replace("-", "&#8209;")


def _fold(s: str) -> str:
    """Case-folded id, for tolerant lookup only."""
    return norm_id(s).casefold()


# --------------------------------------------------------------------------------------
# 1.  Byte-preserving element index
# --------------------------------------------------------------------------------------

VOID = frozenset(
    "area base br col embed hr img input link meta param source track wbr".split()
)


@dataclass
class Node:
    """One element, addressed by byte offsets into the source it came from.

    ``start``/``end`` bound the whole element including its tags; ``inner_start``/
    ``inner_end`` bound its children.  For a void element the inner span is empty.
    """

    tag: str
    attrs: dict[str, str]
    start: int
    end: int
    inner_start: int
    inner_end: int
    depth: int
    parent: Optional["Node"] = field(default=None, repr=False, compare=False)
    children: list["Node"] = field(default_factory=list, repr=False, compare=False)
    line: int = 0

    # -- accessors ---------------------------------------------------------------------
    def outer(self, src: str) -> str:
        return src[self.start : self.end]

    def inner(self, src: str) -> str:
        return src[self.inner_start : self.inner_end]

    def text(self, src: str) -> str:
        return strip_tags(self.inner(src))

    def span(self) -> tuple[int, int]:
        return (self.start, self.end)

    def cls(self) -> list[str]:
        return (self.attrs.get("class") or "").split()

    def has_class(self, name: str) -> bool:
        return name in self.cls()

    def find(self, tag: str = None, cls: str = None, deep: bool = True) -> Iterator["Node"]:
        """Descendants matching tag and/or class, in document order."""
        stack = list(reversed(self.children))
        while stack:
            n = stack.pop()
            if (tag is None or n.tag == tag) and (cls is None or n.has_class(cls)):
                yield n
            if deep:
                stack.extend(reversed(n.children))

    def first(self, tag: str = None, cls: str = None) -> Optional["Node"]:
        return next(self.find(tag, cls), None)


class _StructureWarning(Exception):
    pass


class _Indexer(HTMLParser):
    """HTMLParser subclass that records absolute byte offsets for every element.

    ``convert_charrefs=False`` keeps entities intact in handle_data so nothing shifts.
    Mismatched or unclosed tags are RECORDED rather than coped with silently -- these
    documents are hand-edited and a stray ``</strong>`` is a real and recurring defect
    (one shipped in this very session and was caught only by a manual balance count).
    """

    def __init__(self, src: str):
        super().__init__(convert_charrefs=False)
        self.src = src
        # line -> absolute offset of that line's first character
        self.line_off = [0]
        for ln in src.split("\n")[:-1]:
            self.line_off.append(self.line_off[-1] + len(ln) + 1)
        self.roots: list[Node] = []
        self.all: list[Node] = []
        self.stack: list[Node] = []
        self.problems: list[str] = []

    def _abs(self) -> int:
        line, col = self.getpos()
        return self.line_off[line - 1] + col

    def _open(self, tag: str, attrs, self_closing: bool) -> None:
        start = self._abs()
        raw = self.get_starttag_text() or ""
        inner_start = start + len(raw)
        node = Node(
            tag=tag,
            attrs={k: (v if v is not None else "") for k, v in attrs},
            start=start,
            end=inner_start if (self_closing or tag in VOID) else -1,
            inner_start=inner_start,
            inner_end=inner_start if (self_closing or tag in VOID) else -1,
            depth=len(self.stack),
            parent=self.stack[-1] if self.stack else None,
            line=self.getpos()[0],
        )
        self.all.append(node)
        if node.parent is not None:
            node.parent.children.append(node)
        else:
            self.roots.append(node)
        if not (self_closing or tag in VOID):
            self.stack.append(node)

    # -- HTMLParser hooks --------------------------------------------------------------
    def handle_starttag(self, tag, attrs):
        self._open(tag, attrs, self_closing=False)

    def handle_startendtag(self, tag, attrs):
        self._open(tag, attrs, self_closing=True)

    def handle_endtag(self, tag):
        estart = self._abs()
        gt = self.src.find(">", estart)
        eend = (gt + 1) if gt != -1 else estart + len(tag) + 3

        # Find the nearest matching open element.  Anything above it was left unclosed.
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i].tag == tag:
                for orphan in self.stack[i + 1 :]:
                    self.problems.append(
                        f"line {orphan.line}: <{orphan.tag}> never closed "
                        f"(implicitly closed by </{tag}> at line {self.getpos()[0]})"
                    )
                    orphan.inner_end = estart
                    orphan.end = estart
                node = self.stack[i]
                node.inner_end = estart
                node.end = eend
                del self.stack[i:]
                return
        self.problems.append(
            f"line {self.getpos()[0]}: stray </{tag}> with no matching open tag"
        )

    def close(self):  # noqa: A003 - HTMLParser API
        super().close()
        for orphan in self.stack:
            self.problems.append(f"line {orphan.line}: <{orphan.tag}> never closed (EOF)")
            orphan.inner_end = len(self.src)
            orphan.end = len(self.src)
        self.stack.clear()


# --------------------------------------------------------------------------------------
# 2.  Blocks
# --------------------------------------------------------------------------------------

# [Q1-SERIES v1] ... [/Q1-SERIES]   -- the round-report data block format (rule 5a/5c).
#
# Markup may sit between the marker and the body: the Register puts a whole block inside
# one <div class="knobs">, while the Archive labels each one
# <p class="blab"><b>[Q1-META v1]</b></p> and follows it with the body.  Both are found,
# because matching runs over a tag-MASKED copy of the source in which every tag has been
# replaced by an equal number of spaces -- so offsets stay exact while markup stops
# interrupting the pattern.
_BLOCK_RE = re.compile(
    r"\[(?P<name>[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)+)\s+v(?P<ver>\d+)\][ \t\r]*\n"
    r"(?P<body>.*?)\r?\n?[ \t]*"
    r"\[/(?P=name)\]",
    re.S,
)


def _mask_tags(src: str) -> str:
    """Replace every tag with spaces of the same length.  Offsets are preserved exactly,
    which is what lets a regex span markup without breaking the byte index."""
    return _TAG_RE.sub(lambda m: " " * len(m.group()), src)
_BLOCK_OPEN_RE = re.compile(r"\[([A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)+)\s+v(\d+)\]")
_BLOCK_CLOSE_RE = re.compile(r"\[/([A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)+)\]")

_NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def unwrap_csv(body: str) -> tuple[str, bool]:
    """Rejoin a CSV table that has been hard-wrapped, e.g. by an editor or a paste.

    Returns ``(text, was_wrapped)``.  A 45-column SUMMARY header is ~700 characters and
    survives one careless reflow as three physical lines; the block then stops being a
    table and every downstream check silently skips it.  Found exactly that way in the
    Archive, where four SUMMARY blocks arrived wrapped.

    The join is greedy to the modal comma count, so a table that is already well-formed
    comes back untouched and ``was_wrapped`` is False.
    """
    lines = [l for l in body.split("\n") if l.strip()]
    if len(lines) < 3:
        return body, False
    counts = [l.count(",") for l in lines]
    if all(c == counts[0] for c in counts):
        return body, False  # already a table

    def _ok(recs: list[str]) -> bool:
        return (len(recs) >= 2 and recs[0].count(",") >= 3
                and all(r.count(",") == recs[0].count(",") for r in recs))

    # Strategy 1 -- FIXED-WIDTH WRAP.  The observed case: a 45-column SUMMARY reflowed
    # at 200 characters, so every continuation line is exactly the wrap width and the
    # last piece of each record is shorter.  Rejoin on that boundary.
    width = max(len(l) for l in lines)
    at_width = sum(1 for l in lines if len(l) == width)
    if 2 <= at_width < len(lines):
        recs, buf = [], ""
        for l in lines:
            buf += l
            if len(l) < width:
                recs.append(buf)
                buf = ""
        if buf:
            recs.append(buf)
        if _ok(recs):
            return "\n".join(recs), True

    # Strategy 2 -- greedy join to the modal comma count, for an irregular reflow.
    modal = max(set(counts), key=counts.count)
    if modal >= 3:
        recs, buf = [], ""
        for l in lines:
            buf += l
            if buf.count(",") >= modal:
                recs.append(buf)
                buf = ""
        if buf:
            recs.append(buf)
        if _ok(recs):
            return "\n".join(recs), True

    return body, False


def _as_float(tok: str) -> Optional[float]:
    tok = tok.strip()
    if not tok:
        return None
    if tok.lower() in ("nan", "-nan", "none", "null"):
        return float("nan")
    if _NUM_RE.match(tok):
        try:
            return float(tok)
        except ValueError:
            return None
    return None


def _isnan(x) -> bool:
    return isinstance(x, float) and x != x


@dataclass
class Block:
    """One ``[NAME vN] ... [/NAME]`` data block, with the span it occupied."""

    name: str
    version: int
    body: str  # entity-decoded plain text
    start: int  # offsets into the source the block was found in
    end: int
    batch: str = ""  # 'Q1' from 'Q1-SERIES'
    kind: str = ""  # 'SERIES' from 'Q1-SERIES'
    wrapped: bool = False  # body was hard-wrapped and had to be rejoined
    delimited: bool = True  # had its own [/NAME] closing line

    def __post_init__(self):
        if "-" in self.name:
            self.batch, _, self.kind = self.name.rpartition("-")

    # -- shape -------------------------------------------------------------------------
    def lines(self) -> list[str]:
        return [ln for ln in self.body.split("\n") if ln.strip()]

    def looks_like_data(self) -> bool:
        """A CSV table, or a META block's ``key=value`` lines.

        Guards block RECOVERY: a card may legitimately *specify* a block's columns --
        the DX-2 card writes ``[DX2-SERIES v2]`` above an annotated column list -- and a
        specification of a block that has never been emitted is not a block.
        """
        if self.is_csv():
            return True
        ls = self.lines()
        if len(ls) < 3:
            return False
        kvish = sum(1 for l in ls if re.match(r"^\w+=\S", l))
        return kvish >= 0.6 * len(ls)

    def is_csv(self) -> bool:
        ls = self.lines()
        if len(ls) < 2 or "," not in ls[0]:
            return False
        n = ls[0].count(",")
        return all(l.count(",") == n for l in ls[1:])

    def header(self) -> list[str]:
        return [c.strip() for c in self.lines()[0].split(",")] if self.is_csv() else []

    def rows(self) -> list[dict[str, str]]:
        """CSV rows as dicts.  Empty for a non-CSV block (META, and anything free-form)."""
        if not self.is_csv():
            return []
        hdr = self.header()
        out = []
        for ln in self.lines()[1:]:
            cells = [c.strip() for c in ln.split(",")]
            out.append(dict(zip(hdr, cells)))
        return out

    def kv(self) -> dict[str, str]:
        """``key=value`` lines as a dict.  This is the META block's shape."""
        out = {}
        for ln in self.lines():
            if "=" in ln and "," not in ln.split("=")[0]:
                k, _, v = ln.partition("=")
                out[k.strip()] = v.strip()
        return out

    def column(self, name: str) -> list[str]:
        return [r.get(name, "") for r in self.rows()]

    def numeric_column(self, name: str) -> list[Optional[float]]:
        return [_as_float(v) for v in self.column(name)]

    def __repr__(self):  # pragma: no cover - debugging aid
        shape = f"{len(self.rows())} rows" if self.is_csv() else f"{len(self.lines())} lines"
        return f"<Block {self.name} v{self.version} {shape}>"


def find_blocks(src: str) -> list[Block]:
    """Every well-formed block in ``src``, in document order.

    Block bodies live inside ``<div class="knobs">`` and are therefore HTML-escaped
    (``-&gt;`` in a PAIRS block, ``&lt;-- INERT`` in a knobs listing).  They are decoded
    here so downstream code compares plain text.
    """
    masked = _mask_tags(src)
    out: list[Block] = []
    for m in _BLOCK_RE.finditer(masked):
        # Slice the ORIGINAL source at the masked match's offsets, then drop any markup
        # that fell inside the body, so the block text is what the block says it is.
        raw = src[m.start("body") : m.end("body")]
        text, was_wrapped = unwrap_csv(_html.unescape(_TAG_RE.sub("", raw)))
        out.append(
            Block(
                name=m.group("name"),
                version=int(m.group("ver")),
                body=text,
                start=m.start(),
                end=m.end(),
                wrapped=was_wrapped,
            )
        )
    return out


def emit_block(name: str, version: int, body: str) -> str:
    """The canonical text form of a block, for the agent side.

    ``body`` is plain text (a CSV table, or ``key=value`` lines).  The result is NOT
    HTML-escaped -- use ``wrap_blocks`` when injecting into a document.
    """
    body = body.rstrip("\n")
    return f"[{name} v{version}]\n{body}\n[/{name}]"


def wrap_blocks(blocks: Sequence[str], caption_html: str) -> str:
    """Blocks in the standard ``note calm`` + ``knobs`` container the documents use."""
    inner = _html.escape("\n".join(blocks), quote=False)
    return (
        '    <div class="note calm">\n'
        f"      <p><strong>{caption_html}</strong></p>\n"
        f'      <div class="knobs">{inner}</div>\n'
        "    </div>"
    )


# --------------------------------------------------------------------------------------
# 3.  Document
# --------------------------------------------------------------------------------------

@dataclass
class _Edit:
    start: int
    end: int
    text: str
    why: str


class Doc:
    """An indexed Register or Archive body, with staged byte-preserving edits."""

    def __init__(self, src: str, path: str = "<memory>"):
        self.src = src
        self.path = path
        p = _Indexer(src)
        p.feed(src)
        p.close()
        self.nodes: list[Node] = p.all
        self.roots: list[Node] = p.roots
        self.problems: list[str] = p.problems
        self._edits: list[_Edit] = []

    # -- construction ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str, body_only: bool = True) -> "Doc":
        """Read a file.  ``body_only`` strips a saved artifact's frame-runtime ``<head>``
        if one is present, so a saved artifact and a hand-written body index alike."""
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        if body_only:
            src = body_of(src)
        return cls(src, path)

    # -- element lookup ----------------------------------------------------------------
    def find(self, tag: str = None, cls: str = None) -> Iterator[Node]:
        for n in self.nodes:
            if (tag is None or n.tag == tag) and (cls is None or n.has_class(cls)):
                yield n

    def sections(self) -> list[tuple[str, Node]]:
        """(heading text, node) for every ``<section>``.  Heading is '' if it has none."""
        out = []
        for s in self.find("section"):
            h2 = s.first("h2")
            out.append((strip_tags(h2.inner(self.src)) if h2 else "", s))
        return out

    def section(self, title: str) -> Node:
        """The section whose ``<h2>`` starts with ``title`` (case-insensitive)."""
        want = _fold(title)
        hits = [n for t, n in self.sections() if _fold(t).startswith(want)]
        if not hits:
            raise KeyError(f"no <section> whose heading starts with {title!r}")
        if len(hits) > 1:
            raise KeyError(f"{len(hits)} sections match {title!r}; be more specific")
        return hits[0]

    def cards(self) -> dict[str, Node]:
        """Protocol cards (``div.exp``) keyed by the id in their ``span.id``."""
        out: dict[str, Node] = {}
        for e in self.find("div", "exp"):
            head = e.first("div", "exp-head")
            idn = (head or e).first("span", "id")
            if idn is None:
                continue
            out.setdefault(norm_id(idn.inner(self.src)), e)
        return out

    def card(self, cid: str) -> Node:
        cards = self.cards()
        key = norm_id(cid)
        if key in cards:
            return cards[key]
        fold = {_fold(k): v for k, v in cards.items()}
        if _fold(cid) in fold:
            return fold[_fold(cid)]
        raise KeyError(f"no card {cid!r}; have {sorted(cards)}")

    def tables(self) -> list[tuple[str, Node]]:
        """(caption text, node) for every ``<table>``."""
        out = []
        for t in self.find("table"):
            cap = t.first("caption")
            out.append((strip_tags(cap.inner(self.src)) if cap else "", t))
        return out

    # Short names for the tables this workflow edits, matched against caption text.
    TABLE_ALIASES = {
        "queue": "the queue",
        "push": "push list",
        "claims": "open claims and their readouts",
        "constraints": "constraints only",
        "handoff": "coding-agent handoff",
        "bugs": "open ",
        "predictions": "eee_01 unless noted",
    }

    def table(self, key: str) -> Node:
        """A table by alias (``queue``, ``push``, ``bugs``, ``predictions``, ...) or by a
        substring of its ``<caption>``."""
        want = _fold(self.TABLE_ALIASES.get(key.lower(), key))
        hits = [n for c, n in self.tables() if want in _fold(c)]
        if not hits:
            have = [c for c, _ in self.tables()]
            raise KeyError(f"no table matching {key!r}; captions are {have}")
        if len(hits) > 1:
            raise KeyError(f"{len(hits)} tables match {key!r}; be more specific")
        return hits[0]

    def rows(self, table_key) -> list[Node]:
        """Body ``<tr>`` of a table (header rows in ``<thead>`` excluded)."""
        t = table_key if isinstance(table_key, Node) else self.table(table_key)
        body = t.first("tbody") or t
        return [r for r in body.find("tr") if not (r.parent and r.parent.tag == "thead")]

    def row_id(self, tr: Node) -> str:
        """A row's id: its ``td.id-cell``, or its first cell if it has none."""
        cell = tr.first("td", "id-cell") or tr.first("td")
        return norm_id(cell.inner(self.src)) if cell else ""

    def row_aliases(self, tr: Node) -> list[str]:
        """Every name a row answers to.

        Push-list and bug-ledger id-cells carry a whole phrase -- "C-9 -- name the
        per-frame estimate trajectory in rule 8's retained list" -- while the rest of
        the document cross-references them as bare ``C-9``.  Both resolve here, so a
        reference check does not report the document's own conventions as dangling.
        """
        full = self.row_id(tr)
        if not full:
            return []
        out = [full]
        head = re.split(r"\s+[-\u2013\u2014]{1,2}\s+", full, maxsplit=1)[0].strip()
        if head and head != full:
            out.append(head)
        tok = full.split()[0].strip(".,:;")
        if tok and tok not in out:
            out.append(tok)
        return out

    def row(self, table_key, rid: str) -> Node:
        want, wf = norm_id(rid), _fold(rid)
        rows = self.rows(table_key)
        for r in rows:
            if want in self.row_aliases(r):
                return r
        for r in rows:
            if wf in [_fold(a) for a in self.row_aliases(r)]:
                return r
        for r in rows:  # last resort: prefix, for long free-text ledger ids
            if _fold(self.row_id(r)).startswith(wf):
                return r
        raise KeyError(f"no row {rid!r}; have {[self.row_id(r)[:40] for r in rows]}")

    def blocks(self) -> list[Block]:
        """Every data block, including ones whose closing marker was lost.

        The Register writes a block whole inside one ``div.knobs``.  The Archive, as
        currently built, labels each one ``<p class="blab"><b>[Q1-META v1]</b></p>``
        and puts the body in the following ``div.knobs`` -- and drops the ``[/Q1-META]``
        line entirely.  That is a real defect (``check_block_markers`` reports it), but
        the data is intact, so a second pass recovers those blocks by treating the
        labelled knobs element as the block body.  Extraction should not silently
        return nothing just because the delimiter convention slipped.
        """
        found = find_blocks(self.src)
        seen = {b.name for b in found}
        masked = _mask_tags(self.src)
        knobs = sorted(self.find("div", "knobs"), key=lambda n: n.start)

        for m in _BLOCK_OPEN_RE.finditer(masked):
            name, ver = m.group(1), int(m.group(2))
            if name in seen:
                continue
            body_node = None
            for k in knobs:
                if k.inner_start <= m.start() < k.inner_end:  # label inside the knobs
                    body_node = k
                    break
                if k.start >= m.end() and self.src[m.end() : k.start].strip(" \t\r\n") in (
                    "", "</b>", "</b></p>", "</b></p>\n", "</p>",
                ):
                    body_node = k  # label immediately precedes the knobs
                    break
            if body_node is None:
                continue
            raw = body_node.inner(self.src)
            # if the label lives inside this knobs div, drop everything up to it
            if body_node.inner_start <= m.start():
                raw = self.src[m.end() : body_node.inner_end].lstrip("\r\n")
            text = _html.unescape(_TAG_RE.sub("", raw)).strip("\n")
            text, was_wrapped = unwrap_csv(text)
            cand = Block(name=name, version=ver, body=text,
                         start=m.start(), end=body_node.end,
                         wrapped=was_wrapped, delimited=False)
            if cand.looks_like_data():
                found.append(cand)
                seen.add(name)
        found.sort(key=lambda b: b.start)
        return found

    def block(self, name: str) -> Block:
        for b in self.blocks():
            if b.name.casefold() == name.casefold():
                return b
        raise KeyError(f"no block {name!r}; have {[b.name for b in self.blocks()]}")

    def ids_referenced(self) -> dict[str, list[int]]:
        """Every ``<span class="id">X</span>`` cross-reference, id -> line numbers."""
        out: dict[str, list[int]] = {}
        for n in self.find("span", "id"):
            out.setdefault(norm_id(n.inner(self.src)), []).append(n.line)
        return out

    # -- staged mutation ---------------------------------------------------------------
    def stage(self, start: int, end: int, text: str, why: str = "") -> None:
        if not (0 <= start <= end <= len(self.src)):
            raise ValueError(f"span ({start}, {end}) outside document")
        self._edits.append(_Edit(start, end, text, why))

    def stage_replace(self, node: Node, text: str, why: str = "") -> None:
        self.stage(node.start, node.end, text, why or f"replace <{node.tag}>")

    def stage_cut(self, node: Node, why: str = "") -> None:
        """Remove an element and the blank line it leaves behind."""
        start, end = node.start, node.end
        # swallow the element's own indentation
        ls = self.src.rfind("\n", 0, start) + 1
        if self.src[ls:start].strip() == "":
            start = ls
        # and one trailing newline, so cuts do not accumulate blank lines
        if self.src[end : end + 1] == "\n":
            end += 1
        self.stage(start, end, "", why or f"cut <{node.tag}>")

    def stage_insert(self, at: int, text: str, why: str = "") -> None:
        self.stage(at, at, text, why or "insert")

    def pending(self) -> list[_Edit]:
        return list(self._edits)

    def apply(self) -> str:
        """Apply staged edits and return the new source.

        Descending order, so no offset is invalidated by an earlier splice -- the bug
        this API exists to prevent.  Overlapping edits are refused outright rather than
        silently resolved, because a silent resolution is how one edit eats another.
        """
        edits = sorted(self._edits, key=lambda e: (e.start, e.end))
        for a, b in zip(edits, edits[1:]):
            if b.start < a.end:
                raise ValueError(
                    f"overlapping edits: [{a.start},{a.end}) {a.why!r} "
                    f"and [{b.start},{b.end}) {b.why!r}"
                )
        out = self.src
        for e in reversed(edits):
            out = out[: e.start] + e.text + out[e.end :]
        return out

    def reset(self) -> None:
        self._edits.clear()

    # -- invariant ---------------------------------------------------------------------
    def roundtrip_ok(self) -> bool:
        """Reassembling the document from its own top-level spans reproduces it exactly.

        This is the guarantee that makes every mutation safe: if the index disagreed
        with the bytes anywhere, a splice would corrupt the document silently.
        """
        pos, out = 0, []
        for n in self.roots:
            out.append(self.src[pos : n.start])
            out.append(self.src[n.start : n.end])
            pos = n.end
        out.append(self.src[pos:])
        return "".join(out) == self.src


# --------------------------------------------------------------------------------------
# 4.  Body extraction
# --------------------------------------------------------------------------------------

def body_of(src: str) -> str:
    """Recover the publishable body from a saved artifact file.

    A saved artifact carries a large machine-generated ``<head>`` (the frame runtime).
    Republishing must be built from the saved file, but the body is what the author
    edits, so this strips the wrapper and nothing else.  Idempotent: a file that is
    already a body comes back unchanged.
    """
    i = src.find("<body>")
    if i == -1:
        # Already a body -- but strip a wrapper someone left on the end (see check_wrapper).
        return re.sub(r"\n*</body>\s*</html>\s*$", "\n", src)
    out = src[i + len("<body>") :]
    if out.startswith("\n"):
        out = out[1:]
    m = re.search(r"</body>\s*</html>\s*$", out)
    if m:
        out = out[: m.start()]
    return out


# --------------------------------------------------------------------------------------
# 5.  Validation
# --------------------------------------------------------------------------------------

@dataclass
class Finding:
    level: str  # 'error' | 'warn' | 'note'
    check: str
    message: str
    where: str = ""

    def __str__(self):
        tail = f"  [{self.where}]" if self.where else ""
        return f"{self.level.upper():5s} {self.check:18s} {self.message}{tail}"


# Structural tags whose imbalance corrupts a splice.  <p> and inline tags are checked
# too, because a stray </strong> shipped in this workflow and was caught only by hand.
_BALANCE_TAGS = (
    "div section p h1 h2 h3 table tr td th tbody thead caption "
    "footer header em strong code span b ul ol li"
).split()


def check_balance(src: str) -> list[Finding]:
    out = []
    for tag in _BALANCE_TAGS:
        o = len(re.findall(rf"<{tag}[\s>]", src))
        c = len(re.findall(rf"</{tag}>", src))
        if o != c:
            out.append(
                Finding("error", "tag-balance", f"<{tag}> opened {o}x, closed {c}x")
            )
    return out


# Elements these documents actually use.  An "unclosed tag" whose name is not in here is
# almost always an unescaped '<' in body text -- `dx2_<cell>_eee_01_ex_off` inside a
# knobs listing parses as an element -- which is a different (smaller) defect than a
# genuinely unbalanced structural tag, and is reported as its own check.
_KNOWN_TAGS = frozenset(
    "html head body title link meta style script div section header footer main "
    "h1 h2 h3 h4 h5 h6 p span em strong b i u code pre br hr img a "
    "table caption thead tbody tfoot tr td th ul ol li dl dt dd "
    "sup sub blockquote figure figcaption small mark".split()
)


def check_structure(doc: Doc) -> list[Finding]:
    out = []
    for p in doc.problems:
        m = re.search(r"<(\w+)>|</(\w+)>", p)
        tag = (m.group(1) or m.group(2)) if m else ""
        if tag and tag not in _KNOWN_TAGS:
            out.append(
                Finding("warn", "raw-lt",
                        f"{p}  -- '{tag}' is not an HTML element, so this is an "
                        f"unescaped '<' in text; write &lt;{tag}&gt;")
            )
        else:
            out.append(Finding("error", "structure", p))
    return out


def check_wrapper(src: str) -> list[Finding]:
    """A body file must not carry the document wrapper.

    Publishing wraps the file in its own ``<html><head>...<body>``, so a stray
    ``</body></html>`` left on the end of an extracted body closes the document early
    and everything after it is out of tree.  Found this way in the live Archive.
    """
    out = []
    if re.search(r"</body>", src) and not re.search(r"<body[\s>]", src):
        line = src.count("\n", 0, src.index("</body>")) + 1
        out.append(
            Finding("error", "wrapper",
                    "body file ends with a stray </body></html> -- publishing adds its "
                    "own wrapper, so this closes the document early", f"line {line}")
        )
    return out


_ENTITY_RE = re.compile(r"&(#\d+|#x[0-9a-fA-F]+|[a-zA-Z][a-zA-Z0-9]*);")
_BARE_AMP_RE = re.compile(r"&(?!#\d+;|#x[0-9a-fA-F]+;|[a-zA-Z][a-zA-Z0-9]*;)")


def check_entities(src: str) -> list[Finding]:
    out = []
    src = _mask_tags(src)  # attribute values (e.g. a fonts URL's &family=) are not text
    for m in _BARE_AMP_RE.finditer(src):
        line = src.count("\n", 0, m.start()) + 1
        ctx = src[m.start() : m.start() + 40].replace("\n", " ")
        out.append(Finding("warn", "entity", f"bare '&' -- {ctx!r}", f"line {line}"))
    return out[:20]


def check_block_syntax(src: str) -> list[Finding]:
    """Every opened block closes, and nothing closes that was not opened."""
    out = []
    opened = [(m.group(1), src.count("\n", 0, m.start()) + 1) for m in _BLOCK_OPEN_RE.finditer(src)]
    closed = [(m.group(1), src.count("\n", 0, m.start()) + 1) for m in _BLOCK_CLOSE_RE.finditer(src)]
    matched = {b.name for b in find_blocks(src)}
    for name, line in opened:
        if name not in matched:
            out.append(
                Finding("warn", "block-syntax",
                        f"[{name}] opened but never closed -- either a truncated block, "
                        f"or a block FORMAT SPECIFICATION in a card (legitimate: the "
                        f"DX-2 card specifies [DX2-SERIES v2]'s columns this way)",
                        f"line {line}")
            )
    for name, line in closed:
        if name not in matched:
            out.append(Finding("error", "block-syntax", f"[/{name}] closes nothing", f"line {line}"))
    return out


# The five blocks a round report owes (rule 5a / 31-0a).
EXPECTED_KINDS = ("META", "VERIFY", "SERIES", "SUMMARY", "PAIRS")

# Column pairs whose AGREEMENT is the point of the check.  e_glob_rms matching
# ate_results_lio to 4 dp is [BATCH-VERIFY] doing its job (rule 31-0a); flagging it as a
# tautology would invert the check's meaning.
INTENDED_AGREEMENT = {frozenset(("e_glob_rms", "ate_results_lio"))}


def check_block_markers(doc: Doc) -> list[Finding]:
    """A block whose ``[/NAME]`` line is missing is human-readable and machine-invisible.

    Found in the Archive: filing rewrapped each block as a label plus a knobs div and
    dropped every closing marker, so the Archive's copy of twenty blocks could not be
    re-extracted by the same tool that extracted them from the Register.
    """
    delimited = {b.name for b in find_blocks(doc.src)}
    out = []
    for b in doc.blocks():
        if b.wrapped:
            line = doc.src.count("\n", 0, b.start) + 1
            out.append(
                Finding("error", "wrapped-csv",
                        f"[{b.name}] is hard-wrapped: its {len(b.header())}-column header "
                        f"is split across physical lines, so the block is not a table as "
                        f"written.  Nothing is edited on the way into the Archive "
                        f"(rule 12) -- rejoin it", f"line {line}")
            )
        if b.name not in delimited:
            line = doc.src.count("\n", 0, b.start) + 1
            out.append(
                Finding("warn", "block-marker",
                        f"[{b.name}] has no [/{b.name}] closing line -- recovered from its "
                        f"knobs element, but not machine-extractable as written",
                        f"line {line}")
            )
    return out


def check_block_completeness(doc: Doc) -> list[Finding]:
    out = []
    by_batch: dict[str, set[str]] = {}
    for b in doc.blocks():
        by_batch.setdefault(b.batch, set()).add(b.kind)
    for batch, kinds in sorted(by_batch.items()):
        missing = [k for k in EXPECTED_KINDS if k not in kinds]
        if missing:
            out.append(
                Finding("warn", "block-set", f"batch {batch} is missing {', '.join(missing)}", batch)
            )
    return out


def check_verify_blocks(doc: Doc, ate_tol: float = 0.01) -> list[Finding]:
    """The instrument checks.  These encode three defects this workflow actually shipped.

    * a column that is ``nan`` on every row is a check that never evaluates (C-11);
    * a column equal to another column on every row is tautological -- it reports the
      quantity instead of the error in it, and passes unconditionally (C-11);
    * ``e_glob_rms`` disagreeing with ``ate_results_lio`` past tolerance means the two
      ATE paths are not the same statistic on that cell.  This fires on exp04 at 7.6%
      and is the whole of queue row X-ATE (C-12).
    """
    out = []
    for b in doc.blocks():
        if b.kind != "VERIFY" or not b.is_csv():
            continue
        hdr = [h for h in b.header() if h != "cell"]
        rows = b.rows()
        if not rows:
            continue

        for col in hdr:
            vals = b.numeric_column(col)
            if vals and all(v is None or _isnan(v) for v in vals):
                out.append(
                    Finding("error", "dead-column",
                            f"{b.name}: '{col}' is nan on all {len(vals)} cells -- "
                            f"a check that never evaluates", b.name)
                )

        for i, a in enumerate(hdr):
            for c in hdr[i + 1 :]:
                if frozenset((a, c)) in INTENDED_AGREEMENT:
                    continue  # these two agreeing IS the check passing, not a tautology
                va, vc = b.numeric_column(a), b.numeric_column(c)
                pairs = [(x, y) for x, y in zip(va, vc) if x is not None and y is not None
                         and not _isnan(x) and not _isnan(y)]
                if len(pairs) >= 2 and all(abs(x - y) <= 1e-9 * max(1.0, abs(x)) for x, y in pairs):
                    out.append(
                        Finding("error", "tautological",
                                f"{b.name}: '{a}' equals '{c}' on all {len(pairs)} cells -- "
                                f"the check reports its own input and cannot fail", b.name)
                    )

        if "e_glob_rms" in b.header() and "ate_results_lio" in b.header():
            for r in rows:
                x, y = _as_float(r.get("e_glob_rms", "")), _as_float(r.get("ate_results_lio", ""))
                if x is None or y is None or _isnan(x) or _isnan(y) or y == 0:
                    continue
                rel = abs(x - y) / abs(y)
                if rel > ate_tol:
                    out.append(
                        Finding("error", "ate-disagree",
                                f"{b.name}/{r.get('cell','?')}: e_glob_rms {x:g} vs "
                                f"ate_results_lio {y:g} -- {rel*100:.1f}% apart; the two "
                                f"ATE paths are not the same statistic here",
                                f"{b.name}:{r.get('cell','?')}")
                    )
    return out


# A VERIFY column is supposed to be an ERROR in a quantity.  If it instead equals the
# quantity itself -- even one carried in a different block -- it passes unconditionally.
# gain_telescope_err_m vs SUMMARY's gain_sum is the live instance (C-11).
CROSS_BLOCK_TAUTOLOGIES = [("gain_telescope_err_m", "gain_sum")]


def check_cross_block(doc: Doc, rtol: float = 1e-3) -> list[Finding]:
    """A VERIFY column that reproduces a SUMMARY column is reporting its own input."""
    summ = {b.batch: b for b in doc.blocks() if b.kind == "SUMMARY" and b.is_csv()}
    out = []
    for b in doc.blocks():
        if b.kind != "VERIFY" or not b.is_csv():
            continue
        peer = summ.get(b.batch)
        if peer is None:
            continue
        peer_by_cell = {r.get("cell", ""): r for r in peer.rows()}
        for vcol, scol in CROSS_BLOCK_TAUTOLOGIES:
            if vcol not in b.header() or scol not in peer.header():
                continue
            pairs = []
            for r in b.rows():
                other = peer_by_cell.get(r.get("cell", ""))
                if other is None:
                    continue
                x, y = _as_float(r.get(vcol, "")), _as_float(other.get(scol, ""))
                if None in (x, y) or _isnan(x) or _isnan(y):
                    continue
                pairs.append((x, y))
            if len(pairs) >= 2 and all(
                abs(x - y) <= rtol * max(1.0, abs(y)) for x, y in pairs
            ):
                out.append(
                    Finding("error", "tautological",
                            f"{b.name}: '{vcol}' reproduces [{b.batch}-SUMMARY].'{scol}' on "
                            f"all {len(pairs)} cells -- it reports the quantity rather than "
                            f"the error in it, so the check passes unconditionally", b.name)
                )
    return out


def check_pairs_blocks(doc: Doc, r_floor: float = 0.2) -> list[Finding]:
    """A PAIRS block whose largest |r| never reaches ``r_floor`` is measuring noise or
    destroying its own signal.  Reported as a finding rather than printed silently for
    a fourth round."""
    out = []
    for b in doc.blocks():
        if b.kind != "PAIRS" or not b.is_csv() or "r" not in b.header():
            continue
        vals = [v for v in b.numeric_column("r") if v is not None and not _isnan(v)]
        if not vals:
            continue
        mx = max(abs(v) for v in vals)
        if mx < r_floor:
            out.append(
                Finding("warn", "inert-pairs",
                        f"{b.name}: max |r| = {mx:.2f} over {len(vals)} rows, none reaches "
                        f"{r_floor} -- the instrument is inert as delivered", b.name)
            )
    return out


def check_scalar_series(doc: Doc) -> list[Finding]:
    """Rule 33: every scalar a round quotes ships its time series in the same round.

    Machine form: a ``*_p50`` / ``_p90`` / ``_max`` column in a SUMMARY block whose base
    quantity has no column in the matching SERIES block is a scalar without its series.
    """
    out = []
    series_cols: dict[str, set[str]] = {}
    for b in doc.blocks():
        if b.kind == "SERIES" and b.is_csv():
            series_cols[b.batch] = set(b.header())
    for b in doc.blocks():
        if b.kind != "SUMMARY" or not b.is_csv():
            continue
        have = series_cols.get(b.batch)
        if have is None:
            continue
        missing = set()
        for col in b.header():
            m = re.match(r"^(?P<base>.+?)_(p50|p90|mean|min|max|frac)$", col)
            if not m:
                continue
            base = m.group("base")
            cands = {base, base.replace("_pre", ""), base.replace("_post", ""),
                     base + "_pre", re.sub(r"_(pre|post)$", "", base)}
            if not (cands & have):
                missing.add(col)
        for col in sorted(missing):
            out.append(
                Finding("warn", "rule-33",
                        f"{b.batch}: '{col}' is quoted as a scalar with no matching "
                        f"column in [{b.batch}-SERIES]", b.batch)
            )
    return out


# Rule 31d: the register must not narrate.  Headings and chips that describe completed
# work belong in the Archive.
# Deliberately literal about case: these documents shout a completed status (RUN,
# RESOLVED, CLOSED) and use the same words in lower case in ordinary prose ("neither
# round ran", "the row is otherwise closed").  Matching case-insensitively turns the
# check into noise, which is how a heuristic gets ignored.
_PAST_MARKERS = re.compile(
    r"\b(RUN|RAN|RESOLVED|CLOSED|FIXED|WITHDRAWN|REFUTED|CONFIRMED|DECIDED|DONE)\b"
    r"|\b(what happened|verdict:|reconciliation|post-?mortem)\b",
)


def check_compaction(doc: Doc) -> list[Finding]:
    """Flag anything left in a Register that names something which already happened."""
    out = []
    for title, node in doc.sections():
        if title and _PAST_MARKERS.search(title):
            out.append(
                Finding("warn", "rule-31d",
                        f"section heading describes completed work: {title[:90]!r}",
                        f"line {node.line}")
            )
    for cid, node in doc.cards().items():
        head = node.first("div", "exp-head")
        if head is None:
            continue
        h3 = head.first("h3")
        chip = head.first("span", "chip")
        text = " ".join(
            strip_tags(x.inner(doc.src)) for x in (h3, chip) if x is not None
        )
        if _PAST_MARKERS.search(text):
            out.append(
                Finding("warn", "rule-31d",
                        f"card {cid} describes completed work: {text[:90]!r}",
                        f"line {node.line}")
            )
    return out


def check_references(doc: Doc) -> list[Finding]:
    """Every ``<span class="id">X</span>`` should resolve to a card or a table row."""
    known = set(doc.cards())
    for cap, t in doc.tables():
        for r in doc.rows(t):
            known.update(doc.row_aliases(r))
    out = []
    for rid, lines in sorted(doc.ids_referenced().items()):
        if rid and rid not in known:
            out.append(
                Finding("note", "dangling-ref",
                        f"'{rid}' is referenced {len(lines)}x but has no card or row",
                        f"lines {lines[:4]}")
            )
    return out


def validate(doc: Doc, compaction: bool = True, strict: bool = False) -> list[Finding]:
    """All checks.  ``compaction`` is Register-only; the Archive is supposed to narrate."""
    out: list[Finding] = []
    out += check_structure(doc)
    out += check_wrapper(doc.src)
    out += check_balance(doc.src)
    out += check_entities(doc.src)
    out += check_block_syntax(doc.src)
    out += check_block_markers(doc)
    out += check_block_completeness(doc)
    out += check_verify_blocks(doc)
    out += check_cross_block(doc)
    out += check_pairs_blocks(doc)
    out += check_scalar_series(doc)
    out += check_references(doc)
    if compaction:
        out += check_compaction(doc)
    if not doc.roundtrip_ok():
        out.insert(0, Finding("error", "roundtrip",
                              "index does not reproduce the source -- do not splice this file"))
    if strict:
        for f in out:
            if f.level == "warn":
                f.level = "error"
    return out


# --------------------------------------------------------------------------------------
# 6.  Filing: cut to pointer, move to Archive
# --------------------------------------------------------------------------------------

def pointer_html(title: str, archive_section: str, extra: str = "", indent: str = "  ") -> str:
    """The one-line Archive pointer a cut leaves behind (rule 31c)."""
    tail = f" {extra}" if extra else ""
    return (
        f'{indent}<div class="note calm">\n'
        f"{indent}  <p><strong>{title} &mdash; filed.</strong> "
        f"<em>livo_recon Experiment Archive</em>, &ldquo;{archive_section}&rdquo;."
        f"{tail}</p>\n"
        f"{indent}</div>"
    )


def cut_to_pointer(doc: Doc, node: Node, pointer: Optional[str], why: str = "") -> None:
    """Stage a cut, optionally leaving a pointer in the hole."""
    if pointer is None:
        doc.stage_cut(node, why)
    else:
        doc.stage_replace(node, pointer, why)


def file_to_archive(
    register: Doc,
    archive: Doc,
    nodes: Sequence[Node],
    section_title: str,
    *,
    pointer: Optional[str] = None,
    anchor_section: str = "Closed defects",
    section_tag: str = "",
    lead: str = "",
) -> tuple[str, str]:
    """Move content from the Register into a new Archive section, verbatim.

    Returns ``(new_register_src, new_archive_src)``.  Nothing is edited on the way in
    (rule 12): the moved HTML is the exact bytes the Register held.  The Register keeps
    a one-line pointer per moved element, or nothing if ``pointer`` is None.

    The Archive section is inserted BEFORE ``anchor_section`` so that appended rounds
    stay in order ahead of the standing closed-defect tables.
    """
    moved = "\n\n".join(n.outer(register.src) for n in nodes)
    tag = f' <span class="tag">{section_tag}</span>' if section_tag else ""
    lead_p = f"    <p>{lead}</p>\n\n" if lead else ""
    new_section = (
        "  <section>\n"
        f"    <h2>{section_title}{tag}</h2>\n\n"
        f"{lead_p}"
        f"{moved}\n"
        "  </section>\n\n"
    )
    anchor = archive.section(anchor_section)
    ls = archive.src.rfind("\n", 0, anchor.start) + 1
    archive.stage_insert(ls, new_section, f"file '{section_title}'")

    for n in nodes:
        cut_to_pointer(register, n, pointer, f"file to Archive: {section_title}")

    return register.apply(), archive.apply()


# --------------------------------------------------------------------------------------
# 7.  Structured edits
# --------------------------------------------------------------------------------------

def set_actual(doc: Doc, pred_id: str, html_fragment: str) -> None:
    """Fill a prediction ledger row's Actual column (its last ``<td>``)."""
    tr = doc.row("predictions", pred_id)
    tds = [c for c in tr.find("td") if c.parent is tr]
    if not tds:
        raise ValueError(f"prediction row {pred_id!r} has no cells")
    cell = tds[-1]
    attrs = " ".join(f'{k}="{v}"' for k, v in cell.attrs.items() if k != "class")
    new_cls = 'class="n"'
    doc.stage_replace(
        cell,
        f"<td {new_cls}{(' ' + attrs) if attrs else ''}>{html_fragment}</td>",
        f"Actual for {pred_id}",
    )


def add_row(
    doc: Doc,
    table_key: str,
    row_html: str,
    *,
    at: str = "top",
    before: Optional[str] = None,
    after: Optional[str] = None,
) -> None:
    """Insert a ``<tr>``.  ``before``/``after`` take a row id and win over ``at``."""
    rows = doc.rows(table_key)
    body = (doc.table(table_key).first("tbody") or doc.table(table_key))
    text = row_html if row_html.endswith("\n") else row_html + "\n"
    if before:
        target = doc.row(table_key, before)
        ls = doc.src.rfind("\n", 0, target.start) + 1
        doc.stage_insert(ls, text, f"row before {before}")
    elif after:
        target = doc.row(table_key, after)
        end = target.end + (1 if doc.src[target.end : target.end + 1] == "\n" else 0)
        doc.stage_insert(end, text, f"row after {after}")
    elif at == "bottom" and rows:
        last = rows[-1]
        end = last.end + (1 if doc.src[last.end : last.end + 1] == "\n" else 0)
        doc.stage_insert(end, text, "row at bottom")
    else:
        anchor = rows[0] if rows else None
        at_off = (doc.src.rfind("\n", 0, anchor.start) + 1) if anchor else body.inner_start
        doc.stage_insert(at_off, text, "row at top")


def remove_row(doc: Doc, table_key: str, rid: str) -> None:
    doc.stage_cut(doc.row(table_key, rid), f"remove row {rid}")


def retitle_card(
    doc: Doc, cid: str, *, h3: Optional[str] = None, chip: Optional[str] = None,
    chip_class: Optional[str] = None,
) -> None:
    """Change a card's headline and/or status chip without touching its body."""
    card = doc.card(cid)
    head = card.first("div", "exp-head")
    if head is None:
        raise ValueError(f"card {cid!r} has no exp-head")
    if h3 is not None:
        node = head.first("h3")
        if node is None:
            raise ValueError(f"card {cid!r} has no <h3>")
        doc.stage_replace(node, f"<h3>{h3}</h3>", f"retitle {cid}")
    if chip is not None:
        node = head.first("span", "chip")
        cls = chip_class or " ".join(c for c in (node.cls() if node else ["chip"]))
        if "chip" not in cls.split():
            cls = "chip " + cls
        new = f'<span class="{cls}">{chip}</span>'
        if node is None:
            doc.stage_insert(head.inner_end, new, f"chip for {cid}")
        else:
            doc.stage_replace(node, new, f"chip for {cid}")


def inject_blocks(
    doc: Doc,
    block_texts: Sequence[str],
    caption_html: str,
    *,
    after_card: Optional[str] = None,
    after_section: Optional[str] = None,
) -> None:
    """The agent side of rule 31-0: put data blocks into the Register.

    Blocks are HTML-escaped and wrapped in the standard container, then placed after a
    named card or section so they sit with the result they belong to.
    """
    html_fragment = "\n" + wrap_blocks(block_texts, caption_html) + "\n"
    if after_card:
        anchor = doc.card(after_card)
    elif after_section:
        anchor = doc.section(after_section)
    else:
        raise ValueError("inject_blocks needs after_card= or after_section=")
    end = anchor.end + (1 if doc.src[anchor.end : anchor.end + 1] == "\n" else 0)
    doc.stage_insert(end, html_fragment, f"inject {len(block_texts)} block(s)")


# --------------------------------------------------------------------------------------
# 8.  Structural diff and report
# --------------------------------------------------------------------------------------

def _inventory(doc: Doc) -> dict[str, dict[str, str]]:
    inv: dict[str, dict[str, str]] = {"section": {}, "card": {}, "block": {}}
    for title, n in doc.sections():
        if title:
            inv["section"][title] = str(hash(n.inner(doc.src)))
    for cid, n in doc.cards().items():
        inv["card"][cid] = str(hash(n.inner(doc.src)))
    for b in doc.blocks():
        inv["block"][b.name] = str(hash(b.body))
    for cap, t in doc.tables():
        key = f"row:{cap[:28]}"
        inv.setdefault(key, {})
        for r in doc.rows(t):
            rid = doc.row_id(r)
            if rid:
                inv[key][rid[:60]] = str(hash(r.inner(doc.src)))
    return inv


def structural_diff(old: Doc, new: Doc) -> list[str]:
    """What changed by name, not by line -- the review a text diff cannot give."""
    a, b = _inventory(old), _inventory(new)
    out: list[str] = []
    for kind in sorted(set(a) | set(b)):
        ka, kb = a.get(kind, {}), b.get(kind, {})
        removed = sorted(set(ka) - set(kb))
        added = sorted(set(kb) - set(ka))
        changed = sorted(k for k in set(ka) & set(kb) if ka[k] != kb[k])
        for k in removed:
            out.append(f"- {kind:24s} {k}")
        for k in added:
            out.append(f"+ {kind:24s} {k}")
        for k in changed:
            out.append(f"~ {kind:24s} {k}")
    size = len(new.src) - len(old.src)
    out.append(f"  size                     {len(old.src):,} -> {len(new.src):,} "
               f"({size:+,} bytes, {size/max(1,len(old.src))*100:+.1f}%)")
    return out


def report(doc: Doc) -> str:
    lines = [f"{doc.path}", f"  {len(doc.src):,} bytes, {doc.src.count(chr(10))+1:,} lines"]
    secs = [t for t, _ in doc.sections() if t]
    lines.append(f"  {len(doc.sections())} sections ({len(secs)} titled)")
    for t in secs:
        lines.append(f"      {t[:96]}")
    cards = doc.cards()
    lines.append(f"  {len(cards)} cards: {', '.join(sorted(cards))}")
    for cap, t in doc.tables():
        lines.append(f"  table {cap[:60]!r}: {len(doc.rows(t))} rows")
    bl = doc.blocks()
    if bl:
        lines.append(f"  {len(bl)} blocks:")
        for b in bl:
            shape = f"{len(b.rows())} rows x {len(b.header())} cols" if b.is_csv() \
                else f"{len(b.lines())} lines"
            lines.append(f"      {b.name:24s} v{b.version}  {shape}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# 9.  CLI
# --------------------------------------------------------------------------------------

def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {path}  ({len(text):,} bytes)", file=sys.stderr)


def _print_findings(fs: Sequence[Finding]) -> int:
    for f in fs:
        print(f)
    errs = sum(1 for f in fs if f.level == "error")
    warns = sum(1 for f in fs if f.level == "warn")
    print(f"\n{errs} error(s), {warns} warning(s), {len(fs)-errs-warns} note(s)")
    return 1 if errs else 0


def _read_arg(text: Optional[str], path: Optional[str]) -> str:
    if text is not None:
        return text
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            return fh.read().rstrip("\n")
    raise SystemExit("need --html or --html-file")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="register_io.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def f(p):
        p.add_argument("file")
        return p

    f(sub.add_parser("report", help="inventory a document"))

    p = f(sub.add_parser("validate", help="run all checks"))
    p.add_argument("--strict", action="store_true", help="warnings become errors")
    p.add_argument("--no-compaction", action="store_true",
                   help="skip the rule-31d test (use for the Archive, which may narrate)")

    p = f(sub.add_parser("blocks", help="extract [BATCH-*] data blocks"))
    p.add_argument("--out", help="directory to write one .txt per block")
    p.add_argument("--json", action="store_true")
    p.add_argument("--name", help="only this block")

    p = f(sub.add_parser("show", help="print one addressed element"))
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--card")
    g.add_argument("--section")
    g.add_argument("--row", help="TABLE:ID, e.g. queue:TB-1")
    g.add_argument("--block")
    p.add_argument("--text", action="store_true", help="strip tags")

    p = f(sub.add_parser("cut", help="remove cards/sections, optionally leaving a pointer"))
    p.add_argument("--card", action="append", default=[])
    p.add_argument("--section", action="append", default=[])
    p.add_argument("--pointer", help="HTML to leave behind (omit to cut outright)")
    p.add_argument("-o", "--out", required=True)

    p = sub.add_parser("file", help="move register content into the archive, verbatim")
    p.add_argument("--register", required=True)
    p.add_argument("--archive", required=True)
    p.add_argument("--title", required=True, help="archive section heading")
    p.add_argument("--tag", default="", help="archive section <span class=tag>")
    p.add_argument("--lead", default="", help="archive section lead paragraph HTML")
    p.add_argument("--card", action="append", default=[])
    p.add_argument("--section", action="append", default=[])
    p.add_argument("--pointer", help="HTML to leave in the register (omit to cut outright)")
    p.add_argument("--anchor", default="Closed defects",
                   help="insert the new archive section before this one")
    p.add_argument("-o-register", dest="out_register", required=True)
    p.add_argument("-o-archive", dest="out_archive", required=True)

    p = f(sub.add_parser("actual", help="fill a prediction ledger Actual cell"))
    p.add_argument("--id", required=True)
    p.add_argument("--html")
    p.add_argument("--html-file")
    p.add_argument("-o", "--out", required=True)

    p = f(sub.add_parser("addrow", help="insert a table row"))
    p.add_argument("--table", required=True)
    p.add_argument("--html")
    p.add_argument("--html-file")
    p.add_argument("--at", choices=["top", "bottom"], default="top")
    p.add_argument("--before")
    p.add_argument("--after")
    p.add_argument("-o", "--out", required=True)

    p = f(sub.add_parser("rmrow", help="remove a table row"))
    p.add_argument("--table", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("-o", "--out", required=True)

    p = f(sub.add_parser("body", help="strip a saved artifact's frame-runtime head"))
    p.add_argument("-o", "--out", required=True)

    p = sub.add_parser("diff", help="structural diff by name")
    p.add_argument("old")
    p.add_argument("new")

    p = sub.add_parser("emit", help="format a data block")
    p.add_argument("--name", required=True)
    p.add_argument("--version", type=int, default=1)
    p.add_argument("--csv", required=True, help="file of block body text, or - for stdin")

    p = f(sub.add_parser("inject", help="put data blocks into the register"))
    p.add_argument("--after-card")
    p.add_argument("--after-section")
    p.add_argument("--caption", required=True)
    p.add_argument("--block", action="append", required=True, help="file containing one block")
    p.add_argument("-o", "--out", required=True)

    p = sub.add_parser("self-check", help="round-trip identity plus validation")
    p.add_argument("files", nargs="+")

    a = ap.parse_args(argv)

    # --- dispatch ---------------------------------------------------------------------
    if a.cmd == "report":
        print(report(Doc.load(a.file)))
        return 0

    if a.cmd == "validate":
        d = Doc.load(a.file)
        return _print_findings(validate(d, compaction=not a.no_compaction, strict=a.strict))

    if a.cmd == "blocks":
        d = Doc.load(a.file)
        bs = [b for b in d.blocks() if not a.name or b.name.casefold() == a.name.casefold()]
        if a.json:
            print(json.dumps(
                [{"name": b.name, "version": b.version, "batch": b.batch, "kind": b.kind,
                  "csv": b.is_csv(), "header": b.header(),
                  "rows": b.rows() if b.is_csv() else None,
                  "kv": b.kv() if not b.is_csv() else None,
                  "body": b.body} for b in bs], indent=2))
        elif a.out:
            os.makedirs(a.out, exist_ok=True)
            for b in bs:
                p2 = os.path.join(a.out, f"{b.name}.txt")
                _write(p2, b.body + "\n")
            print(f"{len(bs)} block(s) -> {a.out}", file=sys.stderr)
        else:
            for b in bs:
                print(f"=== {b.name} v{b.version} ===")
                print(b.body)
        return 0

    if a.cmd == "show":
        d = Doc.load(a.file)
        if a.card:
            n = d.card(a.card)
        elif a.section:
            n = d.section(a.section)
        elif a.row:
            t, _, rid = a.row.partition(":")
            n = d.row(t, rid)
        else:
            print(d.block(a.block).body)
            return 0
        print(n.text(d.src) if a.text else n.outer(d.src))
        return 0

    if a.cmd == "cut":
        d = Doc.load(a.file)
        for cid in a.card:
            cut_to_pointer(d, d.card(cid), a.pointer, f"cut card {cid}")
        for st in a.section:
            cut_to_pointer(d, d.section(st), a.pointer, f"cut section {st}")
        out = d.apply()
        fs = validate(Doc(out), compaction=False)
        _print_findings([x for x in fs if x.level == "error"])
        _write(a.out, out)
        return 0

    if a.cmd == "file":
        reg, arc = Doc.load(a.register), Doc.load(a.archive)
        nodes = [reg.card(c) for c in a.card] + [reg.section(s) for s in a.section]
        nodes.sort(key=lambda n: n.start)
        new_reg, new_arc = file_to_archive(
            reg, arc, nodes, a.title, pointer=a.pointer,
            anchor_section=a.anchor, section_tag=a.tag, lead=a.lead)
        for label, text in (("register", new_reg), ("archive", new_arc)):
            errs = [x for x in validate(Doc(text), compaction=False) if x.level == "error"]
            if errs:
                print(f"-- {label} --")
                _print_findings(errs)
        _write(a.out_register, new_reg)
        _write(a.out_archive, new_arc)
        return 0

    if a.cmd == "actual":
        d = Doc.load(a.file)
        set_actual(d, a.id, _read_arg(a.html, a.html_file))
        _write(a.out, d.apply())
        return 0

    if a.cmd == "addrow":
        d = Doc.load(a.file)
        add_row(d, a.table, _read_arg(a.html, a.html_file),
                at=a.at, before=a.before, after=a.after)
        _write(a.out, d.apply())
        return 0

    if a.cmd == "rmrow":
        d = Doc.load(a.file)
        remove_row(d, a.table, a.id)
        _write(a.out, d.apply())
        return 0

    if a.cmd == "body":
        with open(a.file, encoding="utf-8") as fh:
            _write(a.out, body_of(fh.read()))
        return 0

    if a.cmd == "diff":
        for line in structural_diff(Doc.load(a.old), Doc.load(a.new)):
            print(line)
        return 0

    if a.cmd == "emit":
        body = sys.stdin.read() if a.csv == "-" else open(a.csv, encoding="utf-8").read()
        print(emit_block(a.name, a.version, body))
        return 0

    if a.cmd == "inject":
        d = Doc.load(a.file)
        texts = [open(p2, encoding="utf-8").read().rstrip("\n") for p2 in a.block]
        inject_blocks(d, texts, a.caption,
                      after_card=a.after_card, after_section=a.after_section)
        _write(a.out, d.apply())
        return 0

    if a.cmd == "self-check":
        rc = 0
        for path in a.files:
            d = Doc.load(path)
            checks: list[tuple[str, bool, str]] = []

            checks.append(("roundtrip", d.roundtrip_ok(),
                           "index reproduces the source byte for byte"))
            checks.append(("empty-apply", d.apply() == d.src,
                           "apply() with nothing staged is the identity"))

            # A staged cut must remove exactly the element and nothing else.
            cards = d.cards()
            if cards:
                cid = sorted(cards)[0]
                d2 = Doc(d.src, path)
                d2.stage_cut(d2.card(cid))
                after = d2.apply()
                # Locate the removed region independently, by common prefix/suffix.
                i = 0
                while i < min(len(d.src), len(after)) and d.src[i] == after[i]:
                    i += 1
                j = 0
                while (j < min(len(d.src), len(after)) - i
                       and d.src[len(d.src) - 1 - j] == after[len(after) - 1 - j]):
                    j += 1
                removed = d.src[i : len(d.src) - j]
                outer = cards[cid].outer(d.src)
                checks.append(("cut-exact",
                               removed.strip() == outer.strip() and outer not in after,
                               f"cutting card {cid} removes exactly that element "
                               f"({len(removed)} bytes) and nothing else"))

            # Overlapping edits must be refused, not silently resolved.
            d3 = Doc(d.src, path)
            d3.stage(10, 40, "x", "a")
            d3.stage(30, 60, "y", "b")
            try:
                d3.apply()
                overlap_refused = False
            except ValueError:
                overlap_refused = True
            checks.append(("overlap-refused", overlap_refused,
                           "overlapping staged edits raise instead of eating each other"))

            print(f"{path}: {len(d.nodes)} elements, {len(d.blocks())} blocks, "
                  f"{len(d.cards())} cards")
            for name, ok, why in checks:
                print(f"  {'PASS' if ok else 'FAIL'}  {name:16s} {why}")
                if not ok:
                    rc = 1
            errs = [x for x in validate(d, compaction=False) if x.level == "error"]
            for e in errs:
                print(f"  {e}")
            rc |= 1 if errs else 0
        return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
