"""Parameterized, structure-aware chunking.

``\\n`` line breaks are the candidate cut points; ``cap`` (max size) decides *whether* to
cut. ``cap = inf`` ⇒ never cut ⇒ the whole fiche is one chunk. Size is measured by
``length_fn`` (default: whitespace word count, a lightweight token proxy).
"""

import math
import re
from collections.abc import Callable

# Split on whitespace that follows sentence-ending punctuation (used only as a fallback).
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def count_words(text: str) -> int:
    """Default size metric: number of whitespace-separated words (a cheap token proxy)."""
    return len(text.split())


def _split_long(text: str, cap: int, length_fn: Callable[[str], int]) -> list[str]:
    """Fallback for a single line longer than ``cap``: break it on sentence boundaries,
    and if even one sentence is too long, hard-cut it every ``cap`` words."""
    pieces: list[str] = []
    cur: list[str] = []  # sentences accumulating into the current piece
    cur_len = 0
    for sentence in _SENTENCE.split(text):
        slen = length_fn(sentence)
        if slen > cap:
            # A single sentence already exceeds the cap → flush, then hard word-cut it.
            if cur:
                pieces.append(" ".join(cur))
                cur, cur_len = [], 0
            words = sentence.split()
            step = max(1, cap)
            for i in range(0, len(words), step):
                pieces.append(" ".join(words[i : i + step]))
            continue
        # Starting this sentence would overflow the cap → flush the current piece first.
        if cur and cur_len + slen > cap:
            pieces.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(sentence)
        cur_len += slen
    if cur:  # leftover sentences
        pieces.append(" ".join(cur))
    return pieces


def _overlap_tail(
    units: list[str], overlap: int, length_fn: Callable[[str], int]
) -> tuple[list[str], int]:
    """Take trailing units whose cumulative length is ~``overlap`` so they can seed the
    next chunk (this is what makes consecutive chunks share a little context)."""
    if overlap <= 0:
        return [], 0
    tail: list[str] = []
    total = 0
    for unit in reversed(units):  # walk backwards from the end of the chunk
        if total >= overlap:
            break
        tail.insert(0, unit)  # prepend to keep original order
        total += length_fn(unit)
    return tail, total


def chunk_text(
    text: str,
    *,
    cap: float = 256,
    overlap: int = 32,
    length_fn: Callable[[str], int] = count_words,
) -> list[str]:
    """Split ``text`` into chunks of at most ~``cap`` units, cutting on ``\\n`` boundaries
    with ``overlap`` carried between consecutive chunks. ``cap = inf`` returns one chunk."""
    text = (text or "").strip()
    if not text:
        return []
    if math.isinf(cap):  # whole-fiche mode: no size ceiling, so never split
        return [text]
    cap = int(cap)

    # Step 1: build "units" = the lines, guaranteeing none is larger than the cap
    # (over-long lines get pre-split by _split_long so packing below stays simple).
    units: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if length_fn(line) <= cap:
            units.append(line)
        else:
            units.extend(_split_long(line, cap, length_fn))

    # Step 2: greedily pack units into chunks; when the next unit would overflow the cap,
    # emit the current chunk and start the next one seeded with the overlap tail.
    chunks: list[str] = []
    cur: list[str] = []  # units in the chunk being built
    cur_len = 0  # running size of `cur`, per length_fn
    for unit in units:
        ulen = length_fn(unit)
        if cur and cur_len + ulen > cap:
            chunks.append("\n".join(cur))  # finalize the current chunk
            cur, cur_len = _overlap_tail(cur, overlap, length_fn)  # carry overlap forward
        cur.append(unit)
        cur_len += ulen
    if cur:  # final chunk
        chunks.append("\n".join(cur))
    return chunks
