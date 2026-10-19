from __future__ import annotations

import re

STOP = frozenset(
    "a an the of to in on for and or is are was were be by with as at from that this it its "
    "what which who how many much does do did can will should".split()
)


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOP]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
