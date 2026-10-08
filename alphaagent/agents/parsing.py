"""Robust extraction of structured pieces (JSON, code, confidence) from free-form LLM output."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

_FENCE = re.compile(r"```([a-zA-Z0-9_+-]*)[ \t]*\n(.*?)```", re.DOTALL)
_CONFIDENCE = re.compile(r"\[\s*Confidence\s*:\s*([0-9]*\.?[0-9]+)\s*\]", re.IGNORECASE)


def extract_confidence(text: str, default: float = 0.5) -> float:
    matches = _CONFIDENCE.findall(text)
    if not matches:
        return default
    value = float(matches[-1])
    if value > 1.0:  # tolerate "85" meaning 85%
        value = value / 100.0
    return max(0.0, min(1.0, value))


def extract_code(text: str, lang: str = "python") -> str:
    """Last fenced block tagged ``lang`` (else last untagged block, else the raw text)."""
    blocks = _FENCE.findall(text)
    tagged = [body for tag, body in blocks if tag.lower() in {lang, "py"}]
    if tagged:
        return tagged[-1].strip()
    untagged = [body for tag, body in blocks if not tag]
    if untagged:
        return untagged[-1].strip()
    return text.strip()


def _balanced_slice(text: str, start: int) -> Optional[str]:
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json(text: str) -> Optional[Any]:
    """Parse the last ```json block; otherwise the first balanced {...} or [...] that parses."""
    for tag, body in reversed(_FENCE.findall(text)):
        if tag.lower() in {"json", ""}:
            try:
                return json.loads(body)
            except ValueError:
                continue
    for i, ch in enumerate(text):
        if ch in "{[":
            candidate = _balanced_slice(text, i)
            if candidate:
                try:
                    return json.loads(candidate)
                except ValueError:
                    continue
    return None
