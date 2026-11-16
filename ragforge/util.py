from __future__ import annotations

import json
import re


def parse_json(text: str):
    """First JSON object or array found in an LLM reply, or None."""
    m = re.search(r"\{.*\}|\[.*\]", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def clean_lines(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", line).strip()
        if line:
            out.append(line)
    return out
