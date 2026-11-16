"""Eval set format, one JSON object per line:
{"question": "...", "evidence": ["exact phrase that must appear in a relevant chunk"], "reference": "optional answer"}
"""
from __future__ import annotations

import json
from pathlib import Path


def load_eval(path: str | Path) -> list[dict]:
    items = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if "question" not in item or "evidence" not in item:
            raise ValueError(f"{path}:{n} needs 'question' and 'evidence'")
        ev = item["evidence"]
        item["evidence"] = [ev] if isinstance(ev, str) else list(ev)
        items.append(item)
    return items
