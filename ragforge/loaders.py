"""Quest f1: load documents. Supports .txt, .md and .pdf (one Document per PDF page)."""
from __future__ import annotations

from pathlib import Path

from .types import Document

TEXT_EXT = {".txt", ".md", ".markdown"}


def load_path(path: str | Path) -> list[Document]:
    p = Path(path)
    files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())
    docs: list[Document] = []
    for f in files:
        ext = f.suffix.lower()
        if ext in TEXT_EXT:
            text = f.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                docs.append(Document(f.name, text, {"source": f.name}))
        elif ext == ".pdf":
            docs.extend(_load_pdf(f))
    return docs


def _load_pdf(f: Path) -> list[Document]:
    from pypdf import PdfReader

    out = []
    for i, page in enumerate(PdfReader(str(f)).pages, 1):
        text = (page.extract_text() or "").strip()
        if text:
            out.append(Document(f"{f.name}#p{i}", text, {"source": f.name, "page": i}))
    return out
