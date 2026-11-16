"""Quest w5 / multimodal RAG, in its practical form: turn non-text content into searchable text.
Images (standalone files and images embedded in PDFs) are described by a vision model; CSV tables become
row-group documents that repeat the header. Everything is then indexed like any other text, and
metadata['modality'] ('text' | 'image' | 'table') lets you filter by type.
Not included: image-embedding retrieval (CLIP / ColPali), which needs a GPU-friendly vision stack."""
from __future__ import annotations

import base64
import csv
import io
import os
from pathlib import Path

from .embeddings import post_json
from .loaders import load_path
from .types import Document

PROMPT = "Describe this image for a search index. Include any text, numbers, chart axes and trends you can read."
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}


class Captioner:
    name = "base"

    def caption(self, image: bytes, mime: str) -> str:
        raise NotImplementedError


class OllamaCaptioner(Captioner):
    def __init__(self, model: str = "llava"):
        self.model, self.name = model, f"ollama:{model}"
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

    def caption(self, image, mime):
        msg = {"role": "user", "content": PROMPT, "images": [base64.b64encode(image).decode()]}
        return post_json(f"{self.host}/api/chat", {"model": self.model, "stream": False, "messages": [msg]})["message"]["content"].strip()


class AnthropicCaptioner(Captioner):
    def __init__(self, model: str):
        self.model, self.name = model, f"anthropic:{model}"
        self.key = os.environ.get("ANTHROPIC_API_KEY", "")

    def caption(self, image, mime):
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": base64.b64encode(image).decode()}},
            {"type": "text", "text": PROMPT},
        ]
        resp = post_json(
            "https://api.anthropic.com/v1/messages",
            {"model": self.model, "max_tokens": 400, "messages": [{"role": "user", "content": content}]},
            {"x-api-key": self.key, "anthropic-version": "2023-06-01"},
        )
        return "".join(b.get("text", "") for b in resp["content"]).strip()


class OpenAICaptioner(Captioner):
    def __init__(self, model: str):
        self.model, self.name = model, f"openai:{model}"
        self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.key = os.environ.get("OPENAI_API_KEY", "")

    def caption(self, image, mime):
        url = f"data:{mime};base64,{base64.b64encode(image).decode()}"
        content = [{"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": url}}]
        resp = post_json(
            f"{self.base}/chat/completions",
            {"model": self.model, "max_tokens": 400, "messages": [{"role": "user", "content": content}]},
            {"Authorization": f"Bearer {self.key}"},
        )
        return resp["choices"][0]["message"]["content"].strip()


def make_captioner(spec: str) -> Captioner:
    kind, _, rest = spec.partition(":")
    if kind == "ollama":
        return OllamaCaptioner(rest or "llava")
    if kind == "anthropic":
        return AnthropicCaptioner(rest)
    if kind == "openai":
        return OpenAICaptioner(rest)
    raise ValueError(f"unknown captioner spec: {spec!r}")


def _table_docs(path: Path, rows_per_doc: int = 20) -> list[Document]:
    with path.open(newline="", encoding="utf-8", errors="ignore") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return []
    head, body = rows[0], rows[1:]
    docs = []
    for n in range(0, len(body), rows_per_doc):
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        lines += ["| " + " | ".join(r) + " |" for r in body[n : n + rows_per_doc]]
        docs.append(Document(f"{path.name}#rows{n + 1}", "\n".join(lines), {"source": path.name, "modality": "table"}))
    return docs


def load_multimodal(path: str | Path, captioner: Captioner) -> list[Document]:
    p = Path(path)
    docs = [Document(d.id, d.text, {**d.metadata, "modality": "text"}) for d in load_path(p)]
    files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())
    for f in files:
        ext = f.suffix.lower()
        if ext in MIME:
            cap = captioner.caption(f.read_bytes(), MIME[ext])
            docs.append(Document(f"{f.name}#image", f"Image {f.name}: {cap}", {"source": f.name, "modality": "image"}))
        elif ext == ".csv":
            docs.extend(_table_docs(f))
        elif ext == ".pdf":
            docs.extend(_pdf_images(f, captioner))
    return docs


def _pdf_images(f: Path, captioner: Captioner) -> list[Document]:
    from pypdf import PdfReader

    out = []
    try:
        for pno, page in enumerate(PdfReader(str(f)).pages, 1):
            for n, img in enumerate(page.images):
                ext = Path(img.name).suffix.lower()
                cap = captioner.caption(img.data, MIME.get(ext, "image/png"))
                out.append(Document(f"{f.name}#p{pno}img{n}", f"Image on page {pno} of {f.name}: {cap}",
                                    {"source": f.name, "page": pno, "modality": "image"}))
    except Exception:  # damaged images should not break ingestion of the rest
        pass
    return out
