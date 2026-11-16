import pytest

from ragforge.chunking import chunk_documents
from ragforge.embeddings import HashingEmbedder
from ragforge.index import Index
from ragforge.llm import LLM
from ragforge.loaders import load_path


class Scripted(LLM):
    """Fake LLM driven by a function (system, prompt) -> reply; records every call."""

    name = "scripted"

    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def generate(self, system, prompt, max_tokens=700):
        self.calls.append((system, prompt))
        return self.fn(system, prompt)


@pytest.fixture(scope="session")
def docs():
    return load_path("data/sample")


@pytest.fixture(scope="session")
def small_index(docs):
    return Index.build(chunk_documents(docs, 300, 50), HashingEmbedder(512))
