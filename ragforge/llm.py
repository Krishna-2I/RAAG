"""Quest f3: LLM backends behind one `generate(system, prompt)` call.

  ollama:llama3.1        local Ollama
  openai:gpt-4o-mini     any OpenAI-compatible endpoint (also vLLM / LM Studio via OPENAI_BASE_URL)
  anthropic:<model-id>   Anthropic API (ANTHROPIC_API_KEY)
  extractive             offline stand-in that quotes the best context sentences (tests / demos)
"""
from __future__ import annotations

import os
import re

from .bm25 import tokenize
from .embeddings import post_json


class LLM:
    name = "base"

    def generate(self, system: str, prompt: str, max_tokens: int = 700) -> str:
        raise NotImplementedError


class OllamaLLM(LLM):
    def __init__(self, model: str):
        self.model, self.name = model, f"ollama:{model}"
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

    def generate(self, system, prompt, max_tokens=700):
        resp = post_json(
            f"{self.host}/api/chat",
            {
                "model": self.model,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            },
        )
        return resp["message"]["content"].strip()


class OpenAICompatLLM(LLM):
    def __init__(self, model: str):
        self.model, self.name = model, f"openai:{model}"
        self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.key = os.environ.get("OPENAI_API_KEY", "")

    def generate(self, system, prompt, max_tokens=700):
        resp = post_json(
            f"{self.base}/chat/completions",
            {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            },
            {"Authorization": f"Bearer {self.key}"},
        )
        return resp["choices"][0]["message"]["content"].strip()


class AnthropicLLM(LLM):
    def __init__(self, model: str):
        self.model, self.name = model, f"anthropic:{model}"
        self.key = os.environ.get("ANTHROPIC_API_KEY", "")

    def generate(self, system, prompt, max_tokens=700):
        resp = post_json(
            "https://api.anthropic.com/v1/messages",
            {
                "model": self.model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
            {"x-api-key": self.key, "anthropic-version": "2023-06-01"},
        )
        return "".join(b.get("text", "") for b in resp["content"]).strip()


class ExtractiveLLM(LLM):
    name = "extractive"

    def generate(self, system, prompt, max_tokens=700):
        question = prompt.rsplit("Question:", 1)[-1].split("\n")[0]
        qtok = set(tokenize(question))
        blocks = re.findall(r"\[(\d+)\]\s(.+?)(?=\n\[\d+\]|\n\nQuestion:|\Z)", prompt, re.S)
        scored = []
        for num, body in blocks:
            for sent in re.split(r"(?<=[.!?])\s+", body.strip()):
                scored.append((len(qtok & set(tokenize(sent))), num, sent.strip()))
        scored.sort(key=lambda t: -t[0])
        best = [s for s in scored[:2] if s[0] > 0]
        if not best:
            return "I don't know based on the provided context."
        return " ".join(f"{sent} [{num}]" for _, num, sent in best)


def make_llm(spec: str) -> LLM:
    kind, _, rest = spec.partition(":")
    if kind == "ollama":
        return OllamaLLM(rest)
    if kind == "openai":
        return OpenAICompatLLM(rest)
    if kind == "anthropic":
        return AnthropicLLM(rest)
    if kind == "extractive":
        return ExtractiveLLM()
    raise ValueError(f"unknown llm spec: {spec!r}")
