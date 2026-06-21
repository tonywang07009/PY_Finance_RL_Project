"""Lazy Granite sentiment analysis helpers."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from .config import DEFAULT_CONFIG


class GraniteSentimentAnalyzer:
    """Financial-news sentiment analyzer with lazy model initialization."""

    def __init__(
        self,
        model_id: str = DEFAULT_CONFIG.model_id,
        torch_dtype_name: str = "bfloat16",
        device_map: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.torch_dtype_name = torch_dtype_name
        self.device_map = device_map
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        """Initialize tokenizer/model only when sentiment inference is needed."""
        if self._tokenizer is not None and self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = getattr(torch, self.torch_dtype_name)
        print("loading tokenizer...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        print("loading model...")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map=self.device_map,
        )

    @property
    def tokenizer(self):
        self.load()
        return self._tokenizer

    @property
    def model(self):
        self.load()
        return self._model

    def _model_device(self):
        import torch

        model = self.model
        if hasattr(model, "device"):
            return model.device
        return next(model.parameters()).device

    def ask(self, prompt: str, max_new_tokens: int = 64) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(self._model_device())

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def analyze(self, text: str, max_new_tokens: int = 128) -> dict[str, Any]:
        import torch

        system_prompt = (
            "You are an assistant specialized in sentiment analysis for financial news. "
            "Read the news content and determine the overall sentiment as "
            '"positive", "negative", "neutral", or "mixed". '
            "Estimate a confidence score between 0 and 1. "
            "You MUST output STRICT JSON only, with NO extra text, NO explanation, "
            "in this format only:\n"
            '{"label": "positive", "confidence": 0.83}\n'
            "If you cannot decide, choose neutral with a reasonable confidence."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Analyze this news item and output ONLY JSON.\nText:\n" + text,
            },
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(self._model_device())

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        raw = self.tokenizer.decode(generated, skip_special_tokens=True).strip().strip("` \n")
        print("RAW OUTPUT:", repr(raw))

        return parse_sentiment_output(raw)


def parse_sentiment_output(raw: str) -> dict[str, Any]:
    """Parse a strict JSON sentiment response, falling back to neutral."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    json_str = match.group(0) if match else raw

    try:
        result = json.loads(json_str)
    except json.JSONDecodeError:
        result = {"label": "neutral", "confidence": 0.0, "raw_output": raw}

    label = str(result.get("label", "neutral")).lower().strip()
    if label not in {"positive", "negative", "neutral", "mixed"}:
        label = "neutral"

    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    parsed = {
        "label": label,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
    }
    if "raw_output" in result:
        parsed["raw_output"] = raw
    return parsed


_DEFAULT_ANALYZER: GraniteSentimentAnalyzer | None = None


def get_default_analyzer(model_id: str = DEFAULT_CONFIG.model_id) -> GraniteSentimentAnalyzer:
    global _DEFAULT_ANALYZER
    if _DEFAULT_ANALYZER is None or _DEFAULT_ANALYZER.model_id != model_id:
        _DEFAULT_ANALYZER = GraniteSentimentAnalyzer(model_id=model_id)
    return _DEFAULT_ANALYZER


def ask_granite(
    prompt: str,
    max_new_tokens: int = 64,
    analyzer: GraniteSentimentAnalyzer | None = None,
) -> str:
    return (analyzer or get_default_analyzer()).ask(prompt, max_new_tokens=max_new_tokens)


def analyze_sentiment(
    text: str,
    max_new_tokens: int = 128,
    analyzer: GraniteSentimentAnalyzer | None = None,
) -> dict[str, Any]:
    return (analyzer or get_default_analyzer()).analyze(text, max_new_tokens=max_new_tokens)
