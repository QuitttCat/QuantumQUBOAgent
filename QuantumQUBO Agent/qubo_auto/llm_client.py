from __future__ import annotations
import hashlib
import json
import os
import time
from pathlib import Path

from openai import OpenAI, RateLimitError


class TokenBudgetExceeded(Exception):
    pass


def _no_think(model: str, prompt: str) -> str:
    """Prepend /no_think for Qwen3 models to disable chain-of-thought thinking."""
    if "qwen3" in model.lower() or "qwen3" in model.lower():
        return "/no_think\n" + prompt
    return prompt


class LLMClient:
    def __init__(self, transcript_dir: Path, run_id: str, max_tokens: int = 200_000, use_cache: bool = True):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        self.transcript_dir = transcript_dir / run_id
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        self.max_tokens = max_tokens
        self.use_cache = use_cache
        self.tokens_used = 0
        self.cost_used = 0.0
        self._cache: dict[str, str] = {}
        if use_cache:
            self._load_cache()

    def _cache_path(self) -> Path:
        return self.transcript_dir.parent / ".llm_cache.json"

    def _load_cache(self) -> None:
        path = self._cache_path()
        if path.exists():
            try:
                self._cache = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}

    def _save_cache(self) -> None:
        path = self._cache_path()
        path.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")

    def _cache_key(self, model: str, prompt: str) -> str:
        raw = f"{model}||{prompt}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def call(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        step_name: str = "step",
        json_mode: bool = False,
    ) -> str:
        prompt = _no_think(model, prompt)

        cache_key = self._cache_key(model, prompt)
        if self.use_cache and temperature == 0.0 and cache_key in self._cache:
            cached = self._cache[cache_key]
            self._log(step_name, model, temperature, prompt, cached, usage=None, from_cache=True)
            return cached

        if self.tokens_used >= self.max_tokens:
            raise TokenBudgetExceeded(
                f"Token budget of {self.max_tokens} exceeded (used {self.tokens_used})"
            )

        kwargs: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # Retry on 429 with exponential backoff (free-tier rate limits)
        for attempt in range(5):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except RateLimitError:
                if attempt == 4:
                    raise
                wait = 30 * (2 ** attempt)  # 30, 60, 120, 240 s
                print(f"  [rate limit] waiting {wait}s before retry {attempt+2}/5...", flush=True)
                time.sleep(wait)

        content = resp.choices[0].message.content or ""

        usage = resp.usage
        if usage:
            self.tokens_used += usage.total_tokens
            self.cost_used += float(getattr(usage, "cost", 0.0) or 0.0)

        if self.use_cache and temperature == 0.0:
            self._cache[cache_key] = content
            self._save_cache()

        self._log(step_name, model, temperature, prompt, content, usage)
        return content

    def _log(
        self,
        step_name: str,
        model: str,
        temperature: float,
        prompt: str,
        response: str,
        usage,
        from_cache: bool = False,
    ) -> None:
        ts = int(time.time())
        log_path = self.transcript_dir / f"{step_name}_{ts}.json"
        payload = {
            "model": model,
            "temperature": temperature,
            "prompt": prompt,
            "response": response,
            "from_cache": from_cache,
            "usage": usage.model_dump() if usage and hasattr(usage, "model_dump") else None,
            "timestamp": ts,
        }
        log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
