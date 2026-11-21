"""The only module that imports the Gemini SDK (CLAUDE.md §2). Planner/critic code should extend this file.

The API key is read from the repo-root `.env` (GEMINI_API_KEY). It is never printed or logged.
Every call is logged to outputs/logs/gemini_calls.jsonl (model, purpose, latency, tokens, ok/error).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any

from dotenv import load_dotenv

from config import OUTPUTS_DIR, REPO_ROOT

load_dotenv(REPO_ROOT / ".env")
LOG_PATH = OUTPUTS_DIR / "logs" / "gemini_calls.jsonl"

_client = None


class GeminiUnavailable(RuntimeError):
    pass


def available() -> bool:
    key = os.environ.get("GEMINI_API_KEY", "")
    return bool(key) and key != "your_gemini_api_key_here"


def _get_client():
    global _client
    if not available():
        raise GeminiUnavailable("GEMINI_API_KEY is not set. Put it in the repo-root .env file (not .env.example).")
    if _client is None:
        from google import genai

        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def _log(entry: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), **entry}) + "\n")


def generate_json(model: str, prompt: str, schema: dict[str, Any], *, purpose: str, timeout_s: float = 90.0, retries: int = 2) -> dict[str, Any]:
    """Structured output: returns the parsed JSON object that follows `schema` (a plain JSON Schema dict)."""
    from google.genai import types

    client = _get_client()
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=schema,
        http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
    )
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=config)
            text = resp.text or ""
            usage = getattr(resp, "usage_metadata", None)
            _log({"purpose": purpose, "model": model, "latency_s": round(time.time() - t0, 2), "ok": True,
                  "prompt_tokens": getattr(usage, "prompt_token_count", None),
                  "output_tokens": getattr(usage, "candidates_token_count", None)})
            return json.loads(text)
        except Exception as e:  # network/5xx/parse: back off and retry
            last_err = e
            _log({"purpose": purpose, "model": model, "latency_s": round(time.time() - t0, 2), "ok": False,
                  "error": f"{type(e).__name__}: {str(e)[:300]}"})
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Gemini call failed after {retries + 1} attempts: {last_err}")
