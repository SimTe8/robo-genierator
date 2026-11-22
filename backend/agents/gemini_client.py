"""The only module that imports the Gemini SDK (CLAUDE.md §2). Planner/critic code should extend this file.

The API key is read from the repo-root `.env` (GEMINI_API_KEY) or `os.environ`. It is never printed or logged.
Every call is logged to outputs/logs/gemini_calls.jsonl (model, purpose, latency, tokens, thoughts, ok/error).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any

from dotenv import dotenv_values

from config import OUTPUTS_DIR, REPO_ROOT

ENV_PATH = REPO_ROOT / ".env"
LOG_PATH = OUTPUTS_DIR / "logs" / "gemini_calls.jsonl"
PLACEHOLDER = "your_gemini_api_key_here"

_clients: dict[tuple[str, str | None, float], Any] = {}
_active_key: str | None = None


class GeminiUnavailable(RuntimeError):
    pass


def _key() -> str:
    """Re-read .env each time, so a key pasted while the server runs works without a restart."""
    for key in ((dotenv_values(ENV_PATH).get("GEMINI_API_KEY") or ""), os.environ.get("GEMINI_API_KEY", "")):
        key = key.strip()
        if key and key != PLACEHOLDER:
            return key
    return ""


def available() -> bool:
    return bool(_key())


def get_client(*, api_version: str | None = None, timeout_s: float = 120.0):
    """Return a cached genai.Client with bounded HTTP timeout and hot-reloaded GEMINI_API_KEY."""
    global _active_key
    key = _key()
    if not key:
        raise GeminiUnavailable("GEMINI_API_KEY is not set. Put it in the repo-root .env file (not .env.example).")

    if _active_key != key:
        _clients.clear()
        _active_key = key

    timeout_norm = round(max(1.0, float(timeout_s)), 3)
    cache_key = (key, api_version, timeout_norm)
    client = _clients.get(cache_key)
    if client is None:
        from google import genai
        from google.genai import types

        http_kwargs: dict[str, Any] = {"timeout": int(timeout_norm * 1000)}
        if api_version is not None:
            http_kwargs["api_version"] = api_version
        client = genai.Client(api_key=key, http_options=types.HttpOptions(**http_kwargs))
        _clients[cache_key] = client
    return client


def _get_client():
    return get_client(timeout_s=120.0)


def supports_thinking_config(model: str | None) -> bool:
    """Only `gemini-4-*` models support ThinkingConfig(thinking_level=...)."""
    if model is None:
        return True
    return "gemini-4-" in model.lower()


def build_thinking_config(
    thinking_level: str | None,
    *,
    model: str | None = None,
):
    """Build google.genai.types.ThinkingConfig(thinking_level=...) for Gemini 4 models.

    Accepts 'minimal', 'low', 'medium', or 'high' (case-insensitive).
    Returns None when thinking_level is empty/None or when `model` is a non-`gemini-4-*` model.
    """
    if not thinking_level or not supports_thinking_config(model):
        return None
    normalized = thinking_level.strip().upper()
    if not normalized or normalized in ("NONE", "OFF"):
        return None

    from google.genai import types

    level_enum = getattr(types.ThinkingLevel, normalized, None)
    if level_enum is None:
        raise ValueError(
            f"Unsupported thinking_level={thinking_level!r}; expected one of: minimal, low, medium, high."
        )
    return types.ThinkingConfig(thinking_level=level_enum)


def is_thinking_config_rejected(err: Exception) -> bool:
    """Return True if an SDK error indicates HTTP 400 / INVALID_ARGUMENT rejection of thinking_config."""
    code = getattr(err, "code", None) or getattr(err, "status_code", None)
    if code == 400:
        return True
    msg = str(err)
    return "400" in msg or "INVALID_ARGUMENT" in msg.upper()


def _log(entry: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), **entry}) + "\n")


def _usage_field(usage: Any, attr: str) -> Any:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage.get(attr)
    return getattr(usage, attr, None)


def log_call(
    *,
    purpose: str,
    model: str,
    latency_s: float,
    ok: bool,
    usage: Any = None,
    extra: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    """Append a structured Gemini call telemetry line to outputs/logs/gemini_calls.jsonl."""
    entry: dict[str, Any] = {
        "purpose": purpose,
        "model": model,
        "latency_s": round(float(latency_s), 2),
        "ok": bool(ok),
    }
    if usage is not None:
        entry["prompt_tokens"] = _usage_field(usage, "prompt_token_count")
        entry["output_tokens"] = _usage_field(usage, "candidates_token_count")
        entry["thoughts_token_count"] = _usage_field(usage, "thoughts_token_count")
    if extra:
        entry.update(extra)
    if error is not None:
        entry["error"] = error
    _log(entry)


def ensure_call_http_options(client: Any, timeout_s: float = 120.0):
    """Provide per-call HttpOptions timeout if `client` was instantiated without one."""
    client_opts = getattr(getattr(client, "_api_client", None), "_http_options", None)
    if getattr(client_opts, "timeout", None) is not None:
        return None
    from google.genai import types

    kwargs: dict[str, Any] = {"timeout": int(max(1.0, float(timeout_s)) * 1000)}
    api_ver = getattr(client_opts, "api_version", None)
    if api_ver is not None:
        kwargs["api_version"] = api_ver
    return types.HttpOptions(**kwargs)


def generate_json(
    model: str,
    prompt: str,
    schema: dict[str, Any],
    *,
    purpose: str,
    timeout_s: float = 90.0,
    retries: int = 2,
) -> dict[str, Any]:
    """Structured output: returns the parsed JSON object that follows `schema` (a plain JSON Schema dict)."""
    from google.genai import types

    client = get_client(timeout_s=timeout_s)
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
            log_call(
                purpose=purpose,
                model=model,
                latency_s=time.time() - t0,
                ok=True,
                usage=usage,
            )
            return json.loads(text)
        except Exception as e:  # network/5xx/parse: back off and retry
            last_err = e
            log_call(
                purpose=purpose,
                model=model,
                latency_s=time.time() - t0,
                ok=False,
                error=f"{type(e).__name__}: {str(e)[:300]}",
            )
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Gemini call failed after {retries + 1} attempts: {last_err}")
