#!/usr/bin/env python
"""
Quick OpenRouter test client for s1k-style generation.

Supports reasoning models (e.g., `openai/gpt-oss-20b:free`) with
`--reasoning-effort {low,medium,high}` and extracts:
- thinking_trajectory: from `message.reasoning` or `reasoning_details`
- attempt: from `message.content`

Reads API key from env var `OPENROUTER_API_KEY` and prints:
- Raw API JSON payload (optional)
- Parsed thinking_trajectory and attempt
- Basic usage stats and identifiers

Example:
  OPENROUTER_API_KEY=sk-... \
  python scripts/test_openrouter.py --from-jsonl data/s1k-1.1/train_mini.jsonl --index 0

  python scripts/test_openrouter.py --question "What is 13*17?"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import requests


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def load_question(from_jsonl: Optional[str], index: int, fallback_question: Optional[str]) -> Dict[str, Any]:
    if fallback_question:
        return {"question": fallback_question}

    if not from_jsonl:
        # Very small default to keep the tool immediately usable.
        return {"question": "What is the sum of the first 10 positive integers?"}

    path = Path(from_jsonl)
    if not path.exists():
        raise SystemExit(f"JSONL not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                ex = json.loads(line)
                return ex
    raise SystemExit(f"Index {index} out of range for {path}")


def build_messages(question: str) -> list[dict]:
    system = (
        "You are a helpful assistant. You are given a question and you need to answer it."
    )
    user = (
        f"Question:\n{question}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_openrouter(
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 16384,
    temperature: float = 0.7,
    json_mode: bool = False,
    request_timeout: int = 120,
    referer: Optional[str] = None,
    title: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    include_usage_details: bool = True,
) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # OpenRouter recommends setting these when possible (helps with rate limits/telemetry)
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        # This is OpenAI-compatible; many models will comply.
        payload["response_format"] = {"type": "json_object"}

    # Reasoning models (o3/o4, gpt-oss, etc.) accept a reasoning effort setting.
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}

    # Request provider token breakdown details when available
    if include_usage_details:
        payload["usage"] = {"include": True}

    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=request_timeout)
    resp.raise_for_status()
    return resp.json()


def _join_reasoning_details(details: Sequence[dict] | None) -> str:
    if not details:
        return ""
    out = []
    for d in details:
        t = d.get("text") if isinstance(d, dict) else None
        if isinstance(t, str) and t:
            out.append(t)
    return "\n".join(out).strip()


def extract_reasoning_and_answer(resp_json: dict) -> tuple[str, str, dict]:
    """Return (thinking_trajectory, attempt, meta) from a Chat Completions response."""
    choices = resp_json.get("choices") or []
    if not choices:
        return "", "", {}
    msg = choices[0].get("message", {}) or {}
    # Primary sources for reasoning traces seen across providers
    reasoning = (
        msg.get("reasoning")
        or _join_reasoning_details(msg.get("reasoning_details"))
        or _join_reasoning_details(resp_json.get("reasoning_details"))
        or ""
    )
    content = msg.get("content") or ""
    meta = {
        "finish_reason": choices[0].get("finish_reason"),
        "native_finish_reason": choices[0].get("native_finish_reason"),
        "role": msg.get("role"),
    }
    return str(reasoning or ""), str(content or ""), meta


def main():
    ap = argparse.ArgumentParser(description="Send a sample request to OpenRouter and print the response")
    ap.add_argument("--model", default="openai/gpt-oss-20b:free")
    ap.add_argument("--question", default=None, help="Explicit question string")
    ap.add_argument("--from-jsonl", default=str(Path("data") / "s1k-1.1" / "train_mini.jsonl"))
    ap.add_argument("--index", type=int, default=0, help="Row index in JSONL if using --from-jsonl")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--json-mode", action="store_true", help="Enable response_format=json_object for content")
    ap.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default="medium",
        help="Reasoning effort for models that support it (e.g., gpt-oss)",
    )
    ap.add_argument(
        "--no-usage-details",
        action="store_true",
        help="Do not request provider token breakdown details",
    )
    ap.add_argument("--show-payload", action="store_true")

    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Missing OPENROUTER_API_KEY in environment.", file=sys.stderr)
        sys.exit(2)

    ex = load_question(args.from_jsonl, args.index, args.question)
    question = ex.get("question") or ex.get("prompt") or args.question
    if not question:
        raise SystemExit("Could not determine a question to send.")

    messages = build_messages(str(question))

    referer = os.environ.get("OPENROUTER_SITE_URL") or "http://localhost"
    title = os.environ.get("OPENROUTER_TITLE") or "s1k-generation-test"

    if args.show_payload:
        print("=== Request Payload ===")
        print(json.dumps({
            "url": OPENROUTER_URL,
            "model": args.model,
            "messages": messages,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "json_mode": args.json_mode,
            "reasoning_effort": args.reasoning_effort,
            "usage_include": not args.no_usage_details,
        }, ensure_ascii=False, indent=2))

    print("Sending request to OpenRouter ...")
    try:
        resp_json = call_openrouter(
            api_key=api_key,
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            json_mode=args.json_mode,
            referer=referer,
            title=title,
            reasoning_effort=args.reasoning_effort,
            include_usage_details=(not args.no_usage_details),
        )
    except requests.HTTPError as e:
        print("Request failed with HTTP error:", e, file=sys.stderr)
        if e.response is not None:
            try:
                print(e.response.text, file=sys.stderr)
            except Exception:
                pass
        sys.exit(1)

    print("\n=== Raw API Response ===")
    print(json.dumps(resp_json, ensure_ascii=False, indent=2))

    thinking, attempt, meta = extract_reasoning_and_answer(resp_json)

    print("\n=== Parsed Fields ===")
    print(json.dumps({
        "thinking_trajectory": thinking,
        "attempt": attempt,
        "meta": meta,
    }, ensure_ascii=False, indent=2))

    usage = resp_json.get("usage", {})
    rid = resp_json.get("id")
    model = resp_json.get("model")
    print("\n=== Usage / IDs ===")
    print(json.dumps({
        "id": rid,
        "model": model,
        "provider": resp_json.get("provider"),
        "usage": usage,
        "usage_details": {
            "prompt_tokens_details": usage.get("prompt_tokens_details"),
            "completion_tokens_details": usage.get("completion_tokens_details"),
        },
    }, ensure_ascii=False, indent=2))

    # If the provider echoes any reasoning-related fields, show them
    echoed_reasoning = {
        "top_level_reasoning": resp_json.get("reasoning"),
        "choice_reasoning_meta": (resp_json.get("choices", [{}])[0].get("message", {}).get("metadata")
                                   if resp_json.get("choices") else None),
    }
    if any(v for v in echoed_reasoning.values()):
        print("\n=== Provider-Reported Reasoning Meta (if any) ===")
        print(json.dumps(echoed_reasoning, ensure_ascii=False, indent=2))

    # Also echo what we requested (useful for audits)
    print("\n=== Requested Settings (echo) ===")
    print(json.dumps({
        "requested_reasoning_effort": args.reasoning_effort,
        "requested_json_mode": args.json_mode,
        "requested_temperature": args.temperature,
        "requested_max_tokens": args.max_tokens,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
