"""Shared Anthropic call helper for the eval scripts.

Two things every caller in this repo was getting wrong, both discovered on the Sonnet 5 switch:

  1. Sonnet 5 returns extended thinking, so `resp.content[0]` is a ThinkingBlock, not a TextBlock.
     Any code doing `resp.content[0].text` raises AttributeError against Sonnet 5 while working
     fine against 4.6. Text must be gathered from the blocks whose type is "text".

  2. Sonnet 5 rejects `temperature` outright: 400 invalid_request_error, "`temperature` is
     deprecated for this model." So the parameter must be omitted, not merely left at default.

  3. The low-prio key sits behind an org-wide output-token limit and returns sustained 429s, not
     isolated ones. Short backoff cannot ride that out. Policy is: start on low-prio, back off,
     and only swap to the important-experiment key when the run actually matters -- and say so
     out loud when the swap happens, so a run's cost is never silent.

Together those three mean a script that worked verbatim against Sonnet 4.6 fails three
different ways against Sonnet 5, each with a different error class.

Failures return None rather than raising, but they carry the exception name, because an earlier
version collapsed every failure to a bare None and a pure rate-limit outage read as the judge
failing to discriminate -- an infrastructure problem wearing a result's clothing.
"""
import random
import sys
import threading
import time

import os

import anthropic

# Key source per tier: prefer an env var (per repo convention), fall back to a file path.
LOW, FALLBACK = "low", "fb"
_KEY_SRC = {"low": ("ANTHROPIC_API_KEY", "/tmp/.lp"),
            "fb": ("ANTHROPIC_API_KEY_FALLBACK", "/tmp/.fb")}
_clients, _lock, _warned = {}, threading.Lock(), set()


def _resolve_key(tier):
    env, path = _KEY_SRC[tier]
    k = os.environ.get(env)
    if k:
        return k.strip()
    if os.path.exists(path):
        return open(path).read().strip()
    raise RuntimeError(f"no API key for tier {tier!r}: set ${env} or write {path}")


def _client(tier):
    with _lock:
        if tier not in _clients:
            _clients[tier] = anthropic.Anthropic(api_key=_resolve_key(tier))
        return _clients[tier]


def _text(resp):
    """Concatenate the text blocks, skipping thinking blocks."""
    return "\n".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def _note(msg):
    with _lock:
        if msg not in _warned:
            _warned.add(msg)
            print(f"[llm] {msg}", file=sys.stderr, flush=True)


def _kwargs(model, max_tokens, temperature):
    kw = {"model": model, "max_tokens": max_tokens,
          "messages": None}                      # filled by the caller
    # Claude 5 deprecated the sampling temperature; passing it at all is a 400.
    if temperature is not None and not _is_v5(model):
        kw["temperature"] = temperature
    return kw


def _is_v5(model):
    return model.endswith("-5") or "-5-" in model


def call(prompt, model, max_tokens=1024, temperature=0.0, retries=5, allow_fallback=True):
    """Return (text, error_name). text is None on failure."""
    last = ""
    extra = {} if _is_v5(model) else {"temperature": temperature}
    for attempt in range(retries):
        try:
            r = _client(LOW).messages.create(
                model=model, max_tokens=max_tokens, **extra,
                messages=[{"role": "user", "content": prompt}])
            return _text(r), None
        except anthropic.RateLimitError as e:
            last = "RateLimitError"
            if attempt < retries - 1:
                time.sleep(min(45, 3 * 2 ** attempt) * (0.7 + 0.6 * random.random()))
        except Exception as e:
            last = type(e).__name__
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    if allow_fallback and last == "RateLimitError":
        _note("low-prio key persistently rate-limited; SWAPPED TO IMPORTANT-EXPERIMENT KEY "
              "for the rest of this run")
        for attempt in range(3):
            try:
                r = _client(FALLBACK).messages.create(
                    model=model, max_tokens=max_tokens, **extra,
                    messages=[{"role": "user", "content": prompt}])
                return _text(r), None
            except Exception as e:
                last = type(e).__name__
                time.sleep(3 * (attempt + 1))
    return None, last
