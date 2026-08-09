"""Resumable Anthropic Message Batches helper for large offline judge runs."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import anthropic


def _message_text(message) -> str:
    return "\n".join(
        block.text for block in message.content
        if getattr(block, "type", "") == "text"
    )


def batch_call(
    prompts: list[str],
    model: str,
    max_tokens: int,
    state_path: str,
    poll_seconds: int = 60,
) -> list[tuple[str | None, str | None]]:
    """Submit or resume a batch and return results in prompt order.

    ``state_path`` stores the batch ID immediately after submission. A requeued
    Slurm job resumes polling that batch instead of paying for duplicate calls.
    """
    key = (os.environ.get("ANTHROPIC_API_KEY_BATCH")
           or os.environ.get("ANTHROPIC_API_KEY"))
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY_BATCH or ANTHROPIC_API_KEY is required for judge batches"
        )
    client = anthropic.Anthropic(api_key=key)
    state = Path(state_path)
    # The state identity includes generation settings as well as prompts.  A
    # previous version omitted these, so raising max_tokens could silently
    # resume the old truncated batch and reproduce the same parse failures.
    request_identity = {
        "prompts": prompts, "model": model, "max_tokens": max_tokens,
    }
    prompt_hash = hashlib.sha256(json.dumps(
        request_identity, ensure_ascii=False, separators=(",", ":"),
    ).encode()).hexdigest()
    if state.exists():
        saved = json.loads(state.read_text())
        if saved.get("prompt_sha256") != prompt_hash:
            raise RuntimeError(
                f"{state} belongs to different prompts; move it before resubmitting"
            )
        batch_id = saved["batch_id"]
        print(f"[batch] resuming {batch_id}", flush=True)
    else:
        requests = [
            {
                "custom_id": f"item-{i:06d}",
                "params": {
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
            for i, prompt in enumerate(prompts)
        ]
        batch = client.messages.batches.create(requests=requests)
        batch_id = batch.id
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps({
            "batch_id": batch_id,
            "n": len(prompts),
            "prompt_sha256": prompt_hash,
        }, indent=2))
        print(f"[batch] submitted {batch_id}: {len(prompts)} requests", flush=True)

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"[batch] {batch.processing_status}: processing={counts.processing} "
              f"succeeded={counts.succeeded} errored={counts.errored}", flush=True)
        if batch.processing_status == "ended":
            break
        time.sleep(poll_seconds)

    out: list[tuple[str | None, str | None]] = [
        (None, "missing") for _ in prompts
    ]
    for item in client.messages.batches.results(batch_id):
        idx = int(item.custom_id.removeprefix("item-"))
        if item.result.type == "succeeded":
            out[idx] = (_message_text(item.result.message), None)
        else:
            out[idx] = (None, item.result.type)
    return out
