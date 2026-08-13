"""Fit PER-OFFSET averaged Jacobians J^(delta)_{L42->L62} for Qwen3.6-27B.

The multi-slot lens design: at test time an activation h from L42 is
transported to K future-token slots as [Jbar^(0) h, ..., Jbar^(K-1) h] and
fed to the K-marker activation verbalizer. This script fits the Jbar family
(comb estimator with Rademacher tooth signs; see jlens/offset_fitting.py).

Shard across GPUs by launching one process per GPU:
    SHARD=0 N_SHARDS=4 CUDA_VISIBLE_DEVICES=0 python fit_offset_jlens.py &
    SHARD=1 N_SHARDS=4 CUDA_VISIBLE_DEVICES=1 python fit_offset_jlens.py &
    ...
then merge with merge_offset_fit.py (sums shard checkpoints, saves .npy).

Env knobs:
  N_PROMPTS   (default 512)  total prompts across all shards
  N_OFFSETS   (default 16)   offsets delta = 0..N_OFFSETS-1
  COMB        (default 32)   comb tooth spacing (> N_OFFSETS)
  SEQ_LEN     (default 512)  tokens per prompt
  DIM_BATCH   (default 32)   jacobian rows per backward pass
  SHARD / N_SHARDS           this process's prompt slice
  SRC / TGT   (default 42/62)
  OUT_DIR     (default /workspace/results/offset_jlens)
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens import configure_logging
from jlens.hf import from_hf
from jlens.offset_fitting import fit_offsets

BASE = "Qwen/Qwen3.6-27B"
SRC = [int(x) for x in os.environ.get("SRC", "42").split(",")]
TGT = int(os.environ.get("TGT", "62"))
N_PROMPTS = int(os.environ.get("N_PROMPTS", "512"))
N_OFFSETS = int(os.environ.get("N_OFFSETS", "16"))
COMB = int(os.environ.get("COMB", "32"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "512"))
DIM_BATCH = int(os.environ.get("DIM_BATCH", "32"))
SHARD = int(os.environ.get("SHARD", "0"))
N_SHARDS = int(os.environ.get("N_SHARDS", "1"))
OUT_DIR = os.environ.get("OUT_DIR", "/workspace/results/offset_jlens")
PROMPTS_JSON = os.environ.get(
    "PROMPTS_JSON", "/workspace/data/ffw_fit_prompts_512tok.json"
)
PROMPTS_POOL = os.environ.get(
    "PROMPTS_POOL", "/workspace/data/ffw_fit_pool.parquet"
)
os.makedirs(OUT_DIR, exist_ok=True)
configure_logging()


def build_prompt_list(path: str, n_prompts: int, min_tokens: int) -> list[str]:
    """FineFineWeb docs long enough for the comb, cached to json.

    Reads from the fit-prompt pool parquet written by
    pretrain/materialize_ffw_corpus.py — rows DISJOINT from the span-collection
    corpus (fitting must not average over the training docs) but from the same
    corpus family, so the Jacobian average matches the training distribution.
    Deterministic, so every shard sees the same list and slices it disjointly.
    """
    if os.path.exists(path):
        prompts = json.load(open(path))
        if len(prompts) >= n_prompts:
            return prompts[:n_prompts]
    import pyarrow.parquet as pq

    tok = AutoTokenizer.from_pretrained(BASE)
    pool = pq.read_table(PROMPTS_POOL, columns=["text"]).column("text").to_pylist()
    prompts: list[str] = []
    for text in pool:
        if not text or len(text) < min_tokens * 3:  # cheap pre-filter
            continue
        ids = tok(text, truncation=True, max_length=min_tokens + 8)["input_ids"]
        if len(ids) >= min_tokens:
            prompts.append(text)
        if len(prompts) >= n_prompts:
            break
    if len(prompts) < n_prompts:
        raise ValueError(
            f"pool {PROMPTS_POOL} yielded only {len(prompts)}/{n_prompts} prompts "
            f">= {min_tokens} tokens; enlarge the pool"
        )
    tmp = f"{path}.tmp.{os.getpid()}"
    json.dump(prompts, open(tmp, "w"))
    os.replace(tmp, path)
    return prompts


def main() -> None:
    prompts = build_prompt_list(PROMPTS_JSON, N_PROMPTS, SEQ_LEN)
    shard_prompts = prompts[SHARD::N_SHARDS]
    ckpt = f"{OUT_DIR}/offset_fit_shard{SHARD}of{N_SHARDS}.pt"
    print(
        f"[offset-fit] shard {SHARD}/{N_SHARDS}: {len(shard_prompts)} prompts | "
        f"L{SRC}->L{TGT} offsets={N_OFFSETS} comb={COMB} seq={SEQ_LEN} "
        f"dim_batch={DIM_BATCH} ckpt={ckpt}",
        flush=True,
    )
    # Multiple source layers share the SAME backward passes (the estimator takes
    # grads w.r.t. every source at once), so fitting L62->L63 and L42->L63
    # together costs no more than fitting one of them.

    tok = AutoTokenizer.from_pretrained(BASE)
    hf = (
        AutoModelForCausalLM.from_pretrained(
            BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )
    model = from_hf(hf, tok)
    print(
        f"[offset-fit] LensModel n_layers={model.n_layers} d_model={model.d_model}",
        flush=True,
    )

    fit_offsets(
        model,
        shard_prompts,
        source_layers=SRC,
        target_layer=TGT,
        n_offsets=N_OFFSETS,
        comb_spacing=COMB,
        dim_batch=DIM_BATCH,
        max_seq_len=SEQ_LEN,
        checkpoint_path=ckpt,
        checkpoint_every=2,
        resume=True,
    )
    print(f"=== OFFSET FIT SHARD {SHARD} DONE ===", flush=True)


if __name__ == "__main__":
    main()
