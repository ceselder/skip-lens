"""Compositional-NLA RL reward + per-bullet-span advantages.

Given a batch of policy rollouts (each a "* bullet" completion), the injected target
activations, a FROZEN AR critic, and a fitted whitener, produce PER-TOKEN advantages:
each bullet's response-token span carries that bullet's leave-one-out FVE marginal
(cnla.reward.loo_fve_rewards), GRPO-baselined over its prompt group. The AR is a
black-box reward — no gradient flows through it (it is never co-trained).

Design note — text↔token consistency: bullets are derived FROM the response token
stream (spans between "*"-containing tokens), and each bullet's text is the decode of
its own span. So the text the AR scores is exactly what those tokens represent — no
re-tokenization drift between the reward and the advantage assignment.
"""
from __future__ import annotations

import torch

from nla.utils.critic import critic_predict
from cnla.reward import loo_fve_rewards, Whitener

MAX_BULLETS = 4


def bullet_spans(resp_ids, tokenizer, max_bullets=MAX_BULLETS):
    """Split a response token-id list into ≤max_bullets (start,end,text) spans.

    A new bullet begins at each token whose decoded piece contains '*'. Tokens
    before the first '*' belong to no bullet (masked in the advantage). The text of
    each bullet is the decode of its own token span (with the leading '*' stripped),
    guaranteeing the AR scores exactly what the tokens represent.
    """
    starts = [i for i, t in enumerate(resp_ids) if "*" in tokenizer.decode([int(t)])]
    starts = starts[:max_bullets]
    spans = []
    for b, s in enumerate(starts):
        e = starts[b + 1] if b + 1 < len(starts) else len(resp_ids)
        text = tokenizer.decode(resp_ids[s:e]).strip()
        text = text.lstrip("*").strip()
        spans.append((s, e, text))
    return spans


def _ar_score(critic, tokenizer, texts, ar_template, mse_scale_f, device,
              d_model, max_len=1024, batch_size=64):
    """Frozen-AR reconstruction vector for each bullet text. Returns (preds [N,d],
    valid [N] bool). Empty/oversized texts → zero vector, valid=False."""
    n = len(texts)
    ids_list = [None] * n
    for i, t in enumerate(texts):
        if not t:
            continue
        ids = tokenizer.encode(ar_template.format(explanation=t), add_special_tokens=False)
        if 0 < len(ids) <= max_len:
            ids_list[i] = ids
    order = [i for i in range(n) if ids_list[i] is not None]
    preds = torch.zeros(n, d_model, device=device)
    valid = torch.zeros(n, dtype=torch.bool, device=device)
    pad_id = tokenizer.eos_token_id
    for cs in range(0, len(order), batch_size):
        chunk = order[cs:cs + batch_size]
        maxlen = max(len(ids_list[i]) for i in chunk)
        bx = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(chunk), maxlen), dtype=torch.long, device=device)
        for r, i in enumerate(chunk):
            L = len(ids_list[i])
            bx[r, :L] = torch.tensor(ids_list[i], dtype=torch.long, device=device)
            attn[r, :L] = 1
        with torch.no_grad():
            p = critic_predict(critic, bx, attn, mse_scale_f)   # [len(chunk), d]
        for r, i in enumerate(chunk):
            preds[i] = p[r].to(device)
            valid[i] = True
    return preds, valid


def compute_cnla_advantages(
    *, full_ids, prompt_lens, activations, groups, response_texts,
    critic, tokenizer, whitener: Whitener, ar_template, mse_scale_f, device,
    d_model, batch_prompts, adv_clip=10.0,
):
    """Per-token advantages for every rollout + batch metrics.

    Args (all lists indexed by rollout i, length B):
      full_ids[i]     : LongTensor of the full prompt+response token ids
      prompt_lens[i]  : int, response starts at this index
      activations[i]  : [d] the injected target activation for rollout i
      groups[i]       : int prompt-group id (rollouts of the same activation share it)
      response_texts  : decoded response (unused for spans; kept for logging)
    Returns:
      adv_tokens : list[Tensor] — per rollout, a [n_resp] advantage aligned to
                   full_ids[i][p_len:] (0 outside any bullet span)
      per_bullet : dict of metrics (fve_full, mean marginal, uniqueness, n_bullets…)
    """
    B = len(full_ids)
    # 1. spans + bullet texts from the response token stream
    all_spans = []                       # per rollout: list[(s,e,text)]
    flat_texts, flat_where = [], []      # (rollout_i, bullet_b) for each bullet text
    for i in range(B):
        resp = full_ids[i][prompt_lens[i]:].tolist()
        spans = bullet_spans(resp, tokenizer)
        all_spans.append(spans)
        for b, (_s, _e, txt) in enumerate(spans):
            flat_texts.append(txt)
            flat_where.append((i, b))

    # 2. frozen-AR vector per bullet
    if flat_texts:
        preds, valid = _ar_score(critic, tokenizer, flat_texts, ar_template,
                                 mse_scale_f, device, d_model)
    else:
        preds = torch.zeros(0, d_model, device=device)
        valid = torch.zeros(0, dtype=torch.bool, device=device)

    vecs = torch.zeros(B, MAX_BULLETS, d_model, device=device)
    vmask = torch.zeros(B, MAX_BULLETS, dtype=torch.bool, device=device)
    for k, (i, b) in enumerate(flat_where):
        vecs[i, b] = preds[k]
        vmask[i, b] = bool(valid[k])

    # 3. leave-one-out FVE per bullet
    target = torch.stack([torch.as_tensor(a, dtype=torch.float32, device=device)
                          for a in activations], dim=0)         # [B, d]
    out = loo_fve_rewards(vecs, target, whitener, valid=vmask)
    r = out["r"]                                                # [B, MAX_BULLETS]
    fve_full = out["fve_full"]                                  # [B]

    # 4. GRPO baseline over each prompt group's VALID bullets
    groups_t = torch.tensor(groups, dtype=torch.long, device=device)
    bull_adv = torch.zeros(B, MAX_BULLETS, device=device)
    for gi in range(batch_prompts):
        sel = (groups_t == gi).unsqueeze(1) & vmask               # [B, K] bullets in this group
        if sel.sum() == 0:
            continue
        vals = r[sel]
        mu = vals.mean()
        sd = vals.std() if vals.numel() > 1 else torch.tensor(1.0, device=device)
        bull_adv[sel] = ((r[sel] - mu) / (sd + 1e-6)).clamp(-adv_clip, adv_clip)

    # 5. scatter bullet advantages onto response-token spans
    adv_tokens = []
    for i in range(B):
        n_resp = full_ids[i].numel() - prompt_lens[i]
        at = torch.zeros(n_resp, device=device)
        for b, (s, e, _t) in enumerate(all_spans[i]):
            if vmask[i, b]:
                at[s:e] = bull_adv[i, b]
        adv_tokens.append(at)

    n_bull = vmask.sum(1).float()
    # per-rollout mean bullet advantage (for scalar logging)
    adv_scalar = torch.stack([
        bull_adv[i][vmask[i]].mean() if bool(vmask[i].any()) else torch.zeros((), device=device)
        for i in range(B)
    ])
    # within-rollout marginal spread = uniqueness proxy (high → bullets differ)
    uniq = []
    for i in range(B):
        rv = r[i][vmask[i]]
        uniq.append(rv.std().item() if rv.numel() > 1 else 0.0)
    metrics = {
        "cnla/fve_full_mean": fve_full.mean().item(),
        "cnla/marginal_mean": r[vmask].mean().item() if vmask.any() else 0.0,
        "cnla/n_bullets_mean": n_bull.mean().item(),
        "cnla/frac_4bullets": (n_bull == MAX_BULLETS).float().mean().item(),
        "cnla/uniqueness_mean": float(sum(uniq) / max(1, len(uniq))),
    }
    info = {
        "fve_per": fve_full,        # [B] composite reconstruction FVE per rollout
        "adv_scalar": adv_scalar,   # [B] mean bullet advantage per rollout (logging)
        "nbull": vmask.sum(1),      # [B] valid bullet count
        "metrics": metrics,
    }
    return adv_tokens, info


# --------------------------------------------------------------------------- #
# Smoke test (no model): parse/span logic + advantage assembly with fake vecs
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import types

    class FakeTok:
        # each char is a token; "*" tokens start bullets
        def decode(self, ids):
            return "".join(chr(i) for i in ids)
        def encode(self, s, add_special_tokens=False):
            return [ord(c) for c in s]
        eos_token_id = 0

    tok = FakeTok()
    resp = "* aaa* bbb* ccc"          # 3 bullets
    ids = [ord(c) for c in resp]
    spans = bullet_spans(ids, tok)
    print("spans:", [(s, e, t) for s, e, t in spans])
    assert len(spans) == 3 and spans[0][2] == "aaa", spans

    # advantage assembly with a fake AR (identity-ish) via loo_fve directly
    torch.manual_seed(0)
    d = 64
    wh = Whitener.fit(torch.nn.functional.normalize(torch.randn(500, d), dim=-1))
    B = 4
    full_ids = [torch.tensor([9, 9] + [ord(c) for c in resp]) for _ in range(B)]  # 2 prompt toks
    plens = [2] * B
    acts = [torch.randn(d) for _ in range(B)]
    groups = [0, 0, 1, 1]

    # monkeypatch _ar_score (this module's global) to return random vecs
    def fake_ar(critic, tokenizer, texts, *a, **k):
        n = len(texts)
        return torch.randn(n, d), torch.ones(n, dtype=torch.bool)
    globals()["_ar_score"] = fake_ar
    adv, info = compute_cnla_advantages(
        full_ids=full_ids, prompt_lens=plens, activations=acts, groups=groups,
        response_texts=[resp] * B, critic=None, tokenizer=tok, whitener=wh,
        ar_template="{explanation}", mse_scale_f=1.0, device="cpu",
        d_model=d, batch_prompts=2,
    )
    print("adv[0] shape:", adv[0].shape, "n_resp:", full_ids[0].numel() - plens[0])
    print("adv_scalar:", info["adv_scalar"].tolist(), "nbull:", info["nbull"].tolist())
    print("metrics:", {k: round(v, 4) for k, v in info["metrics"].items()})
    assert adv[0].numel() == full_ids[0].numel() - plens[0]
    print("OK: spans + per-token advantage assembly wired correctly")
