#!/bin/bash
# Arm D — skip-lens proper, multi-token.
#
# Training inputs = the REAL penultimate states h62[p..p+7] (forward-only
# collection, no autodiff). Test inputs = Jbar^(d) @ h42[p], which the J-bar
# audit measured at cos +0.50 with the true h62 — i.e. an ESTIMATE of exactly
# what the decoder was trained to read (arm A's JVP transports were only 0.156
# aligned with the test-time vector, a category mismatch this arm removes).
#
# Chain: wait for GPUs -> collect real penultimate states (4 shards)
#     -> fit the affine anchors b_d = E[h62[p+d] - Jbar^(d) h42[p]]
#     -> finalize -> train arm D -> ready for the same judged eval.
set -u
cd /workspace/skip-lens
source /workspace/.keys.env
LOG=/workspace/logs/orchestrator_armD.log
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

NEED_MB=${NEED_MB:-70000}          # forward-only: ~57GB model + small activations
OUT=/workspace/data/spans_pen8
FINAL_T=/workspace/data/final/pen8_multislot_train.parquet
FINAL_V=/workspace/data/final/pen8_multislot_val.parquet
mkdir -p "$OUT" logs

free_mb(){ echo $(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$1") - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1") )); }

log "arm D: waiting for the frozen pass-2 collection to release GPUs"
until [ -f /workspace/data/spans_frozen/shard_3_jvp.parquet ]; do sleep 120; done
sleep 60

# ---- collect real penultimate future states: one shard per free GPU ----
for i in 0 1 2 3; do
  [ -f "$OUT/shard_${i}_jvp.parquet" ] && { log "shard $i done"; continue; }
  g=""
  while [ -z "$g" ]; do
    for c in 0 1 2 3; do
      [ "$(free_mb $c)" -ge "$NEED_MB" ] && { g=$c; break; }
    done
    [ -z "$g" ] && sleep 120
  done
  ( source /workspace/venv/bin/activate
    CUDA_VISIBLE_DEVICES=$g PYTHONPATH=/workspace/skip-lens setsid \
    python pretrain/collect_pen_states.py \
      --in-shards "/workspace/data/spans_raw/shard_${i}.parquet" \
      --out-dir "$OUT" --batch-size 32 \
      > "logs/pen8_shard${i}.log" 2>&1 < /dev/null & )
  log "launched pen-state shard $i on gpu $g"
  sleep 30
done

log "waiting for all 4 pen-state shards"
while true; do
  n=0; for i in 0 1 2 3; do [ -f "$OUT/shard_${i}_jvp.parquet" ] && n=$((n+1)); done
  [ "$n" -eq 4 ] && break
  sleep 120
done
log "all real penultimate states collected"

# ---- affine anchors b_d, and the train/test alignment they buy ----
source /workspace/venv/bin/activate
PYTHONPATH=/workspace/skip-lens python - <<'PY' 2>&1 | tee -a "$LOG"
import glob, json
import numpy as np, pyarrow.parquet as pq, torch
JD = "/workspace/results/offset_jlens"
K = 8
Jb = [torch.from_numpy(np.load(f"{JD}/Jbar_L42_to_L62_off{d}.npy")).float().cuda()
      for d in range(K)]
acc = torch.zeros(K, 5120, dtype=torch.float64).cuda(); n = 0
cos_raw = torch.zeros(K, dtype=torch.float64); cos_aff = torch.zeros(K, dtype=torch.float64)
rows_seen = 0
H, S = [], []
for f in sorted(glob.glob("/workspace/data/spans_pen8/shard_[0-3]_jvp.parquet")):
    t = pq.read_table(f, columns=["activation_vector", "transported_vectors"])
    for b in range(0, t.num_rows, 4096):
        sl = t.slice(b, 4096).to_pylist()
        h = torch.tensor(np.array([r["activation_vector"] for r in sl], dtype=np.float32)).cuda()
        s = torch.tensor(np.array([np.frombuffer(r["transported_vectors"], np.float16)
                                   .reshape(16, -1)[:K] for r in sl], dtype=np.float32)).cuda()
        for d in range(K):
            acc[d] += (s[:, d] - (Jb[d] @ h.T).T).sum(0).double()
        n += h.shape[0]
        if rows_seen < 4096:
            H.append(h.cpu()); S.append(s.cpu()); rows_seen += h.shape[0]
b = (acc / n).float()
np.save(f"{JD}/affine_b_L42_to_L62.npy", b.cpu().numpy())
print(f"[armD] fitted affine anchors on {n} rows; ||b_d|| =",
      [round(float(b[d].norm()), 1) for d in range(K)], flush=True)
Hc, Sc = torch.cat(H).cuda(), torch.cat(S).cuda()
for d in range(K):
    est = (Jb[d] @ Hc.T).T
    cos_raw[d] = torch.nn.functional.cosine_similarity(est, Sc[:, d], dim=-1).mean().double()
    cos_aff[d] = torch.nn.functional.cosine_similarity(est + b[d], Sc[:, d], dim=-1).mean().double()
print("[armD] cos(Jbar h42, real h62) per horizon:      ",
      " ".join(f"{float(x):+.3f}" for x in cos_raw), flush=True)
print("[armD] cos(Jbar h42 + b, real h62) per horizon:  ",
      " ".join(f"{float(x):+.3f}" for x in cos_aff), flush=True)
json.dump({"cos_raw": [float(x) for x in cos_raw],
           "cos_affine": [float(x) for x in cos_aff],
           "b_norms": [float(b[d].norm()) for d in range(K)], "n_rows": n},
          open("/workspace/results/multislot_eval/armD_alignment.json", "w"), indent=2)
PY

# ---- finalize + train (same recipe as arm A; only the slot content differs) ----
if [ ! -f "$FINAL_T" ]; then
  PYTHONPATH=/workspace/skip-lens python pretrain/finalize_jvp_spans.py \
    --shards-glob "$OUT/shard_[0-3]_jvp.parquet" \
    --out-train "$FINAL_T" --out-val "$FINAL_V" --k-slots 8 2>&1 | tee -a "$LOG"
fi
NSTEPS=$(python -c "import pyarrow.parquet as pq;print(max(200,pq.ParquetFile('$FINAL_T').metadata.num_rows//64))")
g=""; while [ -z "$g" ]; do for c in 0 1 2 3; do [ "$(free_mb $c)" -ge 70000 ] && { g=$c; break; }; done; [ -z "$g" ] && sleep 120; done
log "training arm D for $NSTEPS steps on gpu $g"
CUDA_VISIBLE_DEVICES=$g PYTHONPATH=/workspace/skip-lens setsid \
  python -m nla.train_sft --mode av --base-ckpt Qwen/Qwen3.6-27B \
  --parquet "$FINAL_T" --sidecar "$FINAL_T" --n-slots 8 \
  --save-dir ckpts/multislot_armD_pen8 \
  --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all \
  --num-steps "$NSTEPS" --batch-size 16 --gradient-accumulation-steps 4 \
  --save-every 500 --heldout-parquet "$FINAL_V" --heldout-rows 800 \
  --heldout-every 250 --sample-every 500 --n-samples 4 \
  --sample-max-new-tokens 48 \
  --wandb-project skiplens-multislot --wandb-name armD_pen8 \
  > logs/train_armD.log 2>&1 < /dev/null &
log "arm D training launched — orchestrator exiting"
