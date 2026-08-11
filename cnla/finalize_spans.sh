#!/bin/bash
# After ON-POLICY collection: build the AV (activation->span) and AR (span->activation,
# via the <summary> critic template) parquets from the MODEL'S OWN generations
# (rollout_token_ids[0]), sliced to k~U[4,16].
set -eu
cd /workspace/cnla/skip-lens
source /workspace/cnla_venv/bin/activate
source /workspace/.keys.env 2>/dev/null
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/cnla/skip-lens

# 1) AV parquet: response = decode(rollout_token_ids[0][:k]), k~U[4,16] (ON-POLICY)
python pretrain/finalize_span_data.py \
  --shards-glob "data/spans_onpolicy/shard_*.parquet" \
  --meta data/spans_onpolicy/shard_0.parquet.meta.json \
  --out-train data/spans_onpolicy/av_train.parquet \
  --out-val   data/spans_onpolicy/av_val.parquet \
  --span-source rollout --span-min 4 --span-max 16 --seed 0

# 2) AR parquet: wrap response into "Summary of the following text: <text>{span}</text> <summary>"
AR_SRC=data/spans_onpolicy/av_train.parquet AR_OUT=data/spans_onpolicy/ar_train.parquet \
  python scripts/build_ar_big.py
AR_SRC=data/spans_onpolicy/av_val.parquet   AR_OUT=data/spans_onpolicy/ar_heldout.parquet \
  python scripts/build_ar_big.py
echo FINALIZE_SPANS_DONE
