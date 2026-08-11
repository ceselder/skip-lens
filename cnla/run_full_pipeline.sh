#!/bin/bash
# Autonomous on-box pipeline: wait for on-policy collection -> finalize (AV+AR) -> launch LR sweep.
# Detached; logs milestones to logs/full_pipeline.log so the session can grep progress.
set -u
cd /workspace/cnla/skip-lens
LOG=logs/full_pipeline.log
echo "PIPELINE_START $(date)" >> "$LOG"

# 1) wait for all 7 collectors to finish
while [ "$(pgrep -f '[c]ollect_ao_data' | wc -l)" -gt 0 ]; do sleep 120; done
echo "COLLECTION_COMPLETE $(date)" >> "$LOG"
ls -la data/spans_onpolicy/shard_*.parquet >> "$LOG" 2>&1

# 2) finalize -> AV + AR parquets (on-policy spans)
bash cnla/finalize_spans.sh >> "$LOG" 2>&1
if ! grep -q FINALIZE_SPANS_DONE "$LOG"; then
  echo "PIPELINE_ABORT: finalize failed $(date)" >> "$LOG"; exit 1
fi
echo "FINALIZE_COMPLETE $(date)" >> "$LOG"

# 3) launch the 7-GPU LR sweep (this backgrounds the training runs and returns)
bash cnla/launch_span_sweep.sh >> "$LOG" 2>&1
echo "SWEEP_LAUNCHED $(date)" >> "$LOG"
