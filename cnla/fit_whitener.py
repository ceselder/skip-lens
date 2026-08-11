"""Fit the reward whitener from the dataset's L62 activations.

Per-dim mean/std over the L2-NORMED activations (the reward reconstructs the normed
vector), saved for cnla.reward.Whitener. The CNLA-RL loads this so FVE is measured in
a fixed whitened space across steps (not re-fit per batch)."""
import argparse
import numpy as np
import torch
import pyarrow.parquet as pq
from cnla.reward import Whitener


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="cnla_L62.parquet (activation_vector column)")
    ap.add_argument("--out", required=True, help="whitener .pt (mu,std)")
    ap.add_argument("--max-rows", type=int, default=20000)
    args = ap.parse_args()

    t = pq.read_table(args.data, columns=["activation_vector"])
    col = t.column("activation_vector").combine_chunks()
    H = (col.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
         .reshape(len(col), -1))[: args.max_rows]
    H = torch.from_numpy(H)
    Hn = torch.nn.functional.normalize(H, dim=-1)     # normed targets
    wh = Whitener.fit(Hn)
    torch.save(wh.state_dict(), args.out)
    print(f"fit whitener on {H.shape[0]} rows, d={H.shape[1]} -> {args.out} "
          f"(mu|={wh.mu.abs().mean():.4g}, std~{wh.std.mean():.4g})")


if __name__ == "__main__":
    main()
