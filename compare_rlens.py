"""Compare corrected RelP-rules R vs approximate-patches R vs J-lens, per layer."""
import os
import numpy as np
import numpy.linalg as la

RELP = "/workspace/cnla/results/rlens_relp"     # corrected (Identity+Half-rule)
APPROX = "/workspace/cnla/results/rlens_full"   # approximate (detach-one-branch)
JD = "/workspace/cnla/results/jlens_snap"       # J-lens
LAYERS = [10, 18, 26, 34, 42, 48, 55]
SQD = 5120 ** 0.5


def load(dp, l, pfx):
    p = f"{dp}/{pfx}_L{l}_to_L62.npy"
    return np.load(p) if os.path.exists(p) else None


def cos(A, B):
    if A is None or B is None:
        return None
    return round(float((A.ravel() @ B.ravel()) / (la.norm(A) * la.norm(B) + 1e-9)), 3)


print(f"{'L':>3} | {'||R_relp||/vd':>12} | {'cos(relp,approx)':>16} | {'cos(relp,J)':>11} | {'||R_apx||/vd':>12}")
for l in LAYERS:
    Rr, Ra, J = load(RELP, l, "R"), load(APPROX, l, "R"), load(JD, l, "J")
    nr = f"{la.norm(Rr)/SQD:.3f}" if Rr is not None else "n/a"
    na = f"{la.norm(Ra)/SQD:.3f}" if Ra is not None else "n/a"
    print(f"{l:>3} | {nr:>12} | {str(cos(Rr, Ra)):>16} | {str(cos(Rr, J)):>11} | {na:>12}")
print("\n(cos(relp,approx) << 1  =>  the Half-rule / Identity-rule materially changed R, as expected)")
