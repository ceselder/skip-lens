"""My self-fit R variants vs the OFFICIAL camilablank R/J, per source layer."""
import os
import numpy as np
import numpy.linalg as la

RELP = "/workspace/cnla/results/rlens_relp"          # my corrected: 3-rule + AH-rule + GDN rule
APPROX = "/workspace/cnla/results/rlens_full"        # my approximate: detach-one-branch
ROFF = "/workspace/cnla/results/rlens_official"      # OFFICIAL R (3-rule minimal: LN+SiLU+SwiGLU)
JOFF = "/workspace/cnla/results/jlens_official_ws"   # OFFICIAL J
LAYERS = [10, 18, 26, 34, 42, 48, 55]
SQD = 5120 ** 0.5


def ld(dp, l, pfx):
    p = f"{dp}/{pfx}_L{l}_to_L62.npy"
    return np.load(p) if os.path.exists(p) else None


def cos(A, B):
    if A is None or B is None:
        return None
    return round(float((A.ravel() @ B.ravel()) / (la.norm(A) * la.norm(B) + 1e-9)), 3)


print(f"{'L':>3} | {'cos(mineAHGDN,offR)':>19} | {'cos(myApprox,offR)':>18} | {'cos(offR,offJ)':>14} | {'||offR||/vd':>11}")
for l in LAYERS:
    Rrelp, Rapx, Roff, Joff = ld(RELP, l, "R"), ld(APPROX, l, "R"), ld(ROFF, l, "R"), ld(JOFF, l, "J")
    no = f"{la.norm(Roff)/SQD:.3f}" if Roff is not None else "n/a"
    print(f"{l:>3} | {str(cos(Rrelp, Roff)):>19} | {str(cos(Rapx, Roff)):>18} | {str(cos(Roff, Joff)):>14} | {no:>11}")
print("\ncos(mine,offR)~1 => my extra AH+GDN rules barely change R vs the official 3-rule build.")
print("cos(offR,offJ) shows how much R differs from J at each layer (lower at early layers = R's edge).")
