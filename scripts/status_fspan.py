import re, os
base = "/workspace/cnla/skip-lens/logs"
print("Lspan  step        FVE(norm)   cos     FVE(corr)")
for L in ["4", "8", "12", "16"]:
    p = f"{base}/fspan_{L}.log"
    def last(pat, g=1):
        v = None
        if os.path.exists(p):
            for line in open(p, errors="ignore"):
                m = re.search(pat, line)
                if m: v = m.group(g)
        return v
    step = last(r"step (\d+)")
    bl = last(r"baseline \(paper def\) = ([\d.]+)")
    hmse = last(r"heldout@\d+\] mse ([\d.]+)")
    hfve = last(r"heldout@\d+\] mse [\d.]+ \| FVE ([\d.]+)%")
    cos = corr = None
    if hmse:
        cos = 1 - float(hmse) / 2.0
        if bl: corr = 1 - (1 - cos * cos) / float(bl)
    print(f"{L:>5} {str(step or '?'):>5}/1500   {(hfve or '-'):>6}%   "
          f"{('%.3f'%cos) if cos is not None else '-':>6}   "
          f"{('%.1f%%'%(100*corr)) if corr is not None else '-':>8}")
