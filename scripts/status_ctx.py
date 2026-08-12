import re, os
base = "/workspace/cnla/skip-lens/logs"

def last(pat, path, g):
    v = None
    if os.path.exists(path):
        for line in open(path, errors="ignore"):
            m = re.search(pat, line)
            if m:
                v = m.group(g)
    return v

print("ctx  step        FVE(norm-constr)   cos     FVE(corrected)   baseline")
for N in ["0", "32", "128", "full"]:
    p = f"{base}/ctx_{N}.log"
    step = last(r"step (\d+)", p, 1)
    bl = last(r"baseline \(paper def\) = ([\d.]+)", p, 1)
    hmse = last(r"heldout@\d+\] mse ([\d.]+)", p, 1)
    hfve = last(r"heldout@\d+\] mse [\d.]+ \| FVE ([\d.]+)%", p, 1)
    cos = corr = None
    if hmse:
        cos = 1 - float(hmse) / 2.0            # loss = 2(1-cos) since both normed to sqrt(d)
        if bl:
            corr = 1 - (1 - cos * cos) / float(bl)   # standard FVE with optimal magnitude
    print(f"{N:4} {str(step or '?'):>5}/1500   {(hfve or '-'):>6}%           "
          f"{('%.3f'%cos) if cos is not None else '-':>6}   "
          f"{('%.1f%%'%(100*corr)) if corr is not None else '-':>8}        {bl or '?'}")
