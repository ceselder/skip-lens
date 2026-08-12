import re, os
base = "/workspace/cnla/skip-lens/logs"

def last_num(pat, path):
    n = None
    if os.path.exists(path):
        for line in open(path, errors="ignore"):
            m = re.findall(pat, line)
            if m:
                n = m[-1]
    return n

def last_str(pat, path):
    s = None
    if os.path.exists(path):
        for line in open(path, errors="ignore"):
            m = re.search(pat, line)
            if m:
                s = m.group(0)
    return s

arms = [("AR", "ar_lr1e-5", 24249), ("AR", "ar_lr3e-5", 24249),
        ("AR", "ar_lr1e-4", 24249), ("AR", "ar_lr3e-4", 24249),
        ("FL", "av_lr3e-5", 12124), ("FL", "av_lr1e-4", 12124), ("FL", "av_lr3e-4", 12124),
        ("PL", "pastlens", 12124)]
for kind, arm, tot in arms:
    p = f"{base}/{'pastlens_train' if arm=='pastlens' else 'sweep_'+arm}.log"
    st = last_num(r"step (\d+)", p)
    held = last_str(r"heldout@\d+.*?(?:FVE [\d.]+%|val_loss [\d.]+)", p)
    pct = f"{100*int(st)/tot:.0f}%" if st else "?"
    print(f"{kind:2} {arm:10} step {st or '?'}/{tot} ({pct}) | {held or '(no heldout yet)'}")
