import json, glob, re
D = "/workspace/cnla/results/fedlayer_scaling"

# debug: show the L42 structure of one file so we key correctly
one = json.load(open(D + "/scores_fl_0007579.json"))
bfl = one.get("by_fed_layer", {})
v42 = bfl.get("42") or bfl.get(42) or {}
print("[debug] by_fed_layer keys:", list(bfl.keys()))
print("[debug] L42 entry:", json.dumps(v42)[:400])


def L42(d):
    bfl = d.get("by_fed_layer", {})
    v = bfl.get("42") or bfl.get(42) or {}
    def g(*keys):
        for k in keys:
            x = v.get(k)
            if isinstance(x, dict):
                x = x.get("mean", x.get("avg"))
            if x is not None:
                return x
        return None
    return g("agree_jlens", "jlens", "workspace"), g("agree_answer", "answer")


out = {"futurelens": {}, "cnla": {}, "batch_per_step": 64}
for f in sorted(glob.glob(D + "/scores_fl_*.json")):
    it = int(re.search(r"scores_fl_(\d+)", f).group(1))
    aj, aa = L42(json.load(open(f)))
    out["futurelens"][str(it)] = {"agree_jlens": aj, "agree_answer": aa}
for f in sorted(glob.glob(D + "/scores_cnla*.json")):
    m = re.search(r"scores_cnla(\w*?)_(\d+)", f)
    key = ("lh" if "lh" in (m.group(1) or "") else "") + m.group(2)
    aj, aa = L42(json.load(open(f)))
    out["cnla"][key] = {"agree_jlens": aj, "agree_answer": aa}

print("=== FUTURELENS (examples = iter*64) ===")
for it in sorted(out["futurelens"], key=int):
    r = out["futurelens"][it]
    print(f"  {int(it)*64//1000:>3}k  agree_jlens={r['agree_jlens']}  agree_answer={r['agree_answer']}")
print("=== CNLA ===")
for k in sorted(out["cnla"]):
    r = out["cnla"][k]
    print(f"  {k}: agree_jlens={r['agree_jlens']} agree_answer={r['agree_answer']}")
json.dump(out, open("/workspace/cnla/results/workspace_agreement.json", "w"))
print("SAVED /workspace/cnla/results/workspace_agreement.json")
