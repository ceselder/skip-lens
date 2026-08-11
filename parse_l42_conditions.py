"""Print the 6-condition L42 workspace table from the judged.json files."""
import json
O = "/workspace/cnla/results/l42_mismatch_eval"
conds = [
    ("l42matched_raw", "L42-matched  · feed RAW L42"),
    ("l42matched_jac", "L42-matched  · feed J(L42) "),
    ("l42matched_R",   "L42-matched  · feed R(L42) "),
    ("l62mismatch_raw", "L62-mismatch · feed RAW L42"),
    ("l62mismatch_jac", "L62-mismatch · feed J(L42) "),
    ("l62mismatch_R",   "L62-mismatch · feed R(L42) "),
]
print(f"{'condition (all fed @ L42)':32s} {'agree_JLENS(workspace)':>22s} {'agree_ANSWER(surface)':>22s}   n")
for k, lab in conds:
    try:
        d = json.load(open(f"{O}/{k}_judged.json"))["by_fed_layer"]["42"]
        aj = f"{d['agree_jlens']}±{d['agree_jlens_sem']}"
        aa = f"{d['agree_answer']}±{d['agree_answer_sem']}"
        print(f"{lab:32s} {aj:>22s} {aa:>22s}   {d['n']}")
    except Exception as e:
        print(f"{lab:32s}  (missing: {e})")
