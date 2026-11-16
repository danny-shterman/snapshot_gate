"""Merge Gate 0 result files (later files override arms in earlier ones) and re-apply the decision rule.
Usage: python merge_results.py out.json run1.json run2.json ..."""
import json, sys
import snapshot_gate as sg

out, files = sys.argv[1], sys.argv[2:]
results, args = {}, {}
for f in files:
    d = json.load(open(f))
    args[f] = d["args"]
    for C, arms in d["results"].items():
        results.setdefault(int(C), {}).update(arms)
decisions = {C: sg.decide(results, C) for C in results}
json.dump(dict(sources=args, results=results, decisions=decisions), open(out, "w"), indent=1)
print(sg.report(results, decisions))
