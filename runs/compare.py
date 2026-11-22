import json, glob, os, sys
R = os.path.dirname(os.path.abspath(__file__))
S = ["baseline", "double_buffer", "weight_stationary", "auto"]
M = ["basic", "alexnet_small", "gpt2_attn", "gpt2_block"]
def load(m, s):
    d = sorted(glob.glob(f"{R}/{m}__sched-{s}__*"))
    if not d: return None, None
    p = d[-1] + "/summary.json"
    return (json.load(open(p)), os.path.basename(d[-1])) if os.path.exists(p) else (None, None)
keys = ["final_cosine","bytes_loaded","bytes_written","dyn_retired_sdma","dyn_retired_gemm_systolic","dyn_packets","sched_packets","retired_instructions"]
print("| Model | Schedule | run id | cosine | bytes loaded | bytes written | SDMA | gemm+lw.vi | dyn pkts | sched pkts | retired |")
print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for m in M:
    b, _ = load(m, "baseline")
    for s in S:
        x, rid = load(m, s)
        if x is None: print(f"| {m} | {s} | missing |"); continue
        cells = []
        for k in keys:
            v = x[k]
            if k == "final_cosine": cells.append(f"{v:.7f}"); continue
            c = f"{v:,}"
            if b and s != "baseline" and b[k]:
                c += f" ({(v-b[k])/b[k]*100:+.1f}%)"
            cells.append(c)
        print(f"| {m} | {s} | {rid} | " + " | ".join(cells) + " |")
print()
print("Instruction mix (retired): | Model | Schedule | SDMA | branch | scalar ALU | scalar mem | vector mem | vector alu | gemm+lw.vi | move |")
for m in M:
    for s in S:
        x, _ = load(m, s)
        if x is None: continue
        print(f"| {m} | {s} | " + " | ".join(f"{x['dyn_retired_'+k]:,}" for k in ["sdma","branch_control","scalar_alu","scalar_mem","vector_mem","vector_alu","gemm_systolic","move_convert"]) + " |")
print()
print("Per GEMM layer: | Model | node | " + " | ".join(f"{s} ld/wr, sdma, gemm+lw, dyn, sched, cos" for s in S) + " |")
for m in M:
    xs = {s: load(m, s)[0] for s in S}
    if any(v is None for v in xs.values()): continue
    for i, L in enumerate(xs["baseline"]["layers"]):
        if not L.get("dyn_retired_gemm_systolic"): continue
        row = []
        for s in S:
            l = xs[s]["layers"][i]; assert l["node"] == L["node"]
            row.append(f"{l['bytes_loaded']:,}/{l['bytes_written']:,}, {l['dyn_retired_sdma']}, {l['dyn_retired_gemm_systolic']}, {l['dyn_packets']:,}, {l['sched_packets']}, {l['cos_sim']:.7f}")
        print(f"| {m} | {L['node']} | " + " | ".join(row) + " |")
# non-GEMM layers unchanged?
for m in M:
    xs = {s: load(m, s)[0] for s in S}
    if any(v is None for v in xs.values()): continue
    for i, L in enumerate(xs["baseline"]["layers"]):
        if L.get("dyn_retired_gemm_systolic"): continue
        for s in S[1:]:
            l = xs[s]["layers"][i]
            for k in ["dyn_packets","bytes_loaded","cos_sim"]:
                if l[k] != L[k]: print("NONGEMM CHANGED", m, L["node"], s, k, L[k], l[k])
