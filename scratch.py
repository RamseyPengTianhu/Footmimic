import json, numpy as np
from collections import Counter

with open("models/latent_v2/eval/vq_code_semantics.json") as f:
    data = json.load(f)

num_codes = data["num_codes"]
eps = data["episodes"]

all_post = []
all_prior = []
all_ref = []
all_geo = []
for ep in eps:
    # Filter out length 1 episodes which are immediate resets
    if ep["length"] <= 1: continue
    all_post.extend(ep["post_codes"])
    all_prior.extend(ep["prior_codes"])
    all_ref.extend(ep["ref_phases"])
    all_geo.extend(ep["geo_phases"])

phase_names = {0: "approach", 1: "prestrike", 2: "strike", 3: "follow"}
phase_ids = [0, 1, 2, 3]

def print_table(codes, phases, title):
    counts = np.zeros((num_codes, len(phase_ids)), dtype=int)
    for c, p in zip(codes, phases):
        if p in phase_ids:
            counts[c, phase_ids.index(p)] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    pcts = np.where(row_sums > 0, counts / row_sums * 100, 0)
    print(f"\n{title}")
    hdr = f"{'Code':>6s} " + " ".join(f"{phase_names[p]:>10s}" for p in phase_ids) + f" {'Total':>8s}"
    print(hdr)
    for k in range(num_codes):
        row = f"{k:>6d} " + " ".join(f"{pcts[k,j]:>9.1f}%" for j in range(4)) + f" {row_sums[k,0]:>8d}"
        if row_sums[k,0] > 50:
            print(row)

print_table(all_post, all_ref, "POSTERIOR Code x Ref Phase (Only showing codes with N>50)")

print("\nRef Phase -> Posterior Code (Top 3)")
counts = np.zeros((len(phase_ids), num_codes), dtype=int)
for c, p in zip(all_post, all_ref):
    if p in phase_ids:
        counts[phase_ids.index(p), c] += 1
row_sums = counts.sum(axis=1, keepdims=True)
pcts = np.where(row_sums > 0, counts / row_sums * 100, 0)
for i, pid in enumerate(phase_ids):
    order = np.argsort(-pcts[i])
    top = [f"c{c}={pcts[i,c]:.1f}%" for c in order[:3]]
    print(f"{phase_names[pid]:>10s}: {', '.join(top)}")

