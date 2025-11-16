# Generating corrected synthetic dataset with strict uniqueness and exact uniform scores
import sys
import json, pickle, random, numpy as np
from collections import defaultdict, Counter

sys.path.append(r"D:\code\repo\M.tech\sem1\DA\LAB\contest")
from FILE_DIR import *

# Load data and embeddings
with open(TRAIN_DATA,"r",encoding="utf-8") as f:
    train = json.load(f)
N = len(train)

user_emb = np.load(NP_USER_PROMPT_EMB)
resp_emb = np.load(NP_RESPOSE_EMB)
sys_emb = np.load(NP_SYSTEM_EMB)

with open(FIX_MERTIC_DEF,"rb") as f:
    metric_map = pickle.load(f)

metric_names = list(metric_map.keys())
metric_embs = {m: np.array(metric_map[m], dtype=np.float32).reshape(-1) for m in metric_names}

# Combined and normalize
def norm_rows(mat):
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n==0] = 1.0
    return mat / n

combined = (user_emb + resp_emb + sys_emb) / 3.0
combined_norm = norm_rows(combined)
metric_matrix = np.vstack([metric_embs[m] for m in metric_names])
metric_matrix_norm = norm_rows(metric_matrix)

sims = combined_norm @ metric_matrix_norm.T  # (N, M)

# Build expanded candidate pool: for each row, list metrics sorted by increasing sim excluding original
candidates_by_metric = defaultdict(list)
for i, row in enumerate(train):
    orig = row["metric_name"]
    sims_i = sims[i]
    # indices sorted ascending (worst match first)
    idx_order = np.argsort(sims_i)
    for idx in idx_order:
        m = metric_names[idx]
        if m == orig:
            continue
        # store (source_index, sim)
        candidates_by_metric[m].append((i, float(sims_i[idx])))
    # note: pools can be large, that's fine

# Prepare exact uniform scores list (11 buckets * 2500 = 27500)
scores = []
for s in range(11):
    scores += [float(s)] * 2500
random.shuffle(scores)

TOTAL = len(scores)
synthetic = []
seen = set()  # to ensure uniqueness of (user,response,system,metric)

# To make selection fair, randomize candidate order per metric
for m in metric_names:
    random.shuffle(candidates_by_metric[m])

# We'll iterate through scores and try to assign a unique candidate for each
metric_cycle = list(metric_names)
m_idx = 0

# helper to create key
def make_key(src_idx, metric):
    r = train[src_idx]
    return (r.get("user_prompt"), r.get("response"), r.get("system_prompt"), metric)

attempts_limit = 200  # max attempts per score to find unique
for si, score in enumerate(scores):
    found = False
    attempts = 0
    # iterate metrics in round-robin starting from current m_idx to diversify
    start = m_idx % len(metric_cycle)
    for offset in range(len(metric_cycle)):
        metric = metric_cycle[(start + offset) % len(metric_cycle)]
        pool = candidates_by_metric.get(metric, [])
        # try some candidates from pool (random sample of up to 50)
        sample_indices = random.sample(range(len(pool)), min(len(pool), 50)) if pool else []
        for idx_in_pool in sample_indices:
            src_idx, simv = pool[idx_in_pool]
            key = make_key(src_idx, metric)
            if key in seen:
                continue
            # accept this candidate
            seen.add(key)
            synthetic.append({
                "metric_name": metric,
                "score": f"{score:.1f}",
                "user_prompt": train[src_idx]["user_prompt"],
                "response": train[src_idx]["response"],
                "system_prompt": train[src_idx]["system_prompt"],
                "original_metric": train[src_idx]["metric_name"],
                "original_score": train[src_idx]["score"]
            })
            found = True
            # advance m_idx a bit
            m_idx = (m_idx + 1) % len(metric_cycle)
            break
        if found:
            break
        attempts += 1
        if attempts >= attempts_limit:
            break
    # fallback strategies if not found:
    if not found:
        # try random sampling across all rows and metrics until find unique or reach attempts_limit_big
        big_attempts = 0
        while big_attempts < 1000 and not found:
            src_idx = random.randrange(0, N)
            # pick a random metric that's not the original to ensure mismatch
            metric = random.choice(metric_names)
            if metric == train[src_idx]["metric_name"]:
                continue
            key = make_key(src_idx, metric)
            if key in seen:
                big_attempts += 1
                continue
            seen.add(key)
            synthetic.append({
                "metric_name": metric,
                "score": f"{score:.1f}",
                "user_prompt": train[src_idx]["user_prompt"],
                "response": train[src_idx]["response"],
                "system_prompt": train[src_idx]["system_prompt"],
                "original_metric": train[src_idx]["metric_name"],
                "original_score": train[src_idx]["score"]
            })
            found = True
            break
        if not found:
            # last resort: allow slight modification by appending an index suffix to system_prompt to make key unique
            # but we avoid changing text per Q2. If impossible, raise error.
            raise RuntimeError(f"Unable to find unique candidate for score index {si}, score {score}")
    # progress print occasionally
    if (si+1) % 2500 == 0:
        print(f"Assigned {si+1}/{TOTAL} synthetic rows...")

# Final checks
if len(synthetic) != TOTAL:
    raise RuntimeError(f"Expected {TOTAL} synthetic rows, got {len(synthetic)}")

# verify uniqueness
keys = {(r["user_prompt"], r["response"], r["system_prompt"], r["metric_name"]) for r in synthetic}
if len(keys) != TOTAL:
    raise RuntimeError("Duplicates detected after generation")

# verify score distribution
from collections import Counter as Cnt
score_counts = Cnt([r["score"] for r in synthetic])

# Save outputs
with open(f"{OUT_SYN_DIR}/synthetic_dataset.json","w",encoding="utf-8") as f:
    json.dump(synthetic, f, indent=2)

final = train.copy()
for r in synthetic:
    final.append({
        "metric_name": r["metric_name"],
        "score": r["score"],
        "user_prompt": r["user_prompt"],
        "response": r["response"],
        "system_prompt": r["system_prompt"]
    })

with open(f"{OUT_SYN_DIR}/final_balanced_dataset.json","w",encoding="utf-8") as f:
    json.dump(final, f, indent=2)

(len(synthetic), dict(score_counts))
