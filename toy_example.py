# toy_retrieval_demo.py
# 展示 RAL2M 检索阶段是否精准（在 train set 上验证）

import numpy as np
import pandas as pd
import faiss
import json
from sentence_transformers import SentenceTransformer

# Paths
EMB_DIR = "embeddings"
INDEX_DIR = "faiss_index"
CSV_DIR = "datasets"

# Load everything
print("Loading embeddings and index...")
d1_emb = np.load(f"{EMB_DIR}/D1_embeddings.npy")           # (N, 384)
test_emb = np.load(f"{EMB_DIR}/test_query_embeddings.npy") # shared by single/multi
train_emb = np.load(f"{EMB_DIR}/train_embeddings.npy")

d1_meta = pd.read_csv(f"{EMB_DIR}/D1_metadata.csv")
test_meta = pd.read_csv(f"{EMB_DIR}/test_single_with_idx.csv")
train_meta = pd.read_csv(f"{EMB_DIR}/train_metadata.csv")

with open(f"{INDEX_DIR}/D1_mapping.json") as f:
    d1_mapping = json.load(f)

with open(f"{EMB_DIR}/test_query_to_idx.json") as f:
    query_to_idx = json.load(f)

index = faiss.read_index(f"{INDEX_DIR}/D1_index.faiss")

print(f"Loaded D1 with {len(d1_meta):,} entries")
print(f"Train set: {len(train_meta):,} examples")
print(f"Test set: {len(test_meta):,} examples\n")

# Randomly sample ONE example from train set
np.random.seed(42)
idx = np.random.randint(0, len(train_meta))
sample = train_meta.iloc[idx]

user_query = sample['user_query']
true_a_i = sample['true_a_i']        # 正确答案
true_q_i = sample['true_q_i']        # 正确问题

print("="*80)
print("TOY EXAMPLE — Retrieval from TRAIN set")
print("="*80)
print(f"User Query: {user_query}")
print(f"True Answer: {true_a_i}")
print(f"True Question in D1: {true_q_i}\n")

# Get embedding of this query
query_text = sample['user_query']
if query_text in query_to_idx:
    q_emb = test_emb[query_to_idx[query_text]:query_to_idx[query_text]+1]
else:
    # Fallback: re-encode (shouldn't happen in train)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    q_emb = model.encode([query_text], normalize_embeddings=True)

# Retrieve top-10
k = 10
D, I = index.search(q_emb.astype('float32'), k)

print(f"Top-{k} Retrieved Candidates:")
print("-" * 80)
hit = False
for rank, (score, idx) in enumerate(zip(D[0], I[0]), 1):
    cand = d1_mapping[idx]
    marker = "CORRECT" if cand['a_i'] == true_a_i else ""
    print(f"[{rank:2d}] Score: {score:.4f} | Q: {cand['q_i'][:80]:80}... | {marker}")
    if cand['a_i'] == true_a_i:
        hit = True
        hit_rank = rank

if hit:
    print(f"\nSUCCESS: Correct answer found at rank {hit_rank} / {k}")
else:
    print(f"\nFAIL: Correct answer NOT in top-{k}")

# Bonus: show how many train queries have correct answer in top-10
print("\n" + "="*80)
print("Bonus: Train set Top-10 Recall Check (first 1000 samples)")
print("="*80)
hits = 0
for i in range(min(1000, len(train_meta))):
    row = train_meta.iloc[i]
    q_emb = train_emb[i:i+1]
    D, I = index.search(q_emb.astype('float32'), 10)
    retrieved_answers = [d1_mapping[idx]['a_i'] for idx in I[0]]
    if row['true_a_i'] in retrieved_answers:
        hits += 1

print(f"Top-10 Recall on 1000 train samples: {hits}/1000 = {hits/10:.1f}%")