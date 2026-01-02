# 1_retrieval_only_save_top1.py
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from tqdm.auto import tqdm

# ------------------- 加载 D1 向量 + ID -------------------
print("加载 D1 缓存...")
with open('cache/d1_concat_embeddings.pkl', 'rb') as f:
    d1_cache = pickle.load(f)
d1_emb = d1_cache['embeddings'].astype('float32')
d1_ids = np.array(d1_cache['ids'], dtype=int)

# 构建 FAISS
index = faiss.IndexFlatIP(d1_emb.shape[1])
index.add(d1_emb)
print(f"FAISS 索引 ready，{index.ntotal:,} 条")

# ------------------- 加载 data_balanced -------------------
with open("data_with_keywords.jsonl", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

print(f"共 {len(data):,} 条待检索")

# ------------------- 执行 Top-1 检索 -------------------
print("加载 cache/query_concat_embeddings.pkl...")
with open('cache/query_concat_embeddings.pkl', 'rb') as f:
    query_cache = pickle.load(f)
query_emb = query_cache['embeddings'].astype('float32')
print(f"Loaded {query_emb.shape[0]:,} query embeddings")

D, I = index.search(query_emb.astype('float32'), 1)

# ------------------- 添加结果到原始数据 -------------------
output_file = "data_keyword_top1.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for item, score, idx in tqdm(zip(data, D.squeeze(), I.squeeze()), total=len(data), desc="保存"):
        item = item.copy()
        item["bge_keyword_retrieved_top1_id"] = int(d1_ids[idx])
        item["bge_keyword_retrieved_top1_score"] = float(score)
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n检索完成！")
print(f"结果已保存 → {output_file}")
print("新增字段：")
print("  - retrieved_candidate_id (int)")
print("  - retrieval_score (float)")