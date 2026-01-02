import json, os, pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# 配置
EMB_MODEL_NAME = 'BAAI/bge-large-en-v1.5'
D1_CACHE = 'cache/d1_concat_intent_embeddings.pkl'
QUERY_CONCAT_CACHE = 'cache/query_concat_intent_embeddings.pkl'
os.makedirs('cache', exist_ok=True)

print("加载数据...")
with open("data_with_intent.jsonl", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

with open("D1_keyword.jsonl", encoding="utf-8") as f:
    d1 = [json.loads(line) for line in f]

model = SentenceTransformer(EMB_MODEL_NAME)

# ==================== 1. D1 embeddings (Normalized Concat) ====================
if os.path.exists(D1_CACHE):
    print(f"加载 D1 concat embeddings ← {D1_CACHE}")
    with open(D1_CACHE, 'rb') as f:
        cache = pickle.load(f)
    d1_emb = cache['embeddings']
    d1_ids = cache['ids']
    d1_questions = cache['questions']
else:
    print("计算并缓存 D1 (Q + Q_keywords) 拼接 embeddings...")
    d1_questions = [x.get('Q', '') for x in d1]
    d1_ids = [x.get('id', '') for x in d1]
    d1_kw_list = [x.get('Q_intent', []) for x in d1]
    d1_kw_texts = [" ".join(kw) if isinstance(kw, list) else "" for kw in d1_kw_list]

    print("Encoding D1 Components...")
    d1_q_embs = model.encode(d1_questions, normalize_embeddings=True, batch_size=256, show_progress_bar=True)
    d1_kw_embs = model.encode(d1_kw_texts, normalize_embeddings=True, batch_size=256, show_progress_bar=True)

    # 拼接
    d1_emb = np.concatenate([d1_q_embs, d1_kw_embs], axis=1)
    
    # --- CRITICAL UPDATE: Normalize the unified vector ---
    norms = np.linalg.norm(d1_emb, axis=1, keepdims=True)
    d1_emb = d1_emb / (norms + 1e-10) 

    with open(D1_CACHE, 'wb') as f:
        pickle.dump({'embeddings': d1_emb, 'ids': d1_ids, 'questions': d1_questions}, f)
    print(f"D1 concat embeddings 已保存 → {D1_CACHE}")

# ==================== 2. Query + Keywords (Normalized Concat) ====================
print("\n计算并缓存 query + keywords 拼接 embeddings...")
queries = [item.get('user_query', '') for item in data]
keywords_list = [item.get('query_intent', []) for item in data]
kw_texts = [" ".join(kw) if isinstance(kw, list) else "" for kw in keywords_list]

query_embs = model.encode(queries, normalize_embeddings=True, batch_size=256, show_progress_bar=True)
kw_embs = model.encode(kw_texts, normalize_embeddings=True, batch_size=256, show_progress_bar=True)

# 拼接
concat_embs = np.concatenate([query_embs, kw_embs], axis=1)

# --- CRITICAL UPDATE: Normalize the unified vector ---
query_norms = np.linalg.norm(concat_embs, axis=1, keepdims=True)
concat_embs = concat_embs / (query_norms + 1e-10)

query_ids = list(range(len(data)))

with open(QUERY_CONCAT_CACHE, 'wb') as f:
    pickle.dump({'embeddings': concat_embs, 'query_ids': query_ids}, f)
print(f"拼接 query + keywords embeddings 已保存 → {QUERY_CONCAT_CACHE}")