import json, os, pickle
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np

# 配置
EMB_MODEL_NAME = 'BAAI/bge-large-en-v1.5'
D1_CACHE       = 'cache/d1_embeddings.pkl'
QUERY_CACHE    = 'cache/query_embeddings.pkl'   # 所有 FINAL_DATA 的 embedding

os.makedirs('cache', exist_ok=True)

print("加载数据...")
with open("../../data/raw_data.jsonl", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]
with open("../data/D1.jsonl", encoding="utf-8") as f:
    d1 = [json.loads(line) for line in f]

df = pd.DataFrame(data)

print(f"All 样本数 : {len(df):,}")

model = SentenceTransformer(EMB_MODEL_NAME)

# ==================== 1. D1 embeddings (缓存) ====================
if os.path.exists(D1_CACHE):
    print(f"加载 D1 embeddings ← {D1_CACHE}")
    with open(D1_CACHE, 'rb') as f:
        cache = pickle.load(f)
    d1_emb   = cache['embeddings']
    d1_ids   = cache['ids']
    d1_questions = cache['questions']
else:
    print("计算并缓存 D1 embeddings...")
    d1_questions = [x['Q'] for x in d1]
    d1_ids       = [x['id'] for x in d1]
    d1_emb       = model.encode(d1_questions,
                                normalize_embeddings=True,
                                batch_size=64,
                                show_progress_bar=True)
    with open(D1_CACHE, 'wb') as f:
        pickle.dump({'embeddings': d1_emb,
                     'ids': d1_ids,
                     'questions': d1_questions}, f)
    print(f"D1 embeddings 已保存 → {D1_CACHE}")

# ==================== 2. 所有 queries embeddings (缓存) ====================

print("计算并缓存所有 query embeddings...")
queries = df['user_query'].tolist()
query_emb = model.encode(queries,
                         normalize_embeddings=True,
                         batch_size=64,
                         show_progress_bar=True)
query_ids = df.index.tolist()
with open(QUERY_CACHE, 'wb') as f:
    pickle.dump({'embeddings': query_emb,
                 'query_ids': query_ids}, f)
print(f"所有 query embeddings 已保存 → {QUERY_CACHE}")
