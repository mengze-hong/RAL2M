# embed_and_index_ral2m.py
# FINAL OPTIMIZED VERSION — no duplicate embeddings

import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from tqdm.auto import tqdm

# Paths
CSV_DIR = "ral2m_ragbench_csv"
EMB_DIR = "embeddings"
INDEX_DIR = "faiss_index"
os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Model from your paper
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
DIM = model.get_sentence_embedding_dimension()
print(f"Embedding dimension: {DIM}")

# Load D1
print("\n1. Processing D1_all.csv ...")
d1_df = pd.read_csv(os.path.join(CSV_DIR, "D1.csv"))
d1_texts = d1_df['q_i'].tolist()

print("   → Embedding D1 entries:", len(d1_texts))
d1_emb = model.encode(d1_texts, batch_size=512, show_progress_bar=True, normalize_embeddings=True)

np.save(os.path.join(EMB_DIR, "D1_embeddings.npy"), d1_emb)
d1_df.to_csv(os.path.join(EMB_DIR, "D1_metadata.csv"), index=False)
print(f"   → Saved D1 embeddings: {d1_emb.shape}")

# Build FAISS index
print("\n2. Building FAISS index on D1...")
index = faiss.IndexFlatIP(DIM)
index.add(d1_emb.astype('float32'))
faiss.write_index(index, os.path.join(INDEX_DIR, "D1_index.faiss"))
print(f"   → FAISS index saved with {index.ntotal:,} vectors")

# Save D1 mapping
d1_mapping = d1_df[['q_i', 'a_i', 'dataset']].to_dict('records')
with open(os.path.join(INDEX_DIR, "D1_mapping.json"), "w") as f:
    json.dump(d1_mapping, f, indent=2)

# Load train.csv
print("\n3. Embedding train.csv queries...")
train_df = pd.read_csv(os.path.join(CSV_DIR, "train.csv"))
train_queries = train_df['user_query'].tolist()

train_emb = model.encode(train_queries, batch_size=512, show_progress_bar=True, normalize_embeddings=True)
np.save(os.path.join(EMB_DIR, "train_embeddings.npy"), train_emb)
train_df.to_csv(os.path.join(EMB_DIR, "train_metadata.csv"), index=False)
print(f"   → Train embeddings: {train_emb.shape}")

# Load test_single.csv (source of truth for queries)
print("\n4. Embedding test queries (single-turn + multi-turn share the same queries)...")
test_single_df = pd.read_csv(os.path.join(CSV_DIR, "test_single.csv"))

# Use only unique user queries to avoid duplication
unique_queries = test_single_df['user_query'].drop_duplicates().tolist()
print(f"   → Unique test queries: {len(unique_queries):,}")

test_emb = model.encode(unique_queries, batch_size=512, show_progress_bar=True, normalize_embeddings=True)
np.save(os.path.join(EMB_DIR, "test_query_embeddings.npy"), test_emb)

# Save mapping: original index → query text
query_to_idx = {q: i for i, q in enumerate(unique_queries)}
with open(os.path.join(EMB_DIR, "test_query_to_idx.json"), "w") as f:
    json.dump(query_to_idx, f, indent=2)

# Save test metadata with embedding index
test_single_df['query_emb_idx'] = test_single_df['user_query'].map(query_to_idx)
test_single_df.to_csv(os.path.join(EMB_DIR, "test_single_with_idx.csv"), index=False)

# Also save multi-turn if exists
if os.path.exists(os.path.join(CSV_DIR, "test_multi.csv")):
    test_multi_df = pd.read_csv(os.path.join(CSV_DIR, "test_multi.csv"))
    test_multi_df['query_emb_idx'] = test_multi_df['user_query'].map(query_to_idx)
    test_multi_df.to_csv(os.path.join(EMB_DIR, "test_multi_with_idx.csv"), index=False)
    print("   → test_multi_with_idx.csv saved (shares embeddings)")

print("\nAll done! Embedding pipeline complete.")
print(f"""
Your retrieval system is ready:
   • D1 index:           faiss_index/D1_index.faiss
   • D1 embeddings:      embeddings/D1_embeddings.npy
   • Train embeddings:     embeddings/train_embeddings.npy
   • Test query embeddings: embeddings/test_query_embeddings.npy (shared by single & multi-turn)

Next: Run retrieval + RAL2M judgment
""")