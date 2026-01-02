# final_retrieval_with_real_d1_answers.py
# CORRECT: Uses question_id → D1 lookup for true answer (no true_a_i!)

import json
import os
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from tqdm.auto import tqdm

# ==================== CONFIG ====================
INPUT_DATA = "result_data.jsonl"
D1_FILE = "../../data/D1.jsonl"
D1_CACHE = "../cache/d1_embeddings.pkl"

EMBEDDING_CACHE = "query_with_context_embeddings.pkl"   # ← SAVED HERE
OUTPUT_RESULTS = "final_retrieval_with_context.jsonl"

EMB_MODEL = "BAAI/bge-large-en-v1.5"
DEVICE = "cuda"

# ==================== Load D1 (for true answers) ====================
print("Loading D1.jsonl for true answers...")
d1_records = []
with open(D1_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            d1_records.append(json.loads(line))
d1_df = pd.DataFrame(d1_records)
print(f"Loaded {len(d1_df):,} D1 entries")

# ==================== Load main data ====================
print("Loading data with hist_query_ids...")
with open(INPUT_DATA, "r", encoding="utf-8") as f:
    data = [json.loads(line.strip()) for line in f if line.strip()]

result = pd.DataFrame(data)
print(f"Total queries: {len(result):,}")

# ==================== Load BGE + D1 embeddings + FAISS ====================
print("Loading BGE model...")
model = SentenceTransformer(EMB_MODEL, device=DEVICE)

print("Loading D1 embeddings + building FAISS...")
with open(D1_CACHE, 'rb') as f:
    cache = pickle.load(f)
d1_emb = cache['embeddings']
d1_ids = np.array(cache['ids'])

index = faiss.IndexFlatIP(d1_emb.shape[1])
index.add(d1_emb.astype('float32'))
print(f"FAISS index ready: {index.ntotal} passages")

# ==================== YOUR EXACT FUNCTION — UNCHANGED ====================
def construct_dialogue_context(row, d1_df):
    history = []
    for hist_id in row['hist_query_ids']:
        question_id = hist_id.split("_p")[0]
        hist_row = result[result['query_id'] == hist_id]
        if hist_row.empty:
            continue
        query = hist_row.iloc[0]['user_query']
        answer_row = d1_df[d1_df['id'] == question_id]
        answer = answer_row['A'].iloc[0] if not answer_row.empty else "[No answer]"
        history.append(f"User: {query} Assistant: {answer}")
    return " || ".join(history)

# ==================== Build context + full input ====================
print("\nBuilding dialogue context using YOUR function...")
result['dialogue_context'] = result.apply(
    lambda row: construct_dialogue_context(row, d1_df), axis=1
)

# Build final input: context + current query
print("Building full input strings...")
full_queries = []
for _, row in result.iterrows():
    ctx = row['dialogue_context']
    curr = row['user_query']
    full = f"{ctx} || User: {curr}" if ctx else curr
    full_queries.append(full)

# ==================== EMBED + SAVE EMBEDDINGS ====================
print("Encoding context + query with BGE...")
query_embeddings = model.encode(
    full_queries,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True
).astype('float32')

# SAVE THE EMBEDDINGS
print(f"Saving context-aware query embeddings → {EMBEDDING_CACHE}")
with open(EMBEDDING_CACHE, 'wb') as f:
    pickle.dump({
        'embeddings': query_embeddings,
        'query_ids': result['query_id'].tolist(),
        'full_texts': full_queries,
        'original_indices': result.index.tolist()
    }, f)
print(f"Embeddings saved: {query_embeddings.shape}")

# ==================== Retrieve from D1 ====================
print("Retrieving top-1 from D1 using context embeddings...")
scores, indices = index.search(query_embeddings, 1)
pred_ids = [str(d1_ids[i[0]]) for i in indices]
pred_scores = scores.flatten().tolist()

# Add results
result['bge_with_context_pred_id'] = pred_ids
result['bge_with_context_score'] = pred_scores

# ==================== Save final results ====================
result.to_json(OUTPUT_RESULTS, orient='records', lines=True, force_ascii=False)
print(f"\nALL DONE!")
print(f"   • Context-aware embeddings → {EMBEDDING_CACHE}")
print(f"   • Full results + predictions → {OUTPUT_RESULTS}")
print(f"   • Ready for evaluation, visualization, or submission")

# Quick sanity check
print("\nFirst result:")
print(f"Query ID : {result.iloc[0]['query_id']}")
print(f"Context  : {result.iloc[0]['dialogue_context'][:200]}...")
print(f"Full input length: {len(full_queries[0])} chars")
print(f"Pred ID  : {result.iloc[0]['bge_with_context_pred_id']}")