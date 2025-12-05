# preprocess_ragbench_for_ral2m_csv.py
# FINAL VERSION — 100% DATA RETENTION + CSV OUTPUT

import os
import json
import pandas as pd
from datasets import load_dataset

DATASETS = [
    'covidqa', 'cuad', 'delucionqa', 'emanual', 'expertqa',
    'finqa', 'hagrid', 'hotpotqa', 'msmarco', 'pubmedqa', 'tatqa', 'techqa'
]

OUTPUT_DIR = "ral2m_ragbench_csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Final containers
all_d1 = []           # For D1_all.csv
all_train_val = []    # For train_val.csv
all_test = []         # For test.csv

def extract_gt(ex):
    gt = ex.get('ground_truth')
    if not gt:
        return ex.get('response', '').strip()
    if isinstance(gt, str):
        return gt.strip()
    if isinstance(gt, list) and gt:
        return str(gt[0]).strip()
    if isinstance(gt, dict):
        txt = gt.get('text') or gt.get('answer')
        if isinstance(txt, list):
            return str(txt[0]).strip() if txt else ""
        return str(txt).strip() if txt else ""
    return str(gt).strip()

print("Starting FULL RAL2M preprocessing → CSV output (100% retention)\n")

for name in DATASETS:
    print(f"Processing {name.upper():<12} ...")
    ds = load_dataset("rungalileo/ragbench", name)

    # Build D₁ from train + validation
    d1_dict = {}
    for split in ['train', 'validation']:
        for ex in ds[split]:
            q = ex['question'].strip()
            a = extract_gt(ex)
            if len(q) < 3 or len(a) < 3:
                continue
            key = (q.lower(), a.lower())
            if key not in d1_dict:
                d1_dict[key] = (q, a)
                all_d1.append({
                    'q_i': q,
                    'a_i': a,
                    'dataset': name
                })

    # Process train/validation
    for split_name in ['train', 'validation']:
        for ex in ds[split_name]:
            user_q = ex['question'].strip()
            docs = json.dumps(ex['documents'])  # safe string
            true_a = extract_gt(ex)
            if not user_q or not true_a:
                continue

            key = (user_q.lower(), true_a.lower())
            true_q_i = d1_dict.get(key, (user_q, true_a))[0]

            all_train_val.append({
                'user_query': user_q,
                'dialogue_context': "",
                'retrieved_docs': docs,
                'true_q_i': true_q_i,
                'true_a_i': true_a,
                'dataset': name,
                'split': split_name,
                'original_id': ex.get('id', '')
            })

    # Process test — 100% kept
    test_count = 0
    for ex in ds['test']:
        user_q = ex['question'].strip()
        docs = json.dumps(ex['documents'])
        true_a = extract_gt(ex)
        if not user_q or not true_a:
            continue

        key = (user_q.lower(), true_a.lower())
        true_q_i = d1_dict.get(key, (user_q, true_a))[0]

        all_test.append({
            'user_query': user_q,
            'dialogue_context': "",
            'retrieved_docs': docs,
            'true_q_i': true_q_i,
            'true_a_i': true_a,
            'dataset': name,
            'split': 'test',
            'original_id': ex.get('id', '')
        })
        test_count += 1

    print(f"  D₁: {len(d1_dict):>6,} | Train+Val: {len(ds['train']) + len(ds['validation']):>5,} | Test: {test_count:>5,}")

# Save as CSV
print(f"\nSaving three master CSV files to '{OUTPUT_DIR}/' ...")

pd.DataFrame(all_d1).to_csv(os.path.join(OUTPUT_DIR, "D1_all.csv"), index=False)
pd.DataFrame(all_train_val).to_csv(os.path.join(OUTPUT_DIR, "train_val.csv"), index=False)
pd.DataFrame(all_test).to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

print("Done!")
print(f"   • D1_all.csv     → {len(all_d1):,} FAQ entries (for embedding)")
print(f"   • train_val.csv  → {len(all_train_val):,} training examples")
print(f"   • test.csv       → {len(all_test):,} test examples (100% retained)")

# Bonus: per-domain test CSVs
for name in DATASETS:
    subset = [r for r in all_test if r['dataset'] == name]
    if subset:
        pd.DataFrame(subset).to_csv(os.path.join(OUTPUT_DIR, f"test_{name}.csv"), index=False)

print(f"\nAll done! Your RAL2M benchmark is ready in ./{OUTPUT_DIR}/")
print("Now just run your embedding model on D1_all.csv → FAISS index")
print("Then evaluate on test.csv")
print("You're ready to dominate.")