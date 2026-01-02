# llm_rejudge_retrieval_with_gemma8.py
# Use gemma-3-8B-Instruct to judge: "Is the retrieved top-1 passage correct for this query?"
# Output: adds column → gemma_judge_retrieval: True / False

import os
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import torch
import argparse
import re

# Load template
with open("prompt_template_aug.txt", "r", encoding="utf-8") as f:
    template = f.read()
    
# ------------------- 参数 -------------------
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--input_file", type=str, default="judgement_results/ral2m_aug1.jsonl")
parser.add_argument("--output_file", type=str, default="judgement_results/ral2m_aug_gemma.jsonl")
args = parser.parse_args()

# ------------------- 模型加载 -------------------
LOCAL_MODEL_DIR = os.path.expanduser("~/autodl-tmp/models")
gemma_PATH = os.path.join(LOCAL_MODEL_DIR, "google-gemma-2-9b-it")
print("加载 gemma")
tokenizer = AutoTokenizer.from_pretrained(gemma_PATH, local_files_only=True, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    gemma_PATH,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()
# ------------------- 加载 D1 知识库 -------------------
print("Loading D1.jsonl to get passage text by ID...")
id_to_passage = {}
with open("../../data/D1.jsonl", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        id_to_passage[obj['id']] = (obj['Q'], obj['A'])

id_to_doc = {}
with open("../../data/D2.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            doc_id = obj.get('doc_id')
            if doc_id is not None:
                text = obj['text'].strip()
                # Optional cleaning
                text = re.sub(r'\s+', ' ', text)
                id_to_doc[doc_id] = text


# ------------------- 输出文件 -------------------
os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".", exist_ok=True)
f_out = open(args.output_file, "a", encoding="utf-8", buffering=1)

processed_ids = set()
if os.path.exists(args.output_file):
    with open(args.output_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if "query_id" in obj:
                    processed_ids.add(obj["query_id"])

print(f"Already processed: {len(processed_ids):,} samples. Skipping them.")

# ------------------- LLM 判断函数（批量） -------------------
def judge_retrieval_batch(batch_rows):
    prompts = []
    for row in batch_rows:
        user_query = row['user_query']
        retrieved_id = row['bge_retrieved_top1_id']
        doc_id = row['bge_top1_doc_id']
        candidate_text = id_to_doc[doc_id]
        if retrieved_id not in id_to_passage:
            retrieved_q = "[Missing passage]"
            retrieved_a = "[Missing passage]"
        else:
            retrieved_q, retrieved_a = id_to_passage[retrieved_id]
        # Plug values
        prompt = template.format(
            user_query=user_query,
            candidate_Q=retrieved_q,
            candidate_A=retrieved_a,
            document_chunk=candidate_text
        )

        prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" + prompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        prompts.append(prompt)

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1536).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            temperature=0,  # fixed: avoids sampling disabled error
            top_p=1.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    results = []
    for out in outputs:
        text = tokenizer.decode(out[inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().upper()
        results.append("YES" in text)
    return results

# ------------------- 主循环 -------------------
batch = []
total_saved = len(processed_ids)

print("Starting gemma-7B relevance judgment on retrieved top-1...")
with open(args.input_file, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Judging retrieval"):
        item = json.loads(line)
        qid = item.get("query_id")
        if qid in processed_ids:
            continue

        batch.append(item)
        if len(batch) >= args.batch_size:
            judgements = judge_retrieval_batch(batch)
            for row, jud in zip(batch, judgements):
                row = row.copy()
                row["gemma_aug"] = jud
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_saved += len(batch)
            print(f"Saved: {total_saved:,} rows")
            batch.clear()

# 最后一批
if batch:
    judgements = judge_retrieval_batch(batch)
    for row, jud in zip(batch, judgements):
        row = row.copy()
        row["gemma_aug"] = jud
        f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
    total_saved += len(batch)
    print(f"Final batch saved. Total: {total_saved:,}")

f_out.close()
print(f"Done! Results saved to {args.output_file}")
print("New column: gemma_judge_retrieval (True = retrieved answer is correct, False = should reject)")