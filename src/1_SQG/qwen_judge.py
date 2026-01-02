# LLM-as-a-Judge with Opensource LM：重新判断 user_query 是否与 (Q,A) 对齐

import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import torch
import argparse
import re
import json

# ------------------- 参数 -------------------
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=196)
args = parser.parse_args()

# ------------------- 配置 -------------------
INPUT_FILE = "../../data/raw_data.jsonl"
OUTPUT_DIR = "judgment_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = f"{OUTPUT_DIR}/data_quality_judge_qwen.jsonl"

# ------------------- 模型加载（Qwen2.5-7B-Instruct 本地） -------------------
LOCAL_MODEL_DIR = os.path.expanduser("~/autodl-tmp/models") # model path
QWEN_PATH = os.path.join(LOCAL_MODEL_DIR, "Qwen2.5-7B-Instruct")
print("加载 Qwen2.5-7B-Instruct...")
tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, local_files_only=True, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    QWEN_PATH,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# ------------------- 加载 D1 -------------------
print("加载 D1.jsonl...")
d1_list = []
with open("../data/D1.jsonl", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        d1_list.append(obj)
id_to_qa = {item['id']: (item['Q'], item['A']) for item in d1_list}

# ------------------- 断点续跑 -------------------
processed_ids = set()
if os.path.exists(OUTPUT_FILE):
    print(f"检测到已有结果文件，加载已处理记录...")
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                processed_ids.add(obj.get('query_id'))
    print(f"已处理: {len(processed_ids):,} 条 → 将自动跳过")
    
# ------------------- 输出文件（追加模式） -------------------
f_out = open(OUTPUT_FILE, 'a', encoding='utf-8', buffering=1)

# ------------------- LLM 判断函数（批量） -------------------
def judge_batch_consistency(batch_rows):
    queries = [row['user_query'] for row in batch_rows]
    true_ids = [row['question_id'] for row in batch_rows]
    true_qs = []
    true_as = []
    for tid in true_ids:
        q, a = id_to_qa.get(tid, ("[Not found]", "[Not found]"))
        true_qs.append(q)
        true_as.append(a)

    prompts = [
        f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
You are an expert and strict judge. Determine if the user's question is exactly equivalent to the canonical QA question.
Rules (must strictly obey):
- Exactly one word "Yes" or "No", no explanation.
- Use the following criteria to derive the judgement:
    1. correctness: the user question is syntactically sound;
    2. relevancy: the user question is 100% relevant to the canonical FAQ;
    3. coherence: the canonical answer is a suitable and absolutely correct answer for the user's question.
- Output "Yes" if the user question means exactly the same as the canonical question, and the canonical answer is perfectly suitable for the user query; otherwise, "No".
User question: {q}
Canonical FAQ question: {true_q}
Canonical answer: {true_a}
Answer with only "Yes" or "No".<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""" for q, true_q, true_a in zip(queries, true_qs, true_as)
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    results = []
    for out in outputs:
        text = tokenizer.decode(out, skip_special_tokens=True)
        generated = text.split("assistant")[-1].strip().upper()
        if "YES" in generated:
            results.append(True)
        else:
            results.append(False)  # 强制二值！永不 None！

    return results

# ------------------- 主循环：和你成功的代码一模一样！-------------------
batch_rows = []
total_saved = len(processed_ids)

print("开始判断...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Judging"):
        item = json.loads(line)
        qid = item.get("query_id")

        if qid in processed_ids:
            continue

        batch_rows.append(item)

        if len(batch_rows) >= args.batch_size:
            judgements = judge_batch_consistency(batch_rows)
            for row, jud in zip(batch_rows, judgements):
                row = row.copy()
                row["judgement_qwen"] = jud
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_saved += len(batch_rows)
            print(f"已保存 {total_saved:,} 条")
            batch_rows.clear()

# 处理最后一批
if batch_rows:
    judgements = judge_batch_consistency(batch_rows)
    for row, jud in zip(batch_rows, judgements):
        row = row.copy()
        row["judgement_qwen"] = jud
        f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
    total_saved += len(batch_rows)
    print(f"最后一批已保存，总计 {total_saved:,} 条")

f_out.close()
print(f"\n完成！")
print(f"结果已安全保存 → {OUTPUT_FILE}")
print("   新增字段: judgement_qwen (True/False)")