# llm_rejudge_retrieval_with_qwen8.py
# Use qwen-3-8B-Instruct to judge: "Is the retrieved top-1 passage correct for this query?"
# Output: adds column → qwen_judge_retrieval: True / False

import os
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import torch
import argparse
from openai import OpenAI
import time
import httpx

# Load template
with open("prompt_template.txt", "r", encoding="utf-8") as f:
    template = f.read()
    
# ------------------- 参数 -------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default="../5_Analysis/test_results.jsonl")
parser.add_argument("--output_file", type=str, default="../5_Analysis/test_results_with_gpt.jsonl")
args = parser.parse_args()


# ------------------- OpenAI API 配置 -------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-") # API Key

client = OpenAI(
    base_url="https://api.xty.app/v1", # if any (e.g., https://api.xty.app/v1)
    api_key=OPENAI_API_KEY,
    http_client=httpx.Client(base_url="", follow_redirects=True),
)

# ------------------- 加载 D1 知识库 -------------------
print("Loading D1.jsonl to get passage text by ID...")
id_to_passage = {}
with open("../../data/D1.jsonl", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        id_to_passage[obj['id']] = (obj['Q'], obj['A'])

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
def judge_gpt(row):
    user_query = row['user_query']
    retrieved_id = row['bge_retrieved_top1_id']
    if retrieved_id not in id_to_passage:
        retrieved_q = "[Missing passage]"
        retrieved_a = "[Missing passage]"
    else:
        retrieved_q, retrieved_a = id_to_passage[retrieved_id]
    # Plug values
    prompt = template.format(
        user_query=user_query,
        candidate_Q=retrieved_q,
        candidate_A=retrieved_a
    )

    msg = [
        {"role": "system", "content": "You are a meticulous and impartial evaluator with expertise in assessing the accuracy and relevance of question-and-answer pairs."},
        {"role": "user", "content": prompt}
    ]

    for _ in range(3):  # 最多重试3次
        try:
            resp = client.chat.completions.create(
                model="gpt-5-chat",
                temperature=0.0,
                messages=msg
            )
            content = resp.choices[0].message.content.strip().upper()
            if "YES" in content:
                response = True
            else:
                response = False
            break
        except Exception as e:
            print(f"API 错误，重试: {e}")
            time.sleep(1)

    return response

# ------------------- 主循环 -------------------
total_saved = len(processed_ids)
with open(args.input_file, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Judging retrieval"):
        item = json.loads(line)
        qid = item.get("query_id")
        if qid in processed_ids:
            continue
        
        judgement = judge_gpt(item)
        item["gpt5_judge"] = judgement  # True/False
        
        f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
        total_saved += 1
        print(f"已保存 {total_saved:,} 条")

f_out.close()
print(f"Done! Results saved to {args.output_file}")
print("New column: gpt5_judge (True = retrieved answer is correct, False = should reject)")