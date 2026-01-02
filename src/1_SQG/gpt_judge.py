# LLM-as-a-Judge with GPT：重新判断 user_query 是否与 (Q,A) 对齐

import os
import json
import httpx
from openai import OpenAI
from tqdm.auto import tqdm
import argparse
import time

INPUT_FILE = "../../data/raw_data.jsonl"
OUTPUT_DIR = "judgment_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = f"{OUTPUT_DIR}/data_quality_judge_gpt.jsonl"

# ------------------- OpenAI API 配置 -------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-") # API Key

client = OpenAI(
    base_url="", # if any (e.g., https://api.xty.app/v1)
    api_key=OPENAI_API_KEY,
    http_client=httpx.Client(base_url="", follow_redirects=True),
)

# ------------------- 加载 D1 -------------------
print("加载 D1.jsonl...")
d1_list = []
with open("../data/D1.jsonl", encoding="utf-8") as f:
    for line in f:
        d1_list.append(json.loads(line))
id_to_qa = {x["id"]: (x["Q"], x["A"]) for x in d1_list}

# ------------------- 断点续跑 -------------------
processed_ids = set()
if os.path.exists(OUTPUT_FILE):
    print("检测到已有结果，加载已处理 query_id...")
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if "query_id" in obj:
                    processed_ids.add(obj["query_id"])
    print(f"跳过 {len(processed_ids):,} 条")

f_out = open(OUTPUT_FILE, "a", encoding="utf-8", buffering=1)

# ------------------- GPT-4o-mini 批量判断 -------------------
def judge_gpt4o_mini(row):

    true_id = row["question_id"]
    true_q, true_a = id_to_qa.get(true_id, ("[Not found]", "[Not found]"))
    
    system_prompt = "You are an expert semantic judge. Answer only with 'Yes' or 'No'. No explanation."
    user_prompt = f"""User Query: {row['user_query']}

    Canonical Question: {true_q}
    Canonical Answer: {true_a}
    
    Is the user query exactly aligned with the canonical question, with the canonical answer perfectly and completely answer the user's query?
    Answer with only "Yes" or "No".
    """
    
    msg = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = None  # default

    for _ in range(3):  # 最多重试3次
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
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
print("开始使用 gpt-4o-mini 判断...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Judging"):
        item = json.loads(line)
        qid = item.get("query_id")
        if qid in processed_ids:
            continue
        
        judgement = judge_gpt4o_mini(item)
        item["judgement_new"] = judgement
        
        f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
        total_saved += 1

f_out.close()
print(f"\n完成！结果已保存 → {OUTPUT_FILE}")
print("新增字段: judgement_new (True/False)")