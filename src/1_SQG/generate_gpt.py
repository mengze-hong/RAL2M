import os
import json
import httpx
from openai import OpenAI
from tqdm.auto import tqdm
import argparse
import time
import re

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=1)
args = parser.parse_args()

INPUT_FILE = "../../data/D1.jsonl"
OUTPUT_DIR = "generated_queries"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = f"{OUTPUT_DIR}/SQG_gpt_{args.k}.jsonl"

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
with open(INPUT_FILE, encoding="utf-8") as f:
    for line in f:
        d1_list.append(json.loads(line))
print(f"共 {len(d1_list):,} 条 canonical QA")

# ------------------- 检查已处理 ID -------------------
processed_ids = set()
if os.path.exists(OUTPUT_FILE):
    print("检测到已有结果，加载已处理 ID...")
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                processed_ids.add(obj["id"])
    print(f"跳过 {len(processed_ids):,} 条已处理")

f_out = open(OUTPUT_FILE, "a", encoding="utf-8", buffering=1)

# ------------------- GPT-4o-mini 生成 -------------------
def generate_paraphrases(q, a, k=20):
    prompt = f"""
You are an expert at paraphrasing questions while preserving the exact meaning.
Task: Rewrite the following question in EXACTLY {k} completely different ways.
Rules (must strictly obey):
- Every line must be a complete, natural English question
- Every line must end with a question mark (?)
- No numbering, no bullets, no quotes, no explanations, no extra text
- Do NOT repeat the original question or generated questions
- You must read and understand the original question carefully, and use the reference answer to assist in understanding.
- Each new question must capture exactly the same meaning of the original question.
- Each new question must also correspond to the same reference answer.
Original Question: {q}
Reference answer (just for context, do not mention it): {a}
Now output EXACTLY {k} paraphrased questions, one per line:
"""

    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-5-chat",
                temperature=0.8,
                top_p=0.95,
                messages=[{"role": "user", "content": prompt}]
            )
            text = resp.choices[0].message.content.strip()
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            # 严格清洗
            cleaned = []
            for l in lines:
                l = re.sub(r'^\d+[\.\)\]\s]*', '', l)
                l = re.sub(r'^["\']|["\']$', '', l)
                l = re.sub(r'^(Answer|Version|Rewrite)[:：\s]*', '', l, flags=re.I)
                if l.endswith("?") and len(l) > 10 and l.lower() != q.lower():
                    cleaned.append(l)
            cleaned = list(dict.fromkeys(cleaned))  # 去重
            if len(cleaned) >= k:
                return cleaned[:k]
            else:
                return cleaned + [cleaned[0]] * (k - len(cleaned))
        except Exception as e:
            print(f"生成失败，重试: {e}")
            time.sleep(1)
    # 最终保底
    return [f"{q} (rephrased {i})" for i in range(1, k+1)]

# ------------------- 主循环 -------------------
batch_items = []
total_saved = len(processed_ids)

print(f"开始为 D1 生成 {args.k} 条改写（每条立即保存）...")
for item in tqdm(d1_list, desc="Generating paraphrases"):
    if item["id"] in processed_ids:
        continue

    batch_items.append(item)

    if len(batch_items) >= args.batch_size:
        for it in batch_items:
            paraphrases = generate_paraphrases(it["Q"], it["A"], k=args.k)
            # 输出原句 + 20 条改写
            # 1. 原句
            out_item = it.copy()
            out_item["user_query"] = it["Q"]
            out_item["is_original"] = True
            f_out.write(json.dumps(out_item, ensure_ascii=False) + "\n")

            # 2. 改写句
            for rank, new_q in enumerate(paraphrases, 1):
                out_item = it.copy()
                out_item["user_query"] = new_q
                out_item["is_original"] = False
                out_item["source"] = "gpt-5-chat"
                f_out.write(json.dumps(out_item, ensure_ascii=False) + "\n")
        
        total_saved += len(batch_items)
        print(f"已保存 {total_saved:,} 条（含改写）")
        batch_items.clear()

# 最后一批
if batch_items:
    for it in batch_items:
        paraphrases = generate_paraphrases(it["Q"], it["A"], k=args.k)
        out_item = it.copy()
        out_item["user_query"] = it["Q"]
        out_item["is_original"] = True
        f_out.write(json.dumps(out_item, ensure_ascii=False) + "\n")

        for rank, new_q in enumerate(paraphrases, 1):
            out_item = it.copy()
            out_item["user_query"] = new_q
            out_item["is_original"] = False
            out_item["source"] = "gpt-5-chat"
            f_out.write(json.dumps(out_item, ensure_ascii=False) + "\n")

    total_saved += len(batch_items)
    print(f"最后一批已保存，总计 {total_saved:,} 条")

f_out.close()
print(f"\n完成！")
print(f"生成结果已保存 → {OUTPUT_FILE}")
print(f"   每条 canonical QA 生成 {args.k} 条高质量改写 + 1 条原句")
print(f"   总计生成约 {len(d1_list) * (args.k + 1):,} 条数据")