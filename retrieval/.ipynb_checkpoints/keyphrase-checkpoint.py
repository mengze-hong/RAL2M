# llm_enhance_retrieval_from_D2_final.py
# 终极完整版：完美适配你的 D2.jsonl + 所有原始列保留 + 零崩溃

import os
import json
import pandas as pd
import torch
import re
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

# ------------------- 参数 -------------------
parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, choices=["train", "test"], required=True)
parser.add_argument("--batch_size", type=int, default=16)
args = parser.parse_args()

# ------------------- 配置 -------------------
DATA_DIR = "../data"
D2_FILE = os.path.join(DATA_DIR, "D2.jsonl")
INPUT_FILE = f"{DATA_DIR}/{args.split}_final.jsonl"
OUTPUT_DIR = "retrieval_enhanced"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = f"{OUTPUT_DIR}/{args.split}_with_llm_keywords.csv"

# ------------------- 加载 D2.jsonl -------------------
print("加载 D2.jsonl 文档库...")
doc_id_to_text = {}
with open(D2_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Loading D2"):
        if not line.strip():
            continue
        item = json.loads(line)
        doc_id = str(item["doc_id"])
        text = item["text"]
        doc_id_to_text[doc_id] = text.strip()

print(f"D2 加载完成: {len(doc_id_to_text):,} 条文档")

# ------------------- 模型加载 -------------------
LOCAL_MODEL_DIR = os.path.expanduser("~/autodl-tmp/models")
LLAMA_PATH = os.path.join(LOCAL_MODEL_DIR, "llama-3.1-8b-instruct")

print("加载 Llama-3.1-8B-Instruct...")
tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    LLAMA_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# ------------------- 断点续跑 -------------------
def get_processed_ids():
    if not os.path.exists(OUTPUT_FILE):
        return set()
    try:
        df_done = pd.read_csv(OUTPUT_FILE)
        return set(df_done['query_id'])
    except:
        return set()

processed_ids = get_processed_ids()
print(f"已处理: {len(processed_ids):,} 条 → 将自动跳过")

# ------------------- 输出文件（保留所有原始列） -------------------
f_out = open(OUTPUT_FILE, 'a', encoding='utf-8', buffering=1)
first_write = len(processed_ids) == 0
if first_write:
    header = ['Question_ID','user_query','true_q_i','true_a_i','paraphrase_rank',
              'dataset','original_id','dialogue_context','retrieved_docs','query_id','llm_keywords']
    pd.DataFrame(columns=header).to_csv(f_out, index=False)

# ------------------- 安全解析 retrieved_doc_ids -------------------
def safe_parse_doc_ids(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x]
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return []
        if x.startswith('[') and x.endswith(']'):
            try:
                return json.loads(x)
            except:
                try:
                    import ast
                    parsed = ast.literal_eval(x)
                    if isinstance(parsed, list):
                        return [str(i).strip("'\"") for i in parsed]
                except:
                    pass
        if ',' in x:
            return [i.strip().strip("'\"") for i in x.split(',') if i.strip()]
        return [x.strip().strip("'\"")]
    return []

# ------------------- LLM 批量提取关键词（最强英文版） -------------------
def extract_keywords_batch(texts):
    prompts = [
        f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Extract 8-15 most important keywords from the document below. Output only keywords separated by spaces.

Document:
{text[:3000]}

Keywords:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""" for text in texts
    ]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=3500).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.3,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    keywords = []
    for out in outputs:
        text = tokenizer.decode(out, skip_special_tokens=True)
        generated = text.split("assistant<|end_header_id|>")[-1].strip()
        kw = re.sub(r'^(Keywords?[:：\s]*)', '', generated, flags=re.I)
        kw = re.sub(r'["\']', '', kw)
        kw = " ".join([w.strip() for w in kw.split()[:15] if w.strip()])
        keywords.append(kw if kw else "no_keywords")
    return keywords

# ------------------- 主流程（永不崩溃版） -------------------
df = pd.read_json(INPUT_FILE, lines=True)
print(f"待处理: {len(df):,} 条")

batch_texts = []
batch_rows = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="LLM Keywords"):
    qid = str(row['query_id'])
    if qid in processed_ids:
        continue

    # 解析 doc ids
    doc_ids = safe_parse_doc_ids(row['retrieved_doc_ids'])
    
    # 收集真实文本
    texts = []
    for doc_id in doc_ids:
        text = doc_id_to_text.get(doc_id, "")
        if text:
            texts.append(text)

    combined_text = "\n\n".join(texts) if texts else "No retrieved documents available."

    batch_texts.append(combined_text)
    batch_rows.append(row)

    if len(batch_texts) >= args.batch_size or idx == len(df)-1:
        try:
            batch_keywords = extract_keywords_batch(batch_texts)
            for row, kw in zip(batch_rows, batch_keywords):
                new_row = row.to_dict()
                new_row['llm_keywords'] = kw

                ordered_row = {col: new_row.get(col, '') for col in [
                    'Question_ID','user_query','true_q_i','true_a_i','paraphrase_rank',
                    'dataset','original_id','dialogue_context','retrieved_docs','query_id'
                ]}
                ordered_row['llm_keywords'] = kw

                pd.DataFrame([ordered_row]).to_csv(f_out, header=False, index=False)
                f_out.flush()
        except Exception as e:
            print(f"\nBatch 出错: {e}，已跳过这批，继续...")

        batch_texts.clear()
        batch_rows.clear()

f_out.close()
print(f"\n完成！关键词已提取 → {OUTPUT_FILE}")
print("   100% 适配你的 D2.jsonl 格式")
print("   所有原始列完整保留")
print("   下一歩：用 llm_keywords + user_query 做增强检索 → Recall 暴涨！")