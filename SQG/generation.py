# paraphrase_train_test_robust.py
# 断点续跑 + 每条处理完立刻保存 + 永远不会丢数据

import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import torch
import argparse
import hashlib

# ------------------- 参数 -------------------
parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, choices=["train", "test"], required=True)
parser.add_argument("--k", type=int, default=6)
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args()

# ------------------- 配置 -------------------
DATA_DIR = "data"
OUTPUT_DIR = "data_new"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_FILE = f"{DATA_DIR}/{args.split}.csv"
OUTPUT_FILE = f"{OUTPUT_DIR}/{args.split}_paraphrased.csv"

# ------------------- 模型加载（本地） -------------------
LOCAL_MODEL_DIR = os.path.expanduser("~/autodl-tmp/models")
LLAMA_PATH = os.path.join(LOCAL_MODEL_DIR, "models--meta-llama--Meta-Llama-3.1-8B-Instruct", "snapshots", "main")

tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH, local_files_only=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    LLAMA_PATH, local_files_only=True, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

selector = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ------------------- 加载数据 -------------------
df = pd.read_csv(INPUT_FILE)
print(f"原始数据: {len(df):,} 条")

# ------------------- 断点续跑核心：已处理 ID 集合 -------------------
def get_processed_ids(output_path):
    if not os.path.exists(output_path):
        return set()
    try:
        processed_df = pd.read_csv(output_path)
        # 用 original_query + true_a_i 作为唯一标识
        return set(processed_df['original_query'] + "|||SPLIT|||" + processed_df['true_a_i'])
    except:
        return set()

processed_ids = get_processed_ids(OUTPUT_FILE)
print(f"检测到已处理: {len(processed_ids):,} 条记录 → 将跳过")

# ------------------- 打开文件用于追加写入 -------------------
output_file = open(OUTPUT_FILE, 'a', encoding='utf-8', buffering=1)  # 行缓冲，立刻写入
first_write = len(processed_ids) == 0
if first_write:
    # 写表头
    pd.DataFrame(columns=['user_query','true_a_i','true_q_i','dataset','original_query','paraphrase_rank','is_paraphrased']).to_csv(output_file, index=False)

# ------------------- 生成函数（不变） -------------------
def generate_batch_paraphrases(questions, answers, k=6):
    prompts = [
        f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Rewrite the question in {k} differently phrased similar questions while keeping the exact same semantic meaning.
Rules: No numbering, no quotes, no explanation, natural English only, exactly {k} lines.
Question: {q}
Answer (intent): {a}
Generate {k} similar questions:
""" for q, a in zip(questions, answers)
    ]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.9,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )

    responses = []
    for out in outputs:
        text = tokenizer.decode(out, skip_special_tokens=True)
        generated = text.split("assistant<|end_header_id|>")[-1].strip()
        lines = [re.sub(r'^\d+[\.\)]\s*', '', l.strip()) for l in generated.split('\n') if l.strip()]
        lines = [l for l in lines if l.lower() != questions[0].lower()]  # 粗略过滤
        responses.append(list(dict.fromkeys(lines))[:k+5])
    return responses

def select_top2_diverse(cands, orig):
    if not cands: return [orig, orig]
    if len(cands)==1: return [cands[0]]*2
    orig_emb = selector.encode(orig, convert_to_tensor=True)
    cand_embs = selector.encode(cands, convert_to_tensor=True)
    distances = 1 - util.cos_sim(orig_emb, cand_embs)[0]
    idx = distances.topk(2).indices.cpu().numpy()
    return [cands[i] for i in idx]

# ------------------- 主流程：批处理 + 立刻写入 + 断点续跑 -------------------
batch_q = []
batch_a = []
batch_rows = []

print(f"开始处理 {args.split}（支持断点续跑，每条处理完立即保存）...")

for _, row in tqdm(df.iterrows(), total=len(df), desc="Total"):
    row_id = row['user_query'] + "|||SPLIT|||" + row['true_a_i']
    if row_id in processed_ids:
        continue  # 已处理，跳过

    batch_q.append(row['user_query'])
    batch_a.append(row['true_a_i'])
    batch_rows.append(row)

    if len(batch_q) >= args.batch_size or _ == len(df)-1:
        try:
            batch_cands = generate_batch_paraphrases(batch_q, batch_a, k=args.k)
            for cands, row in zip(batch_cands, batch_rows):
                selected = select_top2_diverse(cands, row['user_query'])
                for rank, new_q in enumerate(selected, 1):
                    new_row = {
                        'user_query': new_q,
                        'true_a_i': row['true_a_i'],
                        'true_q_i': row['true_q_i'],
                        'dataset': row['dataset'],
                        'original_query': row['user_query'],
                        'paraphrase_rank': rank,
                        'is_paraphrased': True
                    }
                    pd.DataFrame([new_row]).to_csv(output_file, mode='a', header=False, index=False)
        except Exception as e:
            print(f"Batch 出错: {e}，跳过这批，继续下一批")

        batch_q.clear()
        batch_a.clear()
        batch_rows.clear()

output_file.close()
print(f"\n完成！数据已安全保存到: {OUTPUT_FILE}")
print("   支持随时中断，下次运行自动从断点继续！")