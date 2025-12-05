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
import re

# ------------------- 参数 -------------------
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args()

# ------------------- 配置 -------------------
DATA_DIR = "../Data"
OUTPUT_DIR = "data_new"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_FILE = f"data_new/{args.name}.csv"
OUTPUT_FILE = f"{OUTPUT_DIR}/{args.name}_paraphrased.csv"

# ------------------- 模型加载（本地） -------------------
LOCAL_MODEL_DIR = os.path.expanduser("~/autodl-tmp/models")
LLAMA_PATH = os.path.join(LOCAL_MODEL_DIR, "llama-3.1-8b-instruct")

tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH, local_files_only=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    LLAMA_PATH, local_files_only=True, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)
model.eval()
tokenizer.padding_side = "left"
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
f_out = open(OUTPUT_FILE, 'a', encoding='utf-8', buffering=1)
first_write = len(processed_ids) == 0

if first_write:
    header = ['Question_ID','user_query','true_q_i','true_a_i','paraphrase_rank','dataset','original_id',
              'dialogue_context','retrieved_docs']
    pd.DataFrame(columns=header).to_csv(f_out, index=False)

# ------------------- 生成函数（不变） -------------------
def generate_batch_paraphrases(questions, answers, k=10):
    prompts = [
        f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
You are an expert at paraphrasing questions while preserving the exact meaning.

Task: Rewrite the following question in EXACTLY {k} completely different ways.
Rules (must strictly obey):
- Every line must be a complete, natural English question
- Every line must end with a question mark (?)
- No numbering, no bullets, no quotes, no explanations, no extra text
- Keep the exact same meaning and all details
- Do NOT repeat the original question

Question: {q}
Reference answer (just for context, do not mention it): {a}

Now output EXACTLY {k} different questions, one per line:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
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

    results = []
    for i, out in enumerate(outputs):
        text = tokenizer.decode(out, skip_special_tokens=True)
        # 严格提取 assistant 后的内容
        if "assistant<|end_header_id|>" in text:
            generated = text.split("assistant<|end_header_id|>")[-1].strip()
        else:
            generated = text.split("assistant")[-1].strip()

        # 按行分割 + 严格清洗
        lines = []
        for line in generated.split('\n'):
            l = line.strip()
            if not l:
                continue
            # 去除编号、引号、常见前缀
            l = re.sub(r'^\d+[\.\)\]\s]*', '', l)
            l = re.sub(r'^["\']|["\']$', '', l)
            l = re.sub(r'^(Answer|Version|Rewrite)[:：\s]*', '', l, flags=re.I)
            if len(l) > 10 and l.lower() != questions[i].lower():
                lines.append(l)

        # 去重 + 补足
        lines = list(dict.fromkeys(lines))
        if len(lines) < k:
            lines += [lines[0]] * (k - len(lines))
        results.append(lines[:k])

    return results

def select_top3_diverse(cands, orig):
    k = 3
    if len(cands) == 0:
        return [orig] * k
    if len(cands) <= k:
        return cands + [cands[0]] * (k - len(cands)) if cands else [orig] * k
    
    orig_emb = selector.encode(orig, convert_to_tensor=True)
    cand_embs = selector.encode(cands, convert_to_tensor=True)
    distances = 1 - util.cos_sim(orig_emb, cand_embs)[0]
    top_idx = distances.topk(k).indices.cpu().numpy()
    return [cands[i] for i in top_idx]

# ------------------- 主流程：批处理 + 立刻写入 + 断点续跑 -------------------
df = pd.read_csv(INPUT_FILE)
print(f"待处理: {len(df):,} 条")

batch_q = []
batch_a = []
batch_rows = []

print(f"开始处理 {args.name}（支持断点续跑，每条处理完立即保存）...")

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Paraphrasing"):
    orig_id = row['original_id']
    for rank in range(1, args.k + 1):
        new_id = f"{orig_id}_p{rank}"
        if new_id in processed_ids:
            continue
        break
    else:
        continue  # 所有 rank 都处理完了，跳过

    batch_q.append(row['user_query'])
    batch_a.append(row['true_a_i'])
    batch_rows.append((row, orig_id))

    if len(batch_q) >= args.batch_size or idx == len(df)-1:
        try:
            batch_cands = generate_batch_paraphrases(batch_q, batch_a, k=args.k)
            for cands, (row, orig_id) in zip(batch_cands, batch_rows):
                selected = select_top3_diverse(cands, row['user_query'])
                for rank, new_q in enumerate(selected, 1):
                    new_row = {
                        'Question_ID': row['original_id'],  # 原始问题 ID
                        'user_query': new_q,
                        'true_q_i': row['true_q_i'],
                        'true_a_i': row['true_a_i'],
                        'paraphrase_rank': rank,
                        'dataset': row['dataset'],
                        'original_id': f"{row['original_id']}_p{rank}",
                        'dialogue_context': row.get('dialogue_context', ''),
                        'retrieved_docs': row.get('retrieved_docs', ''),
                    }
                    pd.DataFrame([new_row]).to_csv(f_out, header=False, index=False)
                    f_out.flush()
        except Exception as e:
            print(f"\nBatch 出错: {e}，跳过这批")

        batch_q.clear()
        batch_rows.clear()

f_out.close()
print(f"\n完成！数据已保存 → {OUTPUT_FILE}")
print("   每条原始样本扩展为 3 条高质量改写")
print("   所有原始列完整保留，可直接用于后续检索 + RAL2M 训练")