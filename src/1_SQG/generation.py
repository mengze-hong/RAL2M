import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import torch
import argparse
import re
import json

# ------------------- 参数 -------------------
parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args()

INPUT_FILE = "../../data/D1.jsonl"
OUTPUT_DIR = "generated_queries"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = f"{OUTPUT_DIR}/SQG_llama_{args.k}.jsonl"

# ------------------- 模型加载 -------------------
LOCAL_MODEL_DIR = os.path.expanduser("~/autodl-tmp/models") # model path
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
print("加载数据...")
df = pd.read_json(INPUT_FILE, lines=True)

df = df.drop_duplicates(subset=['Q'], keep='first').copy()

print(f"数据量: {len(df):,} 条")

f_out = open(OUTPUT_FILE, 'a', encoding='utf-8', buffering=1)
header = ['question_id', 'user_query', 'paraphrase_rank']
pd.DataFrame(columns=header).to_csv(f_out, index=False)

# ------------------- 生成函数 -------------------
def generate_batch_paraphrases(questions, answers, k=10):
    prompts = [
        f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
You are an expert at paraphrasing questions while preserving the exact meaning.

Task: Rewrite the following question in EXACTLY {k} completely different ways.
Rules (must strictly obey):
- Every line must be a complete, natural English question
- Every line must end with a question mark (?)
- No numbering, no bullets, no quotes, no explanations, no extra text
- Do NOT repeat the original question
- You must read and understand the original question carefully, and use the reference answer to assist in understanding.
- Each new question must capture exactly the same meaning of the original question.
- Each new question must also correspond to the same reference answer.

Original Question: {q}
Reference answer (just for context, do not mention it): {a}

Now output EXACTLY {k} paraphrased questions, one per line:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""" for q, a in zip(questions, answers)
    ]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )

    results = []
    for i, out in enumerate(outputs):
        text = tokenizer.decode(out, skip_special_tokens=True)
        # 提取 assistant 后的内容
        if "assistant<|end_header_id|>" in text:
            generated = text.split("assistant<|end_header_id|>")[-1].strip()
        else:
            generated = text.split("assistant")[-1].strip()

        # 按行分割 
        lines = []
        for line in generated.split('\n'):
            l = line.strip()
            if not l:
                continue
            # 去除编号、引号、前缀
            l = re.sub(r'^\d+[\.\)\]\s]*', '', l)
            l = re.sub(r'^["\']|["\']$', '', l)
            l = re.sub(r'^(Answer|Version|Rewrite)[:：\s]*', '', l, flags=re.I)
            if len(l) > 10 and l.lower() != questions[i].lower():
                lines.append(l)

        # 去重
        lines = list(dict.fromkeys(lines))
        if len(lines) < k:
            lines += [lines[0]] * (k - len(lines))
        results.append(lines[:k])

    return results


def select_top10_diverse(cands, orig):
    k = 10
    if len(cands) == 0:
        return [orig] * k
    if len(cands) <= k:
        return cands + [cands[0]] * (k - len(cands)) if cands else [orig] * k
    
    orig_emb = selector.encode(orig, convert_to_tensor=True)
    cand_embs = selector.encode(cands, convert_to_tensor=True)
    distances = 1 - util.cos_sim(orig_emb, cand_embs)[0]
    top_idx = distances.topk(k).indices.cpu().numpy()
    return [cands[i] for i in top_idx]


batch_q = []
batch_a = []
batch_rows = []  

print(f"开始处理...")

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Paraphrasing"):
    row_dict = row.to_dict()
    
    batch_q.append(row['Q'])
    batch_a.append(row['A'])
    batch_rows.append(row_dict)

    if len(batch_q) >= args.batch_size or idx == len(df)-1:
        try:
            batch_cands = generate_batch_paraphrases(batch_q, batch_a, k=args.k)
            for cands, orig_row in zip(batch_cands, batch_rows):
                # selected = select_top10_diverse(cands, orig_row['user_query'])
                selected = cands
                for rank, new_q in enumerate(selected, 1):
                    out_row = orig_row.copy()
                    out_row['user_query'] = new_q
                    f_out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    f_out.flush()
        except Exception as e:
            print(f"\nBatch 出错: {e}，跳过这批")

        batch_q.clear()
        batch_a.clear()
        batch_rows.clear()

f_out.close()
print(f"\n完成！")
print(f"输出文件 → {OUTPUT_FILE}")