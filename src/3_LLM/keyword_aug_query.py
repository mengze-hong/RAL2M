import os
import json
import re
import torch
import argparse
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------- ARGUMENTS -------------------
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=196)
parser.add_argument("--input_file", type=str, default="judgement_results/llm_judgments.jsonl")
parser.add_argument("--output_file", type=str, default="judgement_results/llm_judgments_with_keywords.jsonl")
args = parser.parse_args()

# ------------------- PROMPT DESIGN -------------------
# Using a clear structure that forces the model to start with '['
KEYWORD_PROMPT_TEMPLATE = """For the following query, generate a JSON-formatted list of exactly 4 keyphrases that describe its topical focus and user intent.
Constraint: Return ONLY a JSON list of strings. No markdown, no explanation.

Example 1:
Query: "How do I reset my login password?"
Keywords: ["password reset", "forgot credentials", "account access", "security recovery"]

Example 2:
Query: "What are the interest rates for a fixed deposit?"
Keywords: ["fixed deposit", "interest rates", "investment yield", "banking products"]

Query: "{user_query}"
Keywords: ["""

# ------------------- CLEANING LOGIC -------------------
def extract_clean_keywords(raw_text):
    """
    Robustly extracts a list of 4 strings from LLM output.
    Handles markdown blocks, missing brackets, and trailing junk.
    """
    # Force start with [ since we provided it in the prompt
    text = "[" + raw_text.strip()
    
    try:
        # 1. Remove markdown formatting if the model hallucinated it
        text = re.sub(r'```json|```', '', text).strip()
        
        # 2. Extract content between first [ and last ]
        match = re.search(r'(\[.*?\])', text, re.DOTALL)
        if match:
            json_str = match.group(1)
            # Standardize quotes (sometimes models use curly quotes)
            json_str = json_str.replace('“', '"').replace('”', '"')
            keywords = json.loads(json_str)
            if isinstance(keywords, list):
                return [str(k).strip() for k in keywords[:4]]
    except Exception:
        pass

    # Fallback: Regex-based recovery for malformed JSON
    # Finds everything inside quotes
    fallback = re.findall(r'"([^"]*)"', text)
    if not fallback:
        # Last resort: split by comma and strip junk
        fallback = [k.strip() for k in text.replace('[', '').replace(']', '').split(',')]
    
    return [k for k in fallback if len(k) > 1][:4]

# ------------------- MODEL LOADING -------------------
LOCAL_MODEL_DIR = os.path.expanduser("~/autodl-tmp/models")
QWEN_PATH = os.path.join(LOCAL_MODEL_DIR, "Qwen2.5-7B-Instruct")

tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, local_files_only=True, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    QWEN_PATH,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# ------------------- GENERATION FUNCTION -------------------
def generate_keywords_batch(batch_rows):
    prompts = [KEYWORD_PROMPT_TEMPLATE.format(user_query=r['user_query']) for r in batch_rows]
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0.01, # Keep it nearly deterministic
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("]") # Stop immediately at closing bracket
        )

    results = []
    for i, out in enumerate(outputs):
        # Slice out the prompt
        generated_text = tokenizer.decode(out[inputs.input_ids.shape[1]:], skip_special_tokens=True)
        keywords = extract_clean_keywords(generated_text)
        results.append(keywords)
    return results

# ------------------- MAIN EXECUTION -------------------
processed_ids = set()
if os.path.exists(args.output_file):
    with open(args.output_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if "question_id" in obj: processed_ids.add(obj["question_id"])

f_out = open(args.output_file, "a", encoding="utf-8", buffering=1)

batch = []
with open(args.input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in tqdm(lines, desc="Processing"):
        item = json.loads(line)
        if item.get("question_id") in processed_ids: continue

        batch.append(item)
        if len(batch) >= args.batch_size:
            keyword_lists = generate_keywords_batch(batch)
            for row, kws in zip(batch, keyword_lists):
                row["query_keywords"] = kws
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            batch.clear()

if batch:
    keyword_lists = generate_keywords_batch(batch)
    for row, kws in zip(batch, keyword_lists):
        row["query_keywords"] = kws
        f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

f_out.close()
print(f"Success. Data saved to {args.output_file}")