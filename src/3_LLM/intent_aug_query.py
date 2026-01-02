import os
import json
import re
import torch
import argparse
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------- ARGUMENTS -------------------
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--input_file", type=str, default="judgement_results/llm_judgments.jsonl")
parser.add_argument("--output_file", type=str, default="judgement_results/llm_judgments_with_intent_label_new1.jsonl")
args = parser.parse_args()

# ------------------- PROMPT DESIGN -------------------
# Using a clear structure that forces the model to start with '['
KEYWORD_PROMPT_TEMPLATE = """ Summarize the following query into a two word 'Action-Objective' style intent label describing the user intention. Constraint: Return ONLY an 'Action-Objective' intent label, only with action and objective. No markdown, no explanation.

Example 1:
Query: "How do I reset my login password?"
Intent: Reset-password

Example 2:
Query: "What are the interest rates for a fixed deposit?"
Intent: Inquire-interest

Query: "{user_query}"
Intent: """

# ------------------- CLEANING LOGIC -------------------
def extract_clean_intent(raw_text):
    """
    Extracts the Action-Objective intent label (e.g., 'Reset-password') from LLM output.
    Handles common issues: missing quotes, markdown, trailing text.
    """
    text = raw_text.strip()
    
    # Remove markdown if present
    text = re.sub(r'```json|```', '', text).strip()
    
    # Direct match for single intent string (quoted or unquoted)
    match = re.search(r'([A-Za-z][A-Za-z-]*)(?:-[A-Za-z][A-Za-z-]*)?', text)
    if match:
        intent = match.group(1) + (text[match.end():match.end()+1] if text[match.end():match.end()+1] == '-' else '')
        if '-' in text[match.end():]:
            parts = text[match.end()+1:].split()[0]
            intent += parts.split()[0] if parts else ''
        return intent.strip()
    
    # Fallback: take first capitalized phrase before any junk
    match = re.search(r'^([A-Z][a-z-]+(?:-[A-Z][a-z-]+)?)', text)
    if match:
        return match.group(1).strip()
    
    # Last resort: clean and capitalize first meaningful word pair
    cleaned = re.sub(r'[^A-Za-z-]', ' ', text).strip()
    words = [w for w in cleaned.split() if len(w) > 2][:3]
    if words:
        return '-'.join([words[0].capitalize()] + [w.capitalize() for w in words[1:]])
    
    return "Unknown-intent"

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
        intent = extract_clean_intent(generated_text)  # Changed function
        results.append(intent)
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
                row["query_intent"] = kws
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            batch.clear()

if batch:
    keyword_lists = generate_keywords_batch(batch)
    for row, kws in zip(batch, keyword_lists):
        row["query_intent"] = kws
        f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

f_out.close()
print(f"Success. Data saved to {args.output_file}")