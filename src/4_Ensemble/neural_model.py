import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


JSONL_PATH = "../5_Analysis/raw_results/complete_judge_result.jsonl"
SAVE_DIR = "neural_model"
os.makedirs(SAVE_DIR, exist_ok=True)

JUDGE_COLS = ['qwen', 'llama', 'mistral', 'gemma', 'chatglm']

K = len(JUDGE_COLS)

EPOCHS = 50
LR = 1e-3
TRAIN_BATCH = 2048


# --- Data Loading ---
print("Loading data...")
with open(JSONL_PATH) as f:
    df = pd.DataFrame(json.loads(l) for l in f if l.strip())

train_df = df[df["split"] == "train"].reset_index(drop=True)
test_df = df[df["split"] == "test"].reset_index(drop=True)

S_train = torch.tensor(train_df[JUDGE_COLS].fillna(0).values, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(train_df["y_true"].values, dtype=torch.float32).to(DEVICE)
S_test = torch.tensor(test_df[JUDGE_COLS].fillna(0).values, dtype=torch.float32).to(DEVICE)
y_test_cpu = test_df["y_true"].astype(bool).values

e_train = torch.tensor(np.load("sbert_cache/train_emb.npy"), dtype=torch.float32).to(DEVICE)
e_test = torch.tensor(np.load("sbert_cache/test_emb.npy"), dtype=torch.float32).to(DEVICE)
D = e_train.shape[1]


class ResidualRAL2M(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        # Simplified Gate: 1 hidden layer to decide how much to trust each judge
        self.gate = nn.Sequential(
            nn.Linear(d, 64),
            nn.ReLU(),
            nn.Linear(64, k),
            nn.Sigmoid() 
        )
        # Simple classifier that looks at the weighted judge scores
        self.classifier = nn.Linear(k, 1)

    def forward(self, e, S):
        g = self.gate(e)          # Query-dependent weights for each judge
        weighted_S = S * g        # Scale judge outputs by their predicted reliability
        return torch.sigmoid(self.classifier(weighted_S)).squeeze(-1)

model = ResidualRAL2M(D, K).to(DEVICE)

# POS_WEIGHT calculation: Helps with Recall (Pos Correct%)
pos_count = (y_train == 1).sum()
neg_count = (y_train == 0).sum()
pw = neg_count / pos_count
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

# --- Synced Evaluation Function ---
@torch.no_grad()
def evaluate_aggregator(model, S, e, y_true, threshold=0.5):
    model.eval()
    probs = model(e, S)
    accepted = (probs > threshold).cpu().numpy().astype(bool)
    
    acc = (accepted == y_true).mean()
    n_neg = (~y_true).sum()
    neg_correct = (~accepted & ~y_true).sum()
    hallu = 1.0 - (neg_correct / n_neg) if n_neg > 0 else 0.0
    return {"Acc": acc, "Hallu": hallu}

# --- Baseline Performance ---
votes_test = (S_test.cpu().numpy() >= 0.5).sum(axis=1)
baseline_accepted = votes_test >= 3
base_acc = (baseline_accepted == y_test_cpu).mean()
base_hallu = 1.0 - ((~baseline_accepted & ~y_test_cpu).sum() / (~y_test_cpu).sum())

print(f"\nBASELINE (MV>=3) | Acc: {base_acc:.4f} | Hallu: {base_hallu:.1%}")

# --- Training Loop ---
for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(train_df))
    for i in range(0, len(train_df), TRAIN_BATCH):
        idx = perm[i:i+TRAIN_BATCH]
        optimizer.zero_grad()
        probs = model(e_train[idx], S_train[idx])
        # Weighted BCE to force model to learn positive samples
        loss = F.binary_cross_entropy(probs, y_train[idx], 
                                      weight=torch.where(y_train[idx]==1, pw, 1.0))
        loss.backward()
        optimizer.step()
    

torch.save(model.state_dict(), os.path.join(SAVE_DIR, "neural_model.pt"))

# --- Final Inference & File Appending ---
print("\nFinalizing Inference...")
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "neural_model.pt")))
model.eval()

with torch.no_grad():
    # Final threshold shift to 0.45 to recover correct candidates (Improve Recall)
    all_probs_train = model(e_train, S_train)
    all_probs_test = model(e_test, S_test)
    
    train_df['neural_model'] = (all_probs_train > 0.5).cpu().numpy().astype(int)
    test_df['neural_model'] = (all_probs_test > 0.5).cpu().numpy().astype(int)

full_output = pd.concat([train_df, test_df], axis=0).sort_index()
output_path = os.path.join(SAVE_DIR, "final_results_with_neural_model.jsonl")
full_output.to_json(output_path, orient='records', lines=True, force_ascii=False)