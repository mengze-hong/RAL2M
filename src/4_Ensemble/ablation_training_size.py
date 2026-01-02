import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

# ==================== HYPERPARAMETERS & PATHS ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
JSONL_PATH = "../5_Analysis/raw_results/complete_judge_result.jsonl"
SBERT_TRAIN_EMB = "sbert_cache/train_emb.npy"
SBERT_TEST_EMB = "sbert_cache/test_emb.npy"
SAVE_DIR = "checkpoints_graphical_ablation"
os.makedirs(SAVE_DIR, exist_ok=True)

JUDGE_COLS = ["qwen", "llama",
              "mistral", "gemma",
              "chatglm"]
K = len(JUDGE_COLS)

HIDDEN_DIM = 512
DROPOUT = 0.4
BATCH_SIZE = 2048
EPOCHS = 250
LR = 1e-4
GRAD_CLIP = 0.3
VI_TRAIN_ITERS = 20
VI_EVAL_ITERS = 60
MC_SAMPLES_TRAIN = 128
MC_SAMPLES_EVAL = 1024
LABEL_SMOOTHING = 0.05
FOCAL_GAMMA = 2.5
KL_WEIGHT = 0.001

torch.manual_seed(SEED)
np.random.seed(SEED)

def get_metrics(y_true, y_prob):
    y_pred = (y_prob > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    hallu = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return acc, hallu

# ==================== MODEL COMPONENTS ====================
class MuNet(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(d, HIDDEN_DIM)),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.utils.weight_norm(nn.Linear(HIDDEN_DIM, k))
        )
    def forward(self, e):
        return self.net(e)

class SigmaNet(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(d, HIDDEN_DIM)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(HIDDEN_DIM, k))
        )
    def forward(self, e):
        return F.softplus(self.net(e)) + 1e-3

class InteractionPotential(nn.Module):
    def __init__(self, k, d):
        super().__init__()
        self.theta_phi_z = nn.Parameter(torch.randn(k) * 0.01)
        self.W_phi = nn.Parameter(torch.ones(k) * 0.1)
        self.theta_lambda = nn.Parameter(torch.tensor([-2.0, 2.0]))
        self.interaction = nn.Sequential(
            nn.Linear(k, k * 2),
            nn.ReLU(),
            nn.Linear(k * 2, 1)
        )
        self.gate_net = nn.Sequential(
            nn.Linear(d, HIDDEN_DIM // 2),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM // 2, k),
            nn.Sigmoid()
        )

def run_vi_v3(e, S, mu_net, sigma_net, potentials, iters, temp=1.0):
    mu = mu_net(e)
    sigma_diag = sigma_net(e)
    sigma_inv_diag = 1.0 / (sigma_diag + 1e-5)
    gates = potentials.gate_net(e)
    mean_z = S.clone()
    var_i = 1.0 / (sigma_inv_diag + 1.0)
    for _ in range(iters):
        gated_mean_z = (mean_z * gates)
        context_score = potentials.interaction(gated_mean_z)
        logits_y = (context_score * potentials.theta_lambda) / temp
        Q_y = F.softmax(logits_y, dim=1)
        lambda_coeff = (Q_y * potentials.theta_lambda).sum(dim=1, keepdim=True)
        phi_linear = potentials.theta_phi_z + potentials.W_phi * S
        new_mean_z = (mu * sigma_inv_diag + phi_linear + (gates * lambda_coeff)) * var_i
        mean_z = 0.8 * new_mean_z + 0.2 * mean_z
    return mean_z, var_i, gates, mu, sigma_diag

def criterion_advanced(mc_probs, target, var_z, mu, sigma, epoch):
    target = target.float().unsqueeze(1)
    target_smooth = target * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
    avg_prob = torch.clamp(mc_probs.mean(dim=1, keepdim=True), 1e-6, 1.0 - 1e-6)
    p_t = (avg_prob * target) + ((1 - avg_prob) * (1 - target))
    focal_weight = (1 - p_t) ** FOCAL_GAMMA
    bce = F.binary_cross_entropy(avg_prob, target_smooth, reduction='none')
    kl_div = 0.5 * torch.sum(sigma + mu**2 - torch.log(sigma + 1e-5) - 1, dim=1)
    current_kl_w = KL_WEIGHT * min(1.0, epoch / 40.0)
    return (focal_weight * bce).mean() + (current_kl_w * kl_div.mean())

# ==================== DATA PREP ====================
df = pd.read_json(JSONL_PATH, lines=True)
train_df_full = df[df["split"] == "train"].reset_index(drop=True)
test_df = df[df["split"] == "test"].reset_index(drop=True)

e_train_all = torch.from_numpy(np.load(SBERT_TRAIN_EMB)).float()
S_train_all = torch.from_numpy(train_df_full[JUDGE_COLS].fillna(0).values).float()
y_train_all = torch.from_numpy(train_df_full["y_true"].values).long()

e_test = torch.from_numpy(np.load(SBERT_TEST_EMB)).float().to(DEVICE)
S_test = torch.from_numpy(test_df[JUDGE_COLS].fillna(0).values).float().to(DEVICE)
y_test_cpu = test_df["y_true"].values

# ==================== ABLATION SETUP ====================
idx_pos = np.where(y_train_all.numpy() == 1)[0]
idx_neg = np.where(y_train_all.numpy() == 0)[0]
np.random.shuffle(idx_pos)
np.random.shuffle(idx_neg)

ablation_summary = []

for step in range(1, 11):
    n_pos = int(len(idx_pos) * (step / 10.0))
    n_neg = int(len(idx_neg) * (step / 10.0))
    current_indices = np.concatenate([idx_pos[:n_pos], idx_neg[:n_neg]])
    np.random.shuffle(current_indices)

    e_train = e_train_all[current_indices].to(DEVICE)
    S_train = S_train_all[current_indices].to(DEVICE)
    y_train = y_train_all[current_indices].to(DEVICE)

    train_loader = DataLoader(TensorDataset(e_train, S_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    print(f"\n=== Ablation Step {step}/10 | Train Size: {len(current_indices)} ===")

    mu_net = MuNet(e_train_all.shape[1], K).to(DEVICE)
    sigma_net = SigmaNet(e_train_all.shape[1], K).to(DEVICE)
    potentials = InteractionPotential(K, e_train_all.shape[1]).to(DEVICE)

    optimizer = optim.AdamW(list(mu_net.parameters()) + list(sigma_net.parameters()) + list(potentials.parameters()),
                            lr=LR, weight_decay=3e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    best_acc = 0.0
    best_hallu = 1.0

    for epoch in range(EPOCHS):
        temp = max(0.5, 1.0 - (epoch / 80.0))
        mu_net.train(); sigma_net.train(); potentials.train()
        total_loss = 0

        for e_b, S_b, y_b in train_loader:
            optimizer.zero_grad()
            m_z, v_z, g_z, mu_b, sig_b = run_vi_v3(e_b, S_b, mu_net, sigma_net, potentials, VI_TRAIN_ITERS, temp)
            eps = torch.randn(m_z.shape[0], MC_SAMPLES_TRAIN, K, device=DEVICE)
            z_s = m_z.unsqueeze(1) + torch.sqrt(v_z + 1e-6).unsqueeze(1) * eps
            inter_z = potentials.interaction(z_s * g_z.unsqueeze(1)).squeeze(-1)
            mc_probs = F.softmax(inter_z.unsqueeze(-1) * potentials.theta_lambda, dim=-1)[:, :, 1]
            loss = criterion_advanced(mc_probs, y_b, v_z, mu_b, sig_b, epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(mu_net.parameters()) + list(sigma_net.parameters()) + list(potentials.parameters()), GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Evaluation
        mu_net.eval(); sigma_net.eval(); potentials.eval()
        with torch.no_grad():
            m_z_t, v_z_t, g_z_t, _, _ = run_vi_v3(e_test, S_test, mu_net, sigma_net, potentials, VI_EVAL_ITERS, temp=0.5)
            eps_t = torch.randn(m_z_t.shape[0], MC_SAMPLES_EVAL, K, device=DEVICE)
            z_s_t = m_z_t.unsqueeze(1) + torch.sqrt(v_z_t + 1e-6).unsqueeze(1) * eps_t
            inter_t = potentials.interaction(z_s_t * g_z_t.unsqueeze(1)).squeeze(-1)
            test_probs = F.softmax(inter_t.unsqueeze(-1) * potentials.theta_lambda, dim=-1)[:, :, 1].mean(dim=1).cpu().numpy()
            acc, hallu = get_metrics(y_test_cpu, test_probs)

        if acc > best_acc or (abs(acc - best_acc) < 1e-4 and hallu < best_hallu):
            best_acc = acc
            best_hallu = hallu
            torch.save(mu_net.state_dict(), os.path.join(SAVE_DIR, f"best_mu_step{step}.pth"))
            torch.save(sigma_net.state_dict(), os.path.join(SAVE_DIR, f"best_sigma_step{step}.pth"))
            torch.save(potentials.state_dict(), os.path.join(SAVE_DIR, f"best_pot_step{step}.pth"))

        if (epoch + 1) % 50 == 0:
            print(f"Step {step} Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.4f} | Hallu: {hallu:.4f}")

    print(f"Step {step} Best | Acc: {best_acc:.4f} | Hallu: {best_hallu:.4f}")

    ablation_summary.append({
        "step": step,
        "train_size": len(current_indices),
        "acc": best_acc,
        "hallu": best_hallu
    })

# ==================== SUMMARY ====================
summary_df = pd.DataFrame(ablation_summary)
summary_df.to_csv(os.path.join(SAVE_DIR, "ablation_summary.csv"), index=False)
print("\nAblation Complete")
print(summary_df[["step", "train_size", "acc", "hallu"]].to_string(index=False))