import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix

# ==================== HYPERPARAMETERS & PATHS ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
JSONL_PATH = "../5_Analysis/raw_results/complete_judge_result.jsonl"
SBERT_TRAIN_EMB = "sbert_cache/train_emb.npy"
SBERT_TEST_EMB = "sbert_cache/test_emb.npy"
SAVE_DIR = "ablation_model_count"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


HIDDEN_DIM = 512
DROPOUT = 0.4
BATCH_SIZE = 512 
EPOCHS = 200
LR = 1e-4 # Slightly lower for deeper refinement
GRAD_CLIP = 0.3 

VI_TRAIN_ITERS = 20 # Increased for deeper convergence
VI_EVAL_ITERS = 60
MC_SAMPLES_TRAIN = 128
MC_SAMPLES_EVAL = 1024 

LABEL_SMOOTHING = 0.05
FOCAL_GAMMA = 2.5 # Increased focus on hard cases
KL_WEIGHT = 0.001 # Information bottleneck weight

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

ALL_JUDGES = [
    'mistral',
    'gemma',
    'chatglm',
    'qwen',
    'llama',
]

results = {}
for num in [2, 3, 4]:
    print(f"Running ablation with {num} models:")
    JUDGE_COLS = ALL_JUDGES[:num]
    print(JUDGE_COLS)
    K = num

    EPOCHS = num * 50
    df = pd.read_json(JSONL_PATH, lines=True)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    
    e_train = torch.from_numpy(np.load(SBERT_TRAIN_EMB)).float()
    S_train = torch.from_numpy(train_df[JUDGE_COLS].values).float()
    y_train = torch.from_numpy(train_df["y_true"].values).long()
    
    e_test = torch.from_numpy(np.load(SBERT_TEST_EMB)).float().to(DEVICE)
    S_test = torch.from_numpy(test_df[JUDGE_COLS].values).float().to(DEVICE)
    y_test_cpu = test_df["y_true"].values
    
    train_loader = DataLoader(TensorDataset(e_train, S_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    
    
    mu_net = MuNet(e_train.shape[1], K).to(DEVICE)
    sigma_net = SigmaNet(e_train.shape[1], K).to(DEVICE)
    potentials = InteractionPotential(K, e_train.shape[1]).to(DEVICE)
    
    optimizer = optim.AdamW(list(mu_net.parameters()) + list(sigma_net.parameters()) + list(potentials.parameters()), 
                            lr=LR, weight_decay=3e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
    
    # ==================== TRAINING LOOP ====================
    best_acc = 0.0
    best_hallu = 1.0
    best_epoch = 0
    
    for epoch in range(EPOCHS):
        temp = max(0.5, 1.0 - (epoch / 80.0))
        mu_net.train(); sigma_net.train(); potentials.train()
        total_loss = 0
        
        for e_b, S_b, y_b in train_loader:
            e_b, S_b, y_b = e_b.to(DEVICE), S_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            m_z, v_z, g_z, mu_b, sig_b = run_vi_v3(e_b, S_b, mu_net, sigma_net, potentials, VI_TRAIN_ITERS, temp)
            
            eps = torch.randn(m_z.shape[0], MC_SAMPLES_TRAIN, K, device=DEVICE)
            z_s = m_z.unsqueeze(1) + torch.sqrt(v_z + 1e-6).unsqueeze(1) * eps
            inter_z = potentials.interaction(z_s * g_z.unsqueeze(1)).squeeze(-1)
            mc_probs = F.softmax(inter_z.unsqueeze(-1) * potentials.theta_lambda, dim=-1)[:, :, 1]
            
            loss = criterion_advanced(mc_probs, y_b, v_z, mu_b, sig_b, epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], GRAD_CLIP)
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
            
        print(f"Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.4f} | Hallu: {hallu:.4f}")
        
        # Save Best Model
        if acc > best_acc or (np.isclose(acc, best_acc, atol=1e-4) and hallu < best_hallu):
            best_acc = acc
            best_hallu = hallu
            best_epoch = epoch + 1
            torch.save(mu_net.state_dict(), os.path.join(SAVE_DIR, f"{num}_best_mu_net.pth"))
            torch.save(sigma_net.state_dict(), os.path.join(SAVE_DIR, f"{num}_best_sigma_net.pth"))
            torch.save(potentials.state_dict(), os.path.join(SAVE_DIR, f"{num}_best_potentials.pth"))
    
    print(f"\nTraining Complete. Best Epoch: {best_epoch} | Best Acc: {best_acc:.4f} | Best Hallu: {best_hallu:.4f}")
    
    # ==================== FINAL EVALUATION ====================
    # Load best weights
    mu_net.load_state_dict(torch.load(os.path.join(SAVE_DIR, f"{num}_best_mu_net.pth")))
    sigma_net.load_state_dict(torch.load(os.path.join(SAVE_DIR, f"{num}_best_sigma_net.pth")))
    potentials.load_state_dict(torch.load(os.path.join(SAVE_DIR, f"{num}_best_potentials.pth")))
    
    mu_net.eval(); sigma_net.eval(); potentials.eval()
    with torch.no_grad():
        m_z_t, v_z_t, g_z_t, _, _ = run_vi_v3(e_test, S_test, mu_net, sigma_net, potentials, VI_EVAL_ITERS, temp=0.5)
        eps_t = torch.randn(m_z_t.shape[0], MC_SAMPLES_EVAL, K, device=DEVICE)
        z_s_t = m_z_t.unsqueeze(1) + torch.sqrt(v_z_t + 1e-6).unsqueeze(1) * eps_t
        inter_t = potentials.interaction(z_s_t * g_z_t.unsqueeze(1)).squeeze(-1)
        test_probs = F.softmax(inter_t.unsqueeze(-1) * potentials.theta_lambda, dim=-1)[:, :, 1].mean(dim=1).cpu().numpy()
        
        final_acc, final_hallu = get_metrics(y_test_cpu, test_probs)
        cm = confusion_matrix(y_test_cpu, (test_probs > 0.5).astype(int))
    
    print(f"\nFinal Best Model Results:")
    print(f"Accuracy: {final_acc:.4f} | Hallucination Rate: {final_hallu:.4f}")
    print("Confusion Matrix:\n", cm)