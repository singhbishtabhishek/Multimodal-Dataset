import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df_raw = pd.read_csv("Dataset.csv")
y = df_raw["Labels"].values

X_t = np.load("Features/text_features_muril_.npy")
X_v = np.load("Features/video_embeddings_normalized.npy")
X_a = np.load("Features/audio_embeddings_hubert_final.npy")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

counts = Counter(y)
n_classes = len(counts)
class_weights = torch.tensor(
    [len(y) / counts[i] for i in range(n_classes)],
    dtype=torch.float32
).to(device)

class FusionModel(nn.Module):
    def __init__(self, t_dim=768, v_dim=768, a_dim=768, hidden=256):
        super().__init__()

        self.t_proj = nn.Sequential(nn.Linear(t_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.v_proj = nn.Sequential(nn.Linear(v_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.a_proj = nn.Sequential(nn.Linear(a_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())

        self.alpha = nn.Parameter(torch.ones(3))

        self.cls = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes)
        )

    def forward(self, t, v, a, mask):
        t_feat = self.t_proj(t)
        v_feat = self.v_proj(v)
        a_feat = self.a_proj(a)

        feats = torch.stack([t_feat, v_feat, a_feat], dim=1)

        m = mask.float().unsqueeze(0).unsqueeze(-1).to(feats.device)

        raw_alpha = self.alpha * mask.float().to(self.alpha.device)
        w = torch.softmax(raw_alpha + (1 - mask.float().to(self.alpha.device)) * -1e9, dim=0)
        w = w.unsqueeze(0).unsqueeze(-1)

        fused = (feats * m * w).sum(dim=1)
        return self.cls(fused)

masks = {
    "TEXT_ONLY":   torch.tensor([1, 0, 0], dtype=torch.bool),
    "VIDEO_ONLY":  torch.tensor([0, 1, 0], dtype=torch.bool),
    "AUDIO_ONLY":  torch.tensor([0, 0, 1], dtype=torch.bool),
    "TEXT_VIDEO":  torch.tensor([1, 1, 0], dtype=torch.bool),
    "TEXT_AUDIO":  torch.tensor([1, 0, 1], dtype=torch.bool),
    "AUDIO_VIDEO": torch.tensor([0, 1, 1], dtype=torch.bool),  
    "TRIMODAL":    torch.tensor([1, 1, 1], dtype=torch.bool),
}

EPOCHS = 150

def run_cv_early(mask: torch.Tensor):
    accs, precs, recs, f1s = [], [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_t, y)):

        Xt_tr, Xt_te = X_t[train_idx], X_t[test_idx]
        Xv_tr, Xv_te = X_v[train_idx], X_v[test_idx]
        Xa_tr, Xa_te = X_a[train_idx], X_a[test_idx]
        y_tr, y_te   = y[train_idx], y[test_idx]

        model = FusionModel().to(device)
        opt   = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

        Xt_tr_t = torch.FloatTensor(Xt_tr).to(device)
        Xv_tr_t = torch.FloatTensor(Xv_tr).to(device)
        Xa_tr_t = torch.FloatTensor(Xa_tr).to(device)
        y_tr_t  = torch.LongTensor(y_tr).to(device)

        model.train()
        for _ in range(EPOCHS):
            opt.zero_grad()
            out  = model(Xt_tr_t, Xv_tr_t, Xa_tr_t, mask)
            loss = F.cross_entropy(out, y_tr_t, weight=class_weights)
            loss.backward()
            opt.step()
            sched.step()

        model.eval()
        with torch.no_grad():
            out = model(
                torch.FloatTensor(Xt_te).to(device),
                torch.FloatTensor(Xv_te).to(device),
                torch.FloatTensor(Xa_te).to(device),
                mask,
            )
            preds = out.argmax(dim=1).cpu().numpy()

        acc = accuracy_score(y_te, preds)
        p, r, f1, _ = precision_recall_fscore_support(
            y_te, preds, average="macro", zero_division=0
        )
        accs.append(acc); precs.append(p); recs.append(r); f1s.append(f1)

    return {
        "accuracy":  np.mean(accs),
        "precision": np.mean(precs),
        "recall":    np.mean(recs),
        "f1":        np.mean(f1s),
        "acc_std":   np.std(accs),
        "f1_std":    np.std(f1s),
    }

def get_unimodal_fold_probs(X_modal: np.ndarray, modal_idx: int):
    mask = torch.zeros(3, dtype=torch.bool)
    mask[modal_idx] = True

    probs_list, y_list = [], []

    for train_idx, test_idx in skf.split(X_modal, y):
        X_tr, X_te = X_modal[train_idx], X_modal[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model = FusionModel().to(device)
        opt   = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

        X_tr_t = torch.FloatTensor(X_tr).to(device)
        y_tr_t = torch.LongTensor(y_tr).to(device)

        zeros  = torch.zeros_like(X_tr_t)

        model.train()
        for _ in range(EPOCHS):
            opt.zero_grad()
            inputs = [zeros, zeros, zeros]
            inputs[modal_idx] = X_tr_t
            out  = model(*inputs, mask)
            loss = F.cross_entropy(out, y_tr_t, weight=class_weights)
            loss.backward()
            opt.step()
            sched.step()

        model.eval()
        with torch.no_grad():
            X_te_t  = torch.FloatTensor(X_te).to(device)
            zeros_e = torch.zeros_like(X_te_t)
            inputs_e = [zeros_e, zeros_e, zeros_e]
            inputs_e[modal_idx] = X_te_t
            probs = F.softmax(model(*inputs_e, mask), dim=1).cpu().numpy()

        probs_list.append(probs)
        y_list.extend(y_te)

    return np.vstack(probs_list), np.array(y_list)

def eval_late_with_std(probs, y_true):
    fold_accs, fold_f1s = [], []
    fold_sizes = [len(te) for _, te in skf.split(X_t, y)]

    start = 0
    for sz in fold_sizes:
        yp = np.argmax(probs[start:start+sz], axis=1)
        yt = y_true[start:start+sz]
        fold_accs.append(accuracy_score(yt, yp))
        _, _, fold_f1, _ = precision_recall_fscore_support(
            yt, yp, average="macro", zero_division=0
        )
        fold_f1s.append(fold_f1)
        start += sz

    preds = np.argmax(probs, axis=1)
    acc   = accuracy_score(y_true, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="macro", zero_division=0
    )

    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "acc_std": np.std(fold_accs),
        "f1_std": np.std(fold_f1s),
    }

results = {}
print("\nRunning EARLY FUSION...\n")
for name, mask in masks.items():
    print(f"  {name}")
    results[f"{name} (EARLY)"] = run_cv_early(mask)

print("\nRunning LATE FUSION...\n")
p_t, y_true = get_unimodal_fold_probs(X_t, 0)
p_v, _      = get_unimodal_fold_probs(X_v, 1)
p_a, _      = get_unimodal_fold_probs(X_a, 2)

late_configs = {
    "TEXT_ONLY (LATE)":   p_t,
    "VIDEO_ONLY (LATE)":  p_v,
    "AUDIO_ONLY (LATE)":  p_a,
    "TEXT_VIDEO (LATE)":  (p_t + p_v) / 2,
    "TEXT_AUDIO (LATE)":  (p_t + p_a) / 2,
    "AUDIO_VIDEO (LATE)": (p_v + p_a) / 2, 
    "TRIMODAL (LATE)":    (p_t + p_v + p_a) / 3,
}

for name, probs in late_configs.items():
    results[name] = eval_late_with_std(probs, y_true)

df_out = pd.DataFrame(results).T.sort_values("f1", ascending=False)

print("\n===== FINAL RESULTS =====\n")
print(df_out.round(4))