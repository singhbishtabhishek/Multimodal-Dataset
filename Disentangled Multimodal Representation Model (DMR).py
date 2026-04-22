import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T = np.load(r"Features\text_features_muril_.npy")
A = np.load(r"Features\audio_embeddings_hubert_final.npy")
V = np.load(r"Features\video_embeddings_normalized.npy")

df = pd.read_csv("Dataset.csv")
y = df["Labels"].values

le = LabelEncoder()
y = le.fit_transform(y)

T = StandardScaler().fit_transform(T)
A = StandardScaler().fit_transform(A)
V = StandardScaler().fit_transform(V)
class_counts = np.bincount(y)
weights = 1.0 / class_counts
weights = weights / weights.sum()
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

class MISA(nn.Module):
    def __init__(self, d_t, d_a, d_v, hidden=128):
        super().__init__()

        # projections
        self.t_proj = nn.Linear(d_t, hidden)
        self.a_proj = nn.Linear(d_a, hidden)
        self.v_proj = nn.Linear(d_v, hidden)

        # shared encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        # private encoders
        self.private_t = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.private_a = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.private_v = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())

        # reconstruction
        self.recon_t = nn.Linear(hidden * 2, hidden)
        self.recon_a = nn.Linear(hidden * 2, hidden)
        self.recon_v = nn.Linear(hidden * 2, hidden)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 6, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )

    def forward(self, t, a, v):
        # project
        t = self.t_proj(t)
        a = self.a_proj(a)
        v = self.v_proj(v)

        # shared
        t_s = self.shared_encoder(t)
        a_s = self.shared_encoder(a)
        v_s = self.shared_encoder(v)

        # private
        t_p = self.private_t(t)
        a_p = self.private_a(a)
        v_p = self.private_v(v)

        # reconstruction
        t_recon = self.recon_t(torch.cat([t_s, t_p], dim=1))
        a_recon = self.recon_a(torch.cat([a_s, a_p], dim=1))
        v_recon = self.recon_v(torch.cat([v_s, v_p], dim=1))

        # classification
        fused = torch.cat([t_s, a_s, v_s, t_p, a_p, v_p], dim=1)
        out = self.classifier(fused)

        return out, (t, a, v), (t_recon, a_recon, v_recon), (t_s, a_s, v_s), (t_p, a_p, v_p)

def orthogonality_loss(shared, private):
    loss = 0
    for s, p in zip(shared, private):
        loss += torch.mean((s * p) ** 2)
    return loss

def reconstruction_loss(orig, recon):
    loss = 0
    for o, r in zip(orig, recon):
        loss += nn.functional.mse_loss(r, o)
    return loss

def train_eval(T, A, V, y, modalities=('t','a','v'), k=5):

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    all_metrics = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(T)):
        print(f"\nFold {fold+1}")

        t_train = torch.tensor(T[train_idx]).float().to(device)
        a_train = torch.tensor(A[train_idx]).float().to(device)
        v_train = torch.tensor(V[train_idx]).float().to(device)

        t_test = torch.tensor(T[test_idx]).float().to(device)
        a_test = torch.tensor(A[test_idx]).float().to(device)
        v_test = torch.tensor(V[test_idx]).float().to(device)

        y_train = torch.tensor(y[train_idx]).long().to(device)
        y_test = torch.tensor(y[test_idx]).long().to(device)

        if 't' not in modalities:
            t_train *= 0; t_test *= 0
        if 'a' not in modalities:
            a_train *= 0; a_test *= 0
        if 'v' not in modalities:
            v_train *= 0; v_test *= 0

        model = MISA(T.shape[1], A.shape[1], V.shape[1]).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        ce_loss = nn.CrossEntropyLoss(weight=class_weights)

        for epoch in range(25):
            model.train()
            optimizer.zero_grad()

            out, orig, recon, shared, private = model(t_train, a_train, v_train)

            loss_cls = ce_loss(out, y_train)
            loss_recon = reconstruction_loss(orig, recon)
            loss_ortho = orthogonality_loss(shared, private)

            loss = loss_cls + 0.1 * loss_recon + 0.1 * loss_ortho

            loss.backward()
            optimizer.step()

        # eval
        model.eval()
        with torch.no_grad():
            preds, _, _, _, _ = model(t_test, a_test, v_test)
            preds = torch.argmax(preds, dim=1).cpu().numpy()

        y_true = y_test.cpu().numpy()

        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, average='weighted')
        rec = recall_score(y_true, preds, average='weighted')
        f1 = f1_score(y_true, preds, average='weighted')

        print(f"Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

        all_metrics.append([acc, prec, rec, f1])

    return np.mean(all_metrics, axis=0)

configs = {
    "T": ('t',),
    "A": ('a',),
    "V": ('v',),
    "T+A": ('t','a'),
    "T+V": ('t','v'),
    "A+V": ('a','v'),
    "T+A+V": ('t','a','v')
}

results = {}

for name, mods in configs.items():
    print(f"\n==== {name} ====")
    results[name] = train_eval(T, A, V, y, modalities=mods)

print("\nFINAL RESULTS:")
for k, v in results.items():
    print(f"{k}: Acc={v[0]:.4f}, Prec={v[1]:.4f}, Rec={v[2]:.4f}, F1={v[3]:.4f}")
