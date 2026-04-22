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

# normalize
T = StandardScaler().fit_transform(T)
A = StandardScaler().fit_transform(A)
V = StandardScaler().fit_transform(V)

class_counts = np.bincount(y)
weights = 1.0 / class_counts
weights = weights / weights.sum()
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

print("Class counts:", class_counts)
print("Class weights:", class_weights)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, q, k, v):
        attn_out, _ = self.attn(q, k, v)
        x = self.norm1(q + attn_out)
        ff_out = self.ff(x)
        return self.norm2(x + ff_out)

class MULT(nn.Module):
    def __init__(self, d_t, d_a, d_v, hidden=128):
        super().__init__()

        self.t_proj = nn.Linear(d_t, hidden)
        self.a_proj = nn.Linear(d_a, hidden)
        self.v_proj = nn.Linear(d_v, hidden)

        # cross-modal layers (stacked like real MULT)
        self.t_from_a = TransformerBlock(hidden)
        self.t_from_v = TransformerBlock(hidden)

        self.a_from_t = TransformerBlock(hidden)
        self.v_from_t = TransformerBlock(hidden)

        # second layer (deeper interaction)
        self.t_fusion = TransformerBlock(hidden)
        self.a_fusion = TransformerBlock(hidden)
        self.v_fusion = TransformerBlock(hidden)

        # final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

    def forward(self, t, a, v):

        # project
        t = self.t_proj(t).unsqueeze(1)
        a = self.a_proj(a).unsqueeze(1)
        v = self.v_proj(v).unsqueeze(1)

        # cross-modal attention (layer 1)
        t = self.t_from_a(t, a, a)
        t = self.t_from_v(t, v, v)

        a = self.a_from_t(a, t, t)
        v = self.v_from_t(v, t, t)

        # deeper fusion (layer 2)
        t = self.t_fusion(t, t, t)
        a = self.a_fusion(a, a, a)
        v = self.v_fusion(v, v, v)

        # concat
        x = torch.cat([t, a, v], dim=1)
        x = x.reshape(x.size(0), -1)

        return self.classifier(x)

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

        # remove modalities
        if 't' not in modalities:
            t_train *= 0; t_test *= 0
        if 'a' not in modalities:
            a_train *= 0; a_test *= 0
        if 'v' not in modalities:
            v_train *= 0; v_test *= 0

        model = MULT(T.shape[1], A.shape[1], V.shape[1]).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        # training
        for epoch in range(25):
            model.train()
            optimizer.zero_grad()

            out = model(t_train, a_train, v_train)
            loss = loss_fn(out, y_train)

            loss.backward()
            optimizer.step()

        # evaluation
        model.eval()
        with torch.no_grad():
            preds = model(t_test, a_test, v_test)
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

