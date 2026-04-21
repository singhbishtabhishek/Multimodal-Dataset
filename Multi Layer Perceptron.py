import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

torch.manual_seed(42)
np.random.seed(42)

X_text = np.load(r"Features\text_features_muril_.npy")
X_audio = np.load(r"Features\audio_embeddings_hubert_final.npy")
X_video = np.load(r"Features\video_embeddings_normalized.npy")
y = pd.read_csv(r"Dataset.csv")["Labels"].values

class HumorMLP(nn.Module):
    def __init__(self, input_dim):
        super(HumorMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def train_and_evaluate_mlp_cv(X, y, name):
    acc_list, prec_list, rec_list, f1_list = [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        X_test_t = torch.FloatTensor(X_test)

        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

        # Compute class weights dynamically
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

        # Model
        model = HumorMLP(X.shape[1])

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # TRAIN
        model.train()
        for epoch in range(20):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # EVAL
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_t)
            preds = torch.argmax(outputs, dim=1).numpy()

        acc = accuracy_score(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average='macro', zero_division=0
        )

        acc_list.append(acc)
        prec_list.append(precision)
        rec_list.append(recall)
        f1_list.append(f1)

    # Aggregate
    result = {
        "accuracy": np.mean(acc_list),
        "precision": np.mean(prec_list),
        "recall": np.mean(rec_list),
        "f1-score": np.mean(f1_list),
        "acc_std": np.std(acc_list),
        "f1_std": np.std(f1_list)
    }

    print(f"\n=== {name} (5-FOLD CV) ===")
    print(result)

    return result

mlp_results = {}

# Unimodal
mlp_results['Text'] = train_and_evaluate_mlp_cv(X_text, y, "Text Only")
mlp_results['Audio'] = train_and_evaluate_mlp_cv(X_audio, y, "Audio Only")
mlp_results['Video'] = train_and_evaluate_mlp_cv(X_video, y, "Video Only")

# Bimodal
mlp_results['T+A'] = train_and_evaluate_mlp_cv(np.hstack([X_text, X_audio]), y, "Text + Audio")
mlp_results['T+V'] = train_and_evaluate_mlp_cv(np.hstack([X_text, X_video]), y, "Text + Video")
mlp_results['A+V'] = train_and_evaluate_mlp_cv(np.hstack([X_audio, X_video]), y, "Audio + Video")

# Trimodal
mlp_results['T+A+V'] = train_and_evaluate_mlp_cv(
    np.hstack([X_text, X_audio, X_video]), y, "Text + Audio + Video"
)

df_results = pd.DataFrame(mlp_results).T
df_results = df_results.sort_values("f1-score", ascending=False)

print("\n Result")
print(df_results.round(4))
