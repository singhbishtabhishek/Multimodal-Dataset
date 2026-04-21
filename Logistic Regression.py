import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import StratifiedKFold

X_text = np.load(r"Features\text_features_muril_.npy")
X_audio = np.load(r"Features\audio_embeddings_hubert_final.npy")
X_video = np.load(r"Features\video_embeddings_normalized.npy")

y = pd.read_csv(r"Dataset.csv")["Labels"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}

def evaluate(name, X):
    acc_list, prec_list, rec_list, f1_list = [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = LogisticRegression(
            class_weight='balanced',
            max_iter=3000,
            solver='lbfgs'
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro', zero_division=0
        )

        acc_list.append(acc)
        prec_list.append(precision)
        rec_list.append(recall)
        f1_list.append(f1)

    results[name] = {
        "accuracy": np.mean(acc_list),
        "precision": np.mean(prec_list),
        "recall": np.mean(rec_list),
        "f1-score": np.mean(f1_list),
        "acc_std": np.std(acc_list),
        "f1_std": np.std(f1_list)
    }

evaluate("Text Only", X_text)
evaluate("Audio Only", X_audio)
evaluate("Video Only", X_video)

evaluate("Text + Audio", np.hstack([X_text, X_audio]))
evaluate("Text + Video", np.hstack([X_text, X_video]))
evaluate("Audio + Video", np.hstack([X_audio, X_video]))
evaluate("Text + Audio + Video", np.hstack([X_text, X_audio, X_video]))

df_results = pd.DataFrame(results).T
df_results = df_results.sort_values("f1-score", ascending=False)

print("\n===== FINAL CROSS-VALIDATED RESULTS =====\n")
print(df_results.round(4))

print("\nBest Model:", df_results.index[0])


