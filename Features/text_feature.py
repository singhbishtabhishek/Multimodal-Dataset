import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

MODEL_NAME = "google/muril-base-cased" 
DATASET_PATH = "Dataset.csv"
SAVE_PATH = "text_features_muril_.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

def get_normalized_muril_embeddings(texts, batch_size=16):
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting Text"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            
        # 1. Grab [CLS] token (Global context for Humor)
        cls_emb = outputs.last_hidden_state[:, 0, :] 

        # 2. IN-CODE NORMALIZATION (L2)
        # This scales the vector to have a length of 1.0
        norm = cls_emb.norm(p=2, dim=1, keepdim=True)
        cls_emb_normalized = cls_emb / (norm + 1e-9)

        all_embeddings.append(cls_emb_normalized.cpu().numpy())

    return np.vstack(all_embeddings)

df = pd.read_csv(DATASET_PATH)
texts = df["Hindi_utterances"].astype(str).tolist()

features = get_normalized_muril_embeddings(texts)

# Save the finalized, normalized features
np.save(SAVE_PATH, features)

print(f"\nSaved Normalized Features to: {SAVE_PATH}")
print(f"Verification - Mean length of vectors: {np.linalg.norm(features, axis=1).mean():.4f}")