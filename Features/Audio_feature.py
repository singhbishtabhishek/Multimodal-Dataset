import os
import numpy as np
import torch
import librosa
import pandas as pd
from tqdm import tqdm
from transformers import AutoFeatureExtractor, HubertModel

# ======================
# CONFIG
# ======================
model_name = "facebook/hubert-base-ls960"

df_path = "/content/drive/MyDrive/Multimodal/Dataset.csv"

audio_dirs = [
    "/content/drive/MyDrive/Multimodal/Audio_PanchayatS1",
    "/content/drive/MyDrive/Multimodal/Audio_PanchayatS2"
]

save_path = "/content/drive/MyDrive/Multimodal/features/audio_embeddings_hubert_final.npy"

filename_col = "Utterances_id"

# ======================
# DEVICE + MODEL
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = HubertModel.from_pretrained(model_name)

model.to(device)
model.eval()

# ======================
# AUDIO POOLING (IMPROVED)
# ======================
def improved_pool(hidden):
    """
    Stable pooling for small datasets
    """
    mean = hidden.mean(dim=0)
    maxv = hidden.max(dim=0).values
    return (mean + maxv) / 2

# ======================
# AUDIO PROCESSING
# ======================
def extract_audio_embedding(audio_path, target_sr=16000):
    try:
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)

        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=30)

        # Normalize
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        # Pad short audio
        if len(y) < target_sr:
            y = np.pad(y, (0, target_sr - len(y)))

        # Feature extraction
        inputs = feature_extractor(
            y,
            sampling_rate=target_sr,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        hidden = outputs.last_hidden_state.squeeze(0)  # (T, 768)

        emb = improved_pool(hidden)

        # normalize
        emb = emb / (emb.norm(p=2) + 1e-6)

        return emb.cpu().numpy().astype(np.float32)

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return np.zeros(768, dtype=np.float32)

# ======================
# LOAD DATA
# ======================
df = pd.read_csv(df_path)

audio_embeddings = []

# ======================
# EXTRACTION LOOP
# ======================
for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Audio Features"):

    base_name = str(row[filename_col]).strip()
    found = False

    for folder in audio_dirs:
        for ext in [".wav", ".mp3", ".flac", ".m4a", ".WAV"]:
            path = os.path.join(folder, base_name + ext)

            if os.path.exists(path):
                emb = extract_audio_embedding(path)
                audio_embeddings.append(emb)
                found = True
                break

        if found:
            break

    if not found:
        audio_embeddings.append(np.zeros(768, dtype=np.float32))

# ======================
# SAVE
# ======================
audio_embeddings = np.array(audio_embeddings)

np.save(save_path, audio_embeddings)

print("\nSaved at:", save_path)
print("Shape:", audio_embeddings.shape)

# ======================
# STATS CHECK
# ======================
norms = np.linalg.norm(audio_embeddings, axis=1)

print("\nNorm stats:")
print("Mean:", norms.mean())
print("Std:", norms.std())
print("Min:", norms.min())
print("Max:", norms.max())

zero_count = np.sum(np.all(audio_embeddings == 0, axis=1))
print("\nZero embeddings:", zero_count)