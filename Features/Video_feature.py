import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import VideoMAEImageProcessor, VideoMAEModel

MODEL_NAME = "MCG-NJU/videomae-base"
VIDEO_DIRS = [
    "Clips_PanchayatS1",
    "Clips_PanchayatS2"
]
SAVE_PATH = "video_embeddings_normalized.npy"
DATASET_PATH = "Dataset.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
model = VideoMAEModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

def load_video(path, num_frames=16):
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return None

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    current_idx = 0
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret: break
        if current_idx in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        current_idx += 1
            
    cap.release()
    
    # Pad if video is too short or corrupted
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        
    return frames

df = pd.read_csv(DATASET_PATH)
video_features = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting VideoMAE"):
    base_name = str(row["Utterances_id"]).strip()
    found_path = None
    
    for folder in VIDEO_DIRS:
        path = os.path.join(folder, base_name + ".mp4") 
        if os.path.exists(path):
            found_path = path
            break
            
    if found_path:
        frames = load_video(found_path)
        if frames is not None:
            inputs = processor(frames, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Mean pool across sequence length (1568 tokens -> 1 vector)
            emb = outputs.last_hidden_state.mean(dim=1) # Shape: (1, 768)
            
            # This ensures video features are on the same scale as your Text features
            norm = emb.norm(p=2, dim=1, keepdim=True)
            emb_normalized = emb / (norm + 1e-9)
            
            video_features.append(emb_normalized.cpu().numpy())
        else:
            video_features.append(np.zeros((1, 768)))
    else:
        video_features.append(np.zeros((1, 768)))

video_features = np.vstack(video_features)
np.save(SAVE_PATH, video_features)

print(f"\nFinal Normalized Video Feature Shape: {video_features.shape}")
print(f"Verification - Mean length of vectors: {np.linalg.norm(video_features, axis=1).mean():.4f}")