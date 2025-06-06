import torch
import clip
from PIL import Image
import pandas as pd
import os

# Debug: Print working directory to confirm paths
print("Current working directory:", os.getcwd())

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)

# --- Functions ---

def load_captions(filepath="captions.csv"):
    df = pd.read_csv(filepath)

    # Strip whitespace from column names to avoid KeyError
    df.columns = df.columns.str.strip()

    # Check for 'caption' column explicitly
    if 'caption' not in df.columns:
        raise KeyError(f"'caption' column not found! Available columns: {df.columns.tolist()}")

    return df['caption'].dropna().tolist()  # drop any NaN entries just in case

def get_image_embedding(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features / image_features.norm(dim=-1, keepdim=True)

def get_text_embeddings(captions):
    with torch.no_grad():
        text_tokens = clip.tokenize(captions).to(device)
        text_features = model.encode_text(text_tokens)
    return text_features / text_features.norm(dim=-1, keepdim=True)

def recommend_captions(image_path, captions, top_k=5):
    image_feat = get_image_embedding(image_path)
    text_feats = get_text_embeddings(captions)
    similarities = (image_feat @ text_feats.T).squeeze(0)
    top_indices = similarities.topk(top_k).indices.tolist()
    return [captions[i] for i in top_indices]

# --- Optional: Demo/Test Block ---
if __name__ == "__main__":
    image_path = "sample.jpg"  # Change this for standalone testing
    caption_file = "captions.csv"

    captions = load_captions(caption_file)
    top_captions = recommend_captions(image_path, captions, top_k=5)

    print("📸 Recommended Captions:")
    for cap in top_captions:
        print("•", cap)
