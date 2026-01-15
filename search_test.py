"""Search test for Transcendence movie"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import open_clip
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.npz"
METADATA_FILE = OUTPUT_DIR / "metadata.json"

# Load embeddings
data = np.load(EMBEDDINGS_FILE)
embeddings = data['embeddings']

# Load metadata
with open(METADATA_FILE, 'r', encoding='utf-8') as f:
    metadata = json.load(f)
frames = metadata['frames']

print(f"Loaded {len(frames)} frame embeddings")

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device)
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')

def search(query, top_k=5):
    text_tokens = tokenizer([query])
    with torch.no_grad():
        text_emb = model.encode_text(text_tokens.to(device))
        text_emb /= text_emb.norm(dim=-1, keepdim=True)
    text_emb = text_emb.cpu().numpy().flatten()

    similarities = embeddings @ text_emb
    top_indices = np.argsort(similarities)[::-1][:top_k]

    print(f"\n{'='*50}")
    print(f"Query: '{query}'")
    print(f"{'='*50}")
    for i, idx in enumerate(top_indices, 1):
        frame = frames[idx]
        sim = similarities[idx]
        print(f"  {i}. [{frame['time_str']}] {frame['filename']} (score: {sim:.3f})")

# Test searches - Transcendence related
test_queries = [
    "Johnny Depp face",
    "computer screen",
    "laboratory",
    "outdoor garden",
    "woman crying",
    "dark scene",
    "explosion",
    "people talking",
    "sunset",
    "technology equipment",
]

print("\n" + "=" * 60)
print("TRANSCENDENCE SEARCH TEST")
print("=" * 60)

for q in test_queries:
    search(q)
