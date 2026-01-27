"""Search test for Transcendence movie (Hybrid Search)"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import open_clip
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
MOVIE_NAME = "Transcendence"
MOVIE_DIR = OUTPUT_DIR / MOVIE_NAME
EMBEDDINGS_FILE = MOVIE_DIR / "embeddings.npz"
METADATA_FILE = MOVIE_DIR / "metadata.json"

# Load embeddings
data = np.load(EMBEDDINGS_FILE)

# Check for hybrid or legacy format
if 'image_embeddings' in data:
    image_embeddings = data['image_embeddings']
    text_embeddings = data['text_embeddings']
    use_hybrid = True
    print("Mode: Hybrid (image + caption)")
else:
    image_embeddings = data['embeddings']
    text_embeddings = None
    use_hybrid = False
    print("Mode: Image only (legacy)")

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
    # Encode query
    text_tokens = tokenizer([query])
    with torch.no_grad():
        query_emb = model.encode_text(text_tokens.to(device))
        query_emb /= query_emb.norm(dim=-1, keepdim=True)
    query_emb = query_emb.cpu().numpy().flatten()

    # Calculate similarities
    img_sim = image_embeddings @ query_emb
    
    if use_hybrid and text_embeddings is not None:
        txt_sim = text_embeddings @ query_emb
        
        # Dynamic weight based on query length
        if len(query.split()) <= 2:
            w_img, w_txt = 0.7, 0.3
        else:
            w_img, w_txt = 0.5, 0.5
        
        similarities = img_sim * w_img + txt_sim * w_txt
    else:
        similarities = img_sim

    top_indices = np.argsort(similarities)[::-1][:top_k]

    print(f"\n{'='*60}")
    print(f"Query: '{query}'")
    print(f"{'='*60}")
    for i, idx in enumerate(top_indices, 1):
        frame = frames[idx]
        sim = similarities[idx]
        caption = frame.get('caption', 'N/A')
        print(f"  {i}. [{frame['time_str']}] {frame['filename']} (score: {sim:.3f})")
        if caption != 'N/A':
            print(f"      📝 {caption[:70]}...")

# Test searches
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
print("TRANSCENDENCE SEARCH TEST (HYBRID)")
print("=" * 60)

for q in test_queries:
    search(q)
