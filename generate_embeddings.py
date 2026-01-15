"""Generate CLIP embeddings for extracted frames"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import open_clip
from PIL import Image
import json
import time
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
METADATA_FILE = OUTPUT_DIR / "metadata.json"
FRAMES_DIR = OUTPUT_DIR / "frames"
OUTPUT_FILE = OUTPUT_DIR / "embeddings.npz"

def main():
    print("=" * 60)
    print("CLIP Embedding Generator")
    print("=" * 60)

    # Load metadata
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    frames = metadata['frames']
    print(f"\nTotal frames to process: {len(frames)}")

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load CLIP model
    print("\nLoading CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model = model.to(device)
    model.eval()
    print("Model loaded!")

    # Generate embeddings
    print("\n" + "-" * 60)
    print("Generating embeddings...")
    start_time = time.time()

    embeddings = []
    timestamps = []

    for i, frame_info in enumerate(frames):
        frame_path = os.path.join(FRAMES_DIR, frame_info['filename'])

        # Load and preprocess image
        image = Image.open(frame_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Generate embedding
        with torch.no_grad():
            embedding = model.encode_image(image_input)
            embedding /= embedding.norm(dim=-1, keepdim=True)

        embeddings.append(embedding.cpu().numpy().flatten())
        timestamps.append(frame_info['timestamp'])

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed
            remaining = (len(frames) - i - 1) / speed
            print(f"  [{i+1}/{len(frames)}] {speed:.1f} frames/sec, ~{remaining:.0f}s remaining")

    elapsed = time.time() - start_time
    print(f"\nDone! {elapsed:.1f}s total ({len(frames)/elapsed:.1f} frames/sec)")

    # Save embeddings
    embeddings = np.array(embeddings)
    timestamps = np.array(timestamps)

    print(f"\nEmbeddings shape: {embeddings.shape}")
    np.savez(OUTPUT_FILE, embeddings=embeddings, timestamps=timestamps)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
