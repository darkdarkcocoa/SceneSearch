"""
Video Semantic Search Prototype
- Extract frames from video (1 sec interval)
- Generate CLIP embeddings
- Save to numpy file
- Search by natural language
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
import torch
import open_clip
from PIL import Image
import os
import time
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
VIDEO_PATH = BASE_DIR / "movie.mp4"  # 사용 시 영상 파일 필요
EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.npz"
FRAME_INTERVAL = 1  # seconds

# ============================================================
# 1. EXTRACT FRAMES
# ============================================================
def extract_frames(video_path, output_dir, interval=1):
    """Extract frames at fixed interval and return timestamps"""
    
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video: {duration:.1f}s, {fps:.1f}fps")
    print(f"Extracting frames every {interval}s...")
    
    frame_data = []  # [(frame_path, timestamp), ...]
    frame_interval = int(fps * interval)
    
    frame_idx = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / fps
            frame_filename = f"frame_{saved_count:04d}.jpg"
            frame_path = os.path.join(frames_dir, frame_filename)
            
            cv2.imwrite(frame_path, frame)
            frame_data.append((frame_path, timestamp))
            saved_count += 1
        
        frame_idx += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames")
    return frame_data

# ============================================================
# 2. GENERATE EMBEDDINGS
# ============================================================
def generate_embeddings(frame_data, model, preprocess, device):
    """Generate CLIP embeddings for all frames"""
    
    print(f"\nGenerating embeddings for {len(frame_data)} frames...")
    start_time = time.time()
    
    embeddings = []
    timestamps = []
    
    for i, (frame_path, timestamp) in enumerate(frame_data):
        # Load and preprocess image
        image = Image.open(frame_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Generate embedding
        with torch.no_grad():
            embedding = model.encode_image(image_input)
            embedding /= embedding.norm(dim=-1, keepdim=True)  # normalize
        
        embeddings.append(embedding.cpu().numpy().flatten())
        timestamps.append(timestamp)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(frame_data)} frames")
    
    elapsed = time.time() - start_time
    print(f"Done! {elapsed:.1f}s ({len(frame_data)/elapsed:.1f} frames/sec)")
    
    return np.array(embeddings), np.array(timestamps)

# ============================================================
# 3. SAVE / LOAD EMBEDDINGS
# ============================================================
def save_embeddings(embeddings, timestamps, filepath):
    """Save embeddings and timestamps to npz file"""
    np.savez(filepath, embeddings=embeddings, timestamps=timestamps)
    print(f"Saved to {filepath}")

def load_embeddings(filepath):
    """Load embeddings and timestamps from npz file"""
    data = np.load(filepath)
    return data['embeddings'], data['timestamps']

# ============================================================
# 4. SEARCH
# ============================================================
def search(query, embeddings, timestamps, model, tokenizer, device, top_k=5):
    """Search for frames matching the query"""
    
    # Encode query text
    text_tokens = tokenizer([query])
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens.to(device))
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
    
    text_embedding = text_embedding.cpu().numpy().flatten()
    
    # Calculate cosine similarity
    similarities = embeddings @ text_embedding
    
    # Get top-k results
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'timestamp': timestamps[idx],
            'time_str': format_time(timestamps[idx]),
            'similarity': similarities[idx]
        })
    
    return results

def format_time(seconds):
    """Format seconds to MM:SS"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("Video Semantic Search Prototype")
    print("=" * 60)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Load CLIP model
    print("\nLoading CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    print("Model loaded!")
    
    # Check if embeddings already exist
    if os.path.exists(EMBEDDINGS_FILE):
        print(f"\nFound existing embeddings: {EMBEDDINGS_FILE}")
        embeddings, timestamps = load_embeddings(EMBEDDINGS_FILE)
        print(f"Loaded {len(timestamps)} frame embeddings")
    else:
        # Extract frames
        print("\n" + "-" * 60)
        frame_data = extract_frames(VIDEO_PATH, OUTPUT_DIR, FRAME_INTERVAL)
        
        # Generate embeddings
        print("-" * 60)
        embeddings, timestamps = generate_embeddings(frame_data, model, preprocess, device)
        
        # Save embeddings
        print("-" * 60)
        save_embeddings(embeddings, timestamps, EMBEDDINGS_FILE)
    
    # Interactive search
    print("\n" + "=" * 60)
    print("SEARCH MODE")
    print("Type a scene description in English (or 'quit' to exit)")
    print("=" * 60)
    
    while True:
        query = input("\nSearch: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        if not query:
            continue
        
        results = search(query, embeddings, timestamps, model, tokenizer, device)
        
        print(f"\nResults for: '{query}'")
        print("-" * 40)
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['time_str']} (similarity: {r['similarity']:.3f})")

if __name__ == "__main__":
    main()
