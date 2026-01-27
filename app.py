"""SceneSearch - Gradio Web UI with Video Player Integration"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import gradio as gr
import numpy as np
import torch
import open_clip
import json
import time
import subprocess
import shutil
import re
import os
from pathlib import Path
from PIL import Image

# Paths
BASE_DIR = Path(__file__).parent
VIDEO_DIR = BASE_DIR / "video"
OUTPUT_DIR = BASE_DIR / "output"

# Global variables
image_embeddings = None
text_embeddings = None
frames = None
clip_model = None
tokenizer = None
device = None
use_hybrid = False
current_movie = None  # Currently loaded movie name
search_results_timestamps = []  # Store timestamps for gallery clicks

# ============================================================
# UTILS
# ============================================================
def format_time(seconds):
    """Format seconds to HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_movie_name(video_path):
    """Extract movie name from path (without extension)"""
    return Path(video_path).stem

def get_available_movies():
    """Get list of video files in video/ folder"""
    if not VIDEO_DIR.exists():
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        return []

    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.webm'}
    movies = []
    for f in VIDEO_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in video_extensions:
            movies.append(f.stem)
    return sorted(movies)

def get_movie_paths(movie_name):
    """Get all paths for a movie"""
    video_path = VIDEO_DIR / f"{movie_name}.mp4"
    # Try other extensions if mp4 doesn't exist
    if not video_path.exists():
        for ext in ['.avi', '.mkv', '.mov', '.wmv', '.webm']:
            alt_path = VIDEO_DIR / f"{movie_name}{ext}"
            if alt_path.exists():
                video_path = alt_path
                break

    movie_output_dir = OUTPUT_DIR / movie_name
    return {
        'video': video_path,
        'output_dir': movie_output_dir,
        'embeddings': movie_output_dir / 'embeddings.npz',
        'metadata': movie_output_dir / 'metadata.json',
        'frames': movie_output_dir / 'frames'
    }

def is_movie_processed(movie_name):
    """Check if a movie has been processed (embeddings exist)"""
    paths = get_movie_paths(movie_name)
    return paths['embeddings'].exists() and paths['metadata'].exists()

# ============================================================
# SEARCH ENGINE
# ============================================================
def load_clip_model():
    """Load CLIP model (only once)"""
    global clip_model, tokenizer, device

    if clip_model is not None:
        return True

    device = get_device()
    print(f"[+] Using device: {device}")

    try:
        clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        clip_model = clip_model.to(device)
        clip_model.eval()
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        print("[+] CLIP model loaded")
        return True
    except Exception as e:
        print(f"[!] Error loading CLIP model: {e}")
        return False


def load_movie_data(movie_name):
    """Load embeddings and metadata for a specific movie"""
    global image_embeddings, text_embeddings, frames, use_hybrid, current_movie

    if not movie_name:
        return False, "No movie selected"

    paths = get_movie_paths(movie_name)

    if not paths['embeddings'].exists():
        return False, f"Movie '{movie_name}' has not been processed yet"

    # Load Embeddings
    try:
        data = np.load(paths['embeddings'])
        if 'image_embeddings' in data:
            image_embeddings = data['image_embeddings']
            text_embeddings = data['text_embeddings']
            use_hybrid = True
        else:
            image_embeddings = data['embeddings']
            text_embeddings = None
            use_hybrid = False
    except Exception as e:
        return False, f"Error loading embeddings: {e}"

    # Load Metadata
    try:
        with open(paths['metadata'], 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        frames = metadata['frames']
    except Exception as e:
        return False, f"Error loading metadata: {e}"

    current_movie = movie_name
    print(f"[+] Loaded '{movie_name}': {len(frames)} frames")
    return True, f"Loaded {len(frames)} frames"


def load_resources():
    """Initialize - load CLIP model"""
    print("[*] Loading SceneSearch...")
    load_clip_model()
    print("[!] Ready!\n")

def search(query: str, top_k: int, search_mode: str):
    """
    Hybrid search logic
    search_mode: "Visual (Image)", "Hybrid (Smart)", "Conceptual (Text)"
    Returns: (gallery_results, stats_html, timestamps_list)
    """
    global image_embeddings, text_embeddings, frames, use_hybrid, search_results_timestamps

    if image_embeddings is None or frames is None:
        search_results_timestamps = []
        return [], "⚠️ 먼저 '영상 처리' 탭에서 영상을 처리해주세요.", []

    if not query.strip():
        search_results_timestamps = []
        return [], "", []

    # 1. Encode Query
    t0 = time.perf_counter()
    text_tokens = tokenizer([query])
    with torch.no_grad():
        query_emb = clip_model.encode_text(text_tokens.to(device))
        query_emb /= query_emb.norm(dim=-1, keepdim=True)
    query_emb = query_emb.cpu().numpy().flatten()
    encode_time = time.perf_counter() - t0

    # 2. Determine Weights based on Mode
    w_img = 0.6
    w_txt = 0.4
    mode_desc = ""

    if search_mode == "Visual":
        w_img, w_txt = 1.0, 0.0
        mode_desc = "🖼️ Visual"
    elif search_mode == "Caption":
        w_img, w_txt = 0.0, 1.0
        mode_desc = "📝 Caption"
    else:  # Hybrid (Smart)
        if use_hybrid and text_embeddings is not None:
            word_count = len(query.split())
            if word_count <= 2:
                w_img, w_txt = 0.8, 0.2
                mode_desc = "⚡ Hybrid (Short Query)"
            else:
                w_img, w_txt = 0.5, 0.5
                mode_desc = "🧠 Hybrid (Balanced)"
        else:
            w_img, w_txt = 1.0, 0.0
            mode_desc = "🖼️ Image Only (No captions avail)"

    # 3. Calculate Similarity
    t1 = time.perf_counter()
    scores = image_embeddings @ query_emb

    if use_hybrid and text_embeddings is not None and w_txt > 0:
        txt_scores = text_embeddings @ query_emb
        scores = scores * w_img + txt_scores * w_txt

    top_indices = np.argsort(scores)[::-1][:top_k]
    search_time = time.perf_counter() - t1

    # 4. Format Results + Collect Timestamps
    results = []
    timestamps = []
    paths = get_movie_paths(current_movie)
    frames_dir = paths['frames']

    for idx in top_indices:
        frame = frames[idx]
        score = scores[idx]
        image_path = frames_dir / frame['filename']

        if image_path.exists():
            caption_text = frame.get('caption', '')
            timestamp = frame.get('timestamp', 0)
            timestamps.append(timestamp)

            # Display caption with click hint
            display_caption = f"⏱️ {frame['time_str']}  |  Score: {score:.3f}"
            if caption_text:
                short_cap = (caption_text[:60] + '..') if len(caption_text) > 60 else caption_text
                display_caption += f"\n📝 {short_cap}"
            display_caption += "\n🎯 Click to jump"

            results.append((str(image_path), display_caption))

    # Store timestamps globally for gallery click handler
    search_results_timestamps = timestamps
    
    # Collect top scores for visualization
    top_scores = [float(scores[idx]) for idx in top_indices]
    total_time = (encode_time + search_time) * 1000
    
    # ===== Generate Scatter Plot Data =====
    all_scores = scores.tolist()
    min_score, max_score = min(all_scores), max(all_scores)
    score_range = max_score - min_score if max_score > min_score else 0.1
    
    # SVG dimensions
    svg_w, svg_h = 500, 120
    pad_l, pad_r, pad_t, pad_b = 35, 10, 10, 25
    plot_w = svg_w - pad_l - pad_r
    plot_h = svg_h - pad_t - pad_b
    
    # Sample frames for performance (max 200 points)
    total_frames_count = len(all_scores)
    if total_frames_count > 200:
        step = total_frames_count // 200
        sample_idx = list(range(0, total_frames_count, step))
    else:
        sample_idx = list(range(total_frames_count))
    
    # Get max timestamp
    max_ts = max(frames[i].get('timestamp', i) for i in sample_idx) if sample_idx else 1
    
    # Scale functions
    def sx(ts): return pad_l + (ts / max_ts) * plot_w if max_ts > 0 else pad_l
    def sy(sc): return pad_t + plot_h - ((sc - min_score) / score_range) * plot_h
    
    # Generate background dots (sampled frames)
    dots = ""
    for i in sample_idx:
        ts = frames[i].get('timestamp', i)
        sc = all_scores[i]
        x, y = sx(ts), sy(sc)
        dots += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="#3d4555" opacity="0.5"><title>{format_time(ts)} | Score: {sc:.3f}</title></circle>'
    
    # Generate top result stars with glow effect
    top_dots = ""
    for rank, idx in enumerate(top_indices):
        ts = frames[idx].get('timestamp', idx)
        sc = all_scores[idx]
        x, y = sx(ts), sy(sc)
        top_dots += f'''
        <circle cx="{x:.1f}" cy="{y:.1f}" r="14" fill="#667eea" opacity="0.2" class="pulse-ring"/>
        <circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="#667eea" stroke="#a78bfa" stroke-width="2" class="top-dot"/>
        <text x="{x:.1f}" y="{y - 12:.1f}" text-anchor="middle" fill="#e2e8f0" font-size="9" font-weight="bold">#{rank+1}</text>
        <title>#{rank+1} | {format_time(ts)} | Score: {sc:.3f}</title>
        '''
    
    # Top results summary
    top_results_html = ""
    for i, (sc, ts) in enumerate(zip(top_scores[:5], timestamps[:5])):
        top_results_html += f'<div style="display:flex; justify-content:space-between; padding:4px 8px; background:#252a34; border-radius:4px; margin-bottom:4px;"><span style="color:#667eea;">#{i+1}</span><span style="color:#a0aec0;">{format_time(ts)}</span><span style="color:#48bb78;">{sc:.3f}</span></div>'

    # Visualization HTML
    stats = f"""
    <div style="background: linear-gradient(135deg, #1a1d24 0%, #252a34 100%); border-radius: 16px; padding: 20px; border: 1px solid #2d3748; margin-bottom: 20px;">
        
        <div style="text-align: center; margin-bottom: 20px;">
            <span style="font-size: 1.2rem; font-weight: 600; color: #e2e8f0;">🔮 Search Pipeline</span>
        </div>
        
        <div style="display: flex; align-items: center; justify-content: center; gap: 8px; margin-bottom: 24px; flex-wrap: wrap;">
            <div style="background: #2d3748; padding: 12px 16px; border-radius: 10px; text-align: center; min-width: 100px;">
                <div style="font-size: 1.5rem;">📝</div>
                <div style="color: #a0aec0; font-size: 0.75rem; margin-top: 4px;">Query</div>
                <div style="color: #e2e8f0; font-size: 0.85rem; font-weight: 500; margin-top: 2px;">"{query[:15]}{"..." if len(query) > 15 else ""}"</div>
            </div>
            <div style="color: #667eea; font-size: 1.2rem;">→</div>
            <div style="background: #2d3748; padding: 12px 16px; border-radius: 10px; text-align: center; min-width: 100px;">
                <div style="font-size: 1.5rem;">🧠</div>
                <div style="color: #a0aec0; font-size: 0.75rem; margin-top: 4px;">CLIP Encode</div>
                <div style="color: #48bb78; font-size: 0.85rem; font-weight: 500; margin-top: 2px;">{encode_time*1000:.1f}ms</div>
            </div>
            <div style="color: #667eea; font-size: 1.2rem;">→</div>
            <div style="background: #2d3748; padding: 12px 16px; border-radius: 10px; text-align: center; min-width: 100px;">
                <div style="font-size: 1.5rem;">🔢</div>
                <div style="color: #a0aec0; font-size: 0.75rem; margin-top: 4px;">Cosine Similarity</div>
                <div style="color: #48bb78; font-size: 0.85rem; font-weight: 500; margin-top: 2px;">{len(frames):,} frames</div>
            </div>
            <div style="color: #667eea; font-size: 1.2rem;">→</div>
            <div style="background: #2d3748; padding: 12px 16px; border-radius: 10px; text-align: center; min-width: 100px;">
                <div style="font-size: 1.5rem;">📊</div>
                <div style="color: #a0aec0; font-size: 0.75rem; margin-top: 4px;">Rank & Select</div>
                <div style="color: #48bb78; font-size: 0.85rem; font-weight: 500; margin-top: 2px;">Top {top_k}</div>
            </div>
            <div style="color: #667eea; font-size: 1.2rem;">→</div>
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 12px 16px; border-radius: 10px; text-align: center; min-width: 100px;">
                <div style="font-size: 1.5rem;">✨</div>
                <div style="color: rgba(255,255,255,0.8); font-size: 0.75rem; margin-top: 4px;">Results</div>
                <div style="color: white; font-size: 0.85rem; font-weight: 600; margin-top: 2px;">{len(results)} found</div>
            </div>
        </div>
        
        <div style="display: flex; justify-content: center; gap: 24px; margin-bottom: 20px; flex-wrap: wrap;">
            <div style="text-align: center;">
                <div style="color: #a0aec0; font-size: 0.75rem;">Mode</div>
                <div style="color: #667eea; font-weight: 600;">{mode_desc}</div>
            </div>
            <div style="text-align: center;">
                <div style="color: #a0aec0; font-size: 0.75rem;">Weights</div>
                <div style="color: #e2e8f0; font-weight: 500;">IMG {w_img:.0%} / TXT {w_txt:.0%}</div>
            </div>
            <div style="text-align: center;">
                <div style="color: #a0aec0; font-size: 0.75rem;">Total Time</div>
                <div style="color: #48bb78; font-weight: 600;">{total_time:.1f}ms</div>
            </div>
        </div>
        
        <div style="display: flex; gap: 20px; flex-wrap: wrap;">
            <div style="flex: 2; min-width: 280px;">
                <div style="color: #a0aec0; font-size: 0.8rem; margin-bottom: 8px; text-align: center;">
                    📈 Similarity Timeline (★ = top results, hover for details)
                </div>
                <style>
                    .pulse-ring {{ animation: pulse 2s ease-in-out infinite; }}
                    .top-dot {{ filter: drop-shadow(0 0 4px #667eea); }}
                    @keyframes pulse {{ 0%,100% {{ opacity:0.2; r:14; }} 50% {{ opacity:0.4; r:18; }} }}
                </style>
                <svg viewBox="0 0 {svg_w} {svg_h}" style="width:100%; background:#1a1d24; border-radius:8px; overflow:visible;">
                    <line x1="{pad_l}" y1="{svg_h - pad_b}" x2="{svg_w - pad_r}" y2="{svg_h - pad_b}" stroke="#2d3748"/>
                    <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{svg_h - pad_b}" stroke="#2d3748"/>
                    <text x="{pad_l - 3}" y="{pad_t + 4}" text-anchor="end" fill="#718096" font-size="8">{max_score:.2f}</text>
                    <text x="{pad_l - 3}" y="{svg_h - pad_b}" text-anchor="end" fill="#718096" font-size="8">{min_score:.2f}</text>
                    <text x="{pad_l}" y="{svg_h - 8}" fill="#718096" font-size="8">0:00</text>
                    <text x="{svg_w - pad_r}" y="{svg_h - 8}" text-anchor="end" fill="#718096" font-size="8">{format_time(max_ts)}</text>
                    {dots}
                    {top_dots}
                </svg>
            </div>
            <div style="flex: 1; min-width: 180px;">
                <div style="color: #a0aec0; font-size: 0.8rem; margin-bottom: 8px; text-align: center;">
                    🏆 Top Matches
                </div>
                <div style="background: #1a1d24; border-radius: 8px; padding: 8px;">
                    {top_results_html}
                </div>
            </div>
        </div>
        
    </div>
    """

    return results, stats, timestamps


def on_gallery_select(evt: gr.SelectData, timestamps_state):
    """Handle gallery image click - return timestamp to seek"""
    if timestamps_state and evt.index < len(timestamps_state):
        return timestamps_state[evt.index]
    return None

# ============================================================
# PROCESSING ENGINE
# ============================================================
def process_video(video_file, scene_threshold, progress=gr.Progress()):
    """Full pipeline: Extract -> Caption -> Embed"""

    if video_file is None:
        yield "❌ 영상 파일을 선택해주세요.", None
        return

    video_path = Path(video_file)
    movie_name = video_path.stem
    paths = get_movie_paths(movie_name)
    frames_dir = paths['frames']
    output_dir = paths['output_dir']

    log_messages = []

    def log(msg):
        timestamp = time.strftime("%H:%M:%S")
        log_messages.append(f"[{timestamp}] {msg}")
        return "\n".join(log_messages)

    try:
        # 1. Init
        yield log(f"🚀 Processing: {movie_name}"), None
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 2. FFmpeg Extraction
        yield log(f"🎬 Scene detection (threshold: {scene_threshold})"), None

        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", f"select='gt(scene,{scene_threshold})',showinfo",
            "-vsync", "vfr",
            str(frames_dir / "frame_%04d.jpg")
        ]

        proc = subprocess.Popen(
            cmd, stderr=subprocess.PIPE, universal_newlines=True, encoding='utf-8', errors='replace'
        )
        pts_times = []
        for line in proc.stderr:
            if "pts_time:" in line:
                match = re.search(r'pts_time:(\d+\.?\d*)', line)
                if match:
                    pts_times.append(float(match.group(1)))
        proc.wait()

        frame_files = list(frames_dir.glob("frame_*.jpg"))
        if not frame_files:
            yield log("❌ No frames extracted. Try lowering the threshold."), None
            return

        yield log(f"✅ Extracted {len(frame_files)} frames"), None

        # 3. Metadata Structure
        frames_data = []
        for i, pts in enumerate(pts_times):
            if i >= len(frame_files):
                break
            frames_data.append({
                "index": i,
                "filename": f"frame_{i+1:04d}.jpg",
                "timestamp": round(pts, 3),
                "time_str": format_time(pts)
            })

        # 4. AI Processing (CLIP + BLIP)
        yield log("🧠 Loading AI models..."), None

        proc_device = get_device()

        c_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        c_model = c_model.to(proc_device).eval()
        c_tokenizer = open_clip.get_tokenizer('ViT-B-32')

        from transformers import BlipProcessor, BlipForConditionalGeneration
        b_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        b_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(proc_device).eval()

        yield log("✅ Models loaded. Analyzing frames..."), None

        img_embs = []
        txt_embs = []

        start_time = time.time()
        total = len(frames_data)

        for i, frame_info in enumerate(frames_data):
            f_path = frames_dir / frame_info['filename']
            image = Image.open(f_path).convert('RGB')

            # CLIP Image Embedding
            img_in = preprocess(image).unsqueeze(0).to(proc_device)
            with torch.no_grad():
                ie = c_model.encode_image(img_in)
                ie /= ie.norm(dim=-1, keepdim=True)
            img_embs.append(ie.cpu().numpy().flatten())

            # BLIP Caption
            with torch.no_grad():
                inputs = b_processor(image, return_tensors="pt").to(proc_device)
                out = b_model.generate(**inputs, max_new_tokens=50)
                caption = b_processor.decode(out[0], skip_special_tokens=True)
            frame_info['caption'] = caption

            # CLIP Text Embedding (of Caption)
            txt_in = c_tokenizer([caption])
            with torch.no_grad():
                te = c_model.encode_text(txt_in.to(proc_device))
                te /= te.norm(dim=-1, keepdim=True)
            txt_embs.append(te.cpu().numpy().flatten())

            if (i + 1) % 10 == 0 or (i + 1) == total:
                elapsed = time.time() - start_time
                fps = (i + 1) / elapsed
                eta = (total - (i + 1)) / fps if fps > 0 else 0
                progress((i + 1) / total, desc=f"Analyzing... {i+1}/{total}")
                yield log(f"   ▶ [{i+1}/{total}] {fps:.1f} FPS (ETA: {eta:.0f}s)"), None

        # 5. Save Data
        yield log("💾 Saving data..."), None

        np.savez(
            paths['embeddings'],
            image_embeddings=np.array(img_embs),
            text_embeddings=np.array(txt_embs),
            timestamps=np.array([f['timestamp'] for f in frames_data])
        )

        metadata = {
            "video_info": {"filename": video_path.name, "movie_name": movie_name},
            "extraction_config": {"method": "scene_detection", "threshold": scene_threshold},
            "total_frames": total,
            "frames": frames_data
        }
        with open(paths['metadata'], 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        yield log(f"✨ Done! '{movie_name}' is ready for search."), None

        # Show Samples
        samples = []
        for idx in [0, total // 2, total - 1]:
            if idx < total:
                f = frames_data[idx]
                samples.append((str(frames_dir / f['filename']), f"[{f['time_str']}] {f['caption']}"))

        yield log(""), samples

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        yield log(f"❌ Error:\n{err}"), None

# ============================================================ 
# UI BUILDER
# ============================================================ 
def create_app():
    
    # Custom CSS
    css = """
    /* Main Layout */
    .container { max-width: 1200px; margin: auto; padding-top: 30px; }
    
    /* Dark Theme Base */
    .gradio-container {
        background: #0f1117 !important;
    }
    
    /* Header */
    .header { text-align: center; margin-bottom: 40px; }
    .header h1 { 
        font-size: 3em; 
        font-weight: 800; 
        margin-bottom: 10px; 
        background: -webkit-linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .header p { color: #718096; font-size: 1.2em; font-weight: 400; }
    
    /* Search Section */
    .search-row { 
        background: #1a1d24 !important; 
        padding: 30px; 
        border-radius: 20px; 
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        border: 1px solid #2d3748;
        margin-bottom: 30px;
    }
    
    /* Inputs */
    .search-input textarea { 
        font-size: 1.2rem !important; 
        padding: 15px !important; 
        border-radius: 12px !important; 
        border: 2px solid #2d3748 !important;
        background: #0f1117 !important;
        color: #e2e8f0 !important;
        transition: all 0.2s;
    }
    .search-input textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.25) !important;
    }
    
    /* Buttons */
    .primary-btn { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; 
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        border-radius: 12px !important;
        transition: transform 0.1s !important;
    }
    .primary-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Gallery - Dark Theme */
    .result-gallery { 
        border-radius: 16px; 
        overflow: hidden; 
        background: #1a1d24 !important;
        border: 1px solid #2d3748 !important;
        padding: 16px !important;
    }
    .result-gallery .grid-wrap { 
        gap: 16px !important; 
        background: #1a1d24 !important;
    }
    .result-gallery .thumbnail-item {
        background: #252a34 !important;
        border-radius: 12px !important;
        border: 1px solid #2d3748 !important;
    }
    .result-gallery .caption {
        background: #252a34 !important;
        color: #a0aec0 !important;
    }
    
    /* Labels & Text */
    .gr-box, .gr-panel {
        background: #1a1d24 !important;
        border-color: #2d3748 !important;
    }
    label, .label-wrap {
        color: #a0aec0 !important;
    }
    
    /* Radio Buttons */
    .gr-radio {
        background: #1a1d24 !important;
    }
    
    /* Slider */
    .gr-slider input[type="range"] {
        background: #2d3748 !important;
    }
    
    /* Tabs */
    .tabs {
        background: transparent !important;
        border: none !important;
    }
    .tab-nav {
        background: #1a1d24 !important;
        border-radius: 12px !important;
        border: 1px solid #2d3748 !important;
    }
    .tab-nav button {
        color: #a0aec0 !important;
    }
    .tab-nav button.selected {
        background: #667eea !important;
        color: white !important;
    }
    
    /* Logs */
    .log-area textarea { 
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 13px;
        background-color: #0f1117 !important;
        color: #a0aec0 !important;
        border-radius: 12px;
        border: 1px solid #2d3748 !important;
        line-height: 1.5;
    }
    
    /* Examples */
    .gr-examples {
        background: transparent !important;
    }
    .gr-examples button {
        background: #252a34 !important;
        color: #a0aec0 !important;
        border: 1px solid #2d3748 !important;
        border-radius: 8px !important;
    }
    .gr-examples button:hover {
        background: #667eea !important;
        color: white !important;
        border-color: #667eea !important;
    }
    
    /* Video Input */
    .gr-video {
        background: #1a1d24 !important;
        border: 2px dashed #2d3748 !important;
        border-radius: 16px !important;
    }
    
    /* Accordion */
    .gr-accordion {
        background: #1a1d24 !important;
        border: 1px solid #2d3748 !important;
        border-radius: 12px !important;
    }
    
    /* Gallery - Disable modal/lightbox on click */
    .result-gallery .preview,
    .result-gallery .modal,
    .result-gallery .backdrop,
    .result-gallery .fixed {
        display: none !important;
    }

    /* Footer */
    footer { display: none !important; }
    """

    # JavaScript for video seeking
    seek_js = """
    (timestamp) => {
        if (timestamp === null || timestamp === undefined) return timestamp;

        const video = document.querySelector('#main-player video');
        if (video) {
            video.currentTime = timestamp;
            video.pause();
        }

        return timestamp;
    }
    """

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", radius_size="lg"), css=css, title="SceneSearch") as app:

        # State for timestamps
        timestamps_state = gr.State([])

        with gr.Column(elem_classes="container"):

            # Header
            gr.HTML("""
                <div class="header">
                    <h1>SceneSearch</h1>
                    <p>Find any moment with natural language</p>
                </div>
            """)

            with gr.Tabs():

                # ---------------------------------------------------------
                # TAB 1: SEARCH + PLAYER
                # ---------------------------------------------------------
                with gr.Tab("🔍 Search", id="search_tab"):

                    # Movie Selection
                    with gr.Row():
                        movie_dropdown = gr.Dropdown(
                            choices=get_available_movies(),
                            label="Select Movie",
                            value=None,
                            scale=3
                        )
                        refresh_btn = gr.Button("🔄", scale=0, min_width=50)

                    # Status message
                    movie_status = gr.HTML(
                        value="<p style='text-align:center; color:#718096;'>Select a movie to start</p>"
                    )

                    # Video Player Section
                    with gr.Column(elem_classes="player-section"):
                        video_player = gr.Video(
                            label=None,
                            show_label=False,
                            elem_id="main-player",
                            height=400,
                            interactive=False
                        )

                    # Search Section
                    with gr.Column(elem_classes="search-row"):
                        query_input = gr.Textbox(
                            show_label=False,
                            placeholder="Describe the scene... (e.g., 'Johnny Depp smiling', 'explosion scene')",
                            lines=1,
                            elem_classes="search-input",
                            autofocus=True
                        )

                        with gr.Row(equal_height=True):
                            search_mode = gr.Radio(
                                choices=["Hybrid", "Visual", "Caption"],
                                value="Hybrid",
                                label="Mode",
                                scale=2
                            )
                            top_k = gr.Slider(4, 20, value=8, step=4, label="Results", scale=1)
                            search_btn = gr.Button("Search", variant="primary", scale=1, elem_classes="primary-btn")

                    # Results
                    stats_output = gr.HTML()
                    gallery = gr.Gallery(
                        label="Click a frame to jump to that moment",
                        columns=4,
                        height="auto",
                        object_fit="cover",
                        elem_classes="result-gallery",
                        show_label=True,
                        interactive=False,
                        preview=False
                    )

                    # Examples
                    gr.Examples(
                        examples=[
                            ["Johnny Depp face"],
                            ["computer screen"],
                            ["two people talking"],
                            ["explosion or fire"],
                            ["outdoor trees"]
                        ],
                        inputs=query_input,
                        label="Try:"
                    )
                    
                    # Hidden component for JS bridge
                    seek_timestamp = gr.Number(value=-1, visible=False)

                # ---------------------------------------------------------
                # TAB 2: PROCESSING
                # ---------------------------------------------------------
                with gr.Tab("⚙️ Process Video", id="process_tab"):
                    gr.Markdown("### Upload and Process Video")
                    with gr.Row():
                        with gr.Column(scale=1):
                            video_input = gr.Video(label="Video File", sources=["upload"])
                            threshold = gr.Slider(0.1, 0.5, value=0.3, step=0.05, label="Scene Sensitivity", info="Lower = More frames")
                            process_btn = gr.Button("Start Processing", variant="primary")

                        with gr.Column(scale=1):
                            log_output = gr.Textbox(
                                label="Log",
                                lines=15,
                                elem_classes="log-area",
                                interactive=False
                            )

                    gr.Markdown("### Sample Frames")
                    sample_output = gr.Gallery(label="", columns=4, height="auto")

        # ---------------------------------------------------------
        # EVENT HANDLERS
        # ---------------------------------------------------------

        # Movie selection -> load video + embeddings
        def on_movie_select(movie_name):
            if not movie_name:
                return None, "<p style='text-align:center; color:#718096;'>Select a movie to start</p>", [], ""

            paths = get_movie_paths(movie_name)

            # Check if processed
            if not is_movie_processed(movie_name):
                return (
                    None,
                    f"<p style='text-align:center; color:#f6ad55;'>⚠️ '{movie_name}' needs processing. Go to Process Video tab.</p>",
                    [],
                    ""
                )

            # Load embeddings
            success, msg = load_movie_data(movie_name)
            if not success:
                return None, f"<p style='text-align:center; color:#fc8181;'>❌ {msg}</p>", [], ""

            # Return video path and success status
            video_path = str(paths['video']) if paths['video'].exists() else None
            status_html = f"<p style='text-align:center; color:#48bb78;'>✓ Loaded: {movie_name} ({len(frames)} frames)</p>"

            return video_path, status_html, [], ""

        movie_dropdown.change(
            fn=on_movie_select,
            inputs=[movie_dropdown],
            outputs=[video_player, movie_status, gallery, stats_output]
        )

        # Refresh movie list
        def refresh_movies():
            movies = get_available_movies()
            return gr.update(choices=movies)

        refresh_btn.click(
            fn=refresh_movies,
            inputs=[],
            outputs=[movie_dropdown]
        )

        # Search -> update gallery + timestamps
        def do_search(query, k, mode):
            results, stats, ts = search(query, k, mode)
            return results, stats, ts

        search_btn.click(
            fn=do_search,
            inputs=[query_input, top_k, search_mode],
            outputs=[gallery, stats_output, timestamps_state]
        )
        query_input.submit(
            fn=do_search,
            inputs=[query_input, top_k, search_mode],
            outputs=[gallery, stats_output, timestamps_state]
        )

        # Gallery click -> seek video
        gallery.select(
            fn=on_gallery_select,
            inputs=[timestamps_state],
            outputs=[seek_timestamp]
        )
        
        # Trigger seek when timestamp changes
        seek_timestamp.change(
            fn=None,
            inputs=[seek_timestamp],
            outputs=[],
            js=seek_js
        )

        # Process video
        def do_process(video_file, thresh):
            for log_msg, samples in process_video(video_file, thresh):
                yield log_msg, samples

        process_btn.click(
            fn=do_process,
            inputs=[video_input, threshold],
            outputs=[log_output, sample_output]
        )

    return app

if __name__ == "__main__":
    load_resources()
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7880,
        inbrowser=True
    )