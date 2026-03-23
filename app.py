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
import threading
import base64
from pathlib import Path
from PIL import Image
import google.generativeai as genai

# Paths
BASE_DIR = Path(__file__).parent
VIDEO_DIR = BASE_DIR / "video"
OUTPUT_DIR = BASE_DIR / "output"

# Gemini API Setup
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')

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
stop_event = threading.Event()  # Stop signal for processing
current_ffmpeg_proc = None  # Reference to running ffmpeg process

# Benchmark models definition
BENCHMARK_MODELS = {
    'ViT-B-32': {'pretrained': 'openai', 'dim': 512, 'params': '88M', 'patch': 32, 'desc': 'Fastest, lowest accuracy', 'hf_repo': 'timm/vit_base_patch32_clip_224.openai'},
    'ViT-B-16': {'pretrained': 'openai', 'dim': 512, 'params': '86M', 'patch': 16, 'desc': 'Best speed/accuracy balance', 'hf_repo': 'timm/vit_base_patch16_clip_224.openai'},
    'ViT-L-14': {'pretrained': 'openai', 'dim': 768, 'params': '304M', 'patch': 14, 'desc': 'High accuracy, slower', 'hf_repo': 'timm/vit_large_patch14_clip_224.openai'},
    'ViT-H-14': {'pretrained': 'laion2b_s32b_b79k', 'dim': 1024, 'params': '632M', 'patch': 14, 'desc': 'Very high accuracy', 'hf_repo': 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'},
    'ViT-g-14': {'pretrained': 'laion2b_s34b_b88k', 'dim': 1024, 'params': '1.8B', 'patch': 14, 'desc': 'Highest accuracy, slowest', 'hf_repo': 'laion/CLIP-ViT-g-14-laion2B-s34B-b88K'},
}

def is_model_downloaded(model_name):
    """Check if a CLIP model is already downloaded in HuggingFace cache"""
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        hf_repo = BENCHMARK_MODELS.get(model_name, {}).get('hf_repo', '')
        return any(r.repo_id == hf_repo for r in info.repos)
    except Exception:
        return False

def download_model(model_name):
    """Download a CLIP model to local cache (generator for progress)"""
    info = BENCHMARK_MODELS.get(model_name)
    if not info:
        yield f"❌ Unknown model: {model_name}", ""
        return
    
    if is_model_downloaded(model_name):
        yield f"✅ {model_name} is already installed", ""
        return
    
    yield f"⬇️ Downloading {model_name} ({info['params']})...", f"""
    <div style="background:#1a1d24; border-radius:10px; padding:12px; border:1px solid #2d3748;">
        <div style="color:#e2e8f0; font-size:0.85rem; margin-bottom:6px;">⬇️ Downloading {model_name}...</div>
        <div style="background:#2d3748; border-radius:6px; height:6px; overflow:hidden;">
            <div style="background:linear-gradient(90deg,#667eea,#764ba2); height:100%; width:100%; border-radius:6px; animation:pulse 1.5s ease-in-out infinite;"></div>
        </div>
        <div style="color:#718096; font-size:0.7rem; margin-top:4px;">This may take a few minutes depending on your connection</div>
    </div>"""
    
    try:
        m, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=info['pretrained'])
        del m
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        yield f"✅ {model_name} downloaded successfully!", ""
    except Exception as e:
        yield f"❌ Failed to download {model_name}: {e}", ""

# ============================================================
# UTILS
# ============================================================
def contains_korean(text):
    """Check if text contains Korean characters"""
    return bool(re.search(r'[가-힣]', text))

def translate_to_english(query):
    """Translate Korean query to English using Gemini for CLIP search
    Returns: (translated_text, error_message or None)
    """
    try:
        prompt = f"""Translate the following Korean text to English.
The translation should be optimized for image search (CLIP model).
Return ONLY the English translation, nothing else.
Keep it concise and descriptive.

Korean: {query}
English:"""
        response = gemini_model.generate_content(prompt)
        translated = response.text.strip()
        print(f"[Gemini] '{query}' → '{translated}'")
        return translated, None
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            error_msg = "API 할당량 초과 (잠시 후 다시 시도)"
        elif "API_KEY" in error_msg:
            error_msg = "API 키가 유효하지 않음"
        else:
            error_msg = f"번역 실패: {error_msg[:50]}"
        print(f"[Gemini Error] {e}")
        return query, error_msg  # Fallback to original + error

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

def get_video_duration(video_path):
    """Get video duration in seconds using ffprobe"""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception:
        return 0

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

def get_movie_status_html(movie_name):
    """Generate HTML status panel for a movie's processing state"""
    if not movie_name:
        return '<div style="color:#718096; text-align:center; padding:20px;">Select a movie to view status</div>'

    paths = get_movie_paths(movie_name)
    video_exists = paths['video'].exists()
    frames_dir = paths['frames']
    frames_exist = frames_dir.exists() and any(frames_dir.glob("frame_*.jpg"))
    frame_count = len(list(frames_dir.glob("frame_*.jpg"))) if frames_exist else 0
    metadata_exists = paths['metadata'].exists()
    embeddings_exists = paths['embeddings'].exists()

    # Get file sizes
    def fmt_size(path):
        if not path.exists():
            return ""
        size = path.stat().st_size
        if size > 1024 * 1024:
            return f"{size / 1024 / 1024:.1f} MB"
        elif size > 1024:
            return f"{size / 1024:.1f} KB"
        return f"{size} B"

    # Caption count from metadata
    caption_count = 0
    if metadata_exists:
        try:
            with open(paths['metadata'], 'r', encoding='utf-8') as f:
                meta = json.load(f)
            caption_count = sum(1 for f in meta.get('frames', []) if f.get('caption'))
        except:
            pass

    def check_row(label, exists, detail=""):
        icon = "✅" if exists else "⬜"
        color = "#48bb78" if exists else "#718096"
        detail_html = f'<span style="color:#a0aec0; font-size:0.8rem; margin-left:8px;">{detail}</span>' if detail else ""
        return f'''
        <div style="display:flex; align-items:center; gap:10px; padding:8px 12px; background:#252a34; border-radius:8px; margin-bottom:6px;">
            <span style="font-size:1.2rem;">{icon}</span>
            <span style="color:{color}; font-weight:500;">{label}</span>
            {detail_html}
        </div>'''

    rows = []
    rows.append(check_row("Video File", video_exists, fmt_size(paths['video']) if video_exists else "not found"))
    rows.append(check_row("Frames Extracted", frames_exist, f"{frame_count} frames" if frames_exist else ""))
    rows.append(check_row("Metadata", metadata_exists, fmt_size(paths['metadata']) if metadata_exists else ""))
    rows.append(check_row("Captions", caption_count > 0, f"{caption_count} captions" if caption_count > 0 else ""))
    rows.append(check_row("Embeddings", embeddings_exists, fmt_size(paths['embeddings']) if embeddings_exists else ""))

    all_done = video_exists and frames_exist and metadata_exists and embeddings_exists and caption_count > 0
    status_badge = '<span style="background:#48bb78; color:#1a1d24; padding:4px 12px; border-radius:12px; font-weight:600; font-size:0.8rem;">Ready for Search</span>' if all_done else '<span style="background:#f59e0b; color:#1a1d24; padding:4px 12px; border-radius:12px; font-weight:600; font-size:0.8rem;">Needs Processing</span>'

    return f'''
    <div style="background:#1a1d24; border-radius:12px; padding:16px; border:1px solid #2d3748;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <span style="color:#e2e8f0; font-weight:600; font-size:1.1rem;">📁 {movie_name}</span>
            {status_badge}
        </div>
        {"".join(rows)}
    </div>
    '''

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

    # 0. (Translation disabled - English only for now)
    original_query = query
    translated_query = None

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

    # Translation info box
    translation_html = ""
    if translated_query:
        translation_html = f"""
        <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d3748 100%); border-radius: 12px; padding: 14px 18px; margin-bottom: 16px; border: 1px solid #3b82f6; display: flex; align-items: center; gap: 14px;">
            <div style="font-size: 1.8rem;">🌐</div>
            <div style="flex: 1;">
                <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 4px;">Translated for search</div>
                <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                    <span style="color: #e2e8f0; font-size: 1rem;">{original_query}</span>
                    <span style="color: #3b82f6; font-size: 1.2rem;">→</span>
                    <span style="color: #60a5fa; font-size: 1.05rem; font-weight: 600;">{translated_query}</span>
                </div>
            </div>
        </div>
        """

    # Visualization HTML
    stats = f"""
    {translation_html}
    <style>
        @keyframes cardSlideIn {{
            0% {{ opacity: 0; transform: translateY(20px) scale(0.9); }}
            100% {{ opacity: 1; transform: translateY(0) scale(1); }}
        }}
        @keyframes arrowFade {{
            0% {{ opacity: 0; transform: translateX(-10px); }}
            100% {{ opacity: 1; transform: translateX(0); }}
        }}
        .pipeline-card {{
            background: #2d3748;
            padding: 12px 16px;
            border-radius: 10px;
            text-align: center;
            min-width: 100px;
            opacity: 0;
            animation: cardSlideIn 0.4s ease-out forwards;
        }}
        .pipeline-card.final {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .pipeline-arrow {{
            color: #667eea;
            font-size: 1.2rem;
            opacity: 0;
            animation: arrowFade 0.3s ease-out forwards;
        }}
        .card-0 {{ animation-delay: 0s; }}
        .arrow-0 {{ animation-delay: 0.15s; }}
        .card-1 {{ animation-delay: 0.2s; }}
        .arrow-1 {{ animation-delay: 0.35s; }}
        .card-2 {{ animation-delay: 0.4s; }}
        .arrow-2 {{ animation-delay: 0.55s; }}
        .card-3 {{ animation-delay: 0.6s; }}
        .arrow-3 {{ animation-delay: 0.75s; }}
        .card-4 {{ animation-delay: 0.8s; }}
    </style>
    <div style="background: linear-gradient(135deg, #1a1d24 0%, #252a34 100%); border-radius: 16px; padding: 20px; border: 1px solid #2d3748; margin-bottom: 20px;">

        <div style="text-align: center; margin-bottom: 20px;">
            <span style="font-size: 1.2rem; font-weight: 600; color: #e2e8f0;">🔮 Search Pipeline</span>
        </div>

        <div style="display: flex; align-items: center; justify-content: center; gap: 8px; margin-bottom: 24px; flex-wrap: wrap;">
            <div class="pipeline-card card-0">
                <div style="font-size: 1.5rem;">{"🌐" if translated_query else "📝"}</div>
                <div style="color: #a0aec0; font-size: 0.75rem; margin-top: 4px;">{"Translated" if translated_query else "Query"}</div>
                <div style="color: #e2e8f0; font-size: 0.85rem; font-weight: 500; margin-top: 2px;">"{query[:15]}{"..." if len(query) > 15 else ""}"</div>
                {f'<div style="color: #667eea; font-size: 0.65rem; margin-top: 2px;">({original_query[:10]}{"..." if len(original_query) > 10 else ""})</div>' if translated_query else ""}
            </div>
            <div class="pipeline-arrow arrow-0">→</div>
            <div class="pipeline-card card-1">
                <div style="font-size: 1.5rem;">🧠</div>
                <div style="color: #a0aec0; font-size: 0.75rem; margin-top: 4px;">Image Encode</div>
                <div style="color: #48bb78; font-size: 0.85rem; font-weight: 500; margin-top: 2px;">{encode_time*1000:.1f}ms</div>
            </div>
            <div class="pipeline-arrow arrow-1">→</div>
            <div class="pipeline-card card-2">
                <div style="font-size: 1.5rem;">🔢</div>
                <div style="color: #a0aec0; font-size: 0.75rem; margin-top: 4px;">Cosine Similarity</div>
                <div style="color: #48bb78; font-size: 0.85rem; font-weight: 500; margin-top: 2px;">{len(frames):,} frames</div>
            </div>
            <div class="pipeline-arrow arrow-2">→</div>
            <div class="pipeline-card card-3">
                <div style="font-size: 1.5rem;">📊</div>
                <div style="color: #a0aec0; font-size: 0.75rem; margin-top: 4px;">Rank & Select</div>
                <div style="color: #48bb78; font-size: 0.85rem; font-weight: 500; margin-top: 2px;">Top {top_k}</div>
            </div>
            <div class="pipeline-arrow arrow-3">→</div>
            <div class="pipeline-card card-4 final">
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
                <div style="display: flex; align-items: center; justify-content: center; gap: 16px; margin-bottom: 10px; flex-wrap: wrap;">
                    <span style="color: #a0aec0; font-size: 0.85rem;">📈 Similarity Timeline</span>
                    <div style="display: flex; align-items: center; gap: 12px; background: #252a34; padding: 6px 12px; border-radius: 20px;">
                        <div style="display: flex; align-items: center; gap: 4px;">
                            <span style="display: inline-block; width: 8px; height: 8px; background: #3d4555; border-radius: 50%;"></span>
                            <span style="color: #718096; font-size: 0.7rem;">All frames</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 4px;">
                            <span style="display: inline-block; width: 10px; height: 10px; background: #667eea; border-radius: 50%; box-shadow: 0 0 6px #667eea;"></span>
                            <span style="color: #a78bfa; font-size: 0.7rem;">Top results</span>
                        </div>
                    </div>
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
def make_progress_html(phase, current=0, total=0, fps=0, eta=0, caption=""):
    """Generate HTML progress bar with step indicator cards"""
    hint = ""
    caption_html = ""
    
    if phase == "extract":
        pct = (current / total * 100) if total > 0 else 0
        frame_count = int(eta)
        status_icon = "🎬"
        status_text = f"Extracting frames... {frame_count} found"
        if total > 0 and current > 0:
            sub_text = f"Scanning video · {format_time(current)} / {format_time(total)}"
            pulse = False
        else:
            sub_text = "Starting ffmpeg scene detection..."
            pulse = True
        bar_color = "linear-gradient(90deg, #f59e0b, #d97706)"
        hint = f"💡 장면 전환(scene change)이 설정값({caption})을 초과하는 순간만 키프레임으로 추출합니다" if caption else "💡 장면 전환(scene change)이 감지된 순간만 키프레임으로 추출합니다"
    elif phase == "model_load":
        pct = 0
        status_icon = "🧠"
        status_text = "Loading AI models..."
        sub_text = "CLIP + BLIP (first time may take a minute)"
        bar_color = "linear-gradient(90deg, #8b5cf6, #6d28d9)"
        pulse = True
    elif phase == "analyze":
        pct = (current / total * 100) if total > 0 else 0
        status_icon = "⚡"
        status_text = f"Analyzing frames... {current}/{total}"
        eta_str = f"{int(eta//60)}분 {int(eta%60)}초" if eta >= 60 else f"{eta:.0f}초"
        sub_text = f"처리 속도: {fps:.1f} frames/sec · 남은 시간: {eta_str}"
        if caption:
            cap_text = caption[:55] + '...' if len(caption) > 55 else caption
            caption_html = f'<div style="color:#a78bfa; font-size:0.75rem; margin-top:6px; height:20px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">📝 {cap_text}</div>'
        bar_color = "linear-gradient(90deg, #667eea, #764ba2)"
        pulse = False
        hint = "💡 CLIP으로 이미지 임베딩, BLIP으로 캡션 생성 후 캡션도 임베딩합니다"
    elif phase == "save":
        pct = 100
        status_icon = "💾"
        status_text = "Saving data..."
        sub_text = "Writing embeddings and metadata"
        bar_color = "linear-gradient(90deg, #10b981, #059669)"
        pulse = False
    elif phase == "done":
        pct = 100
        status_icon = "✅"
        status_text = f"Complete! {total} frames processed"
        sub_text = "Ready for search"
        bar_color = "linear-gradient(90deg, #10b981, #059669)"
        pulse = False
    elif phase == "stopped":
        pct = (current / total * 100) if total > 0 else 0
        status_icon = "⛔"
        status_text = "Stopped by user"
        sub_text = f"{int(eta)} frames extracted before stop" if eta > 0 else "Processing cancelled"
        bar_color = "linear-gradient(90deg, #ef4444, #b91c1c)"
        pulse = False
    else:
        return ""

    pulse_css = """
        @keyframes barPulse {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
    """ if pulse else ""
    pulse_style = "background-size: 200% 100%; animation: barPulse 2s ease-in-out infinite;" if pulse else ""
    bar_width = "100%" if pulse else f"{pct}%"
    pct_display = f'<div style="color: #667eea; font-size: 1.3rem; font-weight: 700;">{pct:.1f}%</div>' if not pulse else ""

    # Step indicator cards
    steps = [
        ("extract", "🎬", "Extract"),
        ("model_load", "🧠", "Model"),
        ("analyze", "⚡", "Analyze"),
        ("save", "💾", "Save"),
        ("done", "✅", "Done"),
    ]
    step_order = [s[0] for s in steps]
    current_idx = step_order.index(phase) if phase in step_order else -1
    
    step_cards = ""
    for i, (step_id, icon, label) in enumerate(steps):
        if phase == "stopped":
            # stopped state
            bg = "#1a1d24"
            border = "1px solid #2d3748"
            label_color = "#4a5568"
            icon_opacity = "0.4"
            glow = ""
        elif i < current_idx:
            # completed step
            bg = "linear-gradient(135deg, #1a3a2a 0%, #1a2e1a 100%)"
            border = "1px solid #22c55e44"
            label_color = "#48bb78"
            icon_opacity = "0.9"
            glow = ""
        elif i == current_idx:
            # active step
            bg = "linear-gradient(135deg, #1e2a5f 0%, #2d1b69 100%)"
            border = "1px solid #667eea"
            label_color = "#e2e8f0"
            icon_opacity = "1"
            glow = "box-shadow: 0 0 12px rgba(102, 126, 234, 0.4);"
        else:
            # future step
            bg = "#1a1d24"
            border = "1px solid #2d3748"
            label_color = "#4a5568"
            icon_opacity = "0.4"
            glow = ""
        
        connector = ""
        if i < len(steps) - 1:
            conn_color = "#22c55e" if i < current_idx else ("#667eea" if i == current_idx else "#2d3748")
            connector = f'<div style="color:{conn_color}; font-size:0.7rem; display:flex; align-items:center; opacity:0.8;">▸</div>'
        
        step_cards += f'''<div style="display:flex; align-items:center; gap:4px;">
            <div style="background:{bg}; border:{border}; border-radius:10px; padding:6px 10px; text-align:center; min-width:55px; transition: all 0.3s ease; {glow}">
                <div style="font-size:1rem; opacity:{icon_opacity};">{icon}</div>
                <div style="color:{label_color}; font-size:0.65rem; font-weight:600; margin-top:2px;">{label}</div>
            </div>
            {connector}
        </div>'''

    return f"""
    <style>{pulse_css}</style>
    <div style="background: #1a1d24; border-radius: 16px; padding: 20px; border: 1px solid #2d3748;">
        <div style="display:flex; align-items:center; justify-content:center; gap:4px; margin-bottom:14px; flex-wrap:wrap;">
            {step_cards}
        </div>
        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 12px; min-height: 52px;">
            <div style="font-size: 1.8rem;">{status_icon}</div>
            <div style="flex: 1; min-width: 0;">
                <div style="color: #e2e8f0; font-size: 1rem; font-weight: 600;">{status_text}</div>
                <div style="color: #718096; font-size: 0.8rem; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{sub_text}</div>
                {caption_html}
            </div>
            {pct_display}
        </div>
        <div style="background: #2d3748; border-radius: 8px; height: 10px; overflow: hidden;">
            <div style="background: {bar_color}; height: 100%; border-radius: 8px; width: {bar_width}; transition: width 0.3s ease; {pulse_style}"></div>
        </div>
        {f'<div style="color:#5a6577; font-size:0.7rem; margin-top:8px; text-align:center;">{hint}</div>' if hint else ''}
    </div>
    """


def process_video(video_file, scene_threshold):
    """Full pipeline: Extract -> Caption -> Embed
    Yields: (log_text, progress_html, sample_gallery)
    """
    global current_ffmpeg_proc

    if video_file is None:
        yield "❌ 영상 파일을 선택해주세요.", "", None
        return

    stop_event.clear()

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
        # 1. Init - Copy to video/ folder if uploaded from elsewhere
        yield log(f"🚀 Processing: {movie_name}"), "", None

        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        dest_video = VIDEO_DIR / video_path.name
        if not dest_video.exists() or dest_video.resolve() != video_path.resolve():
            yield log(f"📁 Copying video to video/ folder..."), "", None
            shutil.copy2(str(video_path), str(dest_video))
            yield log(f"✅ Saved: {dest_video.name}"), "", None
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 2. FFmpeg Extraction (with live progress)
        video_duration = get_video_duration(video_path)
        duration_str = format_time(video_duration) if video_duration > 0 else "unknown"
        yield log(f"🎬 Scene detection (threshold: {scene_threshold}) · Duration: {duration_str}"), make_progress_html("extract", 0, video_duration, 0, 0, str(scene_threshold)), None

        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", f"select='gt(scene,{scene_threshold})',showinfo",
            "-vsync", "vfr",
            str(frames_dir / "frame_%04d.jpg")
        ]

        proc = subprocess.Popen(
            cmd, stderr=subprocess.PIPE, universal_newlines=True, encoding='utf-8', errors='replace'
        )
        current_ffmpeg_proc = proc
        pts_times = []
        last_update = time.time()
        for line in proc.stderr:
            if stop_event.is_set():
                proc.terminate()
                proc.wait()
                current_ffmpeg_proc = None
                last_pts = pts_times[-1] if pts_times else 0
                yield log("⛔ Stopped by user."), make_progress_html("stopped", last_pts, video_duration, 0, len(pts_times)), None
                return
            if "pts_time:" in line:
                match = re.search(r'pts_time:(\d+\.?\d*)', line)
                if match:
                    pts_times.append(float(match.group(1)))
                    # Update UI every 2 seconds
                    now = time.time()
                    if now - last_update > 2:
                        last_update = now
                        last_pts = pts_times[-1]
                        yield log(f"   ▶ Extracting... {len(pts_times)} frames ({format_time(last_pts)})"), make_progress_html("extract", last_pts, video_duration, 0, len(pts_times), str(scene_threshold)), None
        proc.wait()
        current_ffmpeg_proc = None

        frame_files = list(frames_dir.glob("frame_*.jpg"))
        if not frame_files:
            yield log("❌ No frames extracted. Try lowering the threshold."), "", None
            return

        yield log(f"✅ Extracted {len(frame_files)} frames"), make_progress_html("extract", video_duration, video_duration, 0, len(frame_files), str(scene_threshold)), None

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
        yield log("🧠 Loading AI models..."), make_progress_html("model_load"), None

        proc_device = get_device()

        c_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        c_model = c_model.to(proc_device).eval()
        c_tokenizer = open_clip.get_tokenizer('ViT-B-32')

        from transformers import BlipProcessor, BlipForConditionalGeneration
        b_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        b_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(proc_device).eval()

        yield log("✅ Models loaded. Analyzing frames..."), make_progress_html("analyze", 0, len(frames_data)), None

        img_embs = []
        txt_embs = []

        start_time = time.time()
        total = len(frames_data)

        for i, frame_info in enumerate(frames_data):
            if stop_event.is_set():
                yield log(f"⛔ Stopped by user. ({i}/{total} frames processed)"), make_progress_html("stopped", i, total, 0, i), None
                return

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

            if (i + 1) % 5 == 0 or (i + 1) == total:
                elapsed = time.time() - start_time
                fps = (i + 1) / elapsed
                eta = (total - (i + 1)) / fps if fps > 0 else 0
                yield log(f"   ▶ [{i+1}/{total}] {fps:.1f} FPS (ETA: {eta:.0f}s)"), make_progress_html("analyze", i + 1, total, fps, eta, caption), None

        # 5. Save Data
        yield log("💾 Saving data..."), make_progress_html("save"), None

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

        yield log(f"✨ Done! '{movie_name}' is ready for search."), make_progress_html("done", total=total), None

        # Show Samples
        samples = []
        for idx in [0, total // 2, total - 1]:
            if idx < total:
                f = frames_data[idx]
                samples.append((str(frames_dir / f['filename']), f"[{f['time_str']}] {f['caption']}"))

        yield log(""), make_progress_html("done", total=total), samples

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        yield log(f"❌ Error:\n{err}"), make_progress_html("stopped", 0, 0, 0, 0), None

# ============================================================ 
# BENCHMARK ENGINE
# ============================================================ 
def get_benchmark_emb_path(movie_name, model_name):
    """Get embeddings file path for a specific model"""
    slug = model_name.replace('-', '_')
    return OUTPUT_DIR / movie_name / f'embeddings_{slug}.npz'

def benchmark_check_status(movie_name):
    """Check which models have embeddings for a given movie"""
    if not movie_name:
        return {}
    status = {}
    for model_name in BENCHMARK_MODELS:
        path = get_benchmark_emb_path(movie_name, model_name)
        # ViT-B-32 also uses the default embeddings.npz
        if model_name == 'ViT-B-32':
            default_path = get_movie_paths(movie_name)['embeddings']
            status[model_name] = path.exists() or default_path.exists()
        else:
            status[model_name] = path.exists()
    return status

def benchmark_generate(movie_name, model_name):
    """Generate embeddings for a movie with a specific CLIP model (generator)"""
    paths = get_movie_paths(movie_name)
    frames_dir = paths['frames']
    metadata_path = paths['metadata']
    
    if not frames_dir.exists() or not metadata_path.exists():
        yield "❌ Movie not processed yet. Run Process tab first.", "", None
        return
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    frame_list = meta['frames']
    total = len(frame_list)
    
    if total == 0:
        yield "❌ No frames found.", "", None
        return
    
    model_info = BENCHMARK_MODELS[model_name]
    out_path = get_benchmark_emb_path(movie_name, model_name)
    
    yield f"🧠 Loading {model_name}...", _bench_progress_html(model_name, "loading", 0, total), None
    
    proc_device = get_device()
    try:
        m, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=model_info['pretrained'])
        m = m.to(proc_device).eval()
        tok = open_clip.get_tokenizer(model_name)
    except Exception as e:
        yield f"❌ Failed to load {model_name}: {e}", "", None
        return
    
    # BLIP for captions (reuse if already have captions)
    has_captions = all(f.get('caption') for f in frame_list)
    b_processor, b_model = None, None
    if not has_captions:
        yield f"🧠 Loading BLIP for captions...", _bench_progress_html(model_name, "loading", 0, total), None
        from transformers import BlipProcessor, BlipForConditionalGeneration
        b_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        b_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(proc_device)
    
    yield f"⚡ Encoding frames with {model_name}...", _bench_progress_html(model_name, "encoding", 0, total), None
    
    img_embs = []
    txt_embs = []
    t_start = time.perf_counter()
    
    for i, frame_info in enumerate(frame_list):
        img_path = frames_dir / frame_info['filename']
        if not img_path.exists():
            continue
        
        img = Image.open(img_path).convert('RGB')
        
        # Image embedding
        img_tensor = preprocess(img).unsqueeze(0).to(proc_device)
        with torch.no_grad():
            img_emb = m.encode_image(img_tensor)
            img_emb /= img_emb.norm(dim=-1, keepdim=True)
        img_embs.append(img_emb.cpu().numpy().flatten())
        
        # Caption
        caption = frame_info.get('caption', '')
        if not caption and b_processor and b_model:
            inputs = b_processor(images=img, return_tensors="pt").to(proc_device)
            with torch.no_grad():
                out = b_model.generate(**inputs, max_new_tokens=50)
            caption = b_processor.decode(out[0], skip_special_tokens=True)
            frame_info['caption'] = caption
        
        # Caption text embedding
        if caption:
            cap_tokens = tok([caption])
            with torch.no_grad():
                cap_emb = m.encode_text(cap_tokens.to(proc_device))
                cap_emb /= cap_emb.norm(dim=-1, keepdim=True)
            txt_embs.append(cap_emb.cpu().numpy().flatten())
        else:
            txt_embs.append(np.zeros(model_info['dim']))
        
        if (i + 1) % 5 == 0 or i == total - 1:
            elapsed = time.perf_counter() - t_start
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / fps if fps > 0 else 0
            pct = (i + 1) / total * 100
            cap_preview = caption[:50] + '...' if len(caption) > 50 else caption
            yield (
                f"⚡ {model_name}: {i+1}/{total} ({pct:.0f}%) | {fps:.1f} fps | ETA {eta:.0f}s\n📝 {cap_preview}",
                _bench_progress_html(model_name, "encoding", i + 1, total, fps, eta),
                None
            )
    
    # Save
    elapsed_total = time.perf_counter() - t_start
    np.savez_compressed(str(out_path),
                        image_embeddings=np.array(img_embs),
                        text_embeddings=np.array(txt_embs))
    
    # Update metadata if new captions were generated
    if not has_captions:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    
    # Cleanup GPU
    del m, tok
    if b_model:
        del b_processor, b_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    yield (
        f"✅ {model_name} done! {total} frames in {elapsed_total:.1f}s ({total/elapsed_total:.1f} fps)",
        _bench_progress_html(model_name, "done", total, total),
        None
    )

def _bench_progress_html(model_name, phase, current, total, fps=0, eta=0):
    """Generate compact progress HTML for benchmark"""
    pct = (current / total * 100) if total > 0 else 0
    info = BENCHMARK_MODELS.get(model_name, {})
    
    if phase == "loading":
        status = "Loading model..."
        bar_color = "#f59e0b"
        pct = 0
    elif phase == "encoding":
        eta_str = f"{eta:.0f}s" if eta < 60 else f"{int(eta//60)}m {int(eta%60)}s"
        status = f"{current}/{total} frames · {fps:.1f} fps · ETA {eta_str}"
        bar_color = "#667eea"
    elif phase == "done":
        status = f"Complete — {total} frames"
        bar_color = "#48bb78"
        pct = 100
    else:
        status = "Ready"
        bar_color = "#4a5568"
    
    return f"""
    <div style="background:#1a1d24; border-radius:12px; padding:14px; border:1px solid #2d3748; margin-bottom:8px;">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
            <span style="color:#e2e8f0; font-weight:600;">{model_name}</span>
            <span style="color:#718096; font-size:0.75rem;">{info.get('params','')} · {info.get('dim','')}d</span>
        </div>
        <div style="color:#a0aec0; font-size:0.8rem; margin-bottom:6px;">{status}</div>
        <div style="background:#2d3748; border-radius:6px; height:8px; overflow:hidden;">
            <div style="background:{bar_color}; height:100%; width:{pct}%; border-radius:6px; transition:width 0.3s;"></div>
        </div>
    </div>
    """

def benchmark_search_compare(movie_name, model_names, query, top_k=8):
    """Run search with multiple models and return comparison HTML"""
    if not movie_name or not query.strip():
        return "<p style='color:#718096; text-align:center;'>Select a movie and enter a query</p>", []
    
    paths = get_movie_paths(movie_name)
    frames_dir = paths['frames']
    metadata_path = paths['metadata']
    
    if not metadata_path.exists():
        return "<p style='color:#fc8181; text-align:center;'>Movie not processed</p>", []
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    frame_list = meta['frames']
    
    # (Translation disabled - English only for now)
    search_query = query
    translated = None
    
    results_per_model = {}
    
    for model_name in model_names:
        info = BENCHMARK_MODELS[model_name]
        
        # Load embeddings
        emb_path = get_benchmark_emb_path(movie_name, model_name)
        if not emb_path.exists() and model_name == 'ViT-B-32':
            emb_path = paths['embeddings']
        
        if not emb_path.exists():
            results_per_model[model_name] = {'error': 'Embeddings not found. Prepare this model first.'}
            continue
        
        data = np.load(str(emb_path))
        img_embs = data.get('image_embeddings', data.get('embeddings'))
        txt_embs = data.get('text_embeddings')
        
        # Load model
        t_load = time.perf_counter()
        try:
            m, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=info['pretrained'])
            m = m.to(get_device()).eval()
            tok = open_clip.get_tokenizer(model_name)
        except Exception as e:
            results_per_model[model_name] = {'error': f'Model load failed: {e}'}
            continue
        load_time = time.perf_counter() - t_load
        
        # Encode query
        t_enc = time.perf_counter()
        text_tokens = tok([search_query])
        with torch.no_grad():
            q_emb = m.encode_text(text_tokens.to(get_device()))
            q_emb /= q_emb.norm(dim=-1, keepdim=True)
        q_emb = q_emb.cpu().numpy().flatten()
        encode_time = time.perf_counter() - t_enc
        
        # Search
        t_search = time.perf_counter()
        img_scores = img_embs @ q_emb
        if txt_embs is not None and len(txt_embs) > 0:
            txt_scores = txt_embs @ q_emb
            scores = img_scores * 0.6 + txt_scores * 0.4
        else:
            scores = img_scores
        
        top_idx = np.argsort(scores)[::-1][:top_k]
        search_time = time.perf_counter() - t_search
        
        # VRAM
        vram_mb = 0
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
        
        # Collect results
        top_results = []
        for idx in top_idx:
            frame = frame_list[idx]
            img_path = frames_dir / frame['filename']
            if img_path.exists():
                top_results.append({
                    'path': str(img_path),
                    'score': float(scores[idx]),
                    'time_str': frame.get('time_str', ''),
                    'timestamp': frame.get('timestamp', 0),
                    'caption': frame.get('caption', ''),
                })
        
        results_per_model[model_name] = {
            'results': top_results,
            'load_time': load_time,
            'encode_time': encode_time,
            'search_time': search_time,
            'total_time': (load_time + encode_time + search_time),
            'vram_mb': vram_mb,
            'dim': info['dim'],
            'params': info['params'],
        }
        
        # Cleanup
        del m, tok
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Build comparison HTML
    html = _build_comparison_html(results_per_model, query, translated, top_k)
    
    # Build galleries per model (for Gradio gallery display)
    all_galleries = []
    for model_name in model_names:
        r = results_per_model.get(model_name, {})
        if 'error' in r:
            all_galleries.append([])
        else:
            gallery = [(item['path'], f"⏱️ {item['time_str']} | {item['score']:.3f}") for item in r.get('results', [])]
            all_galleries.append(gallery)
    
    return html, all_galleries

def _img_to_data_uri(path, max_w=320):
    """Convert image file to base64 data URI (resized for thumbnails)"""
    try:
        img = Image.open(path).convert('RGB')
        if img.width > max_w:
            ratio = max_w / img.width
            img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
        import io
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=70)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return ""

def _build_comparison_html(results_per_model, query, translated, top_k):
    """Build rich comparison HTML for benchmark results"""
    model_names = list(results_per_model.keys())
    
    # Translation badge
    trans_html = ""
    if translated:
        trans_html = f"""
        <div style="background:linear-gradient(135deg,#1e3a5f,#2d3748); border-radius:10px; padding:10px 16px; margin-bottom:16px; border:1px solid #3b82f6; display:flex; align-items:center; gap:10px;">
            <span style="font-size:1.2rem;">🌐</span>
            <span style="color:#e2e8f0;">{query}</span>
            <span style="color:#3b82f6;">→</span>
            <span style="color:#60a5fa; font-weight:600;">{translated}</span>
        </div>
        """
    
    # Metrics comparison cards
    metrics_cards = ""
    colors = ['#667eea', '#f59e0b', '#48bb78', '#ec4899', '#ef4444']
    
    for i, model_name in enumerate(model_names):
        r = results_per_model[model_name]
        color = colors[i % len(colors)]
        
        if 'error' in r:
            metrics_cards += f"""
            <div style="flex:1; min-width:200px; background:#1a1d24; border-radius:12px; padding:16px; border:1px solid #2d3748;">
                <div style="color:{color}; font-weight:700; font-size:1rem; margin-bottom:8px;">{model_name}</div>
                <div style="color:#fc8181; font-size:0.85rem;">{r['error']}</div>
            </div>"""
            continue
        
        top_score = r['results'][0]['score'] if r['results'] else 0
        total_ms = r['total_time'] * 1000
        search_ms = r['search_time'] * 1000
        
        metrics_cards += f"""
        <div style="flex:1; min-width:220px; background:#1a1d24; border-radius:12px; padding:16px; border:1px solid {color}33; position:relative; overflow:hidden;">
            <div style="position:absolute; top:0; left:0; right:0; height:3px; background:{color};"></div>
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                <span style="color:{color}; font-weight:700; font-size:1.05rem;">{model_name}</span>
                <span style="color:#4a5568; font-size:0.7rem; background:#252a34; padding:2px 8px; border-radius:10px;">{r['params']} · {r['dim']}d</span>
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
                <div style="background:#252a34; border-radius:8px; padding:8px 10px;">
                    <div style="color:#718096; font-size:0.65rem; text-transform:uppercase; letter-spacing:0.5px;">Top Score</div>
                    <div style="color:#e2e8f0; font-size:1.1rem; font-weight:700;">{top_score:.4f}</div>
                </div>
                <div style="background:#252a34; border-radius:8px; padding:8px 10px;">
                    <div style="color:#718096; font-size:0.65rem; text-transform:uppercase; letter-spacing:0.5px;">Search</div>
                    <div style="color:#48bb78; font-size:1.1rem; font-weight:700;">{search_ms:.1f}ms</div>
                </div>
                <div style="background:#252a34; border-radius:8px; padding:8px 10px;">
                    <div style="color:#718096; font-size:0.65rem; text-transform:uppercase; letter-spacing:0.5px;">Total</div>
                    <div style="color:#a0aec0; font-size:1.1rem; font-weight:700;">{total_ms:.0f}ms</div>
                </div>
                <div style="background:#252a34; border-radius:8px; padding:8px 10px;">
                    <div style="color:#718096; font-size:0.65rem; text-transform:uppercase; letter-spacing:0.5px;">VRAM</div>
                    <div style="color:#a0aec0; font-size:1.1rem; font-weight:700;">{r['vram_mb']:.0f}MB</div>
                </div>
            </div>
        </div>"""
    
    # Score comparison bar chart (top-1 scores)
    bar_chart = ""
    valid_models = [(m, r) for m, r in results_per_model.items() if 'results' in r and r['results']]
    if len(valid_models) >= 2:
        max_score = max(r['results'][0]['score'] for _, r in valid_models)
        bars = ""
        for i, (m, r) in enumerate(valid_models):
            score = r['results'][0]['score']
            width_pct = (score / max_score * 100) if max_score > 0 else 0
            color = colors[i % len(colors)]
            bars += f"""
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
                <div style="width:70px; color:{color}; font-size:0.8rem; font-weight:600; text-align:right;">{m}</div>
                <div style="flex:1; background:#252a34; border-radius:4px; height:24px; overflow:hidden;">
                    <div style="background:{color}; height:100%; width:{width_pct:.1f}%; border-radius:4px; display:flex; align-items:center; justify-content:flex-end; padding-right:8px;">
                        <span style="color:white; font-size:0.7rem; font-weight:600;">{score:.4f}</span>
                    </div>
                </div>
            </div>"""
        
        bar_chart = f"""
        <div style="background:#1a1d24; border-radius:12px; padding:16px; border:1px solid #2d3748; margin-top:16px;">
            <div style="color:#a0aec0; font-size:0.85rem; font-weight:600; margin-bottom:12px;">📊 Top-1 Score Comparison</div>
            {bars}
        </div>"""
    
    # Result thumbnails side by side
    thumbs_html = ""
    if len(valid_models) >= 2:
        cols = ""
        for i, (m, r) in enumerate(valid_models):
            color = colors[i % len(colors)]
            imgs = ""
            for j, item in enumerate(r['results'][:4]):
                data_uri = _img_to_data_uri(item['path'])
                imgs += f"""
                <div style="position:relative; border-radius:8px; overflow:hidden;">
                    <img src="{data_uri}" style="width:100%; aspect-ratio:16/9; object-fit:cover; border-radius:8px;"/>
                    <div style="position:absolute; bottom:0; left:0; right:0; background:linear-gradient(transparent,rgba(0,0,0,0.85)); padding:6px 8px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="color:#e2e8f0; font-size:0.7rem;">#{j+1} ⏱️ {item['time_str']}</span>
                            <span style="color:{color}; font-size:0.7rem; font-weight:600;">{item['score']:.3f}</span>
                        </div>
                    </div>
                </div>"""
            
            cols += f"""
            <div style="flex:1; min-width:250px;">
                <div style="color:{color}; font-weight:700; font-size:0.9rem; margin-bottom:8px; text-align:center;">{m}</div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px;">
                    {imgs}
                </div>
            </div>"""
        
        thumbs_html = f"""
        <div style="background:#1a1d24; border-radius:12px; padding:16px; border:1px solid #2d3748; margin-top:16px;">
            <div style="color:#a0aec0; font-size:0.85rem; font-weight:600; margin-bottom:12px;">🖼️ Top Results Comparison</div>
            <div style="display:flex; gap:16px; flex-wrap:wrap;">
                {cols}
            </div>
        </div>"""
    
    return f"""
    <div style="background:linear-gradient(135deg,#252a34,#1a1d24); border-radius:16px; padding:20px; border:1px solid #2d3748;">
        <div style="text-align:center; margin-bottom:16px;">
            <span style="font-size:1.3rem; font-weight:700; color:#e2e8f0;">⚡ Benchmark Results</span>
            <div style="color:#718096; font-size:0.8rem; margin-top:4px;">Query: "{query}"</div>
        </div>
        {trans_html}
        <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:8px;">
            {metrics_cards}
        </div>
        {bar_chart}
        {thumbs_html}
    </div>
    """

# ============================================================ 
# EXPORT ENGINE
# ============================================================ 
def get_available_embeddings(movie_name):
    """Get list of models that have embeddings for a movie"""
    if not movie_name or not is_movie_processed(movie_name):
        return []
    available = []
    # Check default embeddings (ViT-B-32)
    paths = get_movie_paths(movie_name)
    if paths['embeddings'].exists():
        available.append('ViT-B-32')
    # Check benchmark model embeddings
    for model_name in BENCHMARK_MODELS:
        if model_name == 'ViT-B-32':
            continue
        emb_path = get_benchmark_emb_path(movie_name, model_name)
        if emb_path.exists():
            available.append(model_name)
    return available

def export_embeddings(movie_name, model_name, fmt="jsonl"):
    """Export embeddings as JSONL or JSON (generator for progress)"""
    paths = get_movie_paths(movie_name)
    
    # Load metadata
    with open(paths['metadata'], 'r', encoding='utf-8') as f:
        meta = json.load(f)
    frame_list = meta['frames']
    total = len(frame_list)
    
    # Load embeddings
    if model_name == 'ViT-B-32':
        emb_path = paths['embeddings']
    else:
        emb_path = get_benchmark_emb_path(movie_name, model_name)
    
    data = np.load(str(emb_path))
    img_embs = data.get('image_embeddings', data.get('embeddings'))
    txt_embs = data.get('text_embeddings')
    
    model_info = BENCHMARK_MODELS.get(model_name, {})
    dim = model_info.get('dim', img_embs.shape[1] if img_embs is not None else 0)
    
    # Export directory
    export_dir = OUTPUT_DIR / movie_name / 'export'
    export_dir.mkdir(parents=True, exist_ok=True)
    
    slug = model_name.replace('-', '_')
    
    if fmt == "jsonl":
        out_path = export_dir / f'{movie_name}_{slug}.jsonl'
        with open(out_path, 'w', encoding='utf-8') as f:
            for i, frame in enumerate(frame_list):
                record = {
                    'movie': movie_name,
                    'model': model_name,
                    'dimension': dim,
                    'index': i,
                    'timestamp': frame.get('timestamp', 0),
                    'time_str': frame.get('time_str', ''),
                    'filename': frame.get('filename', ''),
                    'caption': frame.get('caption', ''),
                    'image_embedding': img_embs[i].tolist() if i < len(img_embs) else [],
                    'text_embedding': txt_embs[i].tolist() if txt_embs is not None and i < len(txt_embs) else [],
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                
                if (i + 1) % 20 == 0 or i == total - 1:
                    yield i + 1, total, None
        
        yield total, total, str(out_path)
    
    else:  # json
        out_path = export_dir / f'{movie_name}_{slug}.json'
        records = []
        for i, frame in enumerate(frame_list):
            records.append({
                'index': i,
                'timestamp': frame.get('timestamp', 0),
                'time_str': frame.get('time_str', ''),
                'filename': frame.get('filename', ''),
                'caption': frame.get('caption', ''),
                'image_embedding': img_embs[i].tolist() if i < len(img_embs) else [],
                'text_embedding': txt_embs[i].tolist() if txt_embs is not None and i < len(txt_embs) else [],
            })
            if (i + 1) % 20 == 0 or i == total - 1:
                yield i + 1, total, None
        
        output = {
            'movie': movie_name,
            'model': model_name,
            'dimension': dim,
            'total_frames': total,
            'frames': records
        }
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False)
        
        yield total, total, str(out_path)

# ============================================================ 
# UI BUILDER
# ============================================================ 
def create_app():
    
    # Custom CSS
    css = """
    /* Main Layout */
    .container { max-width: 1200px; margin: auto; padding-top: 0px; }
    
    /* Dark Theme Base */
    .gradio-container {
        background: #0f1117 !important;
    }
    
    /* Header Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    /* Header */
    .header { text-align: center; margin-bottom: 40px; }
    .header h1 {
        font-size: 3em;
        font-weight: 800;
        margin-bottom: 10px;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        opacity: 0;
        animation: fadeIn 1.2s ease forwards;
    }
    .header p {
        color: #718096;
        font-size: 1.2em;
        font-weight: 400;
        opacity: 0;
        animation: fadeIn 1s ease forwards;
        animation-delay: 0.5s;
    }
    
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
    .log-area {
        padding: 8px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
    }
    .log-area textarea { 
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 13px;
        background-color: #0f1117 !important;
        color: #a0aec0 !important;
        border-radius: 12px;
        border: 1px solid #2d3748 !important;
        line-height: 1.5;
        min-height: 240px !important;
        width: 100% !important;
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
    
    /* Hidden benchmark download trigger buttons */
    #bench-dl-hidden-row {
        position: absolute !important;
        left: -9999px !important;
        height: 0 !important;
        overflow: hidden !important;
    }
    
    /* Compact file upload area text */
    .compact-upload * {
        font-size: 0.8rem !important;
    }
    .compact-upload label span {
        font-size: 0.85rem !important;
    }
    
    /* Remove hide-container extra padding from section headers */
    .section-header {
        padding: 0 !important;
        margin: 0 !important;
        min-height: 0 !important;
        height: 20px !important;
        border-width: 0 !important;
        overflow: visible !important;
    }
    .section-header h4 {
        margin: 0 !important;
        padding: 0 !important;
    }
    
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

    # JS: event delegation for install buttons rendered inside gr.HTML cards
    # Gradio strips onclick/data-*/class from HTML, so we match by text content
    # Gradio wraps buttons in a div with the elem_id, so we find the inner <button>
    install_btn_js = """
    () => {
        document.addEventListener('click', (e) => {
            const el = e.target;
            if (!el || !el.textContent) return;
            const txt = el.textContent.trim();
            const map = {
                'install:B32': 'bench-dl-b32',
                'install:B16': 'bench-dl-b16',
                'install:L14': 'bench-dl-l14',
                'install:H14': 'bench-dl-h14',
                'install:g14': 'bench-dl-g14'
            };
            if (map[txt]) {
                const wrapper = document.getElementById(map[txt]);
                if (wrapper) {
                    const btn = wrapper.tagName.toLowerCase() === 'button' ? wrapper : wrapper.querySelector('button');
                    if (btn) btn.click();
                }
            }
        });
    }
    """

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", radius_size="lg"), css=css, js=install_btn_js, title="SceneSearch") as app:

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

                    # Row 1: Video Source | Progress
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1, min_width=400):
                            gr.Markdown("#### 🎬 Video Source")
                            with gr.Group(elem_classes="video-source-group"):
                                process_dropdown = gr.Dropdown(
                                    choices=get_available_movies(),
                                    label="Select from video/ folder",
                                    value=None,
                                )
                                video_input = gr.File(
                                    label="Or upload new video",
                                    file_types=["video"],
                                    height=120,
                                    elem_classes="compact-upload",
                                )
                        with gr.Column(scale=1, min_width=400):
                            gr.Markdown("#### 📊 Progress", elem_classes="section-header")
                            progress_output = gr.HTML(
                                value='<div style="background:#1a1d24; border-radius:12px; padding:40px; border:1px solid #2d3748; text-align:center; color:#718096; display:flex; align-items:center; justify-content:center;">Waiting to start...</div>'
                            )

                    # Row 2: Settings | Log
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1, min_width=400):
                            gr.Markdown("#### ⚙️ Settings", elem_classes="section-header")
                            threshold = gr.Slider(
                                0.1, 0.5, value=0.3, step=0.05,
                                label="Scene Sensitivity",
                                info="ffmpeg의 scene change detection 임계값"
                            )
                            gr.HTML('''<div style="color:#8a94a6; font-size:0.72rem; line-height:1.5; padding:0px 4px; margin-top:-6px;">
                                <div style="margin-bottom:3px;">영상의 연속된 두 프레임 간 <b style="color:#a0aec0;">픽셀 변화량(0~1)</b>을 비교해서, 이 값을 초과하면 "장면이 바뀌었다"고 판단하여 해당 프레임을 추출합니다.</div>
                                <div style="display:flex; gap:12px; flex-wrap:wrap;">
                                    <span>🔹 <b style="color:#f59e0b;">0.1</b> 민감 (프레임 많음)</span>
                                    <span>🔹 <b style="color:#48bb78;">0.3</b> 권장</span>
                                    <span>🔹 <b style="color:#667eea;">0.5</b> 둔감 (프레임 적음)</span>
                                </div>
                            </div>''')
                            with gr.Row():
                                process_btn = gr.Button("▶ Start Processing", variant="primary", scale=3)
                                stop_btn = gr.Button("■ Stop", variant="stop", scale=1, visible=False)
                                process_refresh_btn = gr.Button("🔄", scale=0, min_width=40)
                        with gr.Column(scale=1, min_width=400):
                            gr.Markdown("#### 📋 Log", elem_classes="section-header")
                            log_output = gr.Textbox(
                                show_label=False,
                                max_lines=8,
                                elem_classes="log-area",
                                interactive=False,
                                autoscroll=True
                            )

                    # Sample Frames (full width)
                    sample_output = gr.Gallery(label="Sample Frames", columns=4, height="auto")

                    # Manage Section
                    gr.Markdown("---")
                    gr.Markdown("#### 📁 Manage Processed Movies")
                    with gr.Row():
                        manage_dropdown = gr.Dropdown(
                            choices=get_available_movies(),
                            label="Select Movie",
                            value=None,
                            scale=4
                        )
                        manage_refresh_btn = gr.Button("🔄 Refresh", scale=0, min_width=100)
                        delete_btn = gr.Button("🗑️ Delete", variant="stop", scale=0, min_width=100)
                    movie_status_html = gr.HTML(
                        value='<div style="color:#718096; text-align:center; padding:16px;">Select a movie to view status</div>'
                    )

                # ---------------------------------------------------------
                # TAB 3: BENCHMARK
                # ---------------------------------------------------------
                with gr.Tab("📊 Benchmark", id="benchmark_tab"):

                    gr.HTML("""
                    <div style="text-align:center; padding:8px 0 4px;">
                        <span style="font-size:1.1rem; font-weight:600; color:#e2e8f0;">CLIP Model Benchmark</span>
                        <div style="color:#718096; font-size:0.8rem; margin-top:2px;">Compare search quality & speed across different CLIP models</div>
                    </div>
                    """)

                    # Movie selection + refresh
                    with gr.Row():
                        bench_movie = gr.Dropdown(
                            choices=get_available_movies(),
                            label="Select Movie (must be processed first)",
                            value=None,
                            scale=4,
                        )
                        bench_refresh_btn = gr.Button("🔄", scale=0, min_width=50)

                    # Model cards
                    bench_status = gr.HTML(
                        value='<div style="color:#718096; text-align:center; padding:20px; background:#1a1d24; border-radius:12px; border:1px solid #2d3748;">Select a movie to check model status</div>'
                    )
                    # Download progress
                    bench_dl_progress = gr.HTML(visible=False)
                    
                    # Hidden trigger buttons (off-screen, functional)
                    with gr.Row(elem_id="bench-dl-hidden-row"):
                        bench_dl_b32 = gr.Button("dl_b32", elem_id="bench-dl-b32", size="sm")
                        bench_dl_b16 = gr.Button("dl_b16", elem_id="bench-dl-b16", size="sm")
                        bench_dl_l14 = gr.Button("dl_l14", elem_id="bench-dl-l14", size="sm")
                        bench_dl_h14 = gr.Button("dl_h14", elem_id="bench-dl-h14", size="sm")
                        bench_dl_g14 = gr.Button("dl_g14", elem_id="bench-dl-g14", size="sm")

                    # Model selection for comparison
                    bench_models = gr.CheckboxGroup(
                        choices=list(BENCHMARK_MODELS.keys()),
                        value=['ViT-B-32', 'ViT-B-16'],
                        label="Select models to compare",
                    )

                    # Prepare Embeddings
                    with gr.Row():
                        bench_prepare_btn = gr.Button("🔧 Prepare Selected Models", variant="primary", scale=3)
                    bench_prepare_log = gr.Textbox(
                        show_label=False,
                        max_lines=6,
                        elem_classes="log-area",
                        interactive=False,
                        autoscroll=True,
                        visible=False,
                    )
                    bench_prepare_progress = gr.HTML(visible=False)

                    gr.Markdown("---")

                    # Row 3: Search Comparison
                    gr.Markdown("#### 🔍 Compare Search Results", elem_classes="section-header")
                    with gr.Row():
                        bench_query = gr.Textbox(
                            show_label=False,
                            placeholder="Enter search query to compare across models...",
                            lines=1,
                            elem_classes="search-input",
                            scale=4,
                        )
                        bench_topk = gr.Slider(4, 16, value=8, step=4, label="Results", scale=1)
                        bench_search_btn = gr.Button("⚡ Compare", variant="primary", scale=1, elem_classes="primary-btn")

                    # Comparison output
                    bench_comparison = gr.HTML(
                        value='<div style="background:#1a1d24; border-radius:12px; padding:40px; border:1px solid #2d3748; text-align:center; color:#718096;">Prepare models and enter a query to compare</div>'
                    )

                # ---------------------------------------------------------
                # TAB 4: EXPORT DATA
                # ---------------------------------------------------------
                with gr.Tab("📦 Export Data", id="export_tab"):

                    gr.HTML("""
                    <div style="text-align:center; padding:8px 0 4px;">
                        <span style="font-size:1.1rem; font-weight:600; color:#e2e8f0;">Export Embeddings</span>
                        <div style="color:#718096; font-size:0.8rem; margin-top:2px;">임베딩 데이터를 JSON/JSONL로 내보내기</div>
                    </div>
                    """)

                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1, min_width=300):
                            export_movie = gr.Dropdown(
                                choices=get_available_movies(),
                                label="Movie",
                                value=None,
                            )
                            export_model = gr.Dropdown(
                                choices=[],
                                label="Model (available embeddings)",
                                value=None,
                            )
                        with gr.Column(scale=1, min_width=300):
                            export_format = gr.Radio(
                                choices=["JSONL", "JSON"],
                                value="JSONL",
                                label="Format",
                            )
                            gr.HTML("""
                            <div style="background:#1e222b; border-radius:8px; padding:10px 14px; border:1px solid #2d374833; margin-top:4px;">
                                <div style="color:#a0aec0; font-size:0.75rem; margin-bottom:6px; font-weight:600;">Format Guide</div>
                                <div style="color:#718096; font-size:0.7rem; line-height:1.5;">
                                    <b style="color:#48bb78;">JSONL</b> — 한 줄에 프레임 하나. Elasticsearch bulk 인덱싱에 최적<br/>
                                    <b style="color:#667eea;">JSON</b> — 영화 전체가 하나의 구조. 전체 데이터를 한번에 볼 때 유용
                                </div>
                            </div>
                            """)

                    with gr.Row():
                        export_btn = gr.Button("📦 Export", variant="primary", scale=3)
                        export_refresh = gr.Button("🔄", scale=0, min_width=50)

                    export_progress = gr.HTML(
                        value='<div style="background:#1a1d24; border-radius:12px; padding:30px; border:1px solid #2d3748; text-align:center; color:#718096;">Select a movie and model to export</div>'
                    )
                    export_file = gr.File(label="Download", visible=False, interactive=False)

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
        ).then(
            fn=None,
            inputs=None,
            outputs=None,
            js="""
            () => {
                setTimeout(() => {
                    const video = document.querySelector('#main-player video');
                    if (video && video.src) {
                        video.play().catch(e => console.log('Autoplay blocked:', e));
                    }
                }, 800);
            }
            """
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
        def do_process(selected_movie, uploaded_file, thresh):
            # Determine video path: dropdown takes priority, then upload
            video_path = None
            if selected_movie:
                paths = get_movie_paths(selected_movie)
                video_path = str(paths['video'])
            elif uploaded_file:
                video_path = uploaded_file  # gr.File returns path string
            
            for log_msg, prog_html, samples in process_video(video_path, thresh):
                yield log_msg, prog_html, samples

        def on_process_start():
            """Show stop button, hide start button"""
            return gr.update(visible=False), gr.update(visible=True)

        def on_process_end():
            """Show start button, hide stop button, refresh dropdowns"""
            movies = get_available_movies()
            return (
                gr.update(visible=True),
                gr.update(visible=False, interactive=True, value="■ Stop"),
                gr.update(choices=movies),
                gr.update(choices=movies),
            )

        def on_stop_click():
            """Signal processing to stop"""
            stop_event.set()
            if current_ffmpeg_proc and current_ffmpeg_proc.poll() is None:
                current_ffmpeg_proc.terminate()
            return gr.update(interactive=False, value="■ Stopping...")

        def on_process_refresh():
            return gr.update(choices=get_available_movies())

        process_refresh_btn.click(
            fn=on_process_refresh,
            inputs=[],
            outputs=[process_dropdown]
        )

        process_click_event = process_btn.click(
            fn=on_process_start,
            inputs=[],
            outputs=[process_btn, stop_btn]
        ).then(
            fn=do_process,
            inputs=[process_dropdown, video_input, threshold],
            outputs=[log_output, progress_output, sample_output]
        ).then(
            fn=on_process_end,
            inputs=[],
            outputs=[process_btn, stop_btn, process_dropdown, manage_dropdown]
        )

        stop_btn.click(
            fn=on_stop_click,
            inputs=[],
            outputs=[stop_btn],
            cancels=[process_click_event]
        ).then(
            fn=on_process_end,
            inputs=[],
            outputs=[process_btn, stop_btn, process_dropdown, manage_dropdown]
        )

        # Manage section handlers
        def on_manage_select(movie_name):
            return get_movie_status_html(movie_name)

        def on_manage_refresh():
            movies = get_available_movies()
            return gr.update(choices=movies)

        def delete_movie_data(movie_name):
            if not movie_name:
                return get_movie_status_html(None), gr.update(choices=get_available_movies())
            paths = get_movie_paths(movie_name)
            output_dir = paths['output_dir']
            if output_dir.exists():
                shutil.rmtree(output_dir)
            return get_movie_status_html(movie_name), gr.update(choices=get_available_movies())

        manage_dropdown.change(
            fn=on_manage_select,
            inputs=[manage_dropdown],
            outputs=[movie_status_html]
        )
        manage_refresh_btn.click(
            fn=on_manage_refresh,
            inputs=[],
            outputs=[manage_dropdown]
        )
        delete_btn.click(
            fn=delete_movie_data,
            inputs=[manage_dropdown],
            outputs=[movie_status_html, manage_dropdown]
        )

        # ---------------------------------------------------------
        # BENCHMARK EVENT HANDLERS
        # ---------------------------------------------------------

        def generate_bench_status_html(movie_name, downloading_model=None):
            """Generate model status cards (supports inline progress bar during download)"""
            default_html = '<div style="color:#718096; text-align:center; padding:20px; background:#1a1d24; border-radius:12px; border:1px solid #2d3748;">Select a movie to check model status</div>'
            
            if not movie_name:
                return default_html
            if not is_movie_processed(movie_name):
                return f'<div style="color:#f59e0b; text-align:center; padding:16px; background:#1a1d24; border-radius:12px; border:1px solid #f59e0b33;">⚠️ {movie_name} needs processing first (Process tab)</div>'
            
            emb_status = benchmark_check_status(movie_name)
            colors = {'ViT-B-32': '#667eea', 'ViT-B-16': '#f59e0b', 'ViT-L-14': '#48bb78', 'ViT-H-14': '#ec4899', 'ViT-g-14': '#ef4444'}
            install_keys = {'ViT-B-32': 'install:B32', 'ViT-B-16': 'install:B16', 'ViT-L-14': 'install:L14', 'ViT-H-14': 'install:H14', 'ViT-g-14': 'install:g14'}
            
            cards = ""
            for model_name in BENCHMARK_MODELS:
                info = BENCHMARK_MODELS[model_name]
                color = colors[model_name]
                downloaded = is_model_downloaded(model_name)
                emb_ready = emb_status.get(model_name, False)
                ikey = install_keys[model_name]
                
                # Download status: progress bar / installed / install button
                if model_name == downloading_model:
                    slug = model_name.replace('-', '')
                    dl_html = f'''<div style="display:flex;align-items:center;gap:6px;width:100%;">
                        <span style="font-size:0.65rem;">⬇️</span>
                        <div style="flex:1;background:#2d3748;border-radius:4px;height:6px;overflow:hidden;position:relative;">
                            <style>@keyframes dlSlide_{slug} {{ 0% {{ left:-100%; }} 100% {{ left:100%; }} }}</style>
                            <div style="position:absolute;top:0;bottom:0;width:50%;background:{color};border-radius:4px;animation:dlSlide_{slug} 1.5s linear infinite;"></div>
                        </div>
                        <span style="color:{color};font-size:0.6rem;font-weight:600;white-space:nowrap;">Downloading...</span>
                    </div>'''
                elif downloaded:
                    dl_html = '<div style="display:flex;align-items:center;gap:4px;"><span style="font-size:0.65rem;">📦</span><span style="color:#48bb78;font-size:0.7rem;">Installed</span></div>'
                else:
                    dl_html = f'''<div style="display:flex;align-items:center;gap:6px;">
                        <span style="font-size:0.65rem;">⬇️</span>
                        <span style="color:#f59e0b;font-size:0.7rem;">Not installed</span>
                        <span style="cursor:pointer;background:#f59e0b22;border:1px solid #f59e0b55;border-radius:5px;padding:1px 6px;font-size:0.55rem;color:#f59e0b;font-weight:600;display:inline-flex;align-items:center;gap:2px;user-select:none;">{ikey}</span>
                    </div>'''
                
                emb_html = '<div style="display:flex;align-items:center;gap:4px;"><span style="font-size:0.65rem;">✅</span><span style="color:#48bb78;font-size:0.7rem;">Embeddings ready</span></div>' if emb_ready else '<div style="display:flex;align-items:center;gap:4px;"><span style="font-size:0.65rem;">⬜</span><span style="color:#718096;font-size:0.7rem;">Not prepared</span></div>'
                
                cards += f'''<div style="flex:1;min-width:170px;background:#1e222b;border-radius:10px;padding:12px 14px;border:1px solid {color}33;position:relative;overflow:hidden;">
                    <div style="position:absolute;top:0;left:0;right:0;height:2px;background:{color};"></div>
                    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
                        <span style="color:{color};font-weight:700;font-size:0.9rem;">{model_name}</span>
                        <span style="color:#4a5568;font-size:0.6rem;background:#252a34;padding:2px 6px;border-radius:8px;">{info['params']}</span>
                    </div>
                    <div style="color:#718096;font-size:0.65rem;margin-bottom:8px;">{info['dim']}d · Patch {info['patch']} · {info['desc']}</div>
                    <div style="display:flex;flex-direction:column;gap:3px;min-height:30px;justify-content:center;">{dl_html}{emb_html}</div>
                </div>'''
            
            return f'<div style="display:flex;gap:10px;flex-wrap:wrap;">{cards}</div>'

        def on_bench_movie_select(movie_name):
            return generate_bench_status_html(movie_name)

        bench_movie.change(
            fn=on_bench_movie_select,
            inputs=[bench_movie],
            outputs=[bench_status]
        )

        def on_bench_refresh():
            return gr.update(choices=get_available_movies())

        bench_refresh_btn.click(
            fn=on_bench_refresh,
            inputs=[],
            outputs=[bench_movie]
        )

        def do_bench_dl_single_gen(model_name, movie_name):
            """Download a single model with card-inline progress"""
            if is_model_downloaded(model_name):
                yield generate_bench_status_html(movie_name)
                return
            # Show progress bar inside the card
            yield generate_bench_status_html(movie_name, downloading_model=model_name)
            # Download
            info = BENCHMARK_MODELS[model_name]
            try:
                m, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=info['pretrained'])
                del m
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except Exception as e:
                print(f"[!] Failed to download {model_name}: {e}")
            # Refresh card state
            yield generate_bench_status_html(movie_name)

        # Factory function to avoid lambda+generator issue
        def make_dl_handler(mname):
            def handler(movie_name):
                yield from do_bench_dl_single_gen(mname, movie_name)
            return handler

        for btn, mname in [(bench_dl_b32, 'ViT-B-32'), (bench_dl_b16, 'ViT-B-16'), (bench_dl_l14, 'ViT-L-14'), (bench_dl_h14, 'ViT-H-14'), (bench_dl_g14, 'ViT-g-14')]:
            btn.click(
                fn=make_dl_handler(mname),
                inputs=[bench_movie],
                outputs=[bench_status]
            )

        def do_bench_prepare(movie_name, model_names):
            if not movie_name:
                yield "❌ Select a movie first", "", None
                return
            if not model_names:
                yield "❌ Select at least one model", "", None
                return
            
            for model_name in model_names:
                # Skip if already exists
                emb_path = get_benchmark_emb_path(movie_name, model_name)
                default_path = get_movie_paths(movie_name)['embeddings']
                if emb_path.exists() or (model_name == 'ViT-B-32' and default_path.exists()):
                    yield f"⏭️ {model_name} already prepared, skipping...", _bench_progress_html(model_name, "done", 1, 1), None
                    continue
                
                for log_msg, prog_html, _ in benchmark_generate(movie_name, model_name):
                    yield log_msg, prog_html, None

        bench_prepare_btn.click(
            fn=lambda: (gr.update(visible=True), gr.update(visible=True)),
            inputs=[],
            outputs=[bench_prepare_log, bench_prepare_progress]
        ).then(
            fn=do_bench_prepare,
            inputs=[bench_movie, bench_models],
            outputs=[bench_prepare_log, bench_prepare_progress, bench_status]
        ).then(
            fn=on_bench_movie_select,
            inputs=[bench_movie],
            outputs=[bench_status]
        )

        def do_bench_search(movie_name, model_names, query, top_k):
            if not movie_name or not query.strip() or not model_names:
                return '<div style="color:#718096; text-align:center; padding:40px; background:#1a1d24; border-radius:12px; border:1px solid #2d3748;">Select movie, models, and enter a query</div>'
            html, _ = benchmark_search_compare(movie_name, model_names, query, int(top_k))
            return html

        bench_search_btn.click(
            fn=do_bench_search,
            inputs=[bench_movie, bench_models, bench_query, bench_topk],
            outputs=[bench_comparison]
        )
        bench_query.submit(
            fn=do_bench_search,
            inputs=[bench_movie, bench_models, bench_query, bench_topk],
            outputs=[bench_comparison]
        )

        # ---------------------------------------------------------
        # EXPORT EVENT HANDLERS
        # ---------------------------------------------------------

        def on_export_movie_select(movie_name):
            """Update model dropdown based on available embeddings"""
            models = get_available_embeddings(movie_name)
            default_val = models[0] if models else None
            return gr.update(choices=models, value=default_val)

        export_movie.change(
            fn=on_export_movie_select,
            inputs=[export_movie],
            outputs=[export_model]
        )

        def on_export_refresh():
            return gr.update(choices=get_available_movies())

        export_refresh.click(
            fn=on_export_refresh,
            inputs=[],
            outputs=[export_movie]
        )

        def do_export(movie_name, model_name, fmt):
            """Export embeddings with progress"""
            if not movie_name or not model_name:
                yield '<div style="color:#f59e0b; text-align:center; padding:20px; background:#1a1d24; border-radius:12px; border:1px solid #f59e0b33;">Select a movie and model first</div>', gr.update(visible=False)
                return
            
            fmt_key = "jsonl" if fmt == "JSONL" else "json"
            model_info = BENCHMARK_MODELS.get(model_name, {})
            color = {'ViT-B-32': '#667eea', 'ViT-B-16': '#f59e0b', 'ViT-L-14': '#48bb78', 'ViT-H-14': '#ec4899', 'ViT-g-14': '#ef4444'}.get(model_name, '#667eea')
            
            out_path = None
            for current, total, path in export_embeddings(movie_name, model_name, fmt_key):
                pct = current / total * 100
                if path:
                    out_path = path
                
                if out_path:
                    # Done
                    import os
                    file_size = os.path.getsize(out_path)
                    size_str = f"{file_size / 1024 / 1024:.1f} MB" if file_size > 1024 * 1024 else f"{file_size / 1024:.1f} KB"
                    yield f'''
                    <div style="background:#1a1d24; border-radius:12px; padding:16px; border:1px solid #22c55e44;">
                        <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
                            <span style="font-size:1.5rem;">✅</span>
                            <div>
                                <div style="color:#48bb78; font-size:1rem; font-weight:600;">Export Complete</div>
                                <div style="color:#6ee7b7; font-size:0.8rem;">{total} frames exported</div>
                            </div>
                        </div>
                        <div style="display:flex; gap:12px; flex-wrap:wrap;">
                            <div style="background:#252a34; border-radius:8px; padding:8px 12px;">
                                <div style="color:#718096; font-size:0.65rem;">FILE</div>
                                <div style="color:#e2e8f0; font-size:0.8rem; font-weight:500; word-break:break-all;">{os.path.basename(out_path)}</div>
                            </div>
                            <div style="background:#252a34; border-radius:8px; padding:8px 12px;">
                                <div style="color:#718096; font-size:0.65rem;">SIZE</div>
                                <div style="color:#e2e8f0; font-size:0.8rem; font-weight:500;">{size_str}</div>
                            </div>
                            <div style="background:#252a34; border-radius:8px; padding:8px 12px;">
                                <div style="color:#718096; font-size:0.65rem;">MODEL</div>
                                <div style="color:{color}; font-size:0.8rem; font-weight:500;">{model_name}</div>
                            </div>
                            <div style="background:#252a34; border-radius:8px; padding:8px 12px;">
                                <div style="color:#718096; font-size:0.65rem;">FORMAT</div>
                                <div style="color:#e2e8f0; font-size:0.8rem; font-weight:500;">{fmt}</div>
                            </div>
                        </div>
                    </div>''', gr.update(visible=True, value=out_path)
                else:
                    # In progress
                    yield f'''
                    <div style="background:#1a1d24; border-radius:12px; padding:16px; border:1px solid {color}33;">
                        <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
                            <span style="font-size:1.2rem;">📦</span>
                            <div style="flex:1;">
                                <div style="color:#e2e8f0; font-size:0.9rem; font-weight:600;">Exporting {movie_name}</div>
                                <div style="color:#718096; font-size:0.75rem;">{model_name} · {fmt} · {current}/{total} frames</div>
                            </div>
                            <div style="color:{color}; font-size:1.1rem; font-weight:700;">{pct:.0f}%</div>
                        </div>
                        <div style="background:#2d3748; border-radius:6px; height:8px; overflow:hidden;">
                            <div style="background:{color}; height:100%; width:{pct}%; border-radius:6px; transition:width 0.3s;"></div>
                        </div>
                    </div>''', gr.update(visible=False)

        export_btn.click(
            fn=do_export,
            inputs=[export_movie, export_model, export_format],
            outputs=[export_progress, export_file]
        )

    return app

if __name__ == "__main__":
    load_resources()
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7880,
        inbrowser=True,
        allowed_paths=[str(OUTPUT_DIR)]
    )
