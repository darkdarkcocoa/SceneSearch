"""SceneSearch - Gradio Web UI (Refined & Aesthetic) """
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
OUTPUT_DIR = BASE_DIR / "output"
EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.npz"
METADATA_FILE = OUTPUT_DIR / "metadata.json"
FRAMES_DIR = OUTPUT_DIR / "frames"

# Global variables
image_embeddings = None
text_embeddings = None
frames = None
clip_model = None
tokenizer = None
device = None
use_hybrid = False

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

# ============================================================ 
# SEARCH ENGINE
# ============================================================ 
def load_resources():
    """Load embeddings, metadata, and CLIP model"""
    global image_embeddings, text_embeddings, frames, clip_model, tokenizer, device, use_hybrid

    print("[*] Loading SceneSearch...")

    if not EMBEDDINGS_FILE.exists():
        print("[!] No embeddings found. Please process a video first.")
        return False

    # Load Embeddings
    try:
        data = np.load(EMBEDDINGS_FILE)
        if 'image_embeddings' in data:
            image_embeddings = data['image_embeddings']
            text_embeddings = data['text_embeddings']
            use_hybrid = True
            print(f"[+] Loaded hybrid embeddings (Image + Text)")
        else:
            image_embeddings = data['embeddings']
            text_embeddings = None
            use_hybrid = False
            print(f"[+] Loaded legacy embeddings (Image Only)")
    except Exception as e:
        print(f"[!] Error loading embeddings: {e}")
        return False

    # Load Metadata
    try:
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        frames = metadata['frames']
        print(f"[+] Loaded {len(frames)} frames")
    except Exception as e:
        print(f"[!] Error loading metadata: {e}")
        return False

    # Load Model
    device = get_device()
    print(f"[+] Using device: {device}")

    try:
        clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        clip_model = clip_model.to(device)
        clip_model.eval()
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        print("[+] CLIP model loaded")
    except Exception as e:
        print(f"[!] Error loading CLIP model: {e}")
        return False

    print("[!] Ready!\n")
    return True

def search(query: str, top_k: int, search_mode: str):
    """
    Hybrid search logic
    search_mode: "Visual (Image)", "Hybrid (Smart)", "Conceptual (Text)"
    """
    global image_embeddings, text_embeddings, frames, use_hybrid
    
    if image_embeddings is None or frames is None:
        return [], "⚠️ 먼저 '영상 처리' 탭에서 영상을 처리해주세요."

    if not query.strip():
        return [], ""

    # 1. Encode Query
    t0 = time.perf_counter()
    text_tokens = tokenizer([query])
    with torch.no_grad():
        query_emb = clip_model.encode_text(text_tokens.to(device))
        query_emb /= query_emb.norm(dim=-1, keepdim=True)
    query_emb = query_emb.cpu().numpy().flatten()
    encode_time = time.perf_counter() - t0

    # 2. Determine Weights based on Mode
    # Default Hybrid weights
    w_img = 0.6
    w_txt = 0.4
    
    mode_desc = ""
    
    if "Visual" in search_mode:
        w_img, w_txt = 1.0, 0.0
        mode_desc = "🖼️ Visual Only"
    elif "Conceptual" in search_mode:
        w_img, w_txt = 0.0, 1.0
        mode_desc = "📝 Caption Only"
    else: # Hybrid (Smart)
        if use_hybrid and text_embeddings is not None:
            # Dynamic weighting for Hybrid
            word_count = len(query.split())
            if word_count <= 2:
                w_img, w_txt = 0.8, 0.2  # Short query -> Trust CLIP visuals more
                mode_desc = "⚡ Hybrid (Short Query)"
            else:
                w_img, w_txt = 0.5, 0.5  # Long query -> Trust BLIP captions equally
                mode_desc = "🧠 Hybrid (Balanced)"
        else:
            w_img, w_txt = 1.0, 0.0
            mode_desc = "🖼️ Image Only (No captions avail)"

    # 3. Calculate Similarity
    t1 = time.perf_counter()
    
    # Image Similarity
    scores = image_embeddings @ query_emb
    
    # Text Similarity (if applicable)
    if use_hybrid and text_embeddings is not None and w_txt > 0:
        txt_scores = text_embeddings @ query_emb
        scores = scores * w_img + txt_scores * w_txt
    
    # Top K
    top_indices = np.argsort(scores)[::-1][:top_k]
    search_time = time.perf_counter() - t1

    # 4. Format Results
    results = []
    for idx in top_indices:
        frame = frames[idx]
        score = scores[idx]
        image_path = FRAMES_DIR / frame['filename']

        if image_path.exists():
            caption_text = frame.get('caption', '')
            
            # Smart Caption Display
            display_caption = f"⏱️ {frame['time_str']}  |  Score: {score:.3f}"
            if caption_text:
                # Truncate long captions for cleaner UI
                short_cap = (caption_text[:75] + '..') if len(caption_text) > 75 else caption_text
                display_caption += f"\n📝 {short_cap}"
            
            results.append((str(image_path), display_caption))

    # Stats Bar
    stats = f"""
    <div style='display: flex; justify-content: center; align-items: center; gap: 3rem; padding: 0.8rem 1.5rem; background: linear-gradient(135deg, #1a1d24 0%, #252a34 100%); border-radius: 12px; font-size: 0.95rem; color: #a0aec0; border: 1px solid #2d3748; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);'>
        <span>🔎 <strong style="color: #e2e8f0;">{len(frames):,}</strong> Frames Scanned</span>
        <span>⚙️ Mode: <strong style="color: #667eea;">{mode_desc}</strong></span>
        <span>⚡ Time: <strong style="color: #48bb78;">{(encode_time + search_time)*1000:.1f}ms</strong></span>
    </div>
    """

    return results, stats

# ============================================================ 
# PROCESSING ENGINE
# ============================================================ 
def process_video(video_file, scene_threshold, progress=gr.Progress()):
    """Full pipeline: Extract -> Caption -> Embed"""
    
    if video_file is None:
        yield "❌ 영상 파일을 선택해주세요.", None
        return
    
    video_path = video_file
    log_messages = []
    
    def log(msg):
        timestamp = time.strftime("%H:%M:%S")
        log_messages.append(f"[{timestamp}] {msg}")
        return "\n".join(log_messages)

    try:
        # 1. Init
        yield log("🚀 작업 시작: 환경 초기화 중..."), None
        if FRAMES_DIR.exists(): shutil.rmtree(FRAMES_DIR)
        FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # 2. FFmpeg Extraction
        yield log("🎬 장면 감지 및 프레임 추출 시작 (Threshold: {scene_threshold})"), None
        frame_log_path = OUTPUT_DIR / "frame_log.txt"
        
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"select='gt(scene,{scene_threshold})',showinfo",
            "-vsync", "vfr",
            str(FRAMES_DIR / "frame_%04d.jpg")
        ]
        
        # Run ffmpeg
        proc = subprocess.Popen(
            cmd, stderr=subprocess.PIPE, universal_newlines=True, encoding='utf-8', errors='replace'
        )
        pts_times = []
        for line in proc.stderr:
            if "pts_time:" in line:
                match = re.search(r'pts_time:(\d+\.?\d*)', line)
                if match: pts_times.append(float(match.group(1)))
        proc.wait()
        
        # Check results
        frame_files = list(FRAMES_DIR.glob("frame_*.jpg"))
        if not frame_files:
            yield log("❌ 프레임 추출 실패. 장면 변화가 감지되지 않았거나 FFmpeg 오류입니다."), None
            return
            
        yield log(f"✅ 프레임 추출 완료: {len(frame_files)}장"), None

        # 3. Metadata Structure
        frames_data = []
        for i, pts in enumerate(pts_times):
            if i >= len(frame_files): break
            frames_data.append({
                "index": i,
                "filename": f"frame_{i+1:04d}.jpg",
                "timestamp": round(pts, 3),
                "time_str": format_time(pts)
            })

        # 4. AI Processing (CLIP + BLIP)
        yield log("🧠 AI 모델(CLIP + BLIP) 로드 중..."), None
        
        proc_device = get_device()
        
        # Load CLIP
        c_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        c_model = c_model.to(proc_device).eval()
        c_tokenizer = open_clip.get_tokenizer('ViT-B-32')
        
        # Load BLIP
        from transformers import BlipProcessor, BlipForConditionalGeneration
        b_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        b_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(proc_device).eval()
        
        yield log("✅ 모델 로드 완료. 분석 시작..."), None
        
        img_embs = []
        txt_embs = []
        captions = []
        
        start_time = time.time()
        total = len(frames_data)
        
        for i, frame_info in enumerate(frames_data):
            # Load Image
            f_path = FRAMES_DIR / frame_info['filename']
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
            captions.append(caption)
            frame_info['caption'] = caption # Save to metadata
            
            # CLIP Text Embedding (of Caption)
            txt_in = c_tokenizer([caption])
            with torch.no_grad():
                te = c_model.encode_text(txt_in.to(proc_device))
                te /= te.norm(dim=-1, keepdim=True)
            txt_embs.append(te.cpu().numpy().flatten())
            
            # Update Progress
            if (i+1) % 10 == 0 or (i+1) == total:
                elapsed = time.time() - start_time
                fps = (i+1) / elapsed
                eta = (total - (i+1)) / fps
                progress((i+1)/total, desc=f"분석 중... {i+1}/{total}")
                yield log(f"   ▶ [{i+1}/{total}] {fps:.1f} FPS (남은 시간: {eta:.0f}초)"), None

        # 5. Save Data
        yield log("💾 데이터 저장 중..."), None
        
        # Save Embeddings
        np.savez(
            EMBEDDINGS_FILE,
            image_embeddings=np.array(img_embs),
            text_embeddings=np.array(txt_embs),
            timestamps=np.array([f['timestamp'] for f in frames_data])
        )
        
        # Save Metadata
        metadata = {
            "video_info": {"filename": Path(video_path).name},
            "extraction_config": {"method": "scene_detection", "threshold": scene_threshold},
            "total_frames": total,
            "frames": frames_data
        }
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        yield log("✨ 모든 작업 완료! 검색 탭으로 이동하세요."), None
        
        # Reload Resources
        load_resources()
        
        # Show Samples
        samples = []
        for idx in [0, total//2, total-1]:
            if idx < total:
                f = frames_data[idx]
                samples.append((str(FRAMES_DIR / f['filename']), f"[{f['time_str']}] {f['caption']}"))
        
        yield log(""), samples

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        yield log(f"❌ 치명적 오류 발생:\n{err}"), None

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
    
    /* Footer */
    footer { display: none !important; }
    """

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", radius_size="lg"), css=css, title="SceneSearch") as app:
        
        with gr.Column(elem_classes="container"):
            
            # Header
            gr.HTML("""
                <div class="header">
                    <h1>🎬 SceneSearch</h1>
                    <p>AI that sees, reads, and understands video moments.</p>
                </div>
            """)
            
            with gr.Tabs():
                
                # ---------------------------------------------------------
                # TAB 1: SEARCH (First Priority)
                # ---------------------------------------------------------
                with gr.Tab("🔍 Search Moments", id="search_tab"):
                    
                    with gr.Column(elem_classes="search-row"):
                        # Hero Search Bar
                        query_input = gr.Textbox(
                            show_label=False,
                            placeholder="Describe the scene you are looking for... (e.g., 'A futuristic laboratory with blue lights', 'Johnny Depp smiling')",
                            lines=1,
                            scale=3,
                            elem_classes="search-input",
                            autofocus=True
                        )
                        
                        with gr.Row(equal_height=True):
                            # Intuitive Controls
                            search_mode = gr.Radio(
                                choices=["Hybrid (Smart)", "Visual Match (Image)", "Conceptual Match (Caption)"],
                                value="Hybrid (Smart)",
                                label="Search Logic",
                                info="Smart mode balances visuals and captions.",
                                scale=2
                            )
                            top_k = gr.Slider(4, 20, value=8, step=4, label="Results", scale=1)
                            
                            search_btn = gr.Button("Search", variant="primary", scale=1, elem_classes="primary-btn")

                    # Results Area
                    stats_output = gr.HTML(label="Status")
                    gallery = gr.Gallery(
                        label="Found Scenes", 
                        columns=4, 
                        height="auto",
                        object_fit="cover",
                        elem_classes="result-gallery",
                        show_label=False
                    )
                    
                    # Examples
                    gr.Examples(
                        examples=[
                            ["Johnny Depp face close up"],
                            ["A computer screen displaying code"],
                            ["Two people talking in a room"],
                            ["An explosion or fire"],
                            ["Outdoor scene with trees"]
                        ],
                        inputs=query_input,
                        label="Try these examples:"
                    )

                # ---------------------------------------------------------
                # TAB 2: PROCESSING (Secondary)
                # ---------------------------------------------------------
                with gr.Tab("⚙️ Video Processing", id="process_tab"):
                    gr.Markdown("### Upload and Process New Videos")
                    with gr.Row():
                        with gr.Column(scale=1):
                            video_input = gr.Video(label="Video File", sources=["upload"])
                            threshold = gr.Slider(0.1, 0.5, value=0.3, step=0.05, label="Scene Detection Sensitivity", info="Lower = More frames")
                            process_btn = gr.Button("Start Processing (Extract & Embed)", variant="primary")
                        
                        with gr.Column(scale=1):
                            log_output = gr.Textbox(
                                label="Processing Log", 
                                lines=15, 
                                elem_classes="log-area",
                                interactive=False
                            )
                    
                    gr.Markdown("### Sample Processed Frames")
                    sample_output = gr.Gallery(label="", columns=4, height="auto")

        # Event Handlers
        search_btn.click(
            fn=search,
            inputs=[query_input, top_k, search_mode],
            outputs=[gallery, stats_output]
        )
        query_input.submit(
            fn=search,
            inputs=[query_input, top_k, search_mode],
            outputs=[gallery, stats_output]
        )
        
        process_btn.click(
            fn=process_video,
            inputs=[video_input, threshold],
            outputs=[log_output, sample_output]
        )

    return app

if __name__ == "__main__":
    load_resources()
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True
    )