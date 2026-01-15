"""Parse ffmpeg showinfo log and create metadata.json"""
import re
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
LOG_FILE = OUTPUT_DIR / "frame_log.txt"
OUTPUT_FILE = OUTPUT_DIR / "metadata.json"

def format_time(seconds):
    """Format seconds to HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def parse_log():
    frames = []
    frame_num = 0  # 실제 프레임 번호 (1부터 시작)

    with open(LOG_FILE, 'r') as f:
        for line in f:
            # Extract pts_time from log line
            match = re.search(r'pts_time:(\d+\.?\d*)', line)
            if match:
                timestamp = float(match.group(1))
                frame_num += 1
                frames.append({
                    "index": frame_num - 1,  # 0-based index
                    "filename": f"frame_{frame_num:04d}.jpg",
                    "timestamp": round(timestamp, 3),
                    "time_str": format_time(timestamp)
                })

    return frames

def main():
    print("Parsing frame log...")
    frames = parse_log()

    metadata = {
        "video_info": {
            "filename": "Transcendence.mp4",
            "duration": 7156.59,
            "fps": 29.97,
            "resolution": [1280, 720]
        },
        "extraction_config": {
            "method": "scene_detection",
            "threshold": 0.3
        },
        "total_frames": len(frames),
        "frames": frames
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Created {OUTPUT_FILE}")
    print(f"Total frames: {len(frames)}")
    print(f"\nFirst 3 frames:")
    for frame in frames[:3]:
        print(f"  {frame}")
    print(f"\nLast 3 frames:")
    for frame in frames[-3:]:
        print(f"  {frame}")

if __name__ == "__main__":
    main()
