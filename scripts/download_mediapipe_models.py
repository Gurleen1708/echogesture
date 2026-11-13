"""Download the two MediaPipe landmark models (one-time, ~13 MB). Inference itself is offline."""
import urllib.request
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from echogesture import config

BASE = "https://storage.googleapis.com/mediapipe-models/"
FILES = {
    config.HAND_TASK: BASE + "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    config.POSE_TASK: BASE + "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}

if __name__ == "__main__":
    config.MEDIAPIPE_DIR.mkdir(parents=True, exist_ok=True)
    for dest, url in FILES.items():
        if dest.exists() and dest.stat().st_size > 100_000:
            print(f"ok       {dest.name}")
            continue
        print(f"download {dest.name} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"         {dest.stat().st_size / 1e6:.1f} MB")
