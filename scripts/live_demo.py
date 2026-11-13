"""Live captions from a webcam / phone camera stream (or a video file), fully on-device.

    python scripts/live_demo.py                       # default webcam, Q quits
    python scripts/live_demo.py --mirror --speak      # selfie view + offline speech (pyttsx3)
    python scripts/live_demo.py --source 1            # second camera
    python scripts/live_demo.py --source clip.mp4 --save demo.mp4 --no-window
    python scripts/live_demo.py --source http://<phone-ip>:8080/video   # IP Webcam app on a phone

Keys: Q/Esc quit · C clear caption · S screenshot · SPACE pause
"""
import argparse
import platform
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from echogesture import config
from echogesture.landmarks import LandmarkExtractor
from echogesture.live import Stats, StreamDecoder, draw_hud
from echogesture.predictor import Classifier


def open_source(src: str):
    """Open a camera index, file or URL. On Windows tries DirectShow first (MSMF is flaky)."""
    if not src.isdigit():
        cap = cv2.VideoCapture(src)
        return (cap, src) if cap.isOpened() else (None, src)
    idx = int(src)
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if platform.system() == "Windows" else [cv2.CAP_ANY]
    for i in [idx] + [j for j in range(4) if j != idx]:
        for be in backends:
            cap = cv2.VideoCapture(i, be)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                ok, _ = cap.read()
                if ok:
                    return cap, f"camera {i}"
            cap.release()
    return None, src


class Speaker:
    """Offline TTS in a background thread so speech never stalls the video loop."""

    def __init__(self):
        import pyttsx3
        self.engine = pyttsx3.init()
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            word = self.q.get()
            self.engine.say(word.lower().replace("_", " "))
            self.engine.runAndWait()

    def say(self, word: str):
        self.q.put(word)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index, video file or stream URL")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--mirror", action="store_true", help="flip the preview (selfie view)")
    ap.add_argument("--speak", action="store_true", help="read confirmed words aloud (offline)")
    ap.add_argument("--save", help="write the HUD video to this .mp4")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--title", default="EchoGesture  ·  Team Mudra")
    args = ap.parse_args()

    cap, name = open_source(args.source)
    if cap is None:
        sys.exit(f"Cannot open source '{args.source}'. If it is a webcam: close Meet/Zoom/Teams "
                 "(they lock the camera), or try --source 1.")
    ok, frame = cap.read()
    if not ok:
        sys.exit("No frames from source")
    h, w = frame.shape[:2]
    is_file = not args.source.isdigit()
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (1 <= src_fps <= 240):
        src_fps = 30.0
    print(f"source: {name}  {w}x{h}")

    clf = Classifier()
    model_label = f"INT8 ONNX  ·  {clf.model_mb:.2f} MB  ·  {len(clf.vocab)} words  ·  open weights"
    speaker = Speaker() if args.speak else None
    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), src_fps if is_file else 20.0, (w, h))

    decoder = StreamDecoder(clf, aspect=w / h, threshold=args.threshold)
    stats = Stats()
    words: list[str] = []
    paused = False
    n = 0
    t_start = time.perf_counter()
    with LandmarkExtractor() as ex:
        while ok:
            t0 = time.perf_counter()
            if args.mirror:
                frame = cv2.flip(frame, 1)
            ts = int(n * 1000 / src_fps) if is_file else int((t0 - t_start) * 1000)
            raw = ex.process(frame, ts)
            lm_ms = (time.perf_counter() - t0) * 1000
            if not paused:
                word = decoder.push(raw)
                if word:
                    words.append(word)
                    print(word, flush=True)
                    if speaker:
                        speaker.say(word)
            total_ms = (time.perf_counter() - t0) * 1000
            stats.add(lm_ms, decoder.model_ms, total_ms)
            out = draw_hud(frame, raw, decoder, words, stats, model_label, paused=paused, title=args.title)
            if writer:
                writer.write(out)
            if not args.no_window:
                cv2.imshow("EchoGesture", out)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    words.clear()
                if key == ord(" "):
                    paused = not paused
                if key == ord("s"):
                    Path("results").mkdir(exist_ok=True)
                    p = f"results/screenshot_{int(time.time())}.png"
                    cv2.imwrite(p, out)
                    print("saved", p)
            n += 1
            ok, frame = cap.read()
    cap.release()
    if writer:
        writer.release()
        print("saved", args.save)
    if not args.no_window:
        cv2.destroyAllWindows()
    print("caption:", " ".join(words))
    print("stats:", stats.summary())
