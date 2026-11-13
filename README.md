# EchoGesture — on-device sign language recognition

Samsung PRISM XR Hackathon 2026 · Track 2 · Team Mudra · Thapar Institute

Video of a signer (third-person view) → word, entirely on the device. No frame ever leaves the machine.

```
camera / clip ─► MediaPipe hand + pose landmarks ─► pose-invariant 3D features ─► INT8 temporal conv net ─► word ─► caption HUD / speech
                 (on-device, float16 TFLite)        (154 values per frame)         (ONNX Runtime, < 1 MB)
```

| Brief requirement | Where |
|---|---|
| `predict(video_path) -> dict` | `predict.py` (exact return format from the brief) |
| `predictions.jsonl` | `scripts/run_predictions.py` |
| `results.csv` (+ end-to-end FPS, per-stage latency) | `scripts/benchmark.py` → `results/` |
| Quantized model + conversion script, **open weights** | `models/echogesture_int8.onnx`, `scripts/export_quantize.py` |
| Pose-invariant 3D hand features | `echogesture/features.py` |
| Robustness (angle, occlusion, low light) | `echogesture/augment.py` (training), measured by `benchmark.py` |
| Captions / audio HUD for two-way conversation | `scripts/live_demo.py` |
| 2–3 min demo video | `scripts/make_demo_reel.py` builds it from your clips |
| Zero cloud | no network code in the inference path |

## Setup (Windows PowerShell, once)

```powershell
cd "C:\Users\Asus\Downloads\echogesture (1)\echogesture"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\download_mediapipe_models.py
python tests\smoke_test.py          # must end with: SMOKE TEST PASSED
```

Use Python 3.12 (MediaPipe has no wheels for 3.14 yet). Every later session: `cd` to the folder and run `.\.venv\Scripts\Activate.ps1`.

## Try the demo before training (stub model)

```powershell
cd "C:\Users\Asus\Downloads\echogesture (1)\echogesture"
python scripts\make_stub_model.py     # random 20-word model, UI testing only
python scripts\live_demo.py --mirror  # Q quit · C clear · S screenshot · SPACE pause
```

The stub predicts nonsense; it exists so you can check camera, hand tracking and the HUD. Close Meet/Zoom/Teams first — they lock the webcam. A phone camera works too: install "IP Webcam" on Android and pass `--source http://<phone-ip>:8080/video`.

Delete the stub before real training: `del models\echogesture_int8.onnx models\echogesture_fp32.onnx models\vocab.txt`.

## Train (WLASL)

1. `E:\wlasl\WLASL_v0.3.json` + videos in `E:\wlasl\videos\` (`<video_id>.mp4`). Keep data outside OneDrive.
2. Build the manifest. Use the organisers' vocabulary list when you have it; until then `--top-n 100`.

```powershell
cd "C:\Users\Asus\Downloads\echogesture (1)\echogesture"
python scripts\prepare_wlasl.py --wlasl-json E:\wlasl\WLASL_v0.3.json --videos-dir E:\wlasl\videos --top-n 100
python scripts\extract_landmarks.py --manifest data\manifest.csv --cache data\cache
python scripts\train.py --manifest data\manifest.csv --cache data\cache --epochs 200
python scripts\export_quantize.py --manifest data\manifest.csv --cache data\cache
```

Landmark extraction is the slow step (once per clip, resumable). Training runs on cached landmarks and takes minutes on CPU.
For the final model train on train + val: `--train-splits train val --val-splits test` — only after you have stopped tuning.

## Evaluate and package

```powershell
cd "C:\Users\Asus\Downloads\echogesture (1)\echogesture"
python scripts\extract_landmarks.py --manifest data\manifest.csv --cache data\cache_hard --splits test --degrade all
python scripts\benchmark.py --manifest data\manifest.csv --cache data\cache --hard-cache data\cache_hard --thermal-minutes 10 --device "HP Victus 15, Core i5" --sample-video E:\wlasl\videos\00335.mp4
python scripts\run_predictions.py --manifest <organisers_test_manifest.csv> --videos-dir <test_clips_folder> --out predictions.jsonl
python predict.py <one_clip.mp4>
```

`benchmark.py` writes `results.csv` (the brief's table) and `results/benchmark_details.json` with the extras: median/p95 classifier latency, landmark ms per frame, **end-to-end FPS** and ms per clip (mentor-suggested throughput numbers).

## Demo video, no editor needed

Record short clips (phone on a stand, 2 m back, plain wall, good light): 3 s idle → the sign → 3 s idle. Name each file by the word: `HELLO_01.mp4`, `BOOK_02.mp4` … (word must match `models\vocab.txt`). Then:

```powershell
cd "C:\Users\Asus\Downloads\echogesture (1)\echogesture"
python scripts\make_demo_reel.py --clips E:\demo_footage --out demo_reel.mp4
```

Output: title card → every clip with the live HUD and an EXPECTED pill → a frozen CORRECT / PREDICTED-vs-EXPECTED verdict after each → results card (accuracy, end-to-end ms, FPS, model size). `results/demo_reel_summary.json` has the numbers for the slides.

For a live-webcam recording instead: `python scripts\live_demo.py --mirror --save live.mp4`.

## Notes on the numbers

* `latency_ms` in `predict()` and `latency_p95` in `results.csv` are the **classifier inference time** (the brief's self-reported figure). The MediaPipe stage is timed separately and shown live in the HUD and in `benchmark_details.json`.
* `avg_power` needs a phone power profiler; it stays blank unless you pass `--avg-power-mw`.
* `thermal_drift_10min` is the change in median inference latency between the first and last minute of a 10-minute sustained run.
* Without `--hard-cache`, robustness is estimated by perturbing landmarks (angled signer, one hand hidden for 30 % of the clip, 15 % lost detections); with it, from darkened + occluded video.
* The classifier runs single-threaded on purpose: the brief scores phone power and thermal drift, and the model is already < 1 ms.

## Layout

```
predict.py                      evaluator entry point
echogesture/  config · landmarks · features · augment · model · data · predictor · live
scripts/      download_mediapipe_models · prepare_wlasl · extract_landmarks · train · export_quantize
              run_predictions · benchmark · live_demo · make_demo_reel · make_stub_model
tests/smoke_test.py             end-to-end check on synthetic data (train → INT8 → predict → HUD → reel)
models/                         MediaPipe .task files, vocab.txt, echogesture_int8.onnx (open weights)
```
