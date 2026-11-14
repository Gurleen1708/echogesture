# EchoGesture: complete project brief (paste this whole file into any AI chat)

> **Instructions for the AI reading this.** You are helping a member of Team Mudra with EchoGesture,
> their entry for Track 2 of the Samsung PRISM XR Hackathon Challenge 2026. Everything below is the
> real state of the project as of 14 Nov 2026. Treat the numbers here as measured facts and never invent
> new ones. If a number is not in this file, say it has not been measured. The team are first-year
> students and beginners at coding: explain simply, give one step at a time, and in any Windows
> PowerShell instructions always start with the `cd` into the project folder. Do not suggest cloud
> inference, servers, Jetson boards, LLM/VLM pipelines or other approaches that break the on-device rule.
> First ask the person which part they are working on (README, PPT, video, testing, or the model).

---

## 1. The competition

- **Event:** Samsung PRISM XR Hackathon Challenge 2026 (industry-backed, at Thapar Institute, TIET Patiala).
- **Track 2:** sign-language word recognition on an edge device. A smartphone camera films a signer from the
  front (third-person view). The system must say which ASL word was signed, **entirely on the device, with no cloud**.
- **Hard targets in the brief:** top-1 accuracy **≥ 70%**, latency **≤ 50 ms**, model size **≤ 25 MB**.
  Also reported: accuracy drop under hard conditions **≤ 5 points**, average power **≤ 500 mW**,
  thermal drift over 10 minutes **≤ 20%**.
- **Deliverables:** `predict.py`, `predictions.jsonl`, `results.csv`, the quantized model plus its conversion
  script, a demo video, Samsung's PPT template filled in, a GitHub repo, and the Google Form submission.
- **Final submission deadline:** **22 Nov 2026**.
- **Language:** American Sign Language (ASL), word level (one word per clip). Not Indian Sign Language.
- An email has been sent to Samsung asking for the official vocabulary list, the test clips, and whether the
  50 ms budget covers the whole pipeline or only the classifier. **No reply yet.** If Samsung provides its own
  word list or test clips, the model must be retrained/re-run on those.

## 2. The team

| Person | Role |
|---|---|
| **SB (Sukhman Bhullar)** | Tech lead: model training, benchmarks, submission form |
| **Gurleen** | Owns the GitHub repo; README; turned on the website (GitHub Pages) |
| **Harsh** | PPT design, demo-video script, optional website restyle |
| **Shinjan** | Records and edits the demo video; phone endurance test |

## 3. What EchoGesture does (plain words)

You point a phone or laptop camera at someone signing an ASL word. EchoGesture shows the word on screen,
builds the words into a sentence, and can read it aloud. Everything runs on the device: no video, image or
landmark is uploaded anywhere. It knows **100 ASL words** (listed in section 7). If someone signs a word outside
those 100, it can only pick the closest of the 100 it knows. For example, HELLO is not in the list.

One-line pitch: *"Live sign-language subtitles that run entirely on your phone: no internet needed, nothing uploaded."*

## 4. How it works (technical)

```
camera / video ─► MediaPipe hand + pose landmarks ─► 154 pose-invariant features per frame ─► small TCN ─► INT8 ONNX ─► word
```

1. **Landmarks (MediaPipe Tasks, VIDEO mode).** Per frame: 21 3D points for each hand (both image coordinates
   and metric "world" coordinates) and 33 body points. Hand Landmarker (float16) + Pose Landmarker Lite (float16),
   up to 2 hands, confidence thresholds 0.4. Left hand goes in slot 0, right in slot 1. Pixels are thrown away after this step.
2. **Features (`echogesture/features.py`, 154 numbers per frame).** For each hand (72 numbers):
   63 = the hand's 3D shape in a "canonical palm frame" (wrist at origin, scaled to unit palm length, rotated so the palm
   always faces the same way), so it does not matter where the hand is or how the camera is angled; 3 = palm "up"
   direction; 3 = palm normal direction (palm orientation carries meaning in ASL); 2 = wrist position relative to the
   shoulder centre, in shoulder widths; 1 = hand present flag. Body (10 numbers): nose, elbows, wrists relative to the
   shoulders. Values are clipped to [-4, 4] so INT8 quantization stays accurate.
3. **Sequence.** Each clip is sampled to **32 evenly spaced frames** → input shape 32 × 154.
4. **Model (`echogesture/model.py`).** A Temporal Convolutional Network: 1×1 conv stem to 160 channels, 4 residual
   blocks of dilated 1-D convolutions (dilations 1, 2, 4, 1) with BatchNorm + ReLU + dropout, average over time, linear
   layer to 100 classes. **0.66 million parameters.** Only Conv/BN/ReLU, so it quantizes cleanly.
5. **Training (`scripts/train.py`).** AdamW (lr 2e-3, weight decay 0.05), OneCycle schedule, batch 64, 200 epochs,
   label smoothing 0.1, CPU training on SB's laptop (~30 min). The checkpoint with the best validation accuracy is kept.
   **Augmentation is done on the landmarks, not the pixels:** random mirror (left/right hands swapped), 3D rotation
   (camera roll, signer yaw/pitch), hiding one hand for part of the clip, randomly lost detections, small noise, and
   random temporal cropping.
6. **Export + quantization (`scripts/export_quantize.py`).** PyTorch → ONNX (opset 17) → **static INT8** quantization
   (QDQ format, per-channel weights, calibrated on 300 real training clips). INT8 lost **zero** accuracy versus FP32.
7. **Runtime.** ONNX Runtime on CPU, **single thread** on purpose (realistic for a phone that is also running the
   camera, and saves power).
8. **Streaming (live demo).** A sliding window of about the last 1.3–1.4 s of frames; the model runs every ~4 frames;
   predictions are smoothed (moving average); a word is "confirmed" when the same top word holds for 3 runs with ≥60%
   confidence. When the hands drop out of view for ~0.4 s, the sign is treated as finished and the window clears.

**Why these choices (good answers for judges):**
- *Landmarks instead of pixels:* tiny input, fast, works in different lighting and backgrounds, and needs far less
  data. The cost: fine finger details are lost, so look-alike signs (APPLE / CANDY, COOL / FINE) can get confused.
- *Pose-invariant features:* the brief uses a third-person phone camera, so signers stand at different distances
  and angles. The features cancel those out.
- *INT8 + single thread CPU:* meets size, latency and power targets on ordinary phones without a GPU.

## 5. Data

- **Dataset:** WLASL (Word-Level American Sign Language), Kaggle mirror **`risangbaskoro/wlasl-processed`**
  (11,980 videos, 2,000 words). On SB's laptop: `E:\wlasl\kaggle\` (`WLASL_v0.3.json` + `videos\<video_id>.mp4`).
- **Subset used:** **WLASL100**, the first 100 words of WLASL, a standard research benchmark. 1,013 clips were
  available on disk (1,025 listed clips are missing from the mirror, which is normal for WLASL).
- **Official academic splits (not made up by us):** **train 748 / validation 165 / test 100.** The test clips were
  never used for training or for choosing the model.
- **Licence:** WLASL is for **academic, non-commercial use only**. The code can be open source, but the model
  weights come from WLASL data, so the demo must stay free and non-commercial, with WLASL credited.
- **Future (v2, not now):** Microsoft's **ASL Citizen** dataset (about 84,000 videos, 2,731 signs, many signers)
  to grow the vocabulary.

## 6. Results (measured, version 1 model)

| Metric | Target | Measured | Status |
|---|---|---|---|
| Top-1 accuracy (100 test clips, INT8) | ≥ 70% | **70.0%** | ✅ PASS |
| Top-5 accuracy | — | **87%** on test (browser run); about 93% on validation | — |
| Validation top-1 (INT8 = FP32) | — | **76.4%** | — |
| Classifier latency, p95 (laptop CPU, 1 thread) | ≤ 50 ms | **0.74 ms** (median 0.49 ms) | ✅ PASS |
| Model size (INT8 ONNX) | ≤ 25 MB | **0.70 MB** (FP32: 2.63 MB) | ✅ PASS |
| Accuracy drop under hard conditions | ≤ 5 pt | 10 pt (70% → 60%) | ❌ being fixed |
| Average power | ≤ 500 mW | not measured yet | Shinjan's phone test |
| Thermal drift over 10 min | ≤ 20% | not measured yet | Shinjan's phone test |

- Laptop: Intel CPU, Windows, ONNX Runtime 1.31.1.
- The **"hard conditions" test** replays the 100 test clips with the signer turned 25°, the camera tilted 12° and 10°,
  one hand hidden for 30% of the clip, and 15% of hand detections randomly lost.
- **Overfitting check (answer if asked):** validation accuracy kept rising and then held steady at about 76% for the
  rest of training; it never peaked and fell. Test accuracy (70%) is on clips never seen, only 6 points below
  validation. The final training loss was about 0.9, not near zero, so the model was not memorising.
- **In-progress:** a retrain with harsher augmentation (`--aug strong`) to bring the hard-conditions drop to ≤ 5 points.
  It replaces version 1 **only if it is at least as good on clean accuracy too**. Version 1 is backed up in
  `models\v1_backup\`. If the numbers above change, the README and PPT must use the new ones.

## 7. The 100 words

BOOK, DRINK, COMPUTER, BEFORE, CHAIR, GO, CLOTHES, WHO, CANDY, COUSIN, DEAF, FINE, HELP, NO, THIN, WALK, YEAR, YES,
ALL, BLACK, COOL, FINISH, HOT, LIKE, MANY, MOTHER, NOW, ORANGE, TABLE, THANKSGIVING, WHAT, WOMAN, BED, BLUE, BOWLING,
CAN, DOG, FAMILY, FISH, GRADUATE, HAT, HEARING, KISS, LANGUAGE, LATER, MAN, SHIRT, STUDY, TALL, WHITE, WRONG,
ACCIDENT, APPLE, BIRD, CHANGE, COLOR, CORN, COW, DANCE, DARK, DOCTOR, EAT, ENJOY, FORGET, GIVE, LAST, MEET, PINK,
PIZZA, PLAY, SCHOOL, SECRETARY, SHORT, TIME, WANT, WORK, AFRICA, BASKETBALL, BIRTHDAY, BROWN, BUT, CHEAT, CITY,
COOK, DECIDE, FULL, HOW, JACKET, LETTER, MEDICINE, NEED, PAINT, PAPER, PULL, PURPLE, RIGHT, SAME, SON, TELL, THURSDAY

**Good demo words** (distinct, reliable): DRINK, COMPUTER, BOOK, DOCTOR, PIZZA, EAT.
**Not in the list:** HELLO, THANK YOU, PLEASE, SORRY, I LOVE YOU (these are rarer in WLASL; HELLO has only 4 clips).

## 8. The website (public demo)

- **Link:** https://gurleen1708.github.io/echogesture/ (GitHub Pages, served from the repo's `docs/` folder;
  Gurleen must switch it on: Settings → Pages → Deploy from a branch → main + /docs).
- **It still counts as on-device:** the site is just files. The browser downloads MediaPipe (pinned 1.2.3) and
  ONNX Runtime Web (pinned 1.31.1, same versions as the Python setup) plus the 0.7 MB model, and runs everything
  locally. First load is about 40 MB, cached afterwards.
- **Privacy is enforced, not just promised:** a Content-Security-Policy in the page only allows downloading the
  model/library files. MediaPipe's browser library tries to send anonymous usage counters to Google; the policy
  blocks that. Tested: zero requests go anywhere else. The "Speak" button only uses offline voices.
- **Verified against Python:** the JavaScript feature code matches Python to within 0.000001; on the 100 test videos
  the browser version scores **70/100, same as Python, and gives the same answer on 98/100 clips** (the 2
  differences were wrong in both). A fake webcam replaying DRINK and COMPUTER was recognised correctly every loop.
  In-browser model time is about 3 ms. On the slow test machine the whole page ran at about 9 fps; real phones
  have not been measured yet.
- **Features:** live camera with hand tracking overlay, best guess with top-3 confidence bars, sentence strip,
  Speak / Clear, front/back camera switch, searchable word list with links to how each sign looks (SignASL.org).
- **Files:** `docs/index.html`, `docs/css/style.css`, `docs/js/app.js` (page logic), `docs/js/engine.js` (the engine),
  `docs/js/features.js` (port of features.py), `docs/models/echogesture_int8.onnx`, `docs/models/vocab.txt`.
  Harsh can restyle `index.html` and `style.css` freely. **Do not change `features.js` or `engine.js`** without
  re-running the parity check, or accuracy silently breaks.
- **Engine interface:**
  ```js
  const eg = await EchoGesture.load({ onProgress: p => {} });
  eg.onPrediction = ({ word, confidence, top3 }) => {};  // live best guess (word null = hands lowered)
  eg.onWord       = ({ word, confidence }) => {};        // confirmed word, for the sentence
  eg.onLandmarks  = ({ left, right, pose }) => {};       // points for drawing
  eg.onStats      = ({ fps, landmarkMs, modelMs }) => {};
  eg.start(videoElement); eg.stop(); eg.reset(); eg.vocab;
  ```

## 9. The repository

GitHub: https://github.com/Gurleen1708/echogesture (code licence file came from GitHub's repo template; model
weights are WLASL-derived, non-commercial). On SB's laptop: `C:\Users\Asus\Downloads\echogesture (1)\echogesture\`.

| Path | What it is |
|---|---|
| `predict.py` | `predict(video_path) -> dict` in the brief's format |
| `predictions.jsonl` | predictions for the 100 test clips (one JSON line each: clip_id, prediction, top5, confidence, latency_ms) |
| `results.csv` | the metrics table above |
| `echogesture/config.py` | all settings (sequence length 32, 154 features, 160 channels, targets) |
| `echogesture/landmarks.py` | MediaPipe extraction |
| `echogesture/features.py` | the 154 features |
| `echogesture/augment.py` | landmark augmentations and the hard-conditions test |
| `echogesture/model.py` | the TCN |
| `echogesture/predictor.py` | ONNX Runtime inference |
| `echogesture/live.py` | streaming decoder + on-screen HUD for the Python live demo |
| `scripts/prepare_wlasl.py` | builds `data/manifest.csv` + `models/vocab.txt` |
| `scripts/extract_landmarks.py` | runs MediaPipe once per clip, caches to `data/cache/*.npz` (resumable) |
| `scripts/train.py` | training (`--aug strong` for the robustness version) |
| `scripts/export_quantize.py` | ONNX export + INT8 quantization (the conversion script) |
| `scripts/benchmark.py` | writes `results.csv` |
| `scripts/run_predictions.py` | writes `predictions.jsonl` |
| `scripts/live_demo.py` | Python webcam demo (Q quit, C clear, S screenshot, SPACE pause; `--mirror`, `--speak`) |
| `scripts/make_demo_reel.py` | builds a demo video from clips named like `DRINK_01.mp4` |
| `scripts/make_stub_model.py` | random 20-word model for UI testing only (predicts nonsense by design) |
| `tests/smoke_test.py` | end-to-end check, must print `SMOKE TEST PASSED` |
| `models/` | INT8 + FP32 ONNX, vocab, meta, MediaPipe `.task` files |
| `docs/` | the website |

**Reproduce everything (Windows PowerShell, Python 3.12):**
```powershell
cd "C:\Users\Asus\Downloads\echogesture (1)\echogesture"
.\.venv\Scripts\Activate.ps1
python scripts\prepare_wlasl.py --wlasl-json E:\wlasl\kaggle\WLASL_v0.3.json --videos-dir E:\wlasl\kaggle\videos --top-n 100
python scripts\extract_landmarks.py --manifest data\manifest.csv --cache data\cache
python scripts\train.py --manifest data\manifest.csv --cache data\cache
python scripts\export_quantize.py --manifest data\manifest.csv --cache data\cache
python scripts\benchmark.py --manifest data\manifest.csv --cache data\cache
python scripts\run_predictions.py --manifest data\manifest.csv --splits test
```
Landmark extraction took ~25–40 min for 1,013 clips; training ~30 min. Python 3.12 is required (MediaPipe has no
wheels for 3.14). Close Meet/Zoom before using the webcam; they lock it and slow everything down.

## 10. How we got here (short history)

1. Built the full pipeline and tested it with a random "stub" model (its bad predictions were expected).
2. Set up Python 3.12 + virtual environment on SB's Windows laptop.
3. Tried downloading WLASL from its original links: too many dead links. Switched to the Kaggle mirror;
   the first 2.8 GB download was corrupt, the second (4.9 GB) passed an integrity check.
4. Built the manifest, extracted landmarks (1,013/1,013, zero failures), trained, quantized, benchmarked:
   3 of the 3 hard targets passed.
5. Live testing showed HELLO was not recognised because it is not one of the 100 words; look-alike signs get confused.
6. Pushed to GitHub, built and verified the website, published it to `docs/`.
7. Started a robustness-focused retrain.

## 11. Honest limitations and likely judge questions

- **"Why only 100 words?"** WLASL100 is the standard benchmark size; more words means much less accuracy with any
  model this small (published results on 2,000 WLASL words are around 45% even with large models). 100 words lets
  us meet the 70% target on-device. The pipeline itself scales; ASL Citizen is the v2 plan.
- **"Is it overfitting?"** No (see section 6).
- **"Does it really run offline?"** Yes. Load the site, turn on airplane mode, keep signing. The Python version has no
  network code at all.
- **"Why is real-life accuracy lower than 70%?"** The 70% is on WLASL test clips. New signers, webcams and lighting
  differ from the training videos. More signers in training (ASL Citizen) is the fix.
- **"Which signs get confused?"** Signs with the same location and motion but different finger shapes
  (APPLE / CANDY / ORANGE, COOL / FINE), because landmarks lose fine finger detail.
- **Power and thermal** are not measured yet (phone test pending). Never claim numbers for them until measured.
- The **50 ms** target: the classifier alone is 0.74 ms; with MediaPipe the whole per-frame pipeline is more. Report
  both honestly.

## 12. What is left (owners and dates)

| Task | Owner | By |
|---|---|---|
| Finish robustness retrain, compare, export, re-benchmark, update website model | SB | 15 Nov |
| Turn on GitHub Pages | Gurleen | today |
| Video script (from the draft) | Harsh | 16 Nov |
| README final version (from the draft) | Gurleen | 17 Nov |
| Record demo video (airplane-mode demo, back-camera shot) + phone endurance test | Shinjan | 16–18 Nov |
| Fill Samsung PPT template (content + numbers from this file, QR code to the site) | Harsh | 19 Nov |
| Edit demo video (2–3 min, captions, key numbers, no copyrighted music) | Shinjan | 20 Nov |
| Google Form submission (GitHub link, website link, video, PPT) | SB | before 22 Nov |
| Chase Samsung for the official word list / test clips | team | ongoing |
