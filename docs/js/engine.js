// EchoGesture browser engine: camera frames -> MediaPipe landmarks -> features -> INT8 ONNX model -> words.
// Everything runs on the visitor's device. No frame, landmark or prediction leaves the browser.
//
//   const eg = await EchoGesture.load({ onProgress: p => {} });
//   eg.onPrediction = ({ word, confidence, top3 }) => {};   // live best guesses, several times a second
//   eg.onWord       = ({ word, confidence }) => {};         // a sign was confirmed (use for the sentence strip)
//   eg.onLandmarks  = ({ left, right, pose }) => {};        // [{x, y}] in 0..1 video coords, for drawing
//   eg.onStats      = ({ fps, landmarkMs, modelMs }) => {};
//   eg.start(videoElement); eg.stop(); eg.reset(); eg.vocab;

import { FilesetResolver, HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision';
import * as ort from 'onnxruntime-web';
import { clipToInput, emptyFrame, SEQ_LEN, FEAT_DIM } from './features.js';

const MP_VERSION = '1.2.3';
export const DEFAULTS = {
  visionWasm: `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`,
  handModel: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
  poseModel: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
  classifier: new URL('../models/echogesture_int8.onnx', import.meta.url).href,
  vocab: new URL('../models/vocab.txt', import.meta.url).href,
  delegate: 'CPU',      // same as the Python pipeline; 'GPU' is faster on some phones
  minConf: 0.4,         // MediaPipe confidences, as in landmarks.py
  // Streaming decoder (time-based version of live.StreamDecoder, tuned at ~30 fps)
  windowMs: 1400,       // ~40 frames at 30 fps
  strideMs: 130,        // run the classifier ~every 4 frames
  minSignMs: 400,       // ~12 frames before the first guess
  idleMs: 400,          // hands gone this long -> sign finished, clear the window
  threshold: 0.6,
  stable: 3,
};

async function fetchWithProgress(url, onBytes) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not download ${url} (${res.status})`);
  const total = Number(res.headers.get('content-length')) || 0;
  if (!res.body || !total) { const b = new Uint8Array(await res.arrayBuffer()); onBytes(b.length, b.length); return b; }
  const reader = res.body.getReader();
  const chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length; onBytes(got, total);
  }
  const out = new Uint8Array(got); let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

function softmax(logits) {
  let m = -Infinity; for (const v of logits) m = Math.max(m, v);
  const e = Array.from(logits, v => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / s);
}

export class EchoGesture {
  static async load(options = {}) {
    const o = { ...DEFAULTS, ...options };
    const report = options.onProgress || (() => {});
    // weights: rough share of the total download, so the bar moves smoothly
    const parts = { hand: 0, pose: 0, clf: 0, rt: 0 };
    const weight = { hand: 0.4, pose: 0.3, clf: 0.05, rt: 0.25 };
    const tick = () => report(Object.keys(parts).reduce((a, k) => a + parts[k] * weight[k], 0));

    const [handBuf, poseBuf, clfBuf, vocabText] = await Promise.all([
      fetchWithProgress(o.handModel, (g, t) => { parts.hand = g / t; tick(); }),
      fetchWithProgress(o.poseModel, (g, t) => { parts.pose = g / t; tick(); }),
      fetchWithProgress(o.classifier, (g, t) => { parts.clf = g / t; tick(); }),
      fetch(o.vocab).then(r => r.text()),
    ]);
    const vision = await FilesetResolver.forVisionTasks(o.visionWasm);
    ort.env.wasm.numThreads = 1;  // single thread, like the Python benchmark (and no special server headers needed)
    const session = await ort.InferenceSession.create(clfBuf, { executionProviders: ['wasm'] });
    parts.rt = 1; tick();
    const vocab = vocabText.split(/\r?\n/).map(w => w.trim().toUpperCase()).filter(Boolean);
    const eg = new EchoGesture(o, vision, handBuf, poseBuf, session, vocab);
    await eg.createLandmarkers();
    report(1);
    return eg;
  }

  constructor(o, vision, handBuf, poseBuf, session, vocab) {
    this.o = o; this.vision = vision; this.handBuf = handBuf; this.poseBuf = poseBuf;
    this.session = session; this.inputName = session.inputNames[0]; this.outputName = session.outputNames[0];
    this.vocab = vocab;
    this.modelBytes = 0;
    this.onPrediction = null; this.onWord = null; this.onLandmarks = null; this.onStats = null;
    this.running = false; this.lastTs = -1;
    this.resetDecoder();
    this.stats = { frameTimes: [], lm: [], model: [] };
  }

  /** (Re)create the MediaPipe landmarkers. Recreating clears tracking state between unrelated clips. */
  async createLandmarkers() {
    this.hands?.close(); this.pose?.close();
    const c = this.o.minConf;
    const base = (buf) => ({ modelAssetBuffer: buf, delegate: this.o.delegate });
    this.hands = await HandLandmarker.createFromOptions(this.vision, {
      baseOptions: base(this.handBuf), runningMode: 'VIDEO', numHands: 2,
      minHandDetectionConfidence: c, minHandPresenceConfidence: c, minTrackingConfidence: c,
    });
    this.pose = await PoseLandmarker.createFromOptions(this.vision, {
      baseOptions: base(this.poseBuf), runningMode: 'VIDEO', numPoses: 1,
      minPoseDetectionConfidence: c, minPosePresenceConfidence: c, minTrackingConfidence: c,
    });
    this.lastTs = -1;
  }

  /** One video frame -> raw landmark frame (same layout and slot rules as landmarks.py). */
  processFrame(source, timestampMs) {
    const ts = Math.max(Math.round(timestampMs), this.lastTs + 1); // must strictly increase
    this.lastTs = ts;
    const out = emptyFrame();
    const res = this.hands.detectForVideo(source, ts);
    const handed = res.handedness || res.handednesses || [];
    const pending = [];
    for (let i = 0; i < res.landmarks.length; i++) {
      const cat = handed[i][0];
      const slot = cat.categoryName.toLowerCase().startsWith('l') ? 0 : 1;
      pending.push({ score: cat.score, slot, img: res.landmarks[i], world: res.worldLandmarks[i] });
    }
    // Highest-confidence hand gets its own slot; a second hand with the same label takes the other slot.
    pending.sort((a, b) => b.score - a.score);
    const best = [-1, -1];
    for (const h of pending) {
      let slot = h.slot;
      if (best[slot] >= 0) { slot = 1 - slot; if (best[slot] >= 0) continue; }
      best[slot] = h.score;
      for (let k = 0; k < 21; k++) {
        const o = slot * 63 + 3 * k;
        out.handsImg[o] = h.img[k].x; out.handsImg[o + 1] = h.img[k].y; out.handsImg[o + 2] = h.img[k].z;
        out.handsWorld[o] = h.world[k].x; out.handsWorld[o + 1] = h.world[k].y; out.handsWorld[o + 2] = h.world[k].z;
      }
      out.handPresent[slot] = 1;
    }
    const pres = this.pose.detectForVideo(source, ts);
    if (pres.landmarks && pres.landmarks.length) {
      const p = pres.landmarks[0];
      for (let j = 0; j < 33; j++) { out.pose[3 * j] = p[j].x; out.pose[3 * j + 1] = p[j].y; out.pose[3 * j + 2] = p[j].z; }
      out.posePresent = 1;
    }
    return out;
  }

  /** Raw clip (array of frames) -> probabilities over the vocabulary. */
  async classify(frames, aspect) {
    const t0 = performance.now();
    const x = clipToInput(frames, aspect);
    const out = await this.session.run({ [this.inputName]: new ort.Tensor('float32', x, [1, SEQ_LEN, FEAT_DIM]) });
    const probs = softmax(out[this.outputName].data);
    return { probs, ms: performance.now() - t0 };
  }

  topK(probs, k) {
    return probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, k)
      .map(([p, i]) => ({ word: this.vocab[i], confidence: p }));
  }

  /** Whole-clip prediction, same as Predictor.predict() in Python (used for recorded videos). */
  async predictClip(frames, aspect) {
    const { probs, ms } = await this.classify(frames, aspect);
    const top5 = this.topK(probs, 5);
    return { prediction: top5[0].word, top5: top5.map(t => t.word), confidence: top5[0].confidence, latencyMs: ms };
  }

  // ---- live streaming -----------------------------------------------------------------

  resetDecoder() {
    this.buf = []; this.ema = null; this.candidate = null; this.lastWord = null;
    this.streak = 0; this.lastRun = -Infinity; this.lastHandAt = null; this.top3 = [];
  }
  reset() { this.resetDecoder(); }

  async pushLive(frame, now, aspect) {
    const d = this.o;
    const hasHand = frame.handPresent[0] > 0 || frame.handPresent[1] > 0;
    if (hasHand) this.lastHandAt = now;
    const idleFor = this.lastHandAt === null ? Infinity : now - this.lastHandAt;
    if (!hasHand && idleFor >= d.idleMs) {                       // hands down -> sign finished
      if (this.buf.length || this.top3.length) {
        this.resetDecoder();
        this.onPrediction?.({ word: null, confidence: 0, top3: [] });
      }
      return;
    }
    this.buf.push({ frame, t: now });
    while (this.buf.length && now - this.buf[0].t > d.windowMs) this.buf.shift();
    const span = now - this.buf[0].t;
    const seen = this.buf.filter(b => b.frame.handPresent[0] > 0 || b.frame.handPresent[1] > 0).length / this.buf.length;
    if (this.buf.length < 6 || span < d.minSignMs || seen < 0.5 || now - this.lastRun < d.strideMs) return;
    this.lastRun = now;
    const { probs, ms } = await this.classify(this.buf.map(b => b.frame), aspect);
    this.stats.model.push(ms); if (this.stats.model.length > 30) this.stats.model.shift();
    this.ema = this.ema ? this.ema.map((v, i) => 0.6 * v + 0.4 * probs[i]) : probs;
    this.top3 = this.topK(this.ema, 3);
    const { word, confidence } = this.top3[0];
    this.streak = word === this.candidate ? this.streak + 1 : 1;
    this.candidate = word;
    this.onPrediction?.({ word, confidence, top3: this.top3 });
    if (confidence >= d.threshold && this.streak >= d.stable && word !== this.lastWord) {
      this.lastWord = word;
      this.onWord?.({ word, confidence });
    }
  }

  start(video) {
    this.video = video; this.running = true; this.resetDecoder();
    const loop = async () => {
      if (!this.running) return;
      if (video.readyState >= 2 && video.videoWidth) {
        const t0 = performance.now();
        const frame = this.processFrame(video, t0);
        const lmMs = performance.now() - t0;
        this.emitLandmarks(frame);
        await this.pushLive(frame, t0, video.videoWidth / video.videoHeight);
        const s = this.stats;
        s.frameTimes.push(t0); while (s.frameTimes.length > 30) s.frameTimes.shift();
        s.lm.push(lmMs); if (s.lm.length > 30) s.lm.shift();
        const avg = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0;
        const fps = s.frameTimes.length > 1 ? 1000 * (s.frameTimes.length - 1) / (s.frameTimes.at(-1) - s.frameTimes[0]) : 0;
        this.onStats?.({ fps, landmarkMs: avg(s.lm), modelMs: avg(s.model) });
      }
      if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(loop);
      else requestAnimationFrame(loop);
    };
    loop();
  }

  stop() { this.running = false; }

  emitLandmarks(frame) {
    if (!this.onLandmarks) return;
    const hand = (s) => frame.handPresent[s] ? Array.from({ length: 21 }, (_, k) => ({ x: frame.handsImg[s * 63 + 3 * k], y: frame.handsImg[s * 63 + 3 * k + 1] })) : null;
    const pose = frame.posePresent ? Array.from({ length: 33 }, (_, j) => ({ x: frame.pose[3 * j], y: frame.pose[3 * j + 1] })) : null;
    this.onLandmarks({ left: hand(0), right: hand(1), pose });
  }
}
