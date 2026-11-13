// Exact JavaScript port of echogesture/features.py.
// Raw landmarks -> pose-invariant feature vectors (no learned parts).
//
// A raw frame is:
//   { handsImg:   Float32Array(2*21*3)  normalised image coords, slot 0 = Left, slot 1 = Right
//     handsWorld: Float32Array(2*21*3)  metric 3D hand landmarks
//     handPresent:[0|1, 0|1]
//     pose:       Float32Array(33*3)    normalised image coords
//     posePresent: 0|1 }
// A clip is an array of raw frames plus the video aspect ratio (width / height).

export const SEQ_LEN = 32;
export const HAND_FEATS = 72;
export const POSE_FEATS = 10;
export const FEAT_DIM = 2 * HAND_FEATS + POSE_FEATS; // 154
export const FEAT_CLIP = 4.0;

const EPS = 1e-6;
const POSE_JOINTS = [0, 13, 14, 15, 16]; // nose, elbows, wrists
const L_SHOULDER = 11;
const R_SHOULDER = 12;

export function emptyFrame() {
  return {
    handsImg: new Float32Array(126),
    handsWorld: new Float32Array(126),
    handPresent: [0, 0],
    pose: new Float32Array(99),
    posePresent: 0,
  };
}

function bodyFrame(frames, aspect) {
  const T = frames.length;
  const xy = []; // xy[t] = Float64Array(66)
  const width = new Float64Array(T);
  const valid = new Array(T);
  let anyValid = false;
  for (let t = 0; t < T; t++) {
    const p = frames[t].pose;
    const a = new Float64Array(66);
    for (let j = 0; j < 33; j++) {
      a[2 * j] = p[3 * j] * aspect;
      a[2 * j + 1] = p[3 * j + 1];
    }
    xy.push(a);
    const dx = a[2 * L_SHOULDER] - a[2 * R_SHOULDER];
    const dy = a[2 * L_SHOULDER + 1] - a[2 * R_SHOULDER + 1];
    width[t] = Math.hypot(dx, dy);
    valid[t] = frames[t].posePresent > 0.5 && width[t] > 1e-3;
    if (valid[t]) anyValid = true;
  }
  const center = [];
  if (anyValid) {
    const idx = [];
    for (let t = 0; t < T; t++) if (valid[t]) idx.push(t);
    const xy2 = [];
    const width2 = new Float64Array(T);
    for (let t = 0; t < T; t++) {
      // nearest valid frame; ties go to the earlier one (numpy argmin keeps the first minimum)
      let best = idx[0];
      let bestD = Math.abs(idx[0] - t);
      for (let k = 1; k < idx.length; k++) {
        const d = Math.abs(idx[k] - t);
        if (d < bestD) { bestD = d; best = idx[k]; }
      }
      xy2.push(xy[best]);
      width2[t] = width[best];
      const s = xy[best];
      center.push([(s[2 * L_SHOULDER] + s[2 * R_SHOULDER]) / 2, (s[2 * L_SHOULDER + 1] + s[2 * R_SHOULDER + 1]) / 2]);
    }
    return { xy: xy2, center, width: width2, hasBody: true };
  }
  // no body at all: assume a centred signer filling ~1/3 of the frame
  const c = [0.5 * aspect, 0.45];
  const w = new Float64Array(T).fill(0.3 * aspect);
  const xy3 = [];
  for (let t = 0; t < T; t++) {
    center.push(c);
    const a = new Float64Array(66);
    for (let j = 0; j < 33; j++) { a[2 * j] = c[0]; a[2 * j + 1] = c[1]; }
    xy3.push(a);
  }
  return { xy: xy3, center, width: w, hasBody: false };
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => Math.sqrt(dot(a, a));
const unit = (a) => { const n = norm(a) + EPS; return [a[0] / n, a[1] / n, a[2] / n]; };

// Writes the 72 hand features for one slot of one frame into out[offset..offset+72).
function handBlock(frame, slot, center, width, aspect, out, offset) {
  const present = frame.handPresent[slot];
  if (!(present > 0)) return; // all zeros (out is pre-zeroed)
  const base = slot * 63;
  const W = frame.handsWorld;
  const w0 = [W[base], W[base + 1], W[base + 2]];
  const P = new Array(21);
  for (let k = 0; k < 21; k++) {
    const o = base + 3 * k;
    P[k] = [W[o] - w0[0], W[o + 1] - w0[1], W[o + 2] - w0[2]];
  }
  const scale = norm(P[9]) + EPS;               // wrist -> middle MCP
  const u = unit(P[9]);                         // palm "up"
  const n = unit(cross(u, sub(P[17], P[5])));   // palm normal
  const v = cross(n, u);                        // across the palm
  let i = offset;
  for (let k = 0; k < 21; k++) {
    out[i++] = (dot(v, P[k]) / scale) * present;
    out[i++] = (dot(u, P[k]) / scale) * present;
    out[i++] = (dot(n, P[k]) / scale) * present;
  }
  for (let d = 0; d < 3; d++) out[i++] = u[d] * present;
  for (let d = 0; d < 3; d++) out[i++] = n[d] * present;
  const I = frame.handsImg;
  const wx = I[base] * aspect;
  const wy = I[base + 1];
  out[i++] = ((wx - center[0]) / width) * present;
  out[i++] = ((wy - center[1]) / width) * present;
  out[i++] = 1.0 * present;
}

function clean(x) {
  if (Number.isNaN(x)) return 0;
  return x > FEAT_CLIP ? FEAT_CLIP : x < -FEAT_CLIP ? -FEAT_CLIP : x;
}

/** Raw clip -> Float32Array(T * FEAT_DIM), row-major (frame, feature). */
export function clipFeatures(frames, aspect) {
  const T = frames.length;
  const { xy, center, width, hasBody } = bodyFrame(frames, aspect);
  const out = new Float64Array(T * FEAT_DIM);
  for (let t = 0; t < T; t++) {
    const row = t * FEAT_DIM;
    handBlock(frames[t], 0, center[t], width[t], aspect, out, row);
    handBlock(frames[t], 1, center[t], width[t], aspect, out, row + HAND_FEATS);
    if (hasBody) {
      let i = row + 2 * HAND_FEATS;
      for (const j of POSE_JOINTS) {
        out[i++] = (xy[t][2 * j] - center[t][0]) / width[t];
        out[i++] = (xy[t][2 * j + 1] - center[t][1]) / width[t];
      }
    }
  }
  const f = new Float32Array(out.length);
  for (let k = 0; k < out.length; k++) f[k] = clean(out[k]);
  return f;
}

// numpy.round rounds half to even
function roundHalfEven(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** Evenly spaced frame indices (inference path of features.sample_indices). */
export function sampleIndices(nSrc, nOut = SEQ_LEN) {
  const idx = new Array(nOut);
  const step = (nSrc - 1) / (nOut - 1);
  for (let i = 0; i < nOut; i++) {
    const pos = i === nOut - 1 ? nSrc - 1 : i * step;
    idx[i] = Math.min(Math.max(roundHalfEven(pos), 0), nSrc - 1);
  }
  return idx;
}

/** Raw clip -> Float32Array(SEQ_LEN * FEAT_DIM) model input. */
export function clipToInput(frames, aspect) {
  const feats = clipFeatures(frames, aspect);
  const idx = sampleIndices(frames.length, SEQ_LEN);
  const x = new Float32Array(SEQ_LEN * FEAT_DIM);
  for (let i = 0; i < SEQ_LEN; i++) {
    x.set(feats.subarray(idx[i] * FEAT_DIM, (idx[i] + 1) * FEAT_DIM), i * FEAT_DIM);
  }
  return x;
}
