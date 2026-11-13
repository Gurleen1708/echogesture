import { EchoGesture } from './engine.js';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const HAND_EDGES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],
  [14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const BODY_EDGES = [[11,12],[11,13],[13,15],[12,14],[14,16]];

let eg = null, stream = null, facing = params.get('camera') === 'back' ? 'environment' : 'user';
const words = [];
window.__words = words; // handy for automated tests

// ---------------- word list ----------------
function openVocab() {
  if (!eg) return loadVocabOnly().then(() => $('vocabDialog').showModal());
  $('vocabDialog').showModal();
}
async function loadVocabOnly() {
  const text = await (await fetch('models/vocab.txt')).text();
  renderVocab(text.split(/\r?\n/).map(w => w.trim().toUpperCase()).filter(Boolean));
}
function renderVocab(vocab) {
  const list = $('vocabList'); list.innerHTML = '';
  for (const w of [...vocab].sort()) {
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = `https://www.signasl.org/sign/${encodeURIComponent(w.toLowerCase().replace(/_/g, '-'))}`;
    a.target = '_blank'; a.rel = 'noopener'; a.textContent = w.replace(/_/g, ' ');
    li.appendChild(a); list.appendChild(li);
  }
}
$('vocabSearch').addEventListener('input', (e) => {
  const q = e.target.value.trim().toUpperCase();
  for (const li of $('vocabList').children) li.classList.toggle('hidden', q && !li.textContent.includes(q));
});
$('vocabLink').addEventListener('click', (e) => { e.preventDefault(); openVocab(); });
$('vocabBtn').addEventListener('click', openVocab);
$('closeVocab').addEventListener('click', () => $('vocabDialog').close());

// ---------------- camera ----------------
async function openCamera() {
  stream?.getTracks().forEach(t => t.stop());
  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: facing, width: { ideal: 640 }, height: { ideal: 480 } },
  });
  const v = $('video');
  v.srcObject = stream;
  await v.play();
  await new Promise(r => (v.videoWidth ? r() : v.addEventListener('loadedmetadata', r, { once: true })));
  $('stage').style.aspectRatio = `${v.videoWidth} / ${v.videoHeight}`;
  $('overlay').width = v.videoWidth; $('overlay').height = v.videoHeight;
  $('stage').classList.toggle('mirror', facing === 'user');
}

$('flipBtn').addEventListener('click', async () => {
  facing = facing === 'user' ? 'environment' : 'user';
  try { eg.stop(); await openCamera(); eg.start($('video')); }
  catch (e) { facing = facing === 'user' ? 'environment' : 'user'; alert('Could not switch camera: ' + e.message); }
});

// ---------------- drawing ----------------
function draw({ left, right, pose }) {
  const c = $('overlay'), ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  const P = (p) => [p.x * c.width, p.y * c.height];
  ctx.lineCap = 'round';
  if (pose) {
    ctx.strokeStyle = 'rgba(255,255,255,.35)'; ctx.lineWidth = 3;
    for (const [a, b] of BODY_EDGES) { ctx.beginPath(); ctx.moveTo(...P(pose[a])); ctx.lineTo(...P(pose[b])); ctx.stroke(); }
  }
  for (const [hand, col] of [[left, '#6fe3a0'], [right, '#6aa8ff']]) {
    if (!hand) continue;
    ctx.strokeStyle = col; ctx.lineWidth = 2.5;
    for (const [a, b] of HAND_EDGES) { ctx.beginPath(); ctx.moveTo(...P(hand[a])); ctx.lineTo(...P(hand[b])); ctx.stroke(); }
    ctx.fillStyle = '#fff';
    for (const p of hand) { ctx.beginPath(); ctx.arc(...P(p), 3, 0, 2 * Math.PI); ctx.fill(); }
  }
}

// ---------------- predictions ----------------
function showPrediction({ word, confidence, top3 }) {
  const w = $('word');
  if (!word) {          // hands lowered: keep the last guess on screen, faded, until the next sign starts
    if (!w.classList.contains('idle')) { w.classList.add('faded'); $('top3').classList.add('faded'); }
    return;
  }
  $('top3').classList.remove('faded');
  w.textContent = word.replace(/_/g, ' ');
  w.className = 'word' + (confidence >= 0.6 ? ' confirmed' : '');
  $('top3').innerHTML = top3.map(t => `
    <div class="row"><span>${t.word.replace(/_/g, ' ')}</span>
      <div class="track"><div class="fill" style="width:${(t.confidence * 100).toFixed(0)}%"></div></div>
      <span class="pct">${(t.confidence * 100).toFixed(0)}%</span></div>`).join('');
}
function addWord({ word }) {
  words.push(word.replace(/_/g, ' '));
  $('sentence').textContent = words.join(' ');
}
$('clearBtn').addEventListener('click', () => { words.length = 0; $('sentence').innerHTML = '<span class="muted">Recognised words appear here</span>'; });
$('speakBtn').addEventListener('click', () => {
  if (!words.length || !('speechSynthesis' in window)) return;
  const u = new SpeechSynthesisUtterance(words.join(' '));
  // only on-device voices, so the text is not sent to an online speech service
  const local = speechSynthesis.getVoices().find(v => v.localService && v.lang.startsWith('en'));
  if (!local) { alert('No offline voice is installed in this browser.'); return; }
  u.voice = local; speechSynthesis.speak(u);
});

// ---------------- start ----------------
$('startBtn').addEventListener('click', async () => {
  $('startBtn').disabled = true; $('err').classList.add('hidden'); $('loading').classList.remove('hidden');
  try {
    const camReady = openCamera();
    eg = await EchoGesture.load({
      delegate: params.get('gpu') === '1' ? 'GPU' : 'CPU',
      onProgress: (p) => {
        $('barFill').style.width = `${Math.round(p * 100)}%`;
        $('loadText').textContent = p < 1 ? `Downloading models… ${Math.round(p * 100)}% (one time, then cached)` : 'Starting…';
      },
    });
    window.eg = eg;
    renderVocab(eg.vocab);
    await camReady;
    eg.onLandmarks = draw;
    eg.onPrediction = showPrediction;
    eg.onWord = addWord;
    eg.onStats = ({ fps, landmarkMs, modelMs }) => {
      $('stats').textContent = `${fps.toFixed(0)} fps · landmarks ${landmarkMs.toFixed(0)} ms · model ${modelMs.toFixed(1)} ms · 0.70 MB`;
    };
    $('intro').classList.add('hidden'); $('live').classList.remove('hidden');
    eg.start($('video'));
  } catch (e) {
    console.error(e);
    const msg = /Permission|NotAllowed/i.test(e.name + e.message)
      ? 'Camera permission was blocked. Allow camera access in your browser settings and try again.'
      : 'Something went wrong: ' + e.message;
    $('err').textContent = msg; $('err').classList.remove('hidden');
    $('startBtn').disabled = false; $('loading').classList.add('hidden');
  }
});
