/**
 * ╔══════════════════════════════════════════════════════════════════╗
 * ║  server.js — Teachable Text AI 예측 API 서버                       ║
 * ║                                                                    ║
 * ║  엔드포인트:                                                        ║
 * ║    GET  /              헬스 체크 + 상태 확인                          ║
 * ║    POST /upload        모델 등록/교체 { modelJson, weightsBase64 }    ║
 * ║    POST /predict       텍스트 → { label, confidence, scores }       ║
 * ║                                                                    ║
 * ║  흐름:                                                              ║
 * ║   1) 학습 페이지에서 "서버로 전송" 클릭 → POST /upload                  ║
 * ║      (model.json + weights.bin을 base64로 전송, 파일 다운로드 불필요)  ║
 * ║   2) 다른 사이트에서는 POST /predict { text }만 호출하면 끝            ║
 * ║                                                                    ║
 * ║  주의: 모델은 서버 메모리에만 저장됩니다. 서버 재시작 시 다시           ║
 * ║        /upload 해야 합니다 (재학습 후에도 동일).                        ║
 * ║                                                                    ║
 * ║  배포: Render / Vercel / Railway / Fly.io 등 Node 환경 어디든 OK     ║
 * ╚══════════════════════════════════════════════════════════════════╝
 *
 * ── 로컬 실행 ───────────────────────────────────────────────────────
 *   npm install
 *   node server.js
 *
 * ── 요청 예시 ───────────────────────────────────────────────────────
 *   POST /predict
 *   Content-Type: application/json
 *   { "text": "안녕하세요" }
 *
 *   응답:
 *   { "label": "인사", "confidence": 0.97, "scores": { "인사": 0.97, "음식": 0.03 } }
 */

const express = require('express');
const cors = require('cors');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');
const { pipeline, env } = require('@xenova/transformers');

const app = express();
const PORT = process.env.PORT || 3000;

// Transformers.js 캐시 설정 (서버 환경)
env.cacheDir = path.join(__dirname, '.cache');

app.use(cors());
app.use(express.json({ limit: '20mb' })); // weights.bin을 base64로 받기 위해 넉넉히

// ── 전역 상태 ──────────────────────────────────────────────────────
let extractor = null;    // 임베딩 모델 (multilingual-e5-small) — 서버 시작 시 1회 로드
let classifier = null;   // 분류기 (Dense layers) — /upload로 등록됨
let labels = [];          // 클래스 레이블
let modelReady = false;  // classifier가 등록되었는지
let embedderReady = false; // extractor 로딩 완료 여부

// ── 임베딩 모델은 서버 시작 시 미리 로드 ──────────────────────────────
async function loadEmbedder() {
  console.log('[준비] 임베딩 모델 로딩 중... (최초 1회는 다운로드, 1~2분 소요될 수 있음)');
  extractor = await pipeline('feature-extraction', 'Xenova/multilingual-e5-small');
  embedderReady = true;
  console.log('[준비] 임베딩 모델 준비 완료. /upload 로 모델을 등록하세요.');
}

// ── model.json + weights.bin(base64) 받아서 분류기 메모리에 등록 ─────────
async function registerModel(modelJSON, weightsBase64) {
  if (!modelJSON.userDefinedMetadata?.labels) {
    throw new Error('model.json에 labels 정보가 없습니다. Teachable Text AI에서 내보낸 파일인지 확인하세요.');
  }

  const weightBuffer = Buffer.from(weightsBase64, 'base64');

  const newClassifier = await tf.loadLayersModel(
    tf.io.fromMemory({
      modelTopology: modelJSON.modelTopology,
      weightSpecs: modelJSON.weightsManifest[0].weights,
      weightData: weightBuffer.buffer.slice(
        weightBuffer.byteOffset,
        weightBuffer.byteOffset + weightBuffer.byteLength
      )
    })
  );

  // 기존 모델 메모리 해제
  if (classifier) classifier.dispose();

  classifier = newClassifier;
  labels = modelJSON.userDefinedMetadata.labels;
  modelReady = true;
}

// ── 임베딩 + 분류 ──────────────────────────────────────────────────
async function predictText(text) {
  const out = await extractor([`query: ${text}`], { pooling: 'mean', normalize: true });
  const embedding = Array.from(out.data); // 384차원

  const inputTensor = tf.tensor2d([embedding], [1, embedding.length]);
  const predTensor = classifier.predict(inputTensor);
  const probs = Array.from(predTensor.dataSync());
  inputTensor.dispose();
  predTensor.dispose();

  const maxIdx = probs.indexOf(Math.max(...probs));
  const scores = {};
  probs.forEach((p, i) => { scores[labels[i]] = parseFloat(p.toFixed(4)); });

  return {
    label: labels[maxIdx],
    confidence: parseFloat(probs[maxIdx].toFixed(4)),
    scores
  };
}

// ── 라우트 ─────────────────────────────────────────────────────────

// 헬스 체크 / 상태 확인
app.get('/', (req, res) => {
  res.json({
    embedderReady,
    modelReady,
    labels: modelReady ? labels : [],
    usage: {
      upload: { method: 'POST', path: '/upload', body: { modelJson: 'object', weightsBase64: 'string' } },
      predict: { method: 'POST', path: '/predict', body: { text: 'string' }, response: { label: 'string', confidence: 'number', scores: 'object' } }
    }
  });
});

// 학습 페이지에서 모델 등록 (model.json + weights.bin)
app.post('/upload', async (req, res) => {
  if (!embedderReady) {
    return res.status(503).json({ error: '서버가 아직 초기화 중입니다. 잠시 후 다시 시도하세요.' });
  }

  const { modelJson, weightsBase64 } = req.body;
  if (!modelJson || !weightsBase64) {
    return res.status(400).json({ error: 'modelJson(object)과 weightsBase64(string)가 필요합니다.' });
  }

  try {
    await registerModel(modelJson, weightsBase64);
    res.json({ ok: true, labels });
  } catch (e) {
    console.error('upload error:', e);
    res.status(400).json({ error: '모델 등록 실패: ' + e.message });
  }
});

// 예측 엔드포인트
app.post('/predict', async (req, res) => {
  if (!embedderReady) {
    return res.status(503).json({ error: '서버가 아직 초기화 중입니다. 잠시 후 다시 시도하세요.' });
  }
  if (!modelReady) {
    return res.status(503).json({ error: '등록된 모델이 없습니다. 먼저 /upload로 모델을 등록하세요.' });
  }

  const { text } = req.body;
  if (!text || typeof text !== 'string' || !text.trim()) {
    return res.status(400).json({ error: 'text 필드(문자열)가 필요합니다.' });
  }
  if (text.length > 2000) {
    return res.status(400).json({ error: '텍스트가 너무 길어요 (최대 2000자).' });
  }

  try {
    const result = await predictText(text.trim());
    res.json(result);
  } catch (e) {
    console.error('predict error:', e);
    res.status(500).json({ error: '예측 처리 중 오류가 발생했습니다.' });
  }
});

// ── 서버 시작 ──────────────────────────────────────────────────────
loadEmbedder()
  .then(() => {
    app.listen(PORT, () => {
      console.log(`\n🚀 서버 실행 중: http://localhost:${PORT}`);
      console.log(`   GET  /          → 상태 확인`);
      console.log(`   POST /upload    → { modelJson, weightsBase64 } (모델 등록/교체)`);
      console.log(`   POST /predict   → { "text": "안녕하세요" }`);
    });
  })
  .catch(err => {
    console.error('❌ 임베딩 모델 로딩 실패:', err.message);
    process.exit(1);
  });
