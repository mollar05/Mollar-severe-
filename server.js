/**
 * ╔══════════════════════════════════════════════════════════════════╗
 * ║  server.js — Teachable Text AI 예측 API 서버 (경량 버전)             ║
 * ║                                                                    ║
 * ║  엔드포인트:                                                        ║
 * ║    GET  /              헬스 체크 + 상태 확인                          ║
 * ║    POST /upload        모델 등록/교체 { modelJson, weightsBase64 }    ║
 * ║    POST /predict       임베딩(384차원) → { label, confidence, scores }║
 * ║                                                                    ║
 * ║  ⚠ 변경점: 임베딩(텍스트→벡터)은 더 이상 서버에서 계산하지 않습니다.     ║
 * ║    Render 무료 플랜(512MB)에서 @xenova/transformers를 같이 띄우면     ║
 * ║    메모리 초과(OOM)가 발생하기 때문입니다.                              ║
 * ║    → 클라이언트(브라우저)에서 Transformers.js로 임베딩을 계산한 뒤      ║
 * ║       그 벡터(embedding)를 /predict로 전송합니다.                     ║
 * ║                                                                    ║
 * ║  흐름:                                                              ║
 * ║   1) 학습 페이지에서 "서버로 전송" 클릭 → POST /upload                  ║
 * ║      (model.json + weights.bin을 base64로 전송)                      ║
 * ║   2) 사용 페이지: 브라우저에서 텍스트→임베딩(384차원) 계산 후            ║
 * ║      POST /predict { embedding: number[] } 호출                     ║
 * ║                                                                    ║
 * ║  주의: 모델은 서버 메모리에만 저장됩니다. 서버 재시작 시 다시           ║
 * ║        /upload 해야 합니다 (재학습 후에도 동일).                        ║
 * ╚══════════════════════════════════════════════════════════════════╝
 *
 * ── 로컬 실행 ───────────────────────────────────────────────────────
 *   npm install
 *   node server.js
 *
 * ── 요청 예시 ───────────────────────────────────────────────────────
 *   POST /predict
 *   Content-Type: application/json
 *   { "embedding": [0.01, -0.02, ... 384개 숫자] }
 *
 *   응답:
 *   { "label": "인사", "confidence": 0.97, "scores": { "인사": 0.97, "음식": 0.03 } }
 */

const express = require('express');
const cors = require('cors');
const tf = require('@tensorflow/tfjs-node');

const app = express();
const PORT = process.env.PORT || 3000;

const EMBEDDING_DIM = 384; // multilingual-e5-small 출력 차원

app.use(cors());
app.use(express.json({ limit: '20mb' })); // weights.bin을 base64로 받기 위해 넉넉히

// ── 전역 상태 ──────────────────────────────────────────────────────
let classifier = null;   // 분류기 (Dense layers) — /upload로 등록됨
let labels = [];          // 클래스 레이블
let modelReady = false;  // classifier가 등록되었는지

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

  if (classifier) classifier.dispose();

  classifier = newClassifier;
  labels = modelJSON.userDefinedMetadata.labels;
  modelReady = true;
}

// ── 분류 (임베딩 벡터를 받아서 예측) ───────────────────────────────────
function predictFromEmbedding(embedding) {
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

app.get('/', (req, res) => {
  res.json({
    modelReady,
    labels: modelReady ? labels : [],
    embeddingDim: EMBEDDING_DIM,
    usage: {
      upload: { method: 'POST', path: '/upload', body: { modelJson: 'object', weightsBase64: 'string' } },
      predict: {
        method: 'POST',
        path: '/predict',
        body: { embedding: `number[${EMBEDDING_DIM}] (브라우저에서 Transformers.js로 계산)` },
        response: { label: 'string', confidence: 'number', scores: 'object' }
      }
    }
  });
});

app.post('/upload', async (req, res) => {
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

app.post('/predict', (req, res) => {
  if (!modelReady) {
    return res.status(503).json({ error: '등록된 모델이 없습니다. 먼저 /upload로 모델을 등록하세요.' });
  }

  const { embedding } = req.body;
  if (!Array.isArray(embedding) || embedding.length !== EMBEDDING_DIM) {
    return res.status(400).json({ error: `embedding 필드(${EMBEDDING_DIM}차원 number[])가 필요합니다.` });
  }
  if (!embedding.every(n => typeof n === 'number' && Number.isFinite(n))) {
    return res.status(400).json({ error: 'embedding의 모든 값은 유효한 숫자여야 합니다.' });
  }

  try {
    const result = predictFromEmbedding(embedding);
    res.json(result);
  } catch (e) {
    console.error('predict error:', e);
    res.status(500).json({ error: '예측 처리 중 오류가 발생했습니다.' });
  }
});

// ── 서버 시작 ──────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n🚀 서버 실행 중: http://localhost:${PORT}`);
  console.log(`   GET  /          → 상태 확인`);
  console.log(`   POST /upload    → { modelJson, weightsBase64 } (모델 등록/교체)`);
  console.log(`   POST /predict   → { "embedding": [...${EMBEDDING_DIM}개 숫자] }`);
});
