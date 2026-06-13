/**
 * ╔══════════════════════════════════════════════════════════════════╗
 * ║  teachable-text-ai.js  —  v1.0                                   ║
 * ║  Teachable Text AI 모델을 다른 사이트에서 불러와 쓰는 런타임 라이브러리  ║
 * ║                                                                    ║
 * ║  사용법:                                                            ║
 * ║    1) 이 파일을 <script src="teachable-text-ai.js"> 로 로드           ║
 * ║    2) TF.js와 Transformers.js도 페이지에 포함 (아래 예시 참고)         ║
 * ║    3) TeachableTextAI.load(modelJsonUrl, weightsBinUrl) 호출         ║
 * ║    4) TeachableTextAI.predict(text) 로 분류                          ║
 * ╚══════════════════════════════════════════════════════════════════╝
 *
 * 필수 의존성 (페이지에 먼저 로드):
 *   <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.0/dist/tf.min.js"></script>
 *   (Transformers.js는 이 라이브러리가 내부에서 동적으로 로드합니다)
 *
 * ────────────────────────────────────────────────────────────────────
 *  PUBLIC API
 * ────────────────────────────────────────────────────────────────────
 *
 *  await TeachableTextAI.load(modelJsonUrl, weightsBinUrl [, options])
 *    → 모델 파일 URL을 받아 로드. 로컬 파일 객체도 지원 (아래 참고).
 *    options: {
 *      onProgress: (pct, message) => void   // 진행률 콜백 (0~100)
 *    }
 *
 *  await TeachableTextAI.loadFromFiles(modelJsonFile, weightsBinFile [, options])
 *    → <input type="file"> 로 선택한 File 객체를 직접 전달
 *
 *  await TeachableTextAI.predict(text)
 *    → { label: string, confidence: number, scores: { [label]: number } }
 *
 *  TeachableTextAI.isReady()  → boolean
 *  TeachableTextAI.getLabels() → string[]
 *  TeachableTextAI.onReady(callback)  → callback({ labels })
 *  TeachableTextAI.unload()   → 메모리 해제
 */

(function (global) {
  'use strict';

  // ── 내부 상태 ──────────────────────────────────────────────────────
  let _extractor = null;
  let _classifier = null;
  let _labels = [];
  let _ready = false;
  const _readyCbs = [];

  const TRANSFORMERS_CDN =
    'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js';
  const MODEL_ID = 'Xenova/multilingual-e5-small';

  // ── Transformers.js 동적 로드 (한 번만) ───────────────────────────
  let _transformersPromise = null;
  function _loadTransformers() {
    if (_transformersPromise) return _transformersPromise;
    _transformersPromise = import(TRANSFORMERS_CDN).then(mod => {
      mod.env.allowLocalModels = false;
      mod.env.useBrowserCache = true;
      return mod;
    });
    return _transformersPromise;
  }

  // ── Feature extractor 로드 (한 번만) ──────────────────────────────
  let _extractorPromise = null;
  async function _ensureExtractor(onProgress) {
    if (_extractor) return _extractor;
    if (_extractorPromise) return _extractorPromise;

    _extractorPromise = (async () => {
      const { pipeline } = await _loadTransformers();
      _extractor = await pipeline('feature-extraction', MODEL_ID, {
        progress_callback: (p) => {
          if (!onProgress) return;
          if (p.status === 'downloading' && p.total) {
            const pct = Math.round((p.loaded / p.total) * 60); // 0~60%
            onProgress(pct, `임베딩 모델 다운로드 중... ${(p.loaded / 1024 / 1024).toFixed(1)} MB`);
          } else if (p.status === 'loading') {
            onProgress(65, '임베딩 모델 초기화 중...');
          }
        }
      });
      return _extractor;
    })();
    return _extractorPromise;
  }

  // ── 임베딩 ─────────────────────────────────────────────────────────
  async function _embed(sentences, isQuery = false) {
    const prefix = isQuery ? 'query: ' : 'passage: ';
    const output = await _extractor(
      sentences.map(s => prefix + s),
      { pooling: 'mean', normalize: true }
    );
    // output.data: Float32Array, shape [batch, 384]
    const dim = 384;
    const n = sentences.length;
    const tensors = [];
    for (let i = 0; i < n; i++) {
      tensors.push(tf.tensor2d(output.data.slice(i * dim, (i + 1) * dim), [1, dim]));
    }
    const stacked = tf.concat(tensors, 0);
    tensors.forEach(t => t.dispose());
    return stacked;
  }

  // ── 준비 완료 이벤트 ───────────────────────────────────────────────
  function _fireReady() {
    _ready = true;
    _readyCbs.forEach(cb => { try { cb({ labels: [..._labels] }); } catch (e) { } });
    _readyCbs.length = 0;
  }

  // ── 공개 API 객체 ──────────────────────────────────────────────────
  const TeachableTextAI = {

    /**
     * URL에서 model.json + weights.bin 로드
     * @param {string} modelJsonUrl
     * @param {string} weightsBinUrl
     * @param {{ onProgress?: (pct:number, msg:string)=>void }} [options]
     */
    load: async function (modelJsonUrl, weightsBinUrl, options = {}) {
      const onProgress = options.onProgress || null;

      _ready = false;
      if (_classifier) { _classifier.dispose(); _classifier = null; }

      try {
        // 1) 임베딩 모델 로드
        onProgress && onProgress(5, '임베딩 모델 준비 중...');
        await _ensureExtractor(onProgress);
        onProgress && onProgress(70, 'model.json 다운로드 중...');

        // 2) model.json fetch
        const jsonRes = await fetch(modelJsonUrl);
        if (!jsonRes.ok) throw new Error(`model.json 로드 실패: ${jsonRes.status} ${jsonRes.statusText}`);
        const modelJSON = await jsonRes.json();

        onProgress && onProgress(80, 'weights.bin 다운로드 중...');

        // 3) weights.bin fetch
        const binRes = await fetch(weightsBinUrl);
        if (!binRes.ok) throw new Error(`weights.bin 로드 실패: ${binRes.status} ${binRes.statusText}`);
        const binBuffer = await binRes.arrayBuffer();

        onProgress && onProgress(90, '분류기 모델 빌드 중...');

        // 4) 레이블 추출
        if (modelJSON.userDefinedMetadata?.labels) {
          _labels = modelJSON.userDefinedMetadata.labels;
        } else {
          throw new Error('model.json에 labels 정보가 없습니다. Teachable Text AI에서 내보낸 파일인지 확인하세요.');
        }

        // 5) TF.js 분류기 복원
        _classifier = await tf.loadLayersModel(
          tf.io.fromMemory({
            modelTopology: modelJSON.modelTopology,
            weightSpecs: modelJSON.weightsManifest[0].weights,
            weightData: binBuffer
          })
        );

        onProgress && onProgress(100, '준비 완료!');
        _fireReady();
        return { labels: [..._labels] };

      } catch (e) {
        _ready = false;
        throw e;
      }
    },

    /**
     * File 객체로 직접 로드 (<input type="file"> 사용 시)
     * @param {File} modelJsonFile
     * @param {File} weightsBinFile
     * @param {{ onProgress?: (pct:number, msg:string)=>void }} [options]
     */
    loadFromFiles: async function (modelJsonFile, weightsBinFile, options = {}) {
      const onProgress = options.onProgress || null;

      _ready = false;
      if (_classifier) { _classifier.dispose(); _classifier = null; }

      try {
        onProgress && onProgress(5, '임베딩 모델 준비 중...');
        await _ensureExtractor(onProgress);
        onProgress && onProgress(70, 'model.json 읽는 중...');

        const jsonText = await modelJsonFile.text();
        const modelJSON = JSON.parse(jsonText);

        onProgress && onProgress(80, 'weights.bin 읽는 중...');
        const binBuffer = await weightsBinFile.arrayBuffer();

        onProgress && onProgress(90, '분류기 모델 빌드 중...');

        if (modelJSON.userDefinedMetadata?.labels) {
          _labels = modelJSON.userDefinedMetadata.labels;
        } else {
          throw new Error('model.json에 labels 정보가 없습니다.');
        }

        _classifier = await tf.loadLayersModel(
          tf.io.fromMemory({
            modelTopology: modelJSON.modelTopology,
            weightSpecs: modelJSON.weightsManifest[0].weights,
            weightData: binBuffer
          })
        );

        onProgress && onProgress(100, '준비 완료!');
        _fireReady();
        return { labels: [..._labels] };

      } catch (e) {
        _ready = false;
        throw e;
      }
    },

    /**
     * 텍스트 분류 예측
     * @param {string} text
     * @returns {Promise<{ label: string, confidence: number, scores: Object }>}
     */
    predict: async function (text) {
      if (!_classifier || !_extractor) {
        throw new Error('모델이 준비되지 않았습니다. load() 또는 loadFromFiles() 를 먼저 호출하세요.');
      }
      if (!text || !text.trim()) {
        throw new Error('텍스트를 입력하세요.');
      }

      const feat = await _embed([text.trim()], true);
      const predTensor = _classifier.predict(feat);
      const probs = Array.from(predTensor.dataSync());
      feat.dispose();
      predTensor.dispose();

      const maxIdx = probs.indexOf(Math.max(...probs));
      const scores = {};
      probs.forEach((p, i) => { scores[_labels[i]] = parseFloat(p.toFixed(4)); });

      return {
        label: _labels[maxIdx],
        confidence: parseFloat(probs[maxIdx].toFixed(4)),
        scores
      };
    },

    /**
     * 모델 준비 완료 시 콜백 등록 (이미 준비된 경우 즉시 호출)
     * @param {function({ labels: string[] }): void} callback
     */
    onReady: function (callback) {
      if (_ready) {
        try { callback({ labels: [..._labels] }); } catch (e) { }
      } else {
        _readyCbs.push(callback);
      }
    },

    /** 현재 예측 가능한지 여부 */
    isReady: function () { return _ready && !!_classifier; },

    /** 현재 클래스 레이블 목록 */
    getLabels: function () { return [..._labels]; },

    /** 메모리 해제 */
    unload: function () {
      if (_classifier) { _classifier.dispose(); _classifier = null; }
      _extractor = null;
      _extractorPromise = null;
      _labels = [];
      _ready = false;
    }
  };

  // ── 전역 등록 ──────────────────────────────────────────────────────
  global.TeachableTextAI = TeachableTextAI;

  // ES Module 호환
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = TeachableTextAI;
  }

})(typeof globalThis !== 'undefined' ? globalThis : window);
