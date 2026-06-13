import os, json, base64, struct
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 모든 도메인에서 호출 허용

# ── 모델 로드 ──────────────────────────────────────────────────
MOLLAR_PATH = os.path.join(os.path.dirname(__file__), 'model.mollar')
_model_data = None
_labels     = []
_weights    = {}   # name → (shape, dtype, values)

def load_mollar():
    global _model_data, _labels, _weights
    if not os.path.exists(MOLLAR_PATH):
        print('[mollar] model.mollar 파일이 없습니다. /upload 로 업로드하세요.')
        return
    with open(MOLLAR_PATH, 'r', encoding='utf-8') as f:
        _model_data = json.load(f)
    _labels = _model_data.get('userDefinedMetadata', {}).get('labels', [])
    print(f'[mollar] 모델 로드 완료 — 클래스: {_labels}')

load_mollar()

# ── 예측 헬퍼 (numpy 없이 순수 Python으로 Dense+Softmax) ───────
def _b64_to_floats(b64: str):
    raw = base64.b64decode(b64)
    return list(struct.unpack(f'{len(raw)//4}f', raw))

def _relu(x):
    return [max(0.0, v) for v in x]

def _softmax(x):
    m = max(x)
    e = [2.718281828 ** (v - m) for v in x]
    s = sum(e)
    return [v / s for v in e]

def _matmul_bias(vec, W, b, rows, cols):
    out = list(b)
    for c in range(cols):
        for r in range(rows):
            out[c] += vec[r] * W[r * cols + c]
    return out

def predict_from_embedding(embedding: list) -> dict:
    """384차원 임베딩 벡터를 받아 클래스 확률 반환"""
    if _model_data is None:
        return {'error': '모델이 로드되지 않았습니다'}

    floats = _b64_to_floats(_model_data['weightDataBase64'])
    topology = _model_data['modelTopology']['model_config']['config']['layers']

    # Dense 레이어 순서대로 추출
    dense_layers = [l for l in topology if l['class_name'] == 'Dense']
    ptr = 0
    x = embedding

    for layer in dense_layers:
        cfg    = layer['config']
        units  = cfg['units']
        inp    = len(x)
        W      = floats[ptr: ptr + inp * units];  ptr += inp * units
        b      = floats[ptr: ptr + units];         ptr += units
        x      = _matmul_bias(x, W, b, inp, units)
        act    = cfg.get('activation', 'linear')
        if act == 'relu':    x = _relu(x)
        elif act == 'softmax': x = _softmax(x)

    if max(x) > 1.0 or min(x) < 0.0:
        x = _softmax(x)

    idx    = x.index(max(x))
    scores = {_labels[i]: round(x[i], 4) for i in range(len(_labels))}
    return {
        'label':      _labels[idx] if _labels else str(idx),
        'confidence': round(x[idx], 4),
        'scores':     scores
    }

# ── 라우트 ─────────────────────────────────────────────────────

@app.route('/')
def index():
    return jsonify({
        'status': 'ok',
        'model_loaded': _model_data is not None,
        'labels': _labels,
        'endpoints': {
            'POST /predict': '{"embedding": [0.1, 0.2, ...]}  — 384차원 벡터로 예측',
            'GET  /labels':  '현재 클래스 목록 반환',
            'POST /upload':  'model.mollar 파일 업로드 (form-data: file)'
        }
    })

@app.route('/labels', methods=['GET'])
def get_labels():
    return jsonify({'labels': _labels})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Body (JSON):
      { "embedding": [float x 384] }

    임베딩은 클라이언트(브라우저)에서 Transformers.js로 생성 후 전송.
    서버는 Dense+Softmax만 실행 → 빠르고 의존성 없음.
    """
    data = request.get_json(silent=True)
    if not data or 'embedding' not in data:
        return jsonify({'error': 'embedding 필드가 필요합니다 (384차원 float 배열)'}), 400
    emb = data['embedding']
    if len(emb) != 384:
        return jsonify({'error': f'embedding 길이가 {len(emb)}입니다. 384이어야 합니다.'}), 400
    result = predict_from_embedding(emb)
    return jsonify(result)

@app.route('/upload', methods=['POST'])
def upload():
    """model.mollar 파일 업로드 후 즉시 적용"""
    if 'file' not in request.files:
        return jsonify({'error': 'file 필드가 없습니다'}), 400
    f = request.files['file']
    f.save(MOLLAR_PATH)
    load_mollar()
    return jsonify({'status': 'ok', 'labels': _labels})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
