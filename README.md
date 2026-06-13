# mollar-server

Teachable Text AI `.mollar` 모델을 REST API로 제공하는 Flask 서버입니다.

---

## 배포 (Render)

1. 이 폴더를 GitHub 저장소에 올리기
2. [render.com](https://render.com) → New → Web Service → 저장소 연결
3. 설정:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Environment**: Python 3
4. Deploy → 완료되면 `https://your-app.onrender.com` URL 발급

---

## 모델 업로드

배포 후 `.mollar` 파일을 서버에 올려야 합니다.

```bash
curl -X POST https://your-app.onrender.com/upload \
  -F "file=@model.mollar"
```

또는 아무 HTTP 클라이언트(Postman 등)로 `POST /upload` + form-data로 업로드.

---

## API 사용법

### 클라이언트 (브라우저) 쪽에서 임베딩 생성 후 서버로 전송

```html
<script type="module">
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js';

const SERVER = 'https://your-app.onrender.com';
const extractor = await pipeline('feature-extraction', 'Xenova/multilingual-e5-small');

async function classify(text) {
  // 1. 브라우저에서 임베딩 생성
  const out = await extractor([`query: ${text}`], { pooling: 'mean', normalize: true });
  const embedding = Array.from(out.data);

  // 2. 서버로 전송
  const res = await fetch(`${SERVER}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ embedding })
  });
  return await res.json();
}

const result = await classify('안녕하세요');
console.log(result);
// { label: '인사', confidence: 0.97, scores: { 인사: 0.97, 음식: 0.03 } }
</script>
```

---

## 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/` | 서버 상태 및 엔드포인트 목록 |
| GET | `/labels` | 현재 클래스 목록 |
| POST | `/predict` | 예측 (`{ "embedding": [...] }`) |
| POST | `/upload` | `.mollar` 파일 업로드 (form-data) |

---

## 구조

```
mollar-server/
├── app.py            # Flask 서버
├── requirements.txt  # 의존성
├── Procfile          # Render 시작 명령
├── model.mollar      # 학습된 모델 (직접 넣거나 /upload로 올리기)
└── README.md
```
