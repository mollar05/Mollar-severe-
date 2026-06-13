# Teachable Text AI — 예측 API 서버 (경량 버전)

학습한 모델(`model.json` + `weights.bin`)을 서버 메모리에 등록해두고,
`POST /predict`로 **임베딩 벡터(384차원)**를 보내면 분류 결과를 돌려주는 API 서버입니다.

## ⚠ 중요 변경점 (메모리 절약)

Render 무료 플랜은 메모리 512MB로 제한되는데, 임베딩 모델
(`@xenova/transformers`, multilingual-e5-small ~120MB + 런타임)을 서버에서
같이 띄우면 OOM(Out of Memory)으로 죽습니다.

그래서 이 버전은:
- **서버**: 분류기(Dense layer)만 메모리에 올림 → 매우 가벼움
- **브라우저(클라이언트)**: Transformers.js로 텍스트→임베딩(384차원) 계산 후
  그 벡터를 서버에 전송

## 1. 로컬 테스트

```bash
npm install
node server.js
```

## 2. 모델 등록

학습 페이지(`web/index.html`)에서 학습 완료 → 서버 주소 입력 →
**"🚀 서버로 전송 (API 등록)"** 클릭. `model.json`+`weights.bin`이
base64로 `/upload`에 전송되어 서버 메모리에 등록됩니다.

```bash
curl -X POST http://localhost:3000/upload \
  -H "Content-Type: application/json" \
  -d '{"modelJson": {...}, "weightsBase64": "..."}'
```

## 3. 예측 호출

클라이언트(브라우저)에서 직접 호출:

```javascript
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js';

const extractor = await pipeline('feature-extraction', 'Xenova/multilingual-e5-small');
const out = await extractor([`query: 안녕하세요`], { pooling: 'mean', normalize: true });
const embedding = Array.from(out.data); // 384차원

const res = await fetch('https://your-app.onrender.com/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ embedding })
});
console.log(await res.json());
// → { label: '인사', confidence: 0.97, scores: { 인사: 0.97, 음식: 0.03 } }
```

`web/demo-consumer.html`의 **"🌐 API 모드"** 섹션에 이미 이 로직이 구현되어
있으니, 서버 주소만 입력하면 바로 테스트할 수 있습니다.

## 4. Render 배포

1. 이 폴더(`server.js`, `package.json`)를 GitHub 리포 **루트**에 push
2. Render → New → Web Service
3. **Runtime: Node** (Python 아님!)
4. Build Command: `npm install`
5. Start Command: `node server.js`

## 주의사항

- 모델은 **서버 메모리에만** 저장됩니다. 서버 재시작/재배포 시 `/upload`로
  다시 등록해야 합니다.
- 임베딩 차원은 `multilingual-e5-small` 기준 **384**로 고정되어 있습니다.
