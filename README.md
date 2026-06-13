# Teachable Text AI — 예측 API 서버

학습한 모델(`model.json` + `weights.bin`)을 서버에 올려서, `POST /predict`로
텍스트를 보내면 분류 결과를 돌려주는 완전한 API 서버입니다.

## 1. 모델 파일 준비

원본 Teachable Text AI 페이지에서 학습 → "⬇ model.json + weights.bin" 클릭 →
다운로드한 두 파일을 이 폴더 안의 `model/` 디렉토리에 넣으세요.

```
server/
 ├─ server.js
 ├─ package.json
 └─ model/
     ├─ model.json      ← 여기에 넣기
     └─ weights.bin      ← 여기에 넣기
```

## 2. 로컬 테스트

```bash
npm install
node server.js
```

첫 실행 시 임베딩 모델(multilingual-e5-small, 약 120MB)을 다운로드하므로
1~2분 정도 걸릴 수 있습니다. 이후엔 `.cache/` 폴더에 캐싱되어 빠르게 시작됩니다.

```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"안녕하세요"}'

# → {"label":"인사","confidence":0.97,"scores":{"인사":0.97,"음식":0.03}}
```

## 3. Render에 배포하기

1. 이 `server/` 폴더 전체를 GitHub 리포지토리에 push (model.json, weights.bin 포함)
2. [render.com](https://render.com) → New → Web Service → 해당 리포 연결
3. 설정:
   - **Build Command**: `npm install`
   - **Start Command**: `node server.js`
   - **Instance Type**: Free 가능 (단, 첫 요청은 임베딩 모델 다운로드로 느릴 수 있음)
4. 배포 완료 후 URL 확인 (예: `https://your-app.onrender.com`)

## 4. 클라이언트에서 호출하기

```javascript
const res = await fetch('https://your-app.onrender.com/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: '안녕하세요' })
});
const result = await res.json();
console.log(result); // { label: '인사', confidence: 0.97, scores: {...} }
```

## 주의사항

- **모델을 재학습하면** 새로 내보낸 `model.json`/`weights.bin`으로 `model/` 폴더 내용을
  교체하고 서버를 재시작해야 합니다 (서버는 시작 시 한 번만 모델을 로드함).
- Render 무료 플랜은 일정 시간 미사용 시 슬립 모드에 들어가며, 깨어날 때
  임베딩 모델을 다시 다운로드할 수 있어 첫 응답이 느릴 수 있습니다 (디스크가
  영속되지 않는 플랜의 경우). 영속 디스크를 사용하면 이 문제가 줄어듭니다.
