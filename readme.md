# NoteHub

LLM 모델(ChatGPT, Claude, Gemini)과 연동하여 대화할 수 있는 데스크탑 애플리케이션

## 프로젝트 구조

```
notehub/
├─ packages/
│  ├─ frontend/     # React 프론트엔드
│  ├─ backend/      # Electron 메인/프리로드 프로세스
│  └─ shared/       # 공용 타입 및 유틸리티
├─ package.json     # 워크스페이스 설정
└─ tsconfig.base.json
```

## 시작하기

### 1. 의존성 설치
```bash
npm install
```

### 2. 빌드
```bash
npm run build
```

### 3. 실행

**개발 모드 (Hot Reload):**
```bash
npm run dev:electron
```

**프로덕션 모드:**
```bash
npm run electron
```

## 개발 명령어

- `npm run dev` - 프론트엔드만 개발 서버 실행
- `npm run dev:electron` - 프론트엔드 + Electron 동시 실행 (권장)
- `npm run build` - 전체 빌드
- `npm run build:backend` - 백엔드만 빌드
- `npm run build:frontend` - 프론트엔드만 빌드

## 테스트 계정

- **Email:** test@example.com
- **Password:** 1234

## 필요 조건

- Node.js 18+
- npm 9+

## 개발 환경

- Electron 38+
- React 19+
- TypeScript 5+
- Vite 7+
 