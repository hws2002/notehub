# NoteHub


## 🎯 프로젝트 개요

ChatGPT와 같은 대화형 AI와의 대화 내역을 분석하여, 주제적으로 관련된 대화들을 자동으로 연결하는 지식 그래프를 생성합니다. 이를 통해 방대한 대화 기록에서 패턴과 관계를 시각적으로 파악할 수 있습니다.

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

## How to start

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

---