# 그래프 시각화

대화 데이터를 기반으로 한 3D 노드 관계 그래프 시각화 프로젝트입니다.

## 📁 프로젝트 구조

```
graph_part/
├── index.html              # 메인 HTML 페이지
├── requirments.txt         # Python 의존성 목록
├── css/
│   └── style.css          # 스타일시트
├── js/
│   ├── app.js             # 메인 애플리케이션 로직
│   └── graph.js           # 3D 그래프 시각화 (D3.js + 3D Force Graph)
├── backend/
│   ├── llm.py             # OpenAI API 클라이언트 설정
│   ├── llm_process.py     # LLM 기반 관계 분석 메인 프로세서
│   ├── preprocess_input.py # 입력 데이터 전처리
│   └── process_data.py    # 그래프 데이터 생성 로직
├── input_data/
│   ├── conversations.json  # GPT 대화 내역 (ChatGPT에서 내보낸 JSON)
│   ├── graph_data.json    # 기본 그래프 데이터
│   └── mock_data.json     # 테스트용 모의 데이터
├── graph_data/
│   ├── graph_data.json    # 처리된 기본 그래프 데이터 (127KB)
│   └── graph_data_llm.json # LLM 분석 결과 그래프 데이터 (127KB)
└── documents/
    ├── project_plan.md    # 프로젝트 계획서
    ├── prototype.md       # 프로토타입 설명
    ├── gemini.md         # Gemini API 관련 문서
    └── frontend.md       # 프론트엔드 구현 가이드
```

## 🚀 실행 방법

### 1. 로컬 서버 실행

**Python 내장 서버 사용:**

```bash
# Python 3
python -m http.server 8000
```

### 2. 브라우저에서 확인

서버 실행 후 브라우저에서 접속:

- `http://localhost:8000` (Python 서버)

### 3. 그래프 조작

- **마우스 드래그**: 시점 회전
- **마우스 휠**: 줌 인/아웃
- **노드 클릭**: 노드 정보 표시
- **노드 더블클릭**: 선택한 노드로 줌인

## 📊 데이터 구조

- `graph_data.json` - 기본 그래프 데이터 (127KB)
- `graph_data_llm.json` - LLM 분석 결과 (127KB)
- **노드**: 대화별 라벨과 타입 정보
- **링크**: 노드 간 연결 관계 및 가중치

## 🔄 새 데이터 처리

새로운 대화 데이터로 그래프를 재생성하려면:

1. **의존성 설치**

   ```bash
   pip install -r requirments.txt
   ```

2. **데이터 처리 실행**

   ```bash
   python backend/llm_process.py
   ```

3. **결과 확인**
   - `graph_data/` 디렉토리에 새 JSON 파일 생성
   - 웹 브라우저 새로고침으로 업데이트된 그래프 확인
