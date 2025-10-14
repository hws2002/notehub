# 그래프 시각화

대화 데이터를 기반으로 한 3D 노드 관계 그래프 시각화 프로젝트입니다.

## 📁 구조

- `js/` - 그래프 시각화 로직 (D3.js + 3D Force Graph)
- `backend/` - 데이터 처리 및 LLM 분석
- `input_data/` - 원본 대화 데이터
- `graph_data/` - 생성된 그래프 데이터 (JSON)
- `index.html` - 메인 페이지

## 🚀 실행 방법

### 1. 로컬 서버 실행

**Python 내장 서버 사용:**

```bash
# Python 3
python -m http.server 8000

# 또는 Python 2
python -m SimpleHTTPServer 8000
```

### 2. 브라우저에서 확인

서버 실행 후 브라우저에서 접속:

- `http://localhost:8000` (Python 서버)

### 3. 그래프 조작

- **마우스 드래그**: 시점 회전
- **마우스 휠**: 줌 인/아웃
- **노드 클릭**: 노드 정보 표시

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
