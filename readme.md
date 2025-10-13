# Conversation Graph Generator

대화 내역을 분석하여 관련성 있는 대화들을 연결하는 지식 그래프를 생성하는 Python 프로젝트입니다.


## 🎯 프로젝트 개요

ChatGPT와 같은 대화형 AI와의 대화 내역을 분석하여, 주제적으로 관련된 대화들을 자동으로 연결하는 지식 그래프를 생성합니다. 이를 통해 방대한 대화 기록에서 패턴과 관계를 시각적으로 파악할 수 있습니다.

## ✨ 주요 기능

### 1. 다중 분석 방식
- **키워드 기반 분석**: 공유 키워드를 통한 빠른 연결 감지
- **의미론적 분석**: Sentence Transformers를 활용한 심층 의미 분석
- **LLM 기반 분석**: 대규모 언어 모델을 활용한 고급 관계 판단

### 2. 지능형 처리
- 대용량 대화 자동 요약
- 스트리밍 방식의 메모리 효율적 처리
- 배치 처리를 통한 API 호출 최적화

### 3. 유연한 설정
- 유사도 임계값 조정 가능
- 키워드 필터링 커스터마이징
- 다양한 처리 모드 지원

## 🔧 시스템 요구사항

- Python 3.8 이상
- 최소 4GB RAM (대용량 데이터셋의 경우 8GB 권장)
- 인터넷 연결 (모델 다운로드 및 LLM API 사용 시)

## 📦 설치 방법

### 1. 저장소 클론
```bash
git clone https://github.com/yourusername/conversation-graph-generator.git
cd conversation-graph-generator
```

### 2. 가상환경 생성 (권장)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 디렉토리 구조 생성
```bash
mkdir -p input_data graph_data
```

## 🚀 사용 방법

### 1. 입력 데이터 준비

`input_data/conversations.json` 파일에 대화 데이터를 배치합니다:

```json
[
  {
    "title": "Python 기초 학습",
    "mapping": {
      "msg_1": {
        "message": {
          "content": {
            "parts": ["Python에서 리스트와 튜플의 차이가 뭔가요?"]
          }
        }
      }
    }
  }
]
```

### 2. 프로세싱 실행

#### 기본 모드 (키워드 + 의미론적 분석)
```bash
python backend/process_data.py
```

#### LLM 모드 (고급 분석)
```bash
python backend/llm_process.py
```

### 3. 출력 확인

생성된 그래프 데이터는 다음 위치에 저장됩니다:
- 기본 모드: `graph_data/graph_data.json`
- LLM 모드: `graph_data/graph_data_llm.json`

## 🔄 프로세싱 모드

### Mode 1: 하이브리드 분석 (`process_data.py`)

**특징:**
- LLM 불필요
- 빠른 처리 속도
- 중소규모 데이터셋에 적합

**처리 과정:**
1. 노드 생성 및 텍스트 추출
2. 키워드 기반 링크 생성
3. 의미론적 유사도 계산
4. 하이브리드 링크 병합

**장점:**
- 외부 API 불필요
- 안정적이고 예측 가능한 결과
- 비용 절감

### Mode 2: LLM 증강 분석 (`llm_process.py`)

**특징:**
- LLM API 활용
- 더욱 정확한 관계 판단
- 대규모 데이터셋 최적화

**처리 과정:**
1. 대화 전처리 및 경량화
2. 고신뢰도 의미론적 링크 생성 (임계값 0.6)
3. 키워드 후보 필터링
4. LLM 배치 검토 (임계값 0.4)

**장점:**
- 미묘한 관계 파악
- 컨텍스트 기반 판단
- 배치 처리로 API 효율화

## ⚙️ 설정 및 최적화

### process_data.py 설정

```python
# 의미론적 유사도 임계값 (0.0-1.0)
# 높을수록 더 직접적인 관련성만 연결
SIMILARITY_THRESHOLD = 0.5  

# 제외할 일반 단어
STOP_WORDS = set(['the', 'a', 'is', ...])
```

**키워드 링크 범위 조정:**
```python
# process_data.py의 create_keyword_links() 함수에서
if 1 < len(node_ids) < 10:  # 기본값
    # 더 엄격하게: if 2 < len(node_ids) < 5:
    # 더 느슨하게: if 1 < len(node_ids) < 15:
```

### llm_process.py 설정

```python
# 의미론적 자동 연결 임계값
SEMANTIC_THRESHOLD = 0.6  # 높음 = 고품질 자동 링크

# LLM 검토 임계값
LLM_REVIEW_THRESHOLD = 0.4  # 낮음 = 더 많은 후보 검토

# 키워드 오버랩 최소값
KEYWORD_OVERLAP_THRESHOLD = 3  # 공유 키워드 개수

# 배치 크기 (API 호출 최적화)
BATCH_SIZE = 50  # 한 번에 처리할 후보 쌍 수
```

### LLM API 설정 (.env 파일)

```env
QWEN_API_URL=https://your-api-endpoint.com/v1
QWEN_API_KEY=your-api-key-here
```

**llm.py 설정:**
```python
# SSL 인증서 검증 (자체 서명 인증서 사용 시)
ALLOW_INSECURE_CONNECTIONS = True  # 주의: 보안 위험

# 사용 모델
MODEL_NAME = "Qwen3-8B"

# 타임아웃 설정
timeout=30.0
```

## 📊 출력 형식

생성된 JSON 파일 구조:

```json
{
  "nodes": [
    {
      "id": "conv_0",
      "label": "대화 제목",
      "type": "conversation"
    }
  ],
  "links": [
    {
      "source": "conv_0",
      "target": "conv_1",
      "strength": 0.85,
      "type": "semantic",
      "relationship": "Semantically similar topics"
    }
  ]
}
```

**링크 타입:**
- `semantic`: 의미론적 유사도 기반
- `llm_approved`: LLM이 승인한 연결
- `keyword`: 키워드 공유 기반 (하이브리드 모드)

## 🔍 문제 해결

### 링크 생성이 너무 많거나 적음

**너무 많을 때:**
- `SIMILARITY_THRESHOLD` 증가 (예: 0.6)
- `KEYWORD_OVERLAP_THRESHOLD` 증가 (예: 5)
- 키워드 링크 범위 축소 (예: `if 2 < len(node_ids) < 5`)

**너무 적을 때:**
- `SIMILARITY_THRESHOLD` 감소 (예: 0.4)
- `KEYWORD_OVERLAP_THRESHOLD` 감소 (예: 2)
- 키워드 링크 범위 확대 (예: `if 1 < len(node_ids) < 15`)

### 처리 속도가 느림

**최적화 방법:**
1. LLM 모드에서 `BATCH_SIZE` 증가
2. `SEMANTIC_THRESHOLD` 증가로 LLM 검토 대상 감소
3. 더 작은 임베딩 모델 사용 고려
4. 입력 데이터 전처리로 불필요한 대화 제거

