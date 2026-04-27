# ⚡ 빠른 작업 체크리스트
## AI 행정관 Pro - 환각 탐지 기능 추가

---

## 🎯 5분 요약

### 무엇을 만드는가?
생성형 AI로 작성된 민원의 **환각(허위 정보)을 자동 탐지**하고, 공무원의 **업무 처리를 효율화**하는 기능

### 핵심 기능 3가지
1. 🔍 **AI 환각 탐지**: 날짜, 법령, 수치 등의 오류 자동 발견
2. 📊 **우선순위 분석**: 민원 긴급도 및 업무 복잡도 자동 판단
3. ✅ **자동화 도구**: 체크리스트, 회신문 초안 자동 생성

---

## 📋 작업 단계 (30분 완료 가능)

### Step 1: 파일 준비 (5분)
```bash
cd /path/to/govable-ai
touch hallucination_detection.py
```

### Step 2: 코드 복사 (10분)
- [ ] `hallucination_detection.py`에 제공된 코드 복사
- [ ] `streamlit_app.py` 상단에 임포트 추가
- [ ] 앱 모드 선택 수정 (283번째 줄)
- [ ] 메인 분석 플로우에 환각 검증 추가 (3200번째 줄)
- [ ] 새로운 환각 검증 모드 추가 (3800번째 줄)

### Step 3: 테스트 (10분)
```bash
streamlit run streamlit_app.py
```
- [ ] 메인 모드에서 케이스 분석 실행 → 환각 검증 결과 확인
- [ ] "🔍 AI 민원 검증" 모드 선택 → 테스트 민원 입력
- [ ] 우선순위 분석 결과 확인
- [ ] 체크리스트 생성 확인
- [ ] 회신문 초안 생성 확인

### Step 4: 배포 (5분)
```bash
git add .
git commit -m "feat: AI 환각 탐지 및 업무 효율화 기능 추가"
git push origin main
```

---

## 🔑 핵심 코드 스니펫

### 1. 임포트 추가 (streamlit_app.py 상단)
```python
# Optional hallucination detection module (선택적 의존성)
try:
    from hallucination_detection import (
        detect_hallucination,
        detect_hallucination_cached,
        get_text_hash,
        analyze_petition_priority,
        generate_processing_checklist,
        generate_response_draft,
        render_hallucination_report
    )
    HALLUCINATION_DETECTION_AVAILABLE = True
except ImportError:
    HALLUCINATION_DETECTION_AVAILABLE = False
    # Fallback 함수들 (기본 동작 보장)
```

**중요**: `try-except`로 감싸서 모듈이 없어도 앱이 부팅되도록 합니다!

### 2. 앱 모드 추가
```python
options=[
    "main",
    "admin",
    "revision",
    "duty_manual",
    "hallucination_check"  # ← 이것만 추가
]
```

### 3. 메인 분석에 통합
```python
# 케이스 분석 직후
hallucination_check = detect_hallucination_cached(
    get_text_hash(situation),
    situation,
    {"law": res.get("law", ""), ...},
    llm_service
)
render_hallucination_report(hallucination_check)
```

---

## 🧪 테스트 민원 샘플

### 높은 위험도 (환각 포함)
```
2025년 13월 32일에 ○○구청에 민원을 제기합니다.
주민등록법 제999조에 따르면...
통계청 자료에 따르면 정확히 47.3829472%가...
```

### 낮은 위험도 (정상)
```
2024년 12월 15일에 ○○구청에 민원을 제기합니다.
주민등록법 제7조에 따라 전입신고를 했으나...
```

---

## ⚠️ 주의사항

### 반드시 확인
- [ ] `llm_service`가 전역으로 정의되어 있는지 (86번째 줄)
- [ ] `generate_official_docx` 함수가 임포트되어 있는지 (82번째 줄)
- [ ] Streamlit 버전 1.30+ 사용 중인지

### 선택사항 (나중에 해도 됨)
- [ ] 데이터베이스 테이블 추가 (hallucination_detections, processing_checklists)
- [ ] 사이드바 바로가기 버튼 추가
- [ ] 캐싱 TTL 조정

---

## 🚨 문제 해결

### 에러 1: ModuleNotFoundError
```bash
# 해결: 파일 위치 확인
ls hallucination_detection.py
# streamlit_app.py와 같은 디렉토리에 있어야 함
```

**중요**: 코드는 선택적 의존성으로 작성되어 있어, 모듈이 없어도 앱은 정상 부팅됩니다.
다만 환각 탐지 기능은 비활성화됩니다.

### 에러 2: llm_service not defined
```python
# 해결: 함수 호출 시 llm_service 전달 확인
detect_hallucination(..., llm_service)  # ← 빠뜨리지 않기
```

### 에러 3: JSON parsing error
```python
# 해결: _safe_json_loads 함수 사용
result = _safe_json_loads(llm_response)
```

### 에러 4: HALLUCINATION_DETECTION_AVAILABLE not defined
```python
# 해결: 임포트 섹션이 제대로 추가되었는지 확인
# try-except 블록 전체를 복사했는지 체크
```

### 에러 5: Cannot hash argument 'llm_service' (캐싱 오류) ⚠️
```
TypeError: cannot pickle 'function' object
또는
Cannot hash argument 'llm_service' (of type __main__.LLMService)
```

**원인**: `llm_service` 객체가 해시 불가능(pickle 불가)해서 `@st.cache_data` 오류 발생

**해결**: `hallucination_detection.py`가 이미 이 문제를 처리합니다:
- 패턴 기반 탐지만 캐싱 (빠르고 안전)
- LLM 기반 탐지는 매번 실행 (llm_service 제외)
- 최신 버전의 `hallucination_detection.py` 사용 확인

**수동 해결** (필요시):
```python
# detect_hallucination_cached 함수를 다음과 같이 수정
@st.cache_data(ttl=3600)
def _cached_core(text_hash: str, text: str, context_json: str):
    # llm_service 없이 패턴만 탐지
    pass

def detect_hallucination_cached(text_hash, text, context, llm_service):
    # 1. 캐싱된 패턴 탐지
    pattern_result = _cached_core(text_hash, text, json.dumps(context))
    # 2. LLM 탐지 (매번 실행)
    llm_result = _detect_by_llm(text, context, llm_service)
    # 3. 병합
    return merge_results(pattern_result, llm_result)
```

---

## 📊 예상 성능

| 항목 | 값 |
|------|-----|
| 환각 탐지 소요 시간 | ~5-10초 |
| 우선순위 분석 소요 시간 | ~5초 |
| 체크리스트 생성 소요 시간 | ~5초 |
| 회신문 초안 생성 소요 시간 | ~10초 |
| **전체 워크플로우** | **~30초** |

---

## ✅ 완료 기준

### 최소 요구사항 (MVP)
- [x] 환각 탐지 기능 작동
- [x] 결과를 UI에 표시
- [x] 에러 없이 실행

### 권장 요구사항
- [ ] 우선순위 분석 작동
- [ ] 체크리스트 생성 작동
- [ ] 회신문 초안 생성 작동
- [ ] 모든 UI 깨짐 없음

### 최고 수준
- [ ] DB 저장 기능
- [ ] 검증 결과 통계
- [ ] 관리자 대시보드 연동

---

## 🎓 학습 자료

### 이해해야 할 개념
1. **환각(Hallucination)**: AI가 그럴듯하지만 거짓인 정보를 생성하는 현상
2. **패턴 매칭**: 정규표현식으로 특정 패턴 탐지
3. **LLM 교차 검증**: AI로 AI를 검증하는 메타 접근

### 추천 읽기
- Streamlit 문서: https://docs.streamlit.io
- 정규표현식: https://regexr.com
- LLM 환각 탐지 논문 (선택)

---

**마지막 업데이트**: 2026-02-10
**작성자**: Claude
**검토자**: -
**상태**: ✅ 개발 준비 완료
