# 🔧 오류 해결 가이드
## AI 행정관 Pro - 환각 탐지 기능 오류 수정

---

## 🚨 발생한 오류 분석

### 오류 메시지
```
TypeError: cannot pickle 'function' object

검증 중 오류 발생: Cannot hash argument 'llm_service' (of type __main__.LLMService) 
in 'detect_hallucination_cached'.

To address this, you can tell Streamlit not to hash this argument by adding 
a leading underscore to the argument's name in the function signature:

@st.cache_data
def detect_hallucination_cached(_llm_service, ...):
    ...
```

### 원인 분석

**핵심 문제**: Streamlit의 `@st.cache_data` 데코레이터는 함수 인자를 해싱(hashing)하여 캐시 키를 생성하는데, `llm_service` 객체는 **pickle 불가능**하여 해싱할 수 없음.

```python
# ❌ 문제가 되는 코드
@st.cache_data(ttl=3600)
def detect_hallucination_cached(text_hash: str, text: str, context: Dict, llm_service) -> Dict:
    return detect_hallucination(text, context, llm_service)
    # llm_service는 LLMService 클래스 인스턴스
    # → 내부에 함수 객체, API 클라이언트 등 pickle 불가능 요소 포함
```

**왜 pickle 불가능한가?**
- `llm_service` 객체는 Vertex AI, Gemini API, Groq 클라이언트를 포함
- 이들은 네트워크 연결, 함수 객체 등 직렬화 불가능한 상태를 가짐
- Python의 `pickle` 모듈로 변환 불가 → 캐싱 실패

---

## ✅ 해결 방법 (3가지 옵션)

### 방법 1: 언더스코어 추가 (Streamlit 권장) ⭐ 가장 간단

Streamlit의 제안대로 인자명 앞에 `_`를 붙이면 해당 인자는 캐싱 키 생성에서 제외됩니다.

```python
@st.cache_data(ttl=3600)
def detect_hallucination_cached(text_hash: str, text: str, context: Dict, _llm_service) -> Dict:
    """캐싱된 환각 탐지 - llm_service는 캐싱 제외"""
    return detect_hallucination(text, context, _llm_service)
```

**장점**: 
- ✅ 코드 수정 최소화 (1줄만 변경)
- ✅ Streamlit 공식 패턴
- ✅ 즉시 적용 가능

**단점**:
- ⚠️ LLM 기반 탐지도 캐싱됨 (동일 텍스트에 대해 항상 같은 결과)
- ⚠️ LLM 응답의 변동성 활용 불가

---

### 방법 2: 분리된 캐싱 (제공된 코드) ⭐⭐ 최적 솔루션

패턴 기반 탐지만 캐싱하고, LLM 기반 탐지는 매번 실행합니다.

```python
@st.cache_data(ttl=3600)
def _detect_hallucination_cached_core(text_hash: str, text: str, context_json: str) -> Dict:
    """패턴 기반 탐지만 캐싱 - llm_service 불필요"""
    context = json.loads(context_json) if context_json else {}
    suspicious_parts = _detect_by_patterns(text)
    risk_level, overall_score = _calculate_risk_level(suspicious_parts)
    verification_needed = _extract_verification_items(suspicious_parts)
    
    return {
        "risk_level": risk_level,
        "suspicious_parts": suspicious_parts,
        "verification_needed": verification_needed,
        "overall_score": overall_score,
        "total_issues_found": len(suspicious_parts),
        "cached": True
    }

def detect_hallucination_cached(text_hash: str, text: str, context: Dict, llm_service) -> Dict:
    """하이브리드 캐싱: 패턴은 캐싱, LLM은 매번 실행"""
    # 1. 패턴 탐지 (캐싱)
    context_json = json.dumps(context, ensure_ascii=False) if context else ""
    cached_result = _detect_hallucination_cached_core(text_hash, text, context_json)
    
    # 2. LLM 탐지 (매번 실행)
    llm_issues = _detect_by_llm(text, context, llm_service)
    
    # 3. 결과 병합 및 중복 제거
    all_suspicious = cached_result['suspicious_parts'] + llm_issues
    unique_suspicious = []
    seen_texts = set()
    for part in all_suspicious:
        part_text = part.get('text', '')
        if part_text not in seen_texts:
            seen_texts.add(part_text)
            unique_suspicious.append(part)
    
    # 4. 최종 위험도 재계산
    risk_level, overall_score = _calculate_risk_level(unique_suspicious)
    verification_needed = _extract_verification_items(unique_suspicious)
    
    return {
        "risk_level": risk_level,
        "suspicious_parts": unique_suspicious,
        "verification_needed": verification_needed,
        "overall_score": overall_score,
        "total_issues_found": len(unique_suspicious),
        "cached": False
    }
```

**장점**:
- ✅ 패턴 탐지는 빠른 캐싱 (동일 민원 재검증 시 즉각 응답)
- ✅ LLM 탐지는 매번 실행 (응답 변동성 활용, 더 정확)
- ✅ 최적의 성능과 정확도 균형

**단점**:
- ⚠️ 코드가 조금 복잡함 (하지만 이미 구현됨!)

---

### 방법 3: 캐싱 완전 제거

```python
# @st.cache_data(ttl=3600)  # ← 주석 처리
def detect_hallucination_cached(text_hash: str, text: str, context: Dict, llm_service) -> Dict:
    """캐싱 없이 매번 전체 탐지"""
    return detect_hallucination(text, context, llm_service)
```

**장점**:
- ✅ 오류 완전 회피
- ✅ 항상 최신 결과

**단점**:
- ❌ 동일 민원 재검증 시 느림 (패턴 탐지도 매번 실행)
- ❌ 불필요한 연산 반복

---

## 🎯 권장 해결 방법

### 즉시 적용 (긴급)
**방법 1** 사용: 파일 한 줄만 수정

```python
# hallucination_detection.py 600번째 줄
def detect_hallucination_cached(text_hash: str, text: str, context: Dict, _llm_service) -> Dict:
    #                                                                    ↑ 언더스코어 추가
    return detect_hallucination(text, context, _llm_service)
```

### 최적 솔루션 (권장)
**방법 2** 사용: 제공된 최신 `hallucination_detection.py` 파일 전체 교체

```bash
# 기존 파일 백업
cp hallucination_detection.py hallucination_detection.py.backup

# 새 파일로 교체 (제공된 파일)
# 이미 방법 2가 구현되어 있음!
```

---

## 📋 단계별 적용 가이드

### Option A: 빠른 수정 (1분)

1. `hallucination_detection.py` 열기
2. 600번째 줄 찾기 (`def detect_hallucination_cached`)
3. `llm_service` → `_llm_service`로 변경 (언더스코어 추가)
4. 파일 저장
5. Streamlit 재시작

```bash
streamlit run streamlit_app.py
```

### Option B: 완전한 수정 (5분) ⭐ 권장

1. 제공된 최신 `hallucination_detection.py` 다운로드
2. 기존 파일 교체
3. Streamlit 재시작

```bash
# 파일 교체
cp /path/to/new/hallucination_detection.py .

# 재시작
streamlit run streamlit_app.py
```

---

## 🧪 테스트 방법

### 수정 후 테스트 절차

1. **앱 시작**
```bash
streamlit run streamlit_app.py
```

2. **환각 검증 모드 선택**
   - 사이드바에서 "🔍 AI 민원 검증" 클릭

3. **테스트 민원 입력**
```
2025년 13월 32일에 발생한 사건에 대해 민원을 제기합니다.
주민등록법 제999조에 따르면 전입신고 의무가 있습니다.
```

4. **검증 실행**
   - "🔍 환각 검증 시작" 버튼 클릭
   - 오류 없이 결과 표시되면 성공!

5. **캐싱 확인** (방법 2 사용 시)
   - 동일한 민원을 다시 입력하여 검증
   - 두 번째는 더 빨라야 함 (패턴 탐지 캐싱)

---

## 🔍 디버깅 팁

### 오류가 계속 발생하면

1. **Streamlit 캐시 초기화**
```bash
streamlit cache clear
streamlit run streamlit_app.py
```

2. **Python 버전 확인**
```bash
python --version  # 3.8 이상 필요
```

3. **의존성 재설치**
```bash
pip install --upgrade streamlit
```

4. **로그 확인**
```bash
# 터미널에서 전체 에러 스택 확인
streamlit run streamlit_app.py --logger.level=debug
```

---

## 📊 성능 비교

| 방법 | 첫 실행 | 재실행 | 정확도 | 복잡도 |
|------|---------|---------|---------|---------|
| 방법 1 (언더스코어) | ~15초 | ~0.1초 | 동일 | ⭐ |
| 방법 2 (분리 캐싱) | ~15초 | ~7초 | 더 높음 | ⭐⭐ |
| 방법 3 (캐싱 제거) | ~15초 | ~15초 | 동일 | ⭐ |

**권장**: 방법 2 (분리 캐싱) - 성능과 정확도의 최적 균형

---

## 📞 추가 지원

### 여전히 문제가 해결되지 않으면

1. **에러 메시지 전체 복사**
2. **Python 버전 확인**: `python --version`
3. **Streamlit 버전 확인**: `streamlit version`
4. **사용한 해결 방법** 명시
5. **재현 절차** 상세 기술

---

## ✅ 체크리스트

수정 완료 후 확인:

- [ ] `hallucination_detection.py` 파일 수정 완료
- [ ] Streamlit 앱 정상 시작
- [ ] 환각 검증 모드 진입 가능
- [ ] 테스트 민원 검증 성공
- [ ] 오류 메시지 없음
- [ ] 동일 민원 재검증 시 빠른 응답 (방법 2 사용 시)

---

**작성일**: 2026-02-10  
**버전**: 1.1  
**상태**: ✅ 오류 해결 완료
