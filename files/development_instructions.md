# AI 행정관 Pro - 환각 탐지 및 업무 효율화 기능 추가 개발 지시사항

## 📋 프로젝트 개요

**목적**: 생성형 AI로 작성된 민원에 포함된 환각(허위 정보)을 탐지하고, 공무원의 업무 처리를 효율화하는 기능 추가

**대상 파일**: `streamlit_app.py`

**핵심 요구사항**:
1. AI 생성 민원 내 환각(거짓 정보) 자동 탐지
2. 사실 관계 검증 및 의심 구간 하이라이팅
3. 공무원 업무 효율화 도구 (자동 요약, 우선순위 판단, 체크리스트)

---

## 🎯 구현할 핵심 기능

### 1. AI 환각 탐지 시스템 (Hallucination Detection)

#### 1.1 탐지 메커니즘
```python
# 새로운 함수 추가 위치: HELPERS 섹션 (98번째 줄 이후)

def detect_hallucination(text: str, context: Dict) -> Dict:
    """
    AI 생성 민원의 환각 가능성 탐지
    
    Args:
        text: 민원 원문
        context: 관련 법령, 절차 등의 맥락 정보
    
    Returns:
        {
            "risk_level": "high" | "medium" | "low",
            "suspicious_parts": [
                {
                    "text": "의심 구간 텍스트",
                    "reason": "탐지 이유",
                    "confidence": 0.0~1.0,
                    "line_number": int
                }
            ],
            "verification_needed": ["검증이 필요한 항목들"],
            "overall_score": 0.0~1.0  # 신뢰도 점수
        }
    """
```

#### 1.2 탐지 기준
- **패턴 기반 탐지**:
  - 비현실적인 날짜/시간 (미래 날짜, 존재하지 않는 날짜)
  - 존재하지 않는 법령/조항 참조
  - 수치의 비일관성 (금액, 면적 등)
  - 과도하게 정확한 통계 수치 (AI가 지어낸 가능성)

- **LLM 기반 교차 검증**:
  ```
  프롬프트:
  "다음 민원 내용에서 사실 관계가 의심되는 부분을 찾아라.
  특히 다음을 검증하라:
  1. 법령/조례 인용의 정확성
  2. 날짜/시간의 논리적 일관성
  3. 수치 데이터의 합리성
  4. 행정절차 서술의 정확성
  
  민원 내용: {text}
  관련 법령: {context['law']}
  
  의심되는 부분을 JSON 형식으로 반환하라."
  ```

- **외부 검증**:
  - 법령 DB 실제 조회 (Lawbot API 활용)
  - 날짜 유효성 검증
  - 기관명/직위명 실존 여부

#### 1.3 UI 표시 방법
```python
# 메인 화면에 환각 탐지 결과 표시
def render_hallucination_report(detection_result: Dict):
    """
    환각 탐지 결과를 시각적으로 표시
    """
    risk_colors = {
        "high": "#dc2626",    # 빨강
        "medium": "#f59e0b",  # 주황
        "low": "#10b981"      # 초록
    }
    
    risk_level = detection_result['risk_level']
    color = risk_colors[risk_level]
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {color}22 0%, {color}11 100%);
                padding: 1.5rem; border-radius: 12px; border-left: 4px solid {color};
                margin: 1rem 0;'>
        <h4 style='margin: 0 0 1rem 0; color: {color};'>
            🔍 AI 환각 탐지 결과 (신뢰도: {detection_result['overall_score']*100:.1f}%)
        </h4>
        <p style='color: #374151; font-size: 0.95rem;'>
            위험도: <b>{risk_level.upper()}</b>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 의심 구간 하이라이팅
    if detection_result['suspicious_parts']:
        st.warning("⚠️ 다음 내용은 검증이 필요합니다:")
        for i, part in enumerate(detection_result['suspicious_parts'], 1):
            with st.expander(f"의심 구간 {i}: {part['text'][:50]}..."):
                st.markdown(f"**전체 내용**: {part['text']}")
                st.markdown(f"**탐지 이유**: {part['reason']}")
                st.markdown(f"**신뢰도**: {part['confidence']*100:.1f}%")
                st.markdown(f"**위치**: {part['line_number']}번째 줄")
```

---

### 2. 공무원 업무 효율화 도구

#### 2.1 스마트 민원 분류 및 우선순위

```python
def analyze_petition_priority(petition_text: str, detection_result: Dict) -> Dict:
    """
    민원 긴급도 및 처리 우선순위 자동 판단
    
    Returns:
        {
            "priority": "urgent" | "high" | "normal" | "low",
            "estimated_workload": "간편" | "보통" | "복잡",
            "recommended_deadline": "YYYY-MM-DD",
            "required_departments": ["부서1", "부서2"],
            "auto_tags": ["태그1", "태그2"]
        }
    """
    prompt = f"""
    다음 민원의 처리 우선순위를 분석하라:
    
    민원 내용: {petition_text}
    환각 위험도: {detection_result['risk_level']}
    
    다음 기준으로 판단:
    1. 긴급성 (법정 기한, 인명 관련 등)
    2. 업무 복잡도 (관련 부서 수, 필요 절차)
    3. 민원인 권리 침해 정도
    4. 환각 위험도 (높으면 검증 시간 추가 필요)
    
    JSON 형식으로 반환:
    {{
        "priority": "urgent/high/normal/low",
        "estimated_workload": "간편/보통/복잡",
        "recommended_deadline": "날짜",
        "required_departments": [],
        "auto_tags": [],
        "reasoning": "판단 근거"
    }}
    """
    
    result = llm_service.generate_text(prompt)
    return _safe_json_loads(result)
```

#### 2.2 자동 처리 체크리스트 생성

```python
def generate_processing_checklist(analysis_result: Dict) -> List[Dict]:
    """
    케이스 분석 결과를 바탕으로 단계별 체크리스트 생성
    
    Returns:
        [
            {
                "step": 1,
                "title": "민원 내용 검증",
                "items": [
                    {"task": "환각 의심 구간 사실 관계 확인", "completed": False},
                    {"task": "첨부 서류 진위 확인", "completed": False}
                ],
                "deadline": "접수 후 1일 이내"
            },
            ...
        ]
    """
```

#### 2.3 원클릭 회신문 초안 생성

```python
def generate_response_draft(petition_text: str, analysis: Dict, 
                            response_type: str = "approval") -> str:
    """
    민원 회신문 자동 초안 생성
    
    Args:
        response_type: "approval" | "rejection" | "partial" | "request_info"
    
    Returns:
        회신문 텍스트 (공문서 형식)
    """
    prompt = f"""
    다음 민원에 대한 {response_type} 회신문을 작성하라:
    
    민원 내용: {petition_text}
    케이스 분석: {json.dumps(analysis, ensure_ascii=False)}
    
    회신문 작성 규칙:
    1. 행정안전부 공문서 작성 기준 준수
    2. 법적 근거 명시
    3. 처리 결과 및 사유 명확히 기술
    4. 민원인 권리 구제 방법 안내
    5. 담당자 연락처 포함
    
    다음 형식으로 작성:
    - 제목
    - 수신자
    - 발신자
    - 본문 (문단별)
    - 첨부 (필요시)
    """
    
    return llm_service.generate_text(prompt)
```

---

### 3. 통합 워크플로우 (Main Function 수정)

#### 3.1 새로운 앱 모드 추가
```python
# line 283-294 부분 수정
app_mode = st.sidebar.radio(
    "🎯 기능 선택",
    options=[
        "main",
        "admin",
        "revision",
        "duty_manual",
        "hallucination_check"  # ← 새로운 모드 추가
    ],
    format_func=lambda x: {
        "main": "📋 케이스 분석 (메인)",
        "admin": "👤 관리자 대시보드",
        "revision": "✏️ 기안문 수정",
        "duty_manual": "📚 업무 매뉴얼",
        "hallucination_check": "🔍 AI 민원 검증"  # ← 새 메뉴
    }.get(x, x),
    key="app_mode_radio"
)
```

#### 3.2 환각 검증 모드 구현
```python
# line 3800 이후에 새로운 섹션 추가

elif st.session_state.app_mode == "hallucination_check":
    st.title("🔍 AI 생성 민원 검증 시스템")
    
    st.markdown("""
    ### 이 도구는 무엇을 하나요?
    - ✅ AI로 작성된 민원의 환각(허위 정보) 자동 탐지
    - ✅ 사실 관계 검증 및 의심 구간 표시
    - ✅ 처리 우선순위 자동 판단
    - ✅ 업무 체크리스트 자동 생성
    """)
    
    st.divider()
    
    # 민원 텍스트 입력
    petition_input = st.text_area(
        "📝 검증할 민원 내용을 입력하세요",
        height=300,
        placeholder="민원 전문을 붙여넣으세요..."
    )
    
    # 파일 업로드 옵션
    uploaded_file = st.file_uploader(
        "또는 민원 파일 업로드 (TXT, DOCX, PDF)",
        type=['txt', 'docx', 'pdf']
    )
    
    if uploaded_file:
        # 파일 파싱 로직 (기존 코드 재사용)
        petition_input = parse_uploaded_file(uploaded_file)
        st.text_area("파일 내용", petition_input, height=200, disabled=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        verify_btn = st.button("🔍 환각 검증 시작", type="primary", use_container_width=True)
    with col2:
        st.markdown("**예상 소요**: ~30초")
    
    if verify_btn and petition_input:
        with st.spinner("🤖 AI 환각 탐지 중..."):
            # Step 1: 환각 탐지
            detection_result = detect_hallucination(petition_input, {})
            
            # Step 2: 우선순위 분석
            priority_analysis = analyze_petition_priority(petition_input, detection_result)
            
            # Step 3: 체크리스트 생성
            checklist = generate_processing_checklist({
                "petition": petition_input,
                "detection": detection_result,
                "priority": priority_analysis
            })
        
        # 결과 표시
        render_hallucination_report(detection_result)
        
        st.divider()
        
        # 우선순위 정보
        st.subheader("📊 처리 우선순위 분석")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("긴급도", priority_analysis['priority'].upper())
        with col2:
            st.metric("업무 복잡도", priority_analysis['estimated_workload'])
        with col3:
            st.metric("권장 처리기한", priority_analysis['recommended_deadline'])
        
        st.markdown(f"**관련 부서**: {', '.join(priority_analysis['required_departments'])}")
        st.markdown(f"**자동 태그**: {', '.join(priority_analysis['auto_tags'])}")
        
        with st.expander("📝 판단 근거 보기"):
            st.write(priority_analysis.get('reasoning', ''))
        
        st.divider()
        
        # 처리 체크리스트
        st.subheader("✅ 업무 처리 체크리스트")
        for step in checklist:
            with st.expander(f"Step {step['step']}: {step['title']} (기한: {step['deadline']})", 
                           expanded=True):
                for item in step['items']:
                    checked = st.checkbox(
                        item['task'],
                        value=item['completed'],
                        key=f"check_{step['step']}_{item['task'][:20]}"
                    )
        
        st.divider()
        
        # 회신문 초안 생성
        st.subheader("📄 회신문 자동 초안")
        response_type = st.selectbox(
            "회신 유형 선택",
            ["approval", "rejection", "partial", "request_info"],
            format_func=lambda x: {
                "approval": "승인/수용",
                "rejection": "불가/거부",
                "partial": "부분 수용",
                "request_info": "보완 요청"
            }[x]
        )
        
        if st.button("📝 회신문 초안 생성", use_container_width=True):
            with st.spinner("회신문 작성 중..."):
                draft = generate_response_draft(
                    petition_input,
                    {"detection": detection_result, "priority": priority_analysis},
                    response_type
                )
            
            st.text_area("생성된 회신문 초안", draft, height=400)
            
            # DOCX 다운로드
            from datetime import datetime
            today_str = datetime.now().strftime("%Y%m%d")
            
            # 회신문을 공문서 형식으로 변환하여 DOCX 생성
            doc_data = {
                "title": f"{response_type.upper()} 회신",
                "body_paragraphs": draft.split('\n\n')
            }
            docx_bytes = generate_official_docx(doc_data)
            
            st.download_button(
                "📥 회신문 DOCX 다운로드",
                docx_bytes,
                f"회신문_{today_str}.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )
```

---

### 4. 메인 케이스 분석에 환각 탐지 통합

#### 4.1 기존 분석 플로우에 환각 검증 단계 추가
```python
# line 3200 근처, 케이스 분석 후 결과 표시 전에 추가

# 환각 탐지 실행 (백그라운드)
with st.spinner("🔍 AI 환각 검증 중..."):
    hallucination_check = detect_hallucination(
        situation,
        {
            "law": res.get("law", ""),
            "procedure": res.get("procedure", {}),
            "analysis": res.get("analysis", {})
        }
    )

# 환각 위험이 있으면 경고 표시
if hallucination_check['risk_level'] in ['high', 'medium']:
    st.warning(f"""
    ⚠️ **AI 환각 위험 감지**: 이 민원은 AI로 생성되었을 가능성이 있으며, 
    일부 내용의 사실 관계 검증이 필요합니다. (위험도: {hallucination_check['risk_level']})
    """)
    
    with st.expander("🔍 환각 탐지 상세 결과 보기"):
        render_hallucination_report(hallucination_check)
```

---

## 🔧 구현 세부사항

### 선택적 의존성 처리 (Best Practice)

```python
# streamlit_app.py 상단에 추가
# Streamlit Cloud 등 일부 배포 환경에서는 모듈이 없을 수 있으므로
# 선택적 의존성으로 처리하여 앱 부팅을 보장합니다.

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
except ImportError as e:
    import warnings
    warnings.warn(f"Hallucination detection module not available: {e}")
    HALLUCINATION_DETECTION_AVAILABLE = False
    
    # Fallback 함수들 정의 (기본 동작 보장)
    def detect_hallucination(*args, **kwargs):
        return {
            "risk_level": "unknown",
            "suspicious_parts": [],
            "verification_needed": [],
            "overall_score": 0.5,
            "total_issues_found": 0
        }
    
    def detect_hallucination_cached(*args, **kwargs):
        return detect_hallucination(*args, **kwargs)
    
    def get_text_hash(text: str) -> str:
        import hashlib
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def analyze_petition_priority(*args, **kwargs):
        from datetime import datetime, timedelta
        return {
            "priority": "normal",
            "estimated_workload": "보통",
            "recommended_deadline": (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
            "required_departments": ["담당부서"],
            "auto_tags": [],
            "reasoning": "환각 탐지 모듈 미사용"
        }
    
    def generate_processing_checklist(*args, **kwargs):
        return []
    
    def generate_response_draft(*args, **kwargs):
        return "환각 탐지 모듈이 로드되지 않았습니다."
    
    def render_hallucination_report(detection_result: Dict):
        import streamlit as st
        st.info("💡 환각 탐지 기능이 현재 환경에서 비활성화되어 있습니다.")
```

**장점**:
- ✅ 모듈 누락 시에도 앱이 정상 부팅
- ✅ 사용자에게 명확한 안내 메시지
- ✅ 기본 동작 보장 (Graceful Degradation)
- ✅ govable_ai.ui.premium_animations과 동일한 패턴

### 필수 패키지 추가
```python
# requirements.txt 또는 pyproject.toml에 추가
dateparser>=1.1.0  # 날짜 파싱 및 검증
fuzzywuzzy>=0.18.0  # 텍스트 유사도 검사
python-Levenshtein>=0.12.0  # 문자열 거리 계산
```

### 환경 변수 추가 (필요시)
```bash
# .env 또는 secrets.toml
LAWBOT_API_KEY=your_lawbot_api_key  # 법령 검증용 (있다면)
FACT_CHECK_ENDPOINT=https://...     # 외부 팩트체크 API (선택)
```

### 데이터베이스 스키마 추가
```sql
-- 환각 탐지 결과 저장용 테이블
CREATE TABLE hallucination_detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    archive_id UUID REFERENCES main_archive(id),
    petition_text TEXT NOT NULL,
    risk_level TEXT CHECK (risk_level IN ('low', 'medium', 'high')),
    suspicious_parts JSONB,
    overall_score FLOAT,
    detected_at TIMESTAMP DEFAULT NOW(),
    verified_by TEXT,  -- 담당자가 수동 검증 시
    verification_result TEXT  -- 검증 결과
);

-- 처리 체크리스트 저장용
CREATE TABLE processing_checklists (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    archive_id UUID REFERENCES main_archive(id),
    checklist_data JSONB NOT NULL,
    completion_status JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

---

## 📊 성능 최적화 전략

### 1. 캐싱 활용
```python
# 동일한 민원에 대한 중복 검증 방지
@st.cache_data(ttl=3600)
def detect_hallucination_cached(text_hash: str, text: str, context: Dict) -> Dict:
    return detect_hallucination(text, context)

# 사용
text_hash = hashlib.sha256(petition_text.encode()).hexdigest()
result = detect_hallucination_cached(text_hash, petition_text, context)
```

### 2. 병렬 처리
```python
import concurrent.futures

def analyze_with_parallel_tasks(petition_text: str):
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # 환각 탐지, 우선순위 분석, 법령 검색 동시 실행
        future_hallucination = executor.submit(detect_hallucination, petition_text, {})
        future_priority = executor.submit(analyze_petition_priority, petition_text, {})
        future_law = executor.submit(search_law_references, petition_text)
        
        hallucination_result = future_hallucination.result()
        priority_result = future_priority.result()
        law_result = future_law.result()
    
    return hallucination_result, priority_result, law_result
```

### 3. 프롬프트 최적화
- 토큰 절약을 위해 긴 컨텍스트는 요약 후 전달
- 구조화된 출력 요청 (JSON 스키마 명시)

---

## 🧪 테스트 시나리오

### 0. 캐싱 오류 해결 (중요!) ⚠️

**문제**: `llm_service` 객체가 pickle 불가능하여 `@st.cache_data` 오류 발생

**해결책**: 제공된 `hallucination_detection.py`는 이미 해결됨
- 패턴 기반 탐지만 캐싱 (llm_service 불필요)
- LLM 기반 탐지는 매번 실행 (변동성 활용)

**대안** (간단한 수정):
```python
# llm_service 앞에 언더스코어 추가
def detect_hallucination_cached(text_hash: str, text: str, context: Dict, _llm_service):
    return detect_hallucination(text, context, _llm_service)
```

자세한 내용은 `ERROR_FIX_GUIDE.md` 참조

### 1. 환각 탐지 정확도 테스트
```python
# 테스트용 민원 샘플
test_cases = [
    {
        "text": "2025년 13월 32일에 발생한 사건...",  # 존재하지 않는 날짜
        "expected_risk": "high"
    },
    {
        "text": "주민등록법 제999조에 따르면...",  # 존재하지 않는 조항
        "expected_risk": "high"
    },
    {
        "text": "통계청 자료에 따르면 정확히 47.3829%가...",  # 과도하게 정확한 통계
        "expected_risk": "medium"
    }
]

def run_hallucination_tests():
    for case in test_cases:
        result = detect_hallucination(case['text'], {})
        assert result['risk_level'] == case['expected_risk'], \
            f"Expected {case['expected_risk']}, got {result['risk_level']}"
```

### 2. 엣지 케이스 처리
- 빈 문자열 입력
- 극도로 긴 민원 (10,000자 이상)
- 특수 문자 및 이모지 포함
- 여러 언어 혼용

---

## 📝 사용자 매뉴얼 (앱 내 도움말)

```python
# 환각 검증 모드 상단에 추가
with st.expander("❓ 사용 방법 및 주의사항"):
    st.markdown("""
    ### 🎯 이 기능은 언제 사용하나요?
    - 민원 내용이 지나치게 전문적이거나 상세할 때
    - 인터넷에서 복사한 듯한 정형화된 문장이 많을 때
    - ChatGPT 등 AI로 작성된 것으로 의심될 때
    
    ### 🔍 무엇을 검증하나요?
    1. **날짜/시간의 논리적 타당성**
    2. **법령/조례 인용의 실존 여부**
    3. **수치 데이터의 일관성**
    4. **행정 절차 서술의 정확성**
    
    ### ⚠️ 주의사항
    - 이 도구는 **보조 수단**입니다. 최종 판단은 담당자가 해야 합니다.
    - "환각 위험 높음"이라고 해서 반드시 허위는 아닙니다.
    - 중요한 사안은 반드시 원본 서류 및 관련 법령을 직접 확인하세요.
    
    ### 💡 결과 해석
    - **위험도 낮음**: 일반적인 민원, 정상 처리
    - **위험도 중간**: 일부 검증 권장, 의심 구간 확인
    - **위험도 높음**: 필수 검증 대상, 담당자 면담 권장
    """)
```

---

## 🚀 배포 체크리스트

### 개발 단계
- [ ] `detect_hallucination()` 함수 구현
- [ ] `analyze_petition_priority()` 함수 구현
- [ ] `generate_processing_checklist()` 함수 구현
- [ ] `generate_response_draft()` 함수 구현
- [ ] `render_hallucination_report()` UI 구현
- [ ] 새로운 앱 모드 `hallucination_check` 추가
- [ ] 메인 케이스 분석에 환각 검증 통합

### 테스트 단계
- [ ] 단위 테스트 작성 및 실행
- [ ] 통합 테스트 (전체 워크플로우)
- [ ] 성능 테스트 (응답 시간 < 30초)
- [ ] 엣지 케이스 처리 확인
- [ ] UI/UX 사용성 테스트

### 배포 단계
- [ ] 데이터베이스 마이그레이션 실행
- [ ] 환경 변수 설정 확인
- [ ] 스테이징 환경 배포 및 검증
- [ ] 프로덕션 배포
- [ ] 모니터링 설정 (에러율, 응답 시간)
- [ ] 사용자 피드백 수집 채널 오픈

---

## 📞 지원 및 문의

**개발 담당**: [팀명/담당자명]  
**기술 지원**: [이메일/슬랙 채널]  
**긴급 연락**: [전화번호]

---

## 📚 참고 자료

1. **AI 환각 탐지 논문**:
   - "Detecting Hallucinations in Large Language Models" (2023)
   - "Fact-Checking with LLMs: A Survey" (2024)

2. **행정 실무 가이드**:
   - 행정안전부 민원처리 매뉴얼
   - 공문서 작성 실무 가이드

3. **관련 법령**:
   - 민원처리에 관한 법률
   - 행정절차법

---

**작성일**: 2026-02-10  
**버전**: 1.0.0  
**문서 상태**: 개발 준비 완료
