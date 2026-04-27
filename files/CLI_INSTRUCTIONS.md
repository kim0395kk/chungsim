# 🚀 AI 행정관 Pro - 환각 탐지 기능 추가 개발 지시사항
## CLI/안티그래비티 실행용

---

## 📌 빠른 시작 (Quick Start)

### 1단계: 파일 준비
```bash
# 기존 프로젝트 디렉토리로 이동
cd /path/to/govable-ai

# 새로운 모듈 파일 생성
touch hallucination_detection.py
```

### 2단계: 모듈 코드 복사
- `hallucination_detection.py` 파일에 제공된 전체 코드를 복사
- 이 파일은 환각 탐지 핵심 로직을 포함

### 3단계: 메인 파일 수정
`streamlit_app.py`를 다음과 같이 수정합니다.

---

## 📝 streamlit_app.py 수정 사항

### A. 상단 임포트 추가 (20번째 줄 근처)

**중요**: 선택적 의존성으로 처리하여 배포 환경에서도 안정적으로 작동하도록 합니다.

```python
# 기존 임포트들 아래에 추가
# Optional hallucination detection module
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
    print(f"Warning: hallucination_detection module not available: {e}")
    HALLUCINATION_DETECTION_AVAILABLE = False
    # Fallback 함수들 정의
    def detect_hallucination(*args, **kwargs):
        return {"risk_level": "unknown", "suspicious_parts": [], "verification_needed": [], "overall_score": 0.5, "total_issues_found": 0}
    def detect_hallucination_cached(*args, **kwargs):
        return detect_hallucination(*args, **kwargs)
    def get_text_hash(text):
        import hashlib
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    def analyze_petition_priority(*args, **kwargs):
        return {"priority": "normal", "estimated_workload": "보통", "recommended_deadline": "", "required_departments": ["담당부서"], "auto_tags": [], "reasoning": "모듈 미사용"}
    def generate_processing_checklist(*args, **kwargs):
        return []
    def generate_response_draft(*args, **kwargs):
        return "환각 탐지 모듈이 로드되지 않았습니다."
    def render_hallucination_report(detection_result):
        st.info("환각 탐지 기능이 현재 환경에서 비활성화되어 있습니다.")

from datetime import timedelta  # 이미 있으면 스킵
```

### B. 앱 모드 선택 수정 (283번째 줄 근처)

**기존 코드**:
```python
app_mode = st.sidebar.radio(
    "🎯 기능 선택",
    options=[
        "main",
        "admin",
        "revision",
        "duty_manual"
    ],
    format_func=lambda x: {
        "main": "📋 케이스 분석 (메인)",
        "admin": "👤 관리자 대시보드",
        "revision": "✏️ 기안문 수정",
        "duty_manual": "📚 업무 매뉴얼"
    }.get(x, x),
    key="app_mode_radio"
)
```

**수정 후**:
```python
app_mode = st.sidebar.radio(
    "🎯 기능 선택",
    options=[
        "main",
        "admin",
        "revision",
        "duty_manual",
        "hallucination_check"  # ← 추가
    ],
    format_func=lambda x: {
        "main": "📋 케이스 분석 (메인)",
        "admin": "👤 관리자 대시보드",
        "revision": "✏️ 기안문 수정",
        "duty_manual": "📚 업무 매뉴얼",
        "hallucination_check": "🔍 AI 민원 검증"  # ← 추가
    }.get(x, x),
    key="app_mode_radio"
)
```

### C. 메인 케이스 분석에 환각 검증 통합 (3200번째 줄 근처)

**케이스 분석 결과 표시 직후**에 다음 코드 추가:

```python
# 기존: st.success("✅ 케이스 분석 완료!") 다음에 추가

# === 환각 탐지 실행 (모듈이 사용 가능한 경우에만) ===
if HALLUCINATION_DETECTION_AVAILABLE:
    st.divider()
    st.subheader("🔍 AI 환각 검증")

    with st.spinner("AI 환각 검증 중..."):
        # 텍스트 해시 생성 (캐싱용)
        situation_hash = get_text_hash(situation)
        
        # 환각 탐지 (캐싱 적용)
        hallucination_check = detect_hallucination_cached(
            situation_hash,
            situation,
            {
                "law": res.get("law", ""),
                "procedure": res.get("procedure", {}),
                "analysis": res.get("analysis", {})
            },
            llm_service
        )

    # 결과 표시
    render_hallucination_report(hallucination_check)

    # 위험도에 따른 추가 안내
    if hallucination_check['risk_level'] == 'high':
        st.error("""
        ⚠️ **높은 환각 위험 감지**
        
        이 민원은 AI로 작성되었을 가능성이 높으며, 허위 정보가 포함되어 있을 수 있습니다.
        
        **필수 조치**:
        1. 모든 사실 관계를 원본 서류로 재확인
        2. 법령 참조는 법제처 사이트에서 직접 조회
        3. 민원인과 직접 통화 또는 면담 권장
        """)
    elif hallucination_check['risk_level'] == 'medium':
        st.warning("""
        ⚡ **중간 수준 환각 위험**
        
        일부 내용에 대한 검증이 필요합니다. 의심 구간을 확인하세요.
        """)
    else:
        st.success("""
        ✅ **환각 위험 낮음**
        
        민원 내용이 비교적 신뢰할 수 있습니다. 일반적인 절차대로 진행하세요.
        """)

    # 검증 항목이 있으면 표시
    verification_needed = hallucination_check.get('verification_needed', [])
    if verification_needed:
        with st.expander("📋 검증 체크리스트", expanded=True):
            for i, item in enumerate(verification_needed, 1):
                st.checkbox(f"{item}", key=f"verify_{i}")
else:
    # 모듈이 없는 경우 안내만 표시
    st.info("💡 AI 환각 탐지 기능을 사용하려면 hallucination_detection.py 모듈을 설치하세요.")
```

### D. 새로운 환각 검증 모드 추가 (3800번째 줄 이후)

**기존 `duty_manual` 모드 다음**에 추가:

```python
# =========================================================
# 환각 검증 모드
# =========================================================
elif st.session_state.app_mode == "hallucination_check":
    # 모듈 사용 가능 여부 체크
    if not HALLUCINATION_DETECTION_AVAILABLE:
        st.error("""
        ❌ **환각 탐지 모듈을 사용할 수 없습니다**
        
        `hallucination_detection.py` 파일이 누락되었거나 로드에 실패했습니다.
        
        **해결 방법**:
        1. `hallucination_detection.py` 파일이 `streamlit_app.py`와 같은 디렉토리에 있는지 확인
        2. 파일 권한 확인 (`chmod 644 hallucination_detection.py`)
        3. 오류 메시지 확인 후 재배포
        
        **임시 조치**: 메인 케이스 분석 모드를 사용하세요.
        """)
        
        if st.button("📋 메인 모드로 이동", type="primary"):
            st.session_state.app_mode = "main"
            st.rerun()
        
        st.stop()  # 여기서 실행 중단
    
    st.title("🔍 AI 생성 민원 검증 시스템")
    
    # 사용 안내
    st.markdown("""
    ### 🎯 이 기능은 무엇을 하나요?
    
    생성형 AI(ChatGPT, Claude 등)로 작성된 민원에 포함될 수 있는 **환각(허위 정보)**을 자동으로 탐지합니다.
    
    **주요 기능**:
    - ✅ 날짜/시간의 논리적 타당성 검증
    - ✅ 법령/조례 인용의 실존 여부 확인
    - ✅ 수치 데이터 일관성 검사
    - ✅ 행정 절차 서술의 정확성 평가
    - ✅ 처리 우선순위 자동 판단
    - ✅ 업무 체크리스트 자동 생성
    """)
    
    with st.expander("❓ 사용 방법 및 주의사항"):
        st.markdown("""
        ### 📖 사용 방법
        1. 아래에 검증할 민원 내용을 붙여넣기
        2. 또는 파일 업로드 (TXT, DOCX, PDF)
        3. "🔍 환각 검증 시작" 버튼 클릭
        4. 결과 확인 및 의심 구간 검토
        
        ### ⚠️ 주의사항
        - 이 도구는 **보조 수단**입니다. 최종 판단은 담당자가 해야 합니다.
        - "환각 위험 높음"이라고 해서 반드시 허위는 아닙니다.
        - 중요한 사안은 반드시 원본 서류 및 관련 법령을 직접 확인하세요.
        
        ### 💡 결과 해석
        - **위험도 낮음 (✅)**: 일반적인 민원, 정상 처리
        - **위험도 중간 (⚡)**: 일부 검증 권장, 의심 구간 확인
        - **위험도 높음 (⚠️)**: 필수 검증 대상, 담당자 면담 권장
        """)
    
    st.divider()
    
    # 입력 섹션
    col1, col2 = st.columns([2, 1])
    
    with col1:
        petition_input = st.text_area(
            "📝 검증할 민원 내용을 입력하세요",
            height=300,
            placeholder="""예시:
2024년 13월 32일에 ○○구청에서...
주민등록법 제999조에 따르면...
통계청 자료에 따르면 정확히 47.3829%가..."""
        )
    
    with col2:
        uploaded_file = st.file_uploader(
            "또는 파일 업로드",
            type=['txt', 'docx', 'pdf'],
            help="민원 문서를 업로드하세요"
        )
        
        if uploaded_file:
            try:
                import io
                if uploaded_file.type == "text/plain":
                    petition_input = uploaded_file.read().decode('utf-8')
                elif uploaded_file.type == "application/pdf":
                    # PDF 파싱 (기존 코드 활용)
                    st.info("PDF 파일 파싱 중...")
                    # TODO: PDF 파싱 로직 추가
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    # DOCX 파싱
                    st.info("DOCX 파일 파싱 중...")
                    # TODO: DOCX 파싱 로직 추가
                
                st.success("파일 업로드 완료!")
            except Exception as e:
                st.error(f"파일 읽기 오류: {e}")
    
    # 검증 실행
    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        verify_btn = st.button(
            "🔍 환각 검증 시작", 
            type="primary", 
            use_container_width=True,
            disabled=not petition_input
        )
    with col_btn2:
        if petition_input:
            st.caption(f"📏 {len(petition_input)}자")
    
    if verify_btn and petition_input:
        # 진행 상황 표시
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        try:
            # Step 1: 환각 탐지 (40%)
            progress_text.text("🔍 환각 패턴 탐지 중...")
            progress_bar.progress(20)
            
            text_hash = get_text_hash(petition_input)
            detection_result = detect_hallucination_cached(
                text_hash,
                petition_input,
                {},
                llm_service
            )
            progress_bar.progress(40)
            
            # Step 2: 우선순위 분석 (70%)
            progress_text.text("📊 우선순위 분석 중...")
            priority_analysis = analyze_petition_priority(
                petition_input, 
                detection_result,
                llm_service
            )
            progress_bar.progress(70)
            
            # Step 3: 체크리스트 생성 (100%)
            progress_text.text("✅ 체크리스트 생성 중...")
            checklist = generate_processing_checklist(
                {
                    "petition": petition_input,
                    "detection": detection_result,
                    "priority": priority_analysis
                },
                llm_service
            )
            progress_bar.progress(100)
            
            # 완료
            progress_text.empty()
            progress_bar.empty()
            
            st.success("✅ 검증 완료!")
            
            # === 결과 표시 ===
            st.divider()
            
            # 1. 환각 탐지 결과
            st.subheader("🔍 환각 탐지 결과")
            render_hallucination_report(detection_result)
            
            st.divider()
            
            # 2. 우선순위 정보
            st.subheader("📊 처리 우선순위 분석")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                priority_colors = {
                    "urgent": "🔴",
                    "high": "🟠",
                    "normal": "🟡",
                    "low": "🟢"
                }
                priority = priority_analysis.get('priority', 'normal')
                st.metric(
                    "긴급도", 
                    f"{priority_colors.get(priority, '⚪')} {priority.upper()}"
                )
            
            with col2:
                st.metric(
                    "업무 복잡도", 
                    priority_analysis.get('estimated_workload', '보통')
                )
            
            with col3:
                deadline = priority_analysis.get('recommended_deadline', '')
                st.metric(
                    "권장 처리기한", 
                    deadline
                )
            
            with col4:
                dept_count = len(priority_analysis.get('required_departments', []))
                st.metric(
                    "관련 부서", 
                    f"{dept_count}개"
                )
            
            # 상세 정보
            col_detail1, col_detail2 = st.columns(2)
            
            with col_detail1:
                st.markdown("**📋 관련 부서**")
                departments = priority_analysis.get('required_departments', ['담당부서'])
                st.write(", ".join(departments))
            
            with col_detail2:
                st.markdown("**🏷️ 자동 태그**")
                tags = priority_analysis.get('auto_tags', [])
                if tags:
                    tag_html = " ".join([f"<span style='background: #e5e7eb; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.85rem; margin-right: 0.25rem;'>{tag}</span>" for tag in tags])
                    st.markdown(tag_html, unsafe_allow_html=True)
                else:
                    st.caption("태그 없음")
            
            with st.expander("📝 우선순위 판단 근거"):
                reasoning = priority_analysis.get('reasoning', '분석 중...')
                st.write(reasoning)
            
            st.divider()
            
            # 3. 처리 체크리스트
            st.subheader("✅ 업무 처리 체크리스트")
            
            for step_data in checklist:
                step_num = step_data.get('step', 0)
                step_title = step_data.get('title', '단계')
                step_deadline = step_data.get('deadline', '')
                items = step_data.get('items', [])
                
                with st.expander(
                    f"**Step {step_num}: {step_title}** (기한: {step_deadline})", 
                    expanded=(step_num == 1)
                ):
                    for i, item in enumerate(items):
                        task_text = item.get('task', '')
                        completed = item.get('completed', False)
                        
                        checked = st.checkbox(
                            task_text,
                            value=completed,
                            key=f"check_{step_num}_{i}_{get_text_hash(task_text)[:8]}"
                        )
            
            st.divider()
            
            # 4. 회신문 자동 초안
            st.subheader("📄 회신문 자동 초안 생성")
            
            col_response1, col_response2 = st.columns([2, 1])
            
            with col_response1:
                response_type = st.selectbox(
                    "회신 유형 선택",
                    ["approval", "rejection", "partial", "request_info"],
                    format_func=lambda x: {
                        "approval": "✅ 승인/수용",
                        "rejection": "❌ 불가/거부",
                        "partial": "⚖️ 부분 수용",
                        "request_info": "📝 보완 요청"
                    }[x],
                    key="response_type_select"
                )
            
            with col_response2:
                generate_draft_btn = st.button(
                    "📝 초안 생성",
                    use_container_width=True,
                    type="secondary"
                )
            
            if generate_draft_btn or st.session_state.get('response_draft'):
                if generate_draft_btn:
                    with st.spinner("회신문 작성 중... (약 10초 소요)"):
                        draft = generate_response_draft(
                            petition_input,
                            {
                                "detection": detection_result,
                                "priority": priority_analysis
                            },
                            response_type,
                            llm_service
                        )
                        st.session_state.response_draft = draft
                else:
                    draft = st.session_state.response_draft
                
                st.text_area(
                    "생성된 회신문 초안 (수정 가능)",
                    draft,
                    height=400,
                    key="draft_editor"
                )
                
                # DOCX 다운로드
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    try:
                        from datetime import datetime
                        today_str = datetime.now().strftime("%Y%m%d")
                        
                        # 회신문을 공문서 형식으로 변환
                        doc_data = {
                            "title": f"{response_type.upper()} 회신",
                            "body_paragraphs": draft.split('\n\n')
                        }
                        
                        docx_bytes = generate_official_docx(doc_data)
                        
                        st.download_button(
                            "📥 회신문 DOCX 다운로드",
                            docx_bytes,
                            f"회신문_{response_type}_{today_str}.docx",
                            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"DOCX 생성 오류: {e}")
                
                with col_dl2:
                    # 텍스트 복사
                    if st.button("📋 텍스트 복사", use_container_width=True):
                        st.code(draft, language=None)
                        st.info("👆 위 텍스트를 복사하세요")
        
        except Exception as e:
            st.error(f"❌ 검증 중 오류 발생: {e}")
            import traceback
            with st.expander("🔧 상세 오류 정보"):
                st.code(traceback.format_exc())

```

---

## 🔧 추가 수정 (선택사항)

### 1. 사이드바에 환각 검증 바로가기 추가 (350번째 줄 근처)

```python
# 사이드바 하단에 추가
st.sidebar.divider()
st.sidebar.markdown("### 🚀 빠른 기능")

if st.sidebar.button("🔍 AI 민원 검증", use_container_width=True):
    st.session_state.app_mode = "hallucination_check"
    st.rerun()
```

### 2. 데이터베이스 테이블 추가 (선택)

관리자 모드에서 다음 SQL 실행:

```sql
-- 환각 탐지 결과 저장
CREATE TABLE IF NOT EXISTS hallucination_detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    archive_id UUID REFERENCES main_archive(id),
    petition_text TEXT NOT NULL,
    risk_level TEXT CHECK (risk_level IN ('low', 'medium', 'high')),
    suspicious_parts JSONB,
    overall_score FLOAT,
    detected_at TIMESTAMP DEFAULT NOW(),
    verified_by TEXT,
    verification_result TEXT
);

-- 처리 체크리스트 저장
CREATE TABLE IF NOT EXISTS processing_checklists (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    archive_id UUID REFERENCES main_archive(id),
    checklist_data JSONB NOT NULL,
    completion_status JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 인덱스 생성
CREATE INDEX idx_hallucination_archive ON hallucination_detections(archive_id);
CREATE INDEX idx_hallucination_risk ON hallucination_detections(risk_level);
CREATE INDEX idx_checklist_archive ON processing_checklists(archive_id);
```

---

## ✅ 배포 체크리스트

### 개발 단계
- [ ] `hallucination_detection.py` 파일 생성
- [ ] `streamlit_app.py` 임포트 추가
- [ ] 앱 모드 선택 수정
- [ ] 메인 케이스 분석에 환각 검증 통합
- [ ] 새로운 환각 검증 모드 추가
- [ ] 로컬 환경에서 테스트

### 테스트 시나리오
```python
# 테스트 민원 샘플
test_petitions = [
    # 1. 날짜 오류
    "2025년 13월 32일에 발생한 사건에 대해...",
    
    # 2. 법령 오류
    "주민등록법 제999조에 따르면...",
    
    # 3. 과도한 통계
    "통계청 자료에 따르면 정확히 47.3829472%가...",
    
    # 4. 정상 민원
    "2024년 12월 15일에 ○○구청에 민원을 제기합니다..."
]

# 각 샘플로 테스트 실행
```

### 배포 전 확인
- [ ] 모든 기능 정상 작동
- [ ] 에러 핸들링 확인
- [ ] 응답 시간 < 30초
- [ ] UI 깨짐 없음
- [ ] 모바일 반응형 확인

### 배포
```bash
# Git 커밋
git add hallucination_detection.py streamlit_app.py
git commit -m "feat: AI 환각 탐지 및 업무 효율화 기능 추가"
git push origin main

# Streamlit Cloud 재배포 (자동)
# 또는 수동 배포 명령 실행
```

---

## 🎯 예상 사용 시나리오

### 시나리오 1: AI 작성 의심 민원 검증
1. 공무원이 민원 접수
2. 내용이 지나치게 전문적이거나 정형화됨
3. "🔍 AI 민원 검증" 모드 실행
4. 환각 탐지 결과 "위험도 높음" 확인
5. 의심 구간 재검증 후 민원인에게 보완 요청

### 시나리오 2: 일반 민원 빠른 처리
1. 케이스 분석 실행
2. 자동으로 환각 검증 진행
3. "위험도 낮음" 확인
4. 안심하고 일반 절차대로 진행

### 시나리오 3: 회신문 자동 작성
1. 환각 검증 완료 후
2. 처리 방향 결정 (승인/거부)
3. 회신문 자동 초안 생성
4. 담당자가 최종 검토 및 발송

---

## 📞 문의 및 지원

**개발자**: [이름]
**이메일**: [이메일]
**Slack**: [채널]

---

## 📚 참고 문서

1. **개발 상세 지침**: `development_instructions.md`
2. **핵심 모듈 코드**: `hallucination_detection.py`
3. **Streamlit 문서**: https://docs.streamlit.io

---

**작성일**: 2026-02-10
**버전**: 1.0
**상태**: ✅ 개발 준비 완료
