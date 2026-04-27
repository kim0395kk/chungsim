# 🔍 환각 검증 기능 업그레이드 구현 지시서

> **이 문서는 AI 코딩 에이전트가 읽고 그대로 구현하기 위한 지시서입니다.**
> 모든 파일 경로, 줄 번호, 코드 구조는 실제 프로젝트 기준입니다.

## 🎯 목표

환각 검증 기능을 3가지 측면에서 업그레이드:
1. **검증 근거 투명화** — 어떤 검사를 어떤 기준으로 수행했는지 사용자에게 보여줌
2. **실시간 진행 상황** — 단계별 라이브 로그 + 부분 결과 즉시 표시로 이탈 방지
3. **원문 하이라이트** — 의심 구간을 원문에서 직접 색상으로 표시

---

## 📁 수정 대상 파일 (2개)

| 파일 | 전체 경로 | 현재 줄 수 |
|------|----------|-----------|
| `hallucination_detection.py` | `c:\Users\Mr Kim\Desktop\chungsim\hallucination_detection.py` | 684줄 |
| `streamlit_app.py` | `c:\Users\Mr Kim\Desktop\chungsim\streamlit_app.py` | 4575줄 |

> **주의**: `streamlit_app.py`는 4575줄짜리 대형 파일입니다. 환각 검증 관련 코드는 **3780~4094행**에만 있습니다. 이 범위만 수정하세요. 나머지는 절대 건드리지 마세요.

---

## 📋 현재 코드 구조 (반드시 읽고 이해할 것)

### hallucination_detection.py 구조

```
1-29행:    imports
31-33행:   섹션 1 헤더 "환각 탐지 핵심 함수"
35-69행:   detect_hallucination() — 메인 탐지 함수
72-132행:  _detect_by_patterns() — 패턴 기반 탐지 (4가지 검사)
             ├ 77-89행:   날짜 검증 (date_pattern)
             ├ 91-105행:  법령 조항 번호 검증 (law_pattern, >500 의심)
             ├ 107-117행: 과도하게 정확한 수치 (소수점 4자리 이상 %)
             └ 119-131행: 금액 불일치 (10배 이상 차이)
135-182행: _detect_by_llm() — LLM 기반 교차 검증
185-195행: _is_valid_date() — 날짜 유효성
198-235행: _calculate_risk_level() — 위험도 계산
238-254행: _extract_verification_items() — 검증 필요 항목 추출
257-271행: _safe_json_loads() — 안전한 JSON 파싱
274-276행: 섹션 2 헤더 "업무 효율화 함수"
278-340행: analyze_petition_priority() — 우선순위 분석
343-431행: generate_processing_checklist() — 체크리스트 생성
434-504행: generate_response_draft() — 회신문 초안 생성
507-509행: 섹션 3 헤더 "UI 렌더링 함수"
511-593행: render_hallucination_report() — 결과 렌더링 UI
595-597행: 섹션 4 헤더 "캐싱 및 유틸리티"
599-626행: _detect_hallucination_cached_core() — 패턴 캐싱
629-678행: detect_hallucination_cached() — 메인 캐싱 함수
681-684행: get_text_hash() — 텍스트 해시
```

### streamlit_app.py 환각 검증 영역 (3780~4094행) 구조

```
3780행:        elif "hallucination_check" 모드 시작
3781-3796행:   헤더 + 기능 소개 마크다운
3798-3815행:   사용 방법 expander
3817행:        구분선
3820-3856행:   입력 섹션 (text_area + file_uploader)
3858-3866행:   검증 시작 버튼
3871-3894행:   검증 실행 로직
                ├ 3873-3874행: progress_text, progress_bar 생성
                ├ 3877-3888행: Step1 환각 탐지 (detect_hallucination_cached 호출)
                ├ 3890-3897행: Step2 우선순위 분석
                └ 3899-3908행: Step3 체크리스트 생성
3911-3913행:   progress 제거
3915행:        완료 메시지
3917-3922행:   결과 1: 환각 탐지 (render_hallucination_report 호출)
3926-3982행:   결과 2: 우선순위 분석 표시
3986-4007행:   결과 3: 체크리스트 표시
4011-4088행:   결과 4: 회신문 초안 생성
4090-4094행:   에러 처리
```

---

## 🔧 구현 작업 1: hallucination_detection.py 수정

### 작업 1-1: `_detect_by_patterns()` 수정 (72-132행)

**목표**: 각 패턴 검사 결과에 `detection_method`와 `rule_applied` 필드를 추가합니다.

**현재 반환 형식** (각 issue):
```python
{
    "text": "2025년 13월 32일",
    "reason": "존재하지 않는 날짜",
    "confidence": 0.95,
    "line_number": 1,
    "category": "invalid_date"
}
```

**변경 후 반환 형식** (각 issue):
```python
{
    "text": "2025년 13월 32일",
    "reason": "존재하지 않는 날짜 (예: 13월은 32일까지 없음)",
    "confidence": 0.95,
    "line_number": 1,
    "category": "invalid_date",
    "detection_method": "pattern",           # 추가
    "rule_applied": "날짜 형식 검증: YYYY년 MM월 DD일 패턴에서 유효하지 않은 날짜 감지"  # 추가
}
```

**구체적 수정 내용**: 각 issues.append() 호출에서 `detection_method`와 `rule_applied` 필드를 추가:

| 위치 | category | rule_applied 값 |
|------|----------|----------------|
| 83-89행 (날짜) | `invalid_date` | `"날짜 유효성 검증: YYYY년 MM월 DD일 형식에서 실제 달력에 존재하지 않는 날짜 감지"` |
| 99-105행 (법령) | `suspicious_law_reference` | `"법령 조항 범위 검증: 조항 번호가 500을 초과하면 실존하지 않을 가능성 높음"` |
| 111-117행 (수치) | `overly_precise_stats` | `"통계 정밀도 검증: 소수점 4자리 이상의 백분율은 AI가 생성한 허위 통계일 가능성"` |
| 124-130행 (금액) | `inconsistent_amounts` | `"금액 일관성 검증: 동일 문서 내 금액 차이가 10배 이상이면 모순 의심"` |

모든 항목에 `"detection_method": "pattern"` 추가

### 작업 1-2: `_detect_by_llm()` 수정 (135-182행)

**목표**: LLM 반환 결과에도 `detection_method` 필드 추가.

**현재 코드 (176-178행)**:
```python
for issue in result['issues']:
    issue['line_number'] = 0
return result['issues']
```

**변경 후**:
```python
for issue in result['issues']:
    issue['line_number'] = 0
    issue['detection_method'] = 'llm'
    issue['rule_applied'] = f"AI 교차 검증: {issue.get('category', '종합')} 분석"
return result['issues']
```

### 작업 1-3: 새 함수 추가 — `render_verification_log()`

**위치**: `render_hallucination_report()` 함수 바로 아래 (593행 뒤)에 추가.

**함수 목적**: 검증에 사용된 방법과 각 검사 항목의 수행 결과를 시각적으로 표시.

```python
def render_verification_log(detection_result: Dict, verification_log: Dict):
    """검증 수행 내역을 투명하게 표시"""
    
    st.markdown("### 🔬 검증 수행 내역")
    
    # 규칙 기반 검증 결과
    pattern_checks = verification_log.get('pattern_checks', {})
    pattern_issues_count = verification_log.get('pattern_issues_count', 0)
    
    with st.expander(
        f"✅ 규칙 기반 검증 (4건 수행, {pattern_issues_count}건 이상 감지)", 
        expanded=True
    ):
        check_items = [
            ("날짜 패턴 검사", "invalid_date", "YYYY년 MM월 DD일 형식의 날짜가 실제 달력에 존재하는지 확인"),
            ("법령 조항 번호 검사", "suspicious_law_reference", "인용된 법령의 조항 번호가 합리적 범위(500조 이내)인지 확인"),
            ("과도 정밀 수치 검사", "overly_precise_stats", "소수점 4자리 이상의 통계 수치가 있는지 확인"),
            ("금액 일관성 검사", "inconsistent_amounts", "문서 내 금액 언급이 서로 10배 이상 차이나는지 확인"),
        ]
        
        for check_name, category, description in check_items:
            count = pattern_checks.get(category, 0)
            if count > 0:
                st.markdown(f"- 🔴 **{check_name}**: {count}건 이상 감지")
            else:
                st.markdown(f"- ✅ **{check_name}**: 이상 없음")
            st.caption(f"   검사 기준: {description}")
    
    # LLM 기반 검증 결과
    llm_status = verification_log.get('llm_status', 'not_run')
    llm_issues_count = verification_log.get('llm_issues_count', 0)
    llm_model = verification_log.get('llm_model', '알 수 없음')
    
    if llm_status == 'success':
        with st.expander(f"✅ AI 교차 검증 완료 ({llm_issues_count}건 추가 감지)"):
            st.markdown(f"- **사용 모델**: {llm_model}")
            st.markdown(f"- **검증 기준**: 법령 실존 여부, 행정 절차 정확성, 날짜/기간 타당성, 수치 합리성")
            st.markdown(f"- **AI 분석 결과**: {llm_issues_count}건 추가 의심 구간 발견")
    elif llm_status == 'error':
        with st.expander("⚠️ AI 교차 검증 실패 (패턴 기반 결과만 표시)"):
            st.markdown("- AI 모델 호출에 실패했습니다. 규칙 기반 검증 결과만 표시됩니다.")
            st.markdown("- 네트워크 상태 또는 API 키를 확인해주세요.")
    
    # 법령 원문 대조 상태
    has_law_context = verification_log.get('has_law_context', False)
    if has_law_context:
        with st.expander("✅ 법령 원문 대조 수행됨"):
            st.markdown("- 관련 법령 데이터와 대조 검증을 수행했습니다.")
    else:
        with st.expander("⚪ 법령 원문 대조 (미수행)"):
            st.markdown("- 관련 법령 데이터가 제공되지 않아 원문 대조를 수행하지 못했습니다.")
            st.markdown("- context에 법령 정보를 연결하면 정확도가 향상됩니다.")
```

### 작업 1-4: 새 함수 추가 — `render_highlighted_text()`

**위치**: `render_verification_log()` 바로 아래에 추가.

**함수 목적**: 민원 원문에서 의심 구간을 색상별로 하이라이트하여 표시.

```python
def render_highlighted_text(original_text: str, suspicious_parts: List[Dict]):
    """원문에서 의심 구간을 하이라이트하여 표시"""
    
    if not suspicious_parts:
        st.markdown("### 📄 원문 검증 결과")
        st.success("의심 구간이 발견되지 않았습니다.")
        st.text(original_text)
        return
    
    st.markdown("### 📄 원문 검증 결과 (하이라이트)")
    
    # 범례
    st.markdown("""
    <div style='display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap;'>
        <span style='padding: 0.2rem 0.5rem; background: #fecaca; border-radius: 4px; font-size: 0.85rem;'>🔴 확정 오류 (신뢰도 80%↑)</span>
        <span style='padding: 0.2rem 0.5rem; background: #fed7aa; border-radius: 4px; font-size: 0.85rem;'>🟠 검증 필요 (신뢰도 60~80%)</span>
        <span style='padding: 0.2rem 0.5rem; background: #fef08a; border-radius: 4px; font-size: 0.85rem;'>🟡 주의 (신뢰도 60%↓)</span>
    </div>
    """, unsafe_allow_html=True)
    
    # 신뢰도별 색상 결정
    def get_highlight_color(confidence):
        if confidence >= 0.8:
            return "#fecaca"  # 빨강 (확정 오류)
        elif confidence >= 0.6:
            return "#fed7aa"  # 주황 (검증 필요)
        else:
            return "#fef08a"  # 노랑 (주의)
    
    # 원문에서 의심 구간 하이라이트 적용
    highlighted_html = original_text
    
    # 긴 텍스트부터 먼저 교체 (겹침 방지)
    sorted_parts = sorted(suspicious_parts, key=lambda x: len(x.get('text', '')), reverse=True)
    
    for part in sorted_parts:
        suspect_text = part.get('text', '')
        confidence = part.get('confidence', 0.5)
        reason = part.get('reason', '')
        color = get_highlight_color(confidence)
        
        if suspect_text and suspect_text in highlighted_html:
            tooltip = f"{reason} (신뢰도: {confidence*100:.0f}%)"
            replacement = (
                f'<span style="background: {color}; padding: 0.1rem 0.3rem; '
                f'border-radius: 3px; cursor: help;" '
                f'title="{tooltip}">{suspect_text}</span>'
            )
            highlighted_html = highlighted_html.replace(suspect_text, replacement, 1)
    
    # 줄바꿈 처리
    highlighted_html = highlighted_html.replace('\n', '<br>')
    
    st.markdown(
        f"""<div style='background: white; padding: 1.5rem; border-radius: 8px; 
             border: 1px solid #e5e7eb; line-height: 1.8; font-size: 0.95rem; 
             color: #1f2937;'>{highlighted_html}</div>""",
        unsafe_allow_html=True
    )
```

### 작업 1-5: `detect_hallucination_cached()` 수정 (629-678행)

**목표**: `verification_log`를 반환값에 포함시킵니다.

**현재 코드 (629-678행)을 아래로 전체 교체**:

```python
def detect_hallucination_cached(text_hash: str, text: str, context: Dict, llm_service) -> Dict:
    """
    캐싱된 환각 탐지 (동일 민원 중복 검증 방지)
    
    전략:
    1. 패턴 기반 탐지는 캐싱 (빠르고 결정적)
    2. LLM 기반 탐지는 매번 실행 (llm_service 객체는 pickle 불가)
    """
    import time
    verification_log = {
        "steps": [],
        "pattern_checks": {},
        "pattern_issues_count": 0,
        "llm_status": "not_run",
        "llm_issues_count": 0,
        "llm_model": "unknown",
        "has_law_context": bool(context),
        "start_time": time.time(),
    }
    
    # 1. 패턴 기반 탐지 (캐싱)
    context_json = json.dumps(context, ensure_ascii=False) if context else ""
    cached_result = _detect_hallucination_cached_core(text_hash, text, context_json)
    
    # 패턴 결과에서 카테고리별 집계
    for part in cached_result.get('suspicious_parts', []):
        cat = part.get('category', 'unknown')
        verification_log['pattern_checks'][cat] = verification_log['pattern_checks'].get(cat, 0) + 1
    verification_log['pattern_issues_count'] = cached_result.get('total_issues_found', 0)
    verification_log['steps'].append({
        "name": "패턴 기반 검증",
        "status": "완료",
        "issues_found": cached_result.get('total_issues_found', 0),
        "elapsed": round(time.time() - verification_log['start_time'], 2)
    })
    
    # 2. LLM 기반 탐지 (매번 실행)
    try:
        # LLM 모델명 기록
        if hasattr(llm_service, 'model_name'):
            verification_log['llm_model'] = llm_service.model_name
        elif hasattr(llm_service, 'model'):
            verification_log['llm_model'] = str(llm_service.model)
        else:
            verification_log['llm_model'] = 'AI 모델'
        
        llm_issues = _detect_by_llm(text, context, llm_service)
        verification_log['llm_status'] = 'success'
        verification_log['llm_issues_count'] = len(llm_issues)
        verification_log['steps'].append({
            "name": "AI 교차 검증",
            "status": "완료",
            "issues_found": len(llm_issues),
            "elapsed": round(time.time() - verification_log['start_time'], 2)
        })
        
        # 결과 병합
        all_suspicious = cached_result['suspicious_parts'] + llm_issues
        
        # 중복 제거 (동일 텍스트)
        seen_texts = set()
        unique_suspicious = []
        for part in all_suspicious:
            part_text = part.get('text', '')
            if part_text not in seen_texts:
                seen_texts.add(part_text)
                unique_suspicious.append(part)
        
        # 재계산
        risk_level, overall_score = _calculate_risk_level(unique_suspicious)
        verification_needed = _extract_verification_items(unique_suspicious)
        
        return {
            "risk_level": risk_level,
            "suspicious_parts": unique_suspicious,
            "verification_needed": verification_needed,
            "overall_score": overall_score,
            "total_issues_found": len(unique_suspicious),
            "cached": False,
            "verification_log": verification_log
        }
    except Exception as e:
        verification_log['llm_status'] = 'error'
        verification_log['llm_error'] = str(e)
        verification_log['steps'].append({
            "name": "AI 교차 검증",
            "status": "실패",
            "error": str(e),
            "elapsed": round(time.time() - verification_log['start_time'], 2)
        })
        print(f"LLM 탐지 오류 (패턴 기반 결과만 사용): {e}")
        result = cached_result.copy()
        result['verification_log'] = verification_log
        return result
```

### 작업 1-6: `render_hallucination_report()` 수정 (511-593행)

**목표**: 각 의심 구간에 `detection_method` 와 `rule_applied` 정보 표시 추가.

**현재 코드 (566-585행)**:
```python
for i, part in enumerate(suspicious_parts, 1):
    with st.expander(f"🔎 의심 구간 {i}: {part['text'][:60]}{'...' if len(part['text']) > 60 else ''}"):
        st.markdown(f"**전체 내용**: `{part['text']}`")
        st.markdown(f"**탐지 이유**: {part['reason']}")
        st.markdown(f"**신뢰도**: {part['confidence']*100:.1f}%")
        
        if part.get('line_number', 0) > 0:
            st.markdown(f"**위치**: {part['line_number']}번째 줄")
        
        category_labels = {
            'invalid_date': '날짜 오류',
            'suspicious_law_reference': '법령 참조 의심',
            'overly_precise_stats': '과도한 통계',
            'inconsistent_amounts': '금액 불일치',
            'law_reference': '법령 검증 필요',
            'procedure': '절차 검증 필요'
        }
        category = part.get('category', '')
        if category:
            st.caption(f"카테고리: {category_labels.get(category, category)}")
```

**변경 후**:
```python
for i, part in enumerate(suspicious_parts, 1):
    # 탐지 방법에 따라 아이콘 변경
    method = part.get('detection_method', 'unknown')
    method_icon = "🔧" if method == "pattern" else "🤖" if method == "llm" else "❓"
    method_label = "규칙 기반" if method == "pattern" else "AI 분석" if method == "llm" else "알 수 없음"
    
    with st.expander(f"🔎 의심 구간 {i}: {part['text'][:60]}{'...' if len(part['text']) > 60 else ''}"):
        st.markdown(f"**전체 내용**: `{part['text']}`")
        st.markdown(f"**탐지 이유**: {part['reason']}")
        st.markdown(f"**신뢰도**: {part['confidence']*100:.1f}%")
        
        if part.get('line_number', 0) > 0:
            st.markdown(f"**위치**: {part['line_number']}번째 줄")
        
        # 탐지 근거 정보 (신규)
        st.markdown(f"**{method_icon} 탐지 방법**: {method_label}")
        rule = part.get('rule_applied', '')
        if rule:
            st.markdown(f"**📏 적용 규칙**: {rule}")
        
        # 법령 관련이면 법제처 링크 제공
        category = part.get('category', '')
        if category in ['suspicious_law_reference', 'law_reference']:
            law_text = part.get('text', '')
            law_match = re.search(r'([가-힣]+법)', law_text)
            if law_match:
                law_name = law_match.group(1)
                law_url = f"https://www.law.go.kr/LSW/lsInfoP.do?efYd=20240101&query={law_name}"
                st.markdown(f"**🔗 확인**: [법제처에서 '{law_name}' 검색하기]({law_url})")
        
        category_labels = {
            'invalid_date': '날짜 오류',
            'suspicious_law_reference': '법령 참조 의심',
            'overly_precise_stats': '과도한 통계',
            'inconsistent_amounts': '금액 불일치',
            'law_reference': '법령 검증 필요',
            'procedure': '절차 검증 필요'
        }
        if category:
            st.caption(f"카테고리: {category_labels.get(category, category)}")
```

---

## 🔧 구현 작업 2: streamlit_app.py 수정 (3780~4094행만)

### 작업 2-1: import 확인

파일 상단(82-110행 근처)에 hallucination_detection에서의 import 확인. 새 함수 2개를 추가로 import해야 합니다.

**현재 import (82행 근처 찾기)**:
```python
from hallucination_detection import (
    detect_hallucination_cached,
    render_hallucination_report,
    analyze_petition_priority,
    generate_processing_checklist,
    generate_response_draft,
    get_text_hash
)
```

**변경 후**:
```python
from hallucination_detection import (
    detect_hallucination_cached,
    render_hallucination_report,
    render_verification_log,        # 추가
    render_highlighted_text,         # 추가
    analyze_petition_priority,
    generate_processing_checklist,
    generate_response_draft,
    get_text_hash
)
```

> **주의**: import 위치가 try/except 블록 안에 있을 수 있습니다. 정확한 위치를 확인하고 추가하세요.

### 작업 2-2: 진행 상황 로직 교체 (3871-3913행)

**현재 코드 (3871-3913행)**:

```python
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
```

**전체 교체 (아래 코드로)**:

```python
if verify_btn and petition_input:
    import time
    
    # 실시간 진행 로그 컨테이너
    log_container = st.container()
    progress_bar = st.progress(0)
    log_messages = []
    
    def add_log(icon, message, elapsed=None):
        """실시간 로그 메시지 추가"""
        time_str = f"[{elapsed:.1f}초]" if elapsed is not None else ""
        log_messages.append(f"{icon} {time_str} {message}")
        with log_container:
            st.markdown(
                "<div style='background: #f8fafc; padding: 1rem; border-radius: 8px; "
                "border: 1px solid #e2e8f0; font-family: monospace; font-size: 0.85rem; "
                "max-height: 200px; overflow-y: auto;'>" +
                "<br>".join(log_messages) +
                "</div>",
                unsafe_allow_html=True
            )
    
    start_time = time.time()
    
    try:
        # Step 1: 텍스트 분석 시작
        elapsed = time.time() - start_time
        add_log("🔄", f"민원 텍스트 분석 시작 ({len(petition_input)}자)", elapsed)
        progress_bar.progress(5)
        
        # Step 2: 패턴 기반 탐지
        elapsed = time.time() - start_time
        add_log("🔍", "규칙 기반 검증 수행 중... (날짜, 법령, 수치, 금액)", elapsed)
        progress_bar.progress(10)
        
        text_hash = get_text_hash(petition_input)
        detection_result = detect_hallucination_cached(
            text_hash,
            petition_input,
            {},
            llm_service
        )
        
        # 패턴 결과 로그
        v_log = detection_result.get('verification_log', {})
        pattern_count = v_log.get('pattern_issues_count', 0)
        elapsed = time.time() - start_time
        add_log("✅", f"규칙 기반 검증 완료 → {pattern_count}건 감지", elapsed)
        
        # LLM 결과 로그
        llm_status = v_log.get('llm_status', 'not_run')
        llm_count = v_log.get('llm_issues_count', 0)
        if llm_status == 'success':
            add_log("✅", f"AI 교차 검증 완료 → {llm_count}건 추가 감지", elapsed)
        elif llm_status == 'error':
            add_log("⚠️", "AI 교차 검증 실패 (패턴 결과만 사용)", elapsed)
        
        progress_bar.progress(40)
        
        # Step 3: 우선순위 분석
        elapsed = time.time() - start_time
        add_log("🔄", "우선순위 분석 중...", elapsed)
        progress_bar.progress(50)
        
        priority_analysis = analyze_petition_priority(
            petition_input, 
            detection_result,
            llm_service
        )
        
        elapsed = time.time() - start_time
        priority = priority_analysis.get('priority', 'normal')
        add_log("✅", f"우선순위 분석 완료 → {priority.upper()}", elapsed)
        progress_bar.progress(70)
        
        # Step 4: 체크리스트 생성
        elapsed = time.time() - start_time
        add_log("🔄", "업무 체크리스트 생성 중...", elapsed)
        progress_bar.progress(80)
        
        checklist = generate_processing_checklist(
            {
                "petition": petition_input,
                "detection": detection_result,
                "priority": priority_analysis
            },
            llm_service
        )
        
        elapsed = time.time() - start_time
        add_log("✅", f"체크리스트 생성 완료 ({len(checklist)}단계)", elapsed)
        progress_bar.progress(100)
        
        # 완료 로그
        total_time = time.time() - start_time
        add_log("🎉", f"전체 검증 완료! (총 {total_time:.1f}초 소요)", total_time)
        
        import time as time_module
        time_module.sleep(1)  # 완료 로그를 잠시 보여줌
        progress_bar.empty()
```

> **중요**: `progress_text.empty()` 와 `progress_bar.empty()` 는 기존이고, 새 코드에서는 `log_container`는 그대로 두고 `progress_bar`만 제거합니다.

### 작업 2-3: 결과 표시 영역 수정 (3917-3922행)

**현재 코드**:
```python
st.success("✅ 검증 완료!")

# === 결과 표시 ===
st.divider()

# 1. 환각 탐지 결과
st.subheader("🔍 환각 탐지 결과")
render_hallucination_report(detection_result)
```

**변경 후**:
```python
st.success(f"✅ 검증 완료! (총 {total_time:.1f}초 소요)")

# === 결과 표시 ===
st.divider()

# 0. 원문 하이라이트 (신규)
render_highlighted_text(petition_input, detection_result.get('suspicious_parts', []))

st.divider()

# 1. 검증 수행 내역 (신규)
v_log = detection_result.get('verification_log', {})
if v_log:
    render_verification_log(detection_result, v_log)

st.divider()

# 2. 환각 탐지 결과 (기존)
st.subheader("🔍 환각 탐지 결과")
render_hallucination_report(detection_result)
```

---

## ⚠️ 구현 시 주의사항

1. **`streamlit_app.py`는 3780~4094행만 수정하세요.** 나머지 4400줄 이상의 코드를 절대로 건드리지 마세요.
2. **기존 함수 시그니처(인터페이스)를 변경하지 마세요.** `detect_hallucination_cached(text_hash, text, context, llm_service)` 시그니처는 그대로 유지합니다.
3. **반환값에 `verification_log` 필드를 추가하는 방식**이므로 기존 코드와 하위 호환됩니다.
4. **`re` 모듈**은 이미 import 되어 있습니다 (hallucination_detection.py 15행).
5. **`time` 모듈**은 함수 내에서 `import time`으로 사용합니다.
6. **st.markdown의 `unsafe_allow_html=True`**는 HTML 렌더링에 필수입니다.
7. **Streamlit 재실행 시 전체 스크립트가 재실행됩니다.** 상태 유지가 필요한 경우 `st.session_state`를 사용하세요.

---

## ✅ 구현 완료 후 확인 사항

### 테스트 입력

아래 텍스트로 환각 검증을 실행하세요:

```
2025년 13월 32일에 ○○구청에서 발생한 사건에 대해 민원을 제기합니다.
주민등록법 제999조에 따르면 전입신고 의무가 있으며, 
통계청 자료에 따르면 해당 지역의 전입 비율은 정확히 47.3829%입니다.
보조금 100원을 신청했으나 10,000,000원이 지급되었습니다.
```

### 기대 결과

1. **실시간 로그**: 단계별로 "🔄 수행 중 → ✅ 완료" 메시지가 순차적으로 나타남
2. **원문 하이라이트**: 4개 구간이 각각 색상별로 표시됨
   - 🔴 빨강: "2025년 13월 32일" (신뢰도 95%)
   - 🟠 주황: "주민등록법 제999조" (신뢰도 75%)
   - 🟡 노랑/주황: "47.3829%" (신뢰도 65%)
   - 🟡 노랑/주황: 금액 불일치 (신뢰도 70%)
3. **검증 수행 내역**: 규칙 기반 4건 + AI 교차 검증 결과 표시
4. **각 의심 구간**: `detection_method`, `rule_applied` 정보 포함

---

## 📊 변경 요약

| 구분 | 수정 내용 | 파일 | 줄 범위 |
|------|----------|------|---------|
| 패턴 결과에 근거 추가 | `detection_method`, `rule_applied` 필드 | `hallucination_detection.py` | 72-132행 |
| LLM 결과에 근거 추가 | `detection_method`, `rule_applied` 필드 | `hallucination_detection.py` | 176-178행 |
| 새 함수: render_verification_log | 검증 수행 내역 표시 | `hallucination_detection.py` | 593행 뒤 추가 |
| 새 함수: render_highlighted_text | 원문 하이라이트 | `hallucination_detection.py` | 위 함수 뒤 추가 |
| detect_hallucination_cached 수정 | verification_log 반환 | `hallucination_detection.py` | 629-678행 교체 |
| render_hallucination_report 수정 | 근거 표시 추가 | `hallucination_detection.py` | 566-585행 |
| import 추가 | 새 함수 2개 | `streamlit_app.py` | import 영역 |
| 진행 상황 로직 교체 | 실시간 로그 | `streamlit_app.py` | 3871-3913행 |
| 결과 표시 영역 수정 | 하이라이트+검증내역 추가 | `streamlit_app.py` | 3917-3922행 |
