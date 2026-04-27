# hallucination_detection.py
# AI 환각 탐지 및 업무 효율화 모듈
# -*- coding: utf-8 -*-

"""
AI-generated petition hallucination detection module.

This module provides tools to detect hallucinations (false information) in 
AI-generated petitions and streamline civil servant workflows.

Optional dependencies are handled gracefully to ensure the app boots even
if some packages are missing (e.g., in Streamlit Cloud deployments).
"""

import re
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Core dependencies (always required)
import streamlit as st

# Optional dependencies - handle gracefully
try:
    from datetime import timedelta
except ImportError:
    timedelta = None
    print("Warning: datetime.timedelta not available - some features may be limited")

# =========================================================
# 1) 환각 탐지 핵심 함수
# =========================================================

def detect_hallucination(text: str, context: Dict, llm_service) -> Dict:
    """
    AI 생성 민원의 환각 가능성 탐지
    
    Args:
        text: 민원 원문
        context: 관련 법령, 절차 등의 맥락 정보
        llm_service: LLM 서비스 인스턴스
    
    Returns:
        환각 탐지 결과 딕셔너리
    """
    suspicious_parts = []
    
    # 1. 패턴 기반 탐지
    pattern_issues = _detect_by_patterns(text)
    suspicious_parts.extend(pattern_issues)
    
    # 2. LLM 기반 교차 검증
    llm_issues = _detect_by_llm(text, context, llm_service)
    suspicious_parts.extend(llm_issues)
    
    # 3. 위험도 계산
    risk_level, overall_score = _calculate_risk_level(suspicious_parts)
    
    # 4. 검증 필요 항목 추출
    verification_needed = _extract_verification_items(suspicious_parts)
    
    return {
        "risk_level": risk_level,
        "suspicious_parts": suspicious_parts,
        "verification_needed": verification_needed,
        "overall_score": overall_score,
        "total_issues_found": len(suspicious_parts)
    }


def _detect_by_patterns(text: str) -> List[Dict]:
    """패턴 기반 환각 탐지"""
    issues = []
    lines = text.split('\n')
    
    # 1. 날짜 검증
    date_pattern = r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일'
    for i, line in enumerate(lines, 1):
        for match in re.finditer(date_pattern, line):
            year, month, day = map(int, match.groups())
            if not _is_valid_date(year, month, day):
                issues.append({
                    "text": match.group(0),
                    "reason": f"존재하지 않는 날짜 (예: {month}월은 {day}일까지 없음)",
                    "confidence": 0.95,
                    "line_number": i,
                    "category": "invalid_date"
                })
    
    # 2. 법령 조항 검증 (기본 패턴만)
    law_pattern = r'([가-힣\s]+법)\s*제\s*(\d+)조'
    for i, line in enumerate(lines, 1):
        for match in re.finditer(law_pattern, line):
            law_name, article_num = match.groups()
            article_num = int(article_num)
            # 비현실적으로 큰 조항 번호
            if article_num > 500:
                issues.append({
                    "text": match.group(0),
                    "reason": f"비현실적으로 큰 조항 번호 (제{article_num}조)",
                    "confidence": 0.75,
                    "line_number": i,
                    "category": "suspicious_law_reference"
                })
    
    # 3. 과도하게 정확한 수치
    precise_number_pattern = r'\d+\.\d{4,}%'
    for i, line in enumerate(lines, 1):
        for match in re.finditer(precise_number_pattern, line):
            issues.append({
                "text": match.group(0),
                "reason": "AI가 지어낸 것으로 의심되는 과도하게 정확한 통계",
                "confidence": 0.65,
                "line_number": i,
                "category": "overly_precise_stats"
            })
    
    # 4. 모순된 수치
    amount_mentions = re.findall(r'(\d{1,3}(?:,\d{3})*)\s*원', text)
    if len(amount_mentions) > 1:
        amounts = [int(amt.replace(',', '')) for amt in amount_mentions]
        if len(set(amounts)) > 1 and max(amounts) / min(amounts) > 10:
            issues.append({
                "text": f"금액 언급: {', '.join(amount_mentions)}원",
                "reason": "문서 내 금액이 일관되지 않음 (10배 이상 차이)",
                "confidence": 0.70,
                "line_number": 0,
                "category": "inconsistent_amounts"
            })
    
    return issues


def _detect_by_llm(text: str, context: Dict, llm_service) -> List[Dict]:
    """LLM 기반 환각 탐지"""
    
    law_context = context.get('law', '')[:2000]  # 토큰 제한
    
    prompt = f"""
당신은 행정 민원 검증 전문가입니다. 다음 민원 내용에서 AI가 생성한 환각(허위/부정확 정보) 가능성이 있는 부분을 찾아주세요.

**민원 내용**:
{text[:3000]}

**관련 법령 (참고용)**:
{law_context}

**검증 기준**:
1. 법령/조례 인용이 실제로 존재하는지
2. 행정 절차 서술이 실무와 일치하는지
3. 날짜/기간이 논리적으로 타당한지
4. 수치 데이터가 합리적인지

**응답 형식** (JSON):
{{
  "issues": [
    {{
      "text": "의심되는 구체적인 문장",
      "reason": "왜 의심되는지 설명",
      "confidence": 0.0~1.0,
      "category": "law_reference|procedure|date|number"
    }}
  ]
}}

**중요**: 확실한 오류만 지적하세요. 애매한 경우는 포함하지 마세요.
"""
    
    try:
        response = llm_service.generate_text(prompt, temperature=0.3)
        result = _safe_json_loads(response)
        
        if result and 'issues' in result:
            # line_number는 0으로 설정 (LLM은 줄 번호를 알 수 없음)
            for issue in result['issues']:
                issue['line_number'] = 0
            return result['issues']
    except Exception as e:
        print(f"LLM 탐지 오류: {e}")
    
    return []


def _is_valid_date(year: int, month: int, day: int) -> bool:
    """날짜 유효성 검증"""
    try:
        datetime(year, month, day)
        # 미래 날짜 체크 (너무 먼 미래는 의심)
        target_date = datetime(year, month, day)
        if target_date > datetime.now().replace(year=datetime.now().year + 10):
            return False
        return True
    except ValueError:
        return False


def _calculate_risk_level(suspicious_parts: List[Dict]) -> Tuple[str, float]:
    """위험도 계산"""
    if not suspicious_parts:
        return "low", 1.0
    
    # 가중치 적용
    total_weight = 0
    for part in suspicious_parts:
        confidence = part.get('confidence', 0.5)
        category = part.get('category', '')
        
        # 카테고리별 가중치
        category_weight = {
            'invalid_date': 1.5,
            'suspicious_law_reference': 1.3,
            'overly_precise_stats': 0.8,
            'inconsistent_amounts': 1.2,
            'law_reference': 1.4,
            'procedure': 1.1,
        }.get(category, 1.0)
        
        total_weight += confidence * category_weight
    
    # 정규화 (이슈 개수 고려)
    normalized_score = total_weight / max(len(suspicious_parts), 1)
    
    # 신뢰도 점수 (역수)
    overall_score = max(0, 1 - normalized_score)
    
    # 위험도 분류
    if normalized_score >= 1.0 or len(suspicious_parts) >= 5:
        risk_level = "high"
    elif normalized_score >= 0.5 or len(suspicious_parts) >= 2:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    return risk_level, overall_score


def _extract_verification_items(suspicious_parts: List[Dict]) -> List[str]:
    """검증이 필요한 항목 추출"""
    items = []
    for part in suspicious_parts:
        category = part.get('category', '')
        text = part.get('text', '')[:50]
        
        if category == 'invalid_date':
            items.append(f"날짜 확인: {text}")
        elif category in ['suspicious_law_reference', 'law_reference']:
            items.append(f"법령 실존 확인: {text}")
        elif category == 'inconsistent_amounts':
            items.append("문서 내 금액 일관성 재확인")
        else:
            items.append(f"사실 관계 확인: {text}")
    
    return list(set(items))  # 중복 제거


def _safe_json_loads(text: str) -> Optional[Dict]:
    """안전한 JSON 파싱"""
    if not text:
        return None
    try:
        return json.loads(text)
    except:
        try:
            # JSON 블록 추출 시도
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except:
            pass
    return None


# =========================================================
# 2) 업무 효율화 함수
# =========================================================

def analyze_petition_priority(petition_text: str, detection_result: Dict, llm_service) -> Dict:
    """민원 긴급도 및 처리 우선순위 자동 판단"""
    
    risk_level = detection_result.get('risk_level', 'low')
    
    prompt = f"""
당신은 행정 민원 처리 전문가입니다. 다음 민원의 처리 우선순위를 분석해주세요.

**민원 내용**:
{petition_text[:2000]}

**환각 위험도**: {risk_level}

**판단 기준**:
1. 긴급성: 법정 기한, 인명/재산 관련, 언론 보도 가능성
2. 업무 복잡도: 관련 부서 수, 필요 절차, 법령 검토 범위
3. 민원인 권리 침해 정도
4. 환각 위험도 (높으면 검증 시간 추가 필요)

**응답 형식** (JSON):
{{
  "priority": "urgent|high|normal|low",
  "estimated_workload": "간편|보통|복잡",
  "recommended_deadline": "YYYY-MM-DD",
  "required_departments": ["부서1", "부서2"],
  "auto_tags": ["태그1", "태그2"],
  "reasoning": "판단 근거 2-3줄"
}}

**우선순위 정의**:
- urgent: 24시간 내 처리 (법정 기한 임박, 긴급상황)
- high: 3일 내 처리 (중요도 높음, 민원인 권리 관련)
- normal: 7일 내 처리 (일반적인 민원)
- low: 14일 내 처리 (단순 문의, 정보 제공)
"""
    
    try:
        response = llm_service.generate_text(prompt, temperature=0.3)
        result = _safe_json_loads(response)
        
        if result:
            # 기본값 설정
            result.setdefault('priority', 'normal')
            result.setdefault('estimated_workload', '보통')
            result.setdefault('recommended_deadline', 
                            (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'))
            result.setdefault('required_departments', ['담당부서'])
            result.setdefault('auto_tags', [])
            result.setdefault('reasoning', '')
            
            return result
    except Exception as e:
        print(f"우선순위 분석 오류: {e}")
    
    # 기본값 반환
    return {
        "priority": "normal",
        "estimated_workload": "보통",
        "recommended_deadline": (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
        "required_departments": ["담당부서"],
        "auto_tags": [],
        "reasoning": "자동 분석 실패, 수동 검토 필요"
    }


def generate_processing_checklist(analysis_result: Dict, llm_service) -> List[Dict]:
    """케이스 분석 결과를 바탕으로 단계별 체크리스트 생성"""
    
    petition = analysis_result.get('petition', '')
    detection = analysis_result.get('detection', {})
    priority = analysis_result.get('priority', {})
    
    risk_level = detection.get('risk_level', 'low')
    
    prompt = f"""
다음 민원에 대한 처리 체크리스트를 단계별로 생성해주세요.

**민원 내용**: {petition[:1500]}
**환각 위험도**: {risk_level}
**우선순위**: {priority.get('priority', 'normal')}

**응답 형식** (JSON):
{{
  "steps": [
    {{
      "step": 1,
      "title": "단계 제목",
      "items": [
        {{"task": "구체적인 작업", "completed": false}},
        {{"task": "구체적인 작업", "completed": false}}
      ],
      "deadline": "접수 후 N일 이내"
    }}
  ]
}}

**반드시 포함할 단계**:
1. 민원 내용 검증 (환각 위험도 높으면 사실 확인 강화)
2. 관련 법령 및 규정 검토
3. 유관 부서 협의 (필요시)
4. 처리 방안 결정
5. 회신문 작성 및 발송

각 단계는 3-5개의 구체적인 작업으로 나누세요.
"""
    
    try:
        response = llm_service.generate_text(prompt, temperature=0.4)
        result = _safe_json_loads(response)
        
        if result and 'steps' in result:
            return result['steps']
    except Exception as e:
        print(f"체크리스트 생성 오류: {e}")
    
    # 기본 체크리스트
    return [
        {
            "step": 1,
            "title": "민원 접수 및 검증",
            "items": [
                {"task": "민원 내용 정확성 확인", "completed": False},
                {"task": "첨부 서류 진위 확인", "completed": False}
            ],
            "deadline": "접수 후 1일 이내"
        },
        {
            "step": 2,
            "title": "관련 법령 검토",
            "items": [
                {"task": "적용 법령 확인", "completed": False},
                {"task": "판례 및 선례 조사", "completed": False}
            ],
            "deadline": "접수 후 3일 이내"
        },
        {
            "step": 3,
            "title": "처리 방안 결정",
            "items": [
                {"task": "처리 방향 설정", "completed": False},
                {"task": "결재 진행", "completed": False}
            ],
            "deadline": "접수 후 5일 이내"
        },
        {
            "step": 4,
            "title": "회신 및 종결",
            "items": [
                {"task": "회신문 작성", "completed": False},
                {"task": "민원인에게 통보", "completed": False}
            ],
            "deadline": "접수 후 7일 이내"
        }
    ]


def generate_response_draft(petition_text: str, analysis: Dict, 
                            response_type: str, llm_service) -> str:
    """민원 회신문 자동 초안 생성"""
    
    response_type_kr = {
        "approval": "승인/수용",
        "rejection": "불가/거부",
        "partial": "부분 수용",
        "request_info": "보완 요청"
    }.get(response_type, "일반 회신")
    
    detection = analysis.get('detection', {})
    priority = analysis.get('priority', {})
    
    prompt = f"""
다음 민원에 대한 **{response_type_kr}** 회신문을 작성해주세요.

**민원 내용**:
{petition_text[:2000]}

**환각 탐지 결과**: {detection.get('risk_level', 'low')}
**처리 우선순위**: {priority.get('priority', 'normal')}

**회신문 작성 규칙**:
1. 행정안전부 공문서 작성 기준 준수
2. 법적 근거 명시 (구체적인 법령명, 조항)
3. 처리 결과 및 사유 명확히 기술
4. 민원인 권리 구제 방법 안내 (거부 시)
5. 담당자 연락처 포함

**형식**:
---
[제목]
(간결하고 명확한 제목)

[본문]
1. 민원 요지 요약 (1-2줄)
2. 처리 결과 및 법적 근거
3. 구체적인 처리 내용/불가 사유
4. 향후 조치 또는 구제 방법
5. 문의처

[첨부]
(필요한 경우)
---

**주의사항**:
- 공손하고 명확한 어조
- 전문 용어는 쉽게 풀어 설명
- {response_type_kr} 사유를 설득력 있게 설명
"""
    
    try:
        response = llm_service.generate_text(prompt, temperature=0.5)
        return response
    except Exception as e:
        print(f"회신문 생성 오류: {e}")
        return f"""
[제목] {response_type_kr} 회신

[본문]
귀하께서 제출하신 민원에 대하여 다음과 같이 회신드립니다.

(자동 생성 실패: {str(e)})
담당자가 수동으로 작성해주세요.

[문의처]
담당: [부서명] [담당자]
연락처: [전화번호]
이메일: [이메일]
"""


# =========================================================
# 3) UI 렌더링 함수
# =========================================================

def render_hallucination_report(detection_result: Dict):
    """환각 탐지 결과를 시각적으로 표시"""
    
    risk_colors = {
        "high": "#dc2626",    # 빨강
        "medium": "#f59e0b",  # 주황
        "low": "#10b981"      # 초록
    }
    
    risk_level = detection_result.get('risk_level', 'low')
    color = risk_colors[risk_level]
    overall_score = detection_result.get('overall_score', 0.5)
    total_issues = detection_result.get('total_issues_found', 0)
    
    risk_labels = {
        "high": "높음 ⚠️",
        "medium": "중간 ⚡",
        "low": "낮음 ✅"
    }
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {color}22 0%, {color}11 100%);
                padding: 1.5rem; border-radius: 12px; border-left: 4px solid {color};
                margin: 1rem 0;'>
        <h4 style='margin: 0 0 1rem 0; color: {color};'>
            🔍 AI 환각 탐지 결과
        </h4>
        <div style='display: flex; gap: 2rem; flex-wrap: wrap;'>
            <div>
                <p style='margin: 0; color: #6b7280; font-size: 0.85rem;'>신뢰도 점수</p>
                <p style='margin: 0.25rem 0 0 0; color: #1f2937; font-size: 1.5rem; font-weight: 700;'>
                    {overall_score*100:.1f}%
                </p>
            </div>
            <div>
                <p style='margin: 0; color: #6b7280; font-size: 0.85rem;'>위험도</p>
                <p style='margin: 0.25rem 0 0 0; color: {color}; font-size: 1.5rem; font-weight: 700;'>
                    {risk_labels[risk_level]}
                </p>
            </div>
            <div>
                <p style='margin: 0; color: #6b7280; font-size: 0.85rem;'>발견된 이슈</p>
                <p style='margin: 0.25rem 0 0 0; color: #1f2937; font-size: 1.5rem; font-weight: 700;'>
                    {total_issues}건
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 의심 구간 표시
    suspicious_parts = detection_result.get('suspicious_parts', [])
    if suspicious_parts:
        st.warning(f"⚠️ **{len(suspicious_parts)}개의 의심 구간이 발견되었습니다.** 아래 내용을 검증하세요:")
        
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
    
    # 검증 필요 항목
    verification_needed = detection_result.get('verification_needed', [])
    if verification_needed:
        st.info("📋 **검증이 필요한 항목**:")
        for item in verification_needed:
            st.markdown(f"- {item}")


# =========================================================
# 4) 캐싱 및 유틸리티
# =========================================================

@st.cache_data(ttl=3600)
def _detect_hallucination_cached_core(text_hash: str, text: str, context_json: str) -> Dict:
    """
    내부 캐싱 함수 - llm_service 없이 호출
    
    주의: llm_service는 해시 불가능(pickle 불가)하므로 캐싱 대상에서 제외
    """
    # LLM 기반 탐지는 캐싱하지 않고, 패턴 기반만 캐싱
    context = json.loads(context_json) if context_json else {}
    
    suspicious_parts = []
    # 패턴 기반 탐지만 수행 (빠르고 결정적)
    suspicious_parts.extend(_detect_by_patterns(text))
    
    # 위험도 계산
    risk_level, overall_score = _calculate_risk_level(suspicious_parts)
    
    # 검증 필요 항목 추출
    verification_needed = _extract_verification_items(suspicious_parts)
    
    return {
        "risk_level": risk_level,
        "suspicious_parts": suspicious_parts,
        "verification_needed": verification_needed,
        "overall_score": overall_score,
        "total_issues_found": len(suspicious_parts),
        "cached": True  # 캐싱된 결과임을 표시
    }


def detect_hallucination_cached(text_hash: str, text: str, context: Dict, llm_service) -> Dict:
    """
    캐싱된 환각 탐지 (동일 민원 중복 검증 방지)
    
    전략:
    1. 패턴 기반 탐지는 캐싱 (빠르고 결정적)
    2. LLM 기반 탐지는 매번 실행 (llm_service 객체는 pickle 불가)
    
    Args:
        text_hash: 텍스트 해시 (캐싱 키)
        text: 민원 원문
        context: 관련 컨텍스트
        llm_service: LLM 서비스 인스턴스 (캐싱 제외)
    """
    # 1. 패턴 기반 탐지 (캐싱)
    context_json = json.dumps(context, ensure_ascii=False) if context else ""
    cached_result = _detect_hallucination_cached_core(text_hash, text, context_json)
    
    # 2. LLM 기반 탐지 (매번 실행 - 캐싱 안 함)
    try:
        llm_issues = _detect_by_llm(text, context, llm_service)
        
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
            "cached": False
        }
    except Exception as e:
        # LLM 오류 시 캐싱된 결과만 반환
        print(f"LLM 탐지 오류 (패턴 기반 결과만 사용): {e}")
        return cached_result


def get_text_hash(text: str) -> str:
    """텍스트 해시 생성"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
