# -*- coding: utf-8 -*-
"""
Govable AI - 재사용 가능한 UI 컴포넌트

이 모듈에서만 streamlit import 허용
"""
from typing import List, Optional

import streamlit as st

from govable_ai.helpers import md_bold_to_html_safe


def render_header(title: str) -> None:
    """섹션 헤더 렌더링"""
    st.markdown(
        f"""
        <div style='background: white; padding: 0.8rem 1rem; border-radius: 10px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 0.8rem; border: 1px solid #f3f4f6;'>
            <h3 style='margin: 0; color: #1f2937; font-size: 1.1rem; font-weight: 700; display: flex; align-items: center; gap: 0.5rem;'>
                {title}
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_lawbot_button(url: str) -> None:
    """Lawbot 검색 버튼 렌더링"""
    st.markdown(
        f"""
        <a href="{url}" target="_blank" class="lawbot-btn">
            🤖 Lawbot에서 추가 검색
            <span class="lawbot-sub">AI 법률 상담 및 판례·법령 심층 탐색 (국가법령정보센터)</span>
        </a>
        """,
        unsafe_allow_html=True
    )


def render_agent_logs(logs: List[dict], placeholder=None) -> None:
    """
    에이전트 로그 렌더링
    
    Args:
        logs: 로그 리스트 [{"msg": str, "style": str, "status": str, "elapsed": float}, ...]
        placeholder: st.empty() 플레이스홀더 (None이면 직접 렌더링)
    """
    log_html = ""
    
    for log in logs:
        style = log.get("style", "sys")
        status = log.get("status", "done")
        msg = log.get("msg", "")
        elapsed = log.get("elapsed", 0)
        
        # CSS 클래스 매핑
        css_class = "log-sys"
        if style == "legal":
            css_class = "log-legal"
        elif style == "search":
            css_class = "log-search"
        elif style == "strat":
            css_class = "log-strat"
        elif style == "calc":
            css_class = "log-calc"
        elif style == "draft":
            css_class = "log-draft"
        
        # 상태별 아이콘
        if status == "active":
            icon = "<span class='spinner-icon'>⏳</span>"
            css_class += " log-active"
        else:
            icon = "✅"
        
        # 경과 시간 표시
        time_str = f"<span style='float: right; color: #9ca3af; font-size: 0.85rem;'>{elapsed:.1f}s</span>" if elapsed else ""
        
        log_html += f"<div class='agent-log {css_class}'>{icon} {msg}{time_str}</div>"
    
    if placeholder:
        placeholder.markdown(log_html, unsafe_allow_html=True)
    else:
        st.markdown(log_html, unsafe_allow_html=True)


def render_document_paper(
    doc: dict,
    meta: dict,
    department_name: str = "OOO시 OOO과",
) -> None:
    """
    공문서 A4 스타일 렌더링
    
    Args:
        doc: 공문서 딕셔너리 (title, receiver, body_paragraphs, department_head)
        meta: 메타데이터 (today_str, doc_num)
        department_name: 부서명
    """
    title = doc.get("title", "민원 처리 결과 회신(안)")
    receiver = doc.get("receiver", "수신자 참조")
    body_paragraphs = doc.get("body_paragraphs", [])
    department_head = doc.get("department_head", "OOO과장")
    
    today_str = meta.get("today_str", "")
    doc_num = meta.get("doc_num", "")
    
    # 본문 HTML 생성
    body_html = ""
    for para in body_paragraphs:
        if para.strip():
            body_html += md_bold_to_html_safe(para) + "<br><br>"
    
    html = f"""
    <div class="paper-sheet">
        <div class="doc-header">{title}</div>
        <div class="doc-info">
            <div>📋 <span>수신</span>: {receiver}</div>
            <div>📅 <span>시행일자</span>: {today_str}</div>
            <div>📝 <span>문서번호</span>: {doc_num}</div>
        </div>
        <div class="doc-body">{body_html}</div>
        <div class="doc-footer">{department_name}<br>{department_head}</div>
        <div class="stamp">결 재</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_info_card(
    title: str,
    content: str,
    icon: str = "📌",
    color: str = "primary",
) -> None:
    """
    정보 카드 렌더링
    
    Args:
        title: 카드 제목
        content: 카드 내용 (마크다운)
        icon: 아이콘 이모지
        color: 색상 테마 (primary, success, warning, error, info)
    """
    color_map = {
        "primary": ("#1e4a7a", "#f0f4f8"),
        "success": ("#2e7d32", "#e8f5e9"),
        "warning": ("#ed6c02", "#fff3e0"),
        "error": ("#d32f2f", "#ffebee"),
        "info": ("#0288d1", "#e3f2fd"),
    }
    
    border_color, bg_color = color_map.get(color, color_map["primary"])
    
    st.markdown(
        f"""
        <div style='background: {bg_color}; border-left: 4px solid {border_color}; 
                    padding: 1rem 1.25rem; border-radius: 0.375rem; margin-bottom: 1rem;'>
            <div style='font-weight: 700; color: {border_color}; margin-bottom: 0.5rem;'>
                {icon} {title}
            </div>
            <div style='color: #424242; line-height: 1.6;'>
                {content}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_analysis_result(analysis: dict) -> None:
    """
    민원 분석 결과 렌더링
    
    Args:
        analysis: CaseAnalyzer.analyze() 결과
    """
    case_type = analysis.get("case_type", "기타")
    core_issues = analysis.get("core_issue", [])
    required_facts = analysis.get("required_facts", [])
    risk_flags = analysis.get("risk_flags", [])
    
    st.markdown(f"### 📋 케이스 유형: `{case_type}`")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 핵심 쟁점")
        for issue in core_issues:
            st.markdown(f"- {issue}")
        
        st.markdown("#### ⚠️ 리스크 플래그")
        for risk in risk_flags:
            st.markdown(f"- 🔴 {risk}")
    
    with col2:
        st.markdown("#### ❓ 추가 확인 필요 사항")
        for fact in required_facts:
            st.markdown(f"- {fact}")


def render_procedure_timeline(procedure: dict) -> None:
    """
    절차 타임라인 렌더링
    
    Args:
        procedure: ProcedurePlanner.plan() 결과
    """
    timeline = procedure.get("timeline", [])
    
    for step in timeline:
        step_num = step.get("step", 0)
        name = step.get("name", "")
        goal = step.get("goal", "")
        actions = step.get("actions", [])
        legal_note = step.get("legal_note", "")
        
        with st.expander(f"📍 단계 {step_num}: {name}", expanded=(step_num == 1)):
            st.markdown(f"**목표**: {goal}")
            st.markdown("**행동**:")
            for action in actions:
                st.markdown(f"  - {action}")
            if legal_note:
                st.info(f"💡 {legal_note}")
