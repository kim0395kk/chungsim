# streamlit_app.py
# -*- coding: utf-8 -*-
import json
import os
import re
import time
import uuid
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta
from html import escape as _escape
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# ---------------------------
# Optional deps
# ---------------------------
try:
    import requests
except Exception:
    requests = None

# Vertex AI imports
vertexai = None
GenerativeModel = None
try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from groq import Groq
except Exception:
    Groq = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from supabase import create_client
except Exception:
    create_client = None


# =========================================================
# 0) SETTINGS
# =========================================================
APP_VERSION = "2026-01-15-agentboost-fixed"
MAX_FOLLOWUP_Q = 5
ADMIN_EMAIL = "kim0395kk@korea.kr"
LAW_BOT_SEARCH_URL = "https://www.law.go.kr/LSW/ais/searchList.do?query="

# 모델별 토큰 가격 ($/1M tokens)
MODEL_PRICING = {
    "gemini-2.5-flash": 0.15,
    "gemini-2.5-flash-lite": 0.075,
    "gemini-2.0-flash": 0.10,
    "gemini-2.0-flash (Gemini API)": 0.10,
    "gemini-2.5-flash (Gemini API)": 0.15,
    "gemini-2.5-flash (Vertex AI)": 0.15,
    "llama-3.3-70b-versatile": 0.59,
    "llama-3.3-70b-versatile (Groq)": 0.59,
    "(unknown)": 0.10,
}

# 업무별 모델 라우팅
MODEL_WORK_INSTRUCTION = "gemini-2.0-flash-lite"
MODEL_REVISION = "gemini-2.5-flash"

# 선택적 모듈 임포트
try:
    from govable_ai.features.duty_manual import render_duty_manual_button
except Exception:
    def render_duty_manual_button(*args, **kwargs):
        pass

try:
    from govable_ai.features.document_revision import render_revision_sidebar_button, run_revision_workflow
except Exception:
    def render_revision_sidebar_button(*args, **kwargs):
        pass
    def run_revision_workflow(*args, **kwargs):
        return {"error": "Document revision module is not available"}

# 환각 탐지 모듈 임포트
from hallucination_detection import (
    detect_hallucination,
    detect_hallucination_cached,
    get_text_hash,
    analyze_petition_priority,
    generate_processing_checklist,
    generate_response_draft,
    render_hallucination_report,
    render_verification_log,
    render_highlighted_text
)
from govable_ai.ui.premium_animations import render_revision_animation
from govable_ai.export import generate_official_docx, generate_guide_docx

try:
    from govable_ai.core.llm_service import LLMService
except Exception:
    # 더미 LLMService 클래스 - secrets 접근 없이 작동
    class LLMService:
        def __init__(self, *args, **kwargs):
            # 더미 클래스는 초기화 시 아무 작업도 하지 않음
            self.vertex_config = None
            self.gemini_key = None
            self.groq_key = None
        def is_available(self):
            return False
        def generate(self, *args, **kwargs):
            return "LLM service is not available"
        def generate_text(self, *args, **kwargs):
            return "LLM service is not available"
        def generate_json(self, *args, **kwargs):
            return {}

try:
    from govable_ai.config import get_secret, get_vertex_config
except Exception:
    def get_secret(*args, **kwargs):
        return None
    def get_vertex_config(*args, **kwargs):
        return None

# [NEW] Civil Engineering Imports
try:
    from civil_engineering.rag_system import load_rag_system
    from civil_engineering.dashboard import render_civil_dashboard
except ImportError:
    load_rag_system = None
    render_civil_dashboard = None

# Initialize LLM Service Globally
def _build_llm_service():
    try:
        return LLMService(
            vertex_config=get_vertex_config(),
            gemini_key=get_secret("general", "GEMINI_API_KEY"),
            groq_key=get_secret("general", "GROQ_API_KEY"),
        )
    except Exception as e:
        print(f"Warning: LLMService initialization failed: {e}")
        return LLMService()  # 더미/폴백 인스턴스


class _LazyLLMService:
    """첫 호출 시 실제 LLMService를 초기화해 부팅 지연을 줄인다."""

    def __init__(self):
        self._svc = None

    def _get(self):
        if self._svc is None:
            self._svc = _build_llm_service()
        return self._svc

    def __getattr__(self, name):
        return getattr(self._get(), name)


llm_service = _LazyLLMService()

# Heavy user / Long latency 임계값
HEAVY_USER_PERCENTILE = 95  # 상위 5% = 과다 사용자
LONG_LATENCY_THRESHOLD = 120  # 초

# =========================================================
# 1) HELPERS
# =========================================================
def make_lawbot_url(query: str) -> str:
    return LAW_BOT_SEARCH_URL + urllib.parse.quote((query or "").strip())

def shorten_one_line(text: str, max_len: int = 28) -> str:
    s = re.sub(r"\s+", " ", (text or "").strip())
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return int(len(text) * 0.7)

def safe_now_utc_iso():
    return datetime.utcnow().isoformat() + "Z"

def _safe_json_loads(text: str) -> Optional[Any]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        m = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception:
        return None
    return None

def strip_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return text

def ensure_anon_session_id() -> str:
    if "anon_session_id" not in st.session_state:
        st.session_state.anon_session_id = str(uuid.uuid4())
    return st.session_state.anon_session_id

def is_admin_user(email: str) -> bool:
    """
    1) 하드코딩 ADMIN_EMAIL
    2) (선택) app_admins 테이블 체크 결과를 session_state에 저장해두면 반영
    """
    e = (email or "").strip().lower()
    if e == ADMIN_EMAIL.lower():
        return True
    return bool(st.session_state.get("is_admin_db", False))

def md_bold_to_html_safe(text: str) -> str:
    s = text or ""
    out = []
    pos = 0
    for m in re.finditer(r"\*\*(.+?)\*\*", s):
        out.append(_escape(s[pos:m.start()]))
        out.append(f"<b>{_escape(m.group(1))}</b>")
        pos = m.end()
    out.append(_escape(s[pos:]))
    html = "".join(out).replace("\n", "<br>")
    return html

def mask_sensitive(text: str) -> str:
    if not text:
        return ""
    t = text
    t = re.sub(r"\b0\d{1,2}-\d{3,4}-\d{4}\b", "0**-****-****", t)
    t = re.sub(r"\b\d{6}-\d{7}\b", "******-*******", t)
    t = re.sub(r"\b\d{2,3}[가-힣]\d{4}\b", "***(차량번호)", t)
    return t

def _short_for_context(s: str, limit: int = 2500) -> str:
    s = (s or "").strip()
    if len(s) <= limit:
        return s
    return s[:limit] + "\n...(생략)"

def render_header(title):
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

def render_civil_response_panel(payload: dict) -> None:
    """토목직 답변을 사용자 친화적으로 렌더링."""
    answer_text = (payload or {}).get("content") or (payload or {}).get("answer") or ""
    summary = (payload or {}).get("summary") or ""
    sources = (payload or {}).get("sources") or []
    fact_rows = (payload or {}).get("fact_rows") or []
    confidence = float((payload or {}).get("confidence") or 0.0)
    quality = (payload or {}).get("quality") or {}
    retrieval_meta = (payload or {}).get("retrieval_meta") or {}

    c1, c2, c3 = st.columns(3)
    c1.metric("신뢰도", f"{int(confidence * 100)}%")
    c2.metric("근거 수", str(len(sources)))
    c3.metric("추출 항목", str(len(fact_rows)))

    if retrieval_meta.get("fallback_general_knowledge"):
        st.warning("내부 문서 근거가 부족해 일반 지식 기반 답변이 포함되었습니다. 최종 판단 전 원문 확인이 필요합니다.")
    if quality.get("location_only_risk"):
        st.warning("위치 편중 응답 위험을 감지해 핵심 항목(면적/예산/기간 등)을 보강했습니다.")
    if confidence < 0.45:
        st.info("신뢰도가 낮습니다. 질문에 대상/기간/사업명 등 구체 정보를 추가하면 정확도가 올라갑니다.")

    tab1, tab2, tab3 = st.tabs(["핵심 답변", "근거 자료", "구조화 데이터"])
    with tab1:
        if summary:
            st.caption(f"요약: {summary}")
        st.markdown(answer_text)
    with tab2:
        if sources:
            for src in sources:
                st.markdown(f"- {src}")
        else:
            st.caption("근거 자료가 없습니다.")
    with tab3:
        if fact_rows:
            st.table(fact_rows)
        else:
            st.caption("질문과 직접 매칭된 구조화 데이터가 없습니다.")

def _compute_result_confidence(res: dict) -> Tuple[int, str, List[str]]:
    """결과 품질 신호를 점수화해 신뢰도 레이블을 계산."""
    score = 40
    warnings: List[str] = []

    law_pack = (res or {}).get("law_pack") or {}
    law_items = law_pack.get("items") or []
    valid_links = 0
    for it in law_items:
        if (it or {}).get("current_link"):
            valid_links += 1
    if valid_links >= 2:
        score += 25
    elif valid_links == 1:
        score += 12
    else:
        warnings.append("법령 원문 링크가 부족합니다. 최종 처리 전 원문 대조를 권장합니다.")

    analysis = (res or {}).get("analysis") or {}
    if analysis.get("core_issue"):
        score += 10
    else:
        warnings.append("핵심 쟁점이 충분히 구조화되지 않았습니다.")

    procedure = (res or {}).get("procedure") or {}
    timeline = procedure.get("timeline") or []
    if len(timeline) >= 2:
        score += 10
    else:
        warnings.append("절차 플랜 단계가 적어 누락 가능성이 있습니다.")

    strategy = (res or {}).get("strategy") or ""
    if str(strategy).strip():
        score += 8
    else:
        warnings.append("처리 가이드가 비어 있습니다.")

    if (res or {}).get("search"):
        score += 7

    score = max(0, min(100, score))
    if score >= 80:
        label = "높음"
    elif score >= 60:
        label = "중간"
    else:
        label = "낮음"
    return score, label, warnings

FEEDBACK_REASON_OPTIONS = [
    "근거 링크 부족",
    "답변이 장황함",
    "답변이 너무 짧음",
    "절차 플랜 부족",
    "법령 설명 부족",
    "실무 적용성 낮음",
    "결과 형식 불만족",
    "속도 느림",
]

def _infer_feedback_tags(score: int, selected_reasons: List[str], comment: str) -> List[str]:
    """낮은 점수 피드백에 대해 자동 원인 태그를 보강."""
    tags = list(selected_reasons or [])
    c = (comment or "").strip()
    lc = c.lower()

    if score <= 2 and not tags:
        tags.extend(["근거 링크 부족", "절차 플랜 부족"])
    if "근거" in c or "링크" in c:
        tags.append("근거 링크 부족")
    if "길" in c and ("너무" in c or "장황" in c):
        tags.append("답변이 장황함")
    if "짧" in c:
        tags.append("답변이 너무 짧음")
    if "절차" in c or "단계" in c:
        tags.append("절차 플랜 부족")
    if "법령" in c or "조문" in c:
        tags.append("법령 설명 부족")
    if "느리" in c or "slow" in lc:
        tags.append("속도 느림")

    # 순서 보존 중복 제거
    out: List[str] = []
    seen = set()
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def _apply_autotune_from_feedback(tags: List[str]) -> None:
    """피드백 태그를 다음 요청 UI 가이드에 반영."""
    prefs = st.session_state.get("ux_autotune_prefs", {})
    for t in tags:
        if t == "근거 링크 부족":
            prefs["emphasize_evidence"] = True
        elif t == "답변이 장황함":
            prefs["prefer_concise"] = True
        elif t == "답변이 너무 짧음":
            prefs["prefer_detailed"] = True
        elif t == "절차 플랜 부족":
            prefs["emphasize_procedure"] = True
        elif t == "법령 설명 부족":
            prefs["emphasize_law"] = True
        elif t == "실무 적용성 낮음":
            prefs["emphasize_actionable"] = True
        elif t == "속도 느림":
            prefs["prefer_fast_mode"] = True
    st.session_state["ux_autotune_prefs"] = prefs

def render_autotune_hint() -> None:
    prefs = st.session_state.get("ux_autotune_prefs", {})
    if not prefs:
        return
    hints: List[str] = []
    if prefs.get("emphasize_evidence"):
        hints.append("근거 링크/법령 원문을 우선 노출")
    if prefs.get("prefer_concise"):
        hints.append("응답 길이를 간결하게")
    if prefs.get("prefer_detailed"):
        hints.append("핵심 근거와 절차를 더 자세히")
    if prefs.get("emphasize_procedure"):
        hints.append("단계별 절차 플랜 강화")
    if prefs.get("emphasize_law"):
        hints.append("법령/조문 설명 강화")
    if prefs.get("emphasize_actionable"):
        hints.append("실무 적용 가능한 액션 중심")
    if prefs.get("prefer_fast_mode"):
        hints.append("응답 속도 우선 모드 권장")
    if not hints:
        return

    st.markdown(
        """
        <div style='background: #f0fdf4; border-left: 4px solid #16a34a; padding: 0.8rem 0.9rem; border-radius: 8px; margin: 0.5rem 0 0.8rem 0;'>
            <p style='margin: 0; color: #166534; font-weight: 700; font-size: 0.92rem;'>최근 피드백 기반 자동 튜닝 적용됨</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for h in hints[:5]:
        st.caption(f"• {h}")

def _autotune_instruction_text() -> str:
    prefs = st.session_state.get("ux_autotune_prefs", {})
    if not prefs:
        return ""
    lines: List[str] = []
    if prefs.get("emphasize_evidence"):
        lines.append("- 법령/근거 링크를 우선 제시")
    if prefs.get("prefer_concise"):
        lines.append("- 응답은 간결하게")
    if prefs.get("prefer_detailed"):
        lines.append("- 핵심 근거와 절차를 충분히 상세화")
    if prefs.get("emphasize_procedure"):
        lines.append("- 단계별 절차/기록 포인트를 강화")
    if prefs.get("emphasize_law"):
        lines.append("- 법령/조문 설명을 보강")
    if prefs.get("emphasize_actionable"):
        lines.append("- 실무 즉시 적용 가능한 액션 위주")
    if prefs.get("prefer_fast_mode"):
        lines.append("- 지연을 줄이기 위해 핵심만 우선 응답")
    if not lines:
        return ""
    return "\n\n[사용자 피드백 기반 우선 지침]\n" + "\n".join(lines)

def render_service_health(sb, llm_svc) -> None:
    llm_ok = bool(llm_svc and getattr(llm_svc, "is_available", lambda: False)())
    db_ok = bool(sb)
    net_ok = bool(requests)

    c1, c2, c3 = st.columns(3)
    c1.metric("LLM 상태", "정상" if llm_ok else "제한")
    c2.metric("DB 상태", "정상" if db_ok else "오프라인")
    c3.metric("외부 API", "가능" if net_ok else "제한")

    if not llm_ok:
        st.caption("참고: LLM 키/연결이 없어 일부 생성 기능이 제한될 수 있습니다.")

def render_trust_panel(res: dict) -> None:
    score, label, warnings = _compute_result_confidence(res or {})
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, #ecfeff 0%, #f0f9ff 100%);
                    border: 1px solid #bae6fd; border-radius: 12px; padding: 0.9rem 1rem; margin: 0.5rem 0 1rem 0;'>
            <div style='display:flex; justify-content:space-between; align-items:center; gap: 1rem;'>
                <div style='font-weight:700; color:#0c4a6e;'>신뢰도 점수: {score}/100</div>
                <div style='font-weight:700; color:#075985;'>판정: {label}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if warnings:
        with st.expander("⚠️ 검토 권장 항목", expanded=(label != "높음")):
            for w in warnings:
                st.write("- ", w)

def render_feedback_panel(
    sb,
    archive_id: Optional[str],
    mode: str = "default",
    confidence_score: Optional[int] = None,
) -> None:
    feedback_key = f"feedback_submitted_{archive_id or 'temp'}_{mode}"
    if st.session_state.get(feedback_key):
        st.success("피드백이 저장되었습니다. 감사합니다.")
        return

    with st.expander("🗳️ 사용자 만족도 피드백", expanded=False):
        score = st.slider("이번 결과 만족도", min_value=1, max_value=5, value=4, key=f"fb_score_{archive_id}_{mode}")
        reasons = st.multiselect(
            "불만/개선 원인 태그 (복수 선택 가능)",
            options=FEEDBACK_REASON_OPTIONS,
            key=f"fb_reasons_{archive_id}_{mode}",
        )
        comment = st.text_input("개선 요청(선택)", key=f"fb_comment_{archive_id}_{mode}", placeholder="예: 근거 링크를 더 강조해 주세요.")
        if st.button("피드백 제출", key=f"fb_submit_{archive_id}_{mode}", use_container_width=True):
            inferred_tags = _infer_feedback_tags(score=score, selected_reasons=reasons, comment=comment)
            _apply_autotune_from_feedback(inferred_tags)
            if sb:
                log_event(
                    sb,
                    "ux_feedback",
                    archive_id=archive_id,
                    meta={
                        "score": score,
                        "comment": comment[:300],
                        "mode": mode,
                        "reason_tags": inferred_tags,
                        "confidence_score": confidence_score,
                    },
                )
            st.session_state[feedback_key] = True
            st.rerun()


# =========================================================
# PERF HELPERS (Cache)
# =========================================================
@st.cache_data(ttl=300, show_spinner=False)
def _cached_naver_news(
    query: str,
    display: int = 10,
    sort: str = "sim",
    _client_id: Optional[str] = None,
    _client_secret: Optional[str] = None,
) -> List[dict]:
    if not requests or not _client_id or not _client_secret or not query:
        return []
    headers = {"X-Naver-Client-Id": _client_id, "X-Naver-Client-Secret": _client_secret}
    params = {"query": query, "display": display, "sort": sort}
    res = requests.get("https://openapi.naver.com/v1/search/news.json", headers=headers, params=params, timeout=8)
    res.raise_for_status()
    return res.json().get("items", []) or []


@st.cache_data(ttl=600, show_spinner=False)
def _cached_law_search_first_mst(api_id: str, law_name: str) -> Optional[str]:
    if not requests or not api_id or not law_name:
        return None
    params = {"OC": api_id, "target": "law", "type": "XML", "query": law_name, "display": 1}
    res = requests.get("https://www.law.go.kr/DRF/lawSearch.do", params=params, timeout=6)
    res.raise_for_status()
    root = ET.fromstring(res.content)
    law_node = root.find(".//law")
    if law_node is None:
        return None
    return (law_node.findtext("법령일련번호") or "").strip() or None


@st.cache_data(ttl=600, show_spinner=False)
def _cached_law_detail_xml(api_id: str, mst_id: str) -> Optional[str]:
    if not requests or not api_id or not mst_id:
        return None
    detail_params = {"OC": api_id, "target": "law", "type": "XML", "MST": mst_id}
    res_detail = requests.get("https://www.law.go.kr/DRF/lawService.do", params=detail_params, timeout=10)
    res_detail.raise_for_status()
    return res_detail.text


@st.cache_data(ttl=20, show_spinner=False)
def _cached_db_fetch_history(_sb, anon_id: str, limit: int = 80) -> List[dict]:
    _sb.postgrest.headers.update({'x-session-id': anon_id})
    resp = (
        _sb.table("work_archive")
        .select("id,prompt,created_at,user_email,anon_session_id")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return getattr(resp, "data", None) or []


@st.cache_data(ttl=60, show_spinner=False)
def _cached_db_fetch_payload(_sb, anon_id: str, archive_id: str) -> Optional[dict]:
    _sb.postgrest.headers.update({'x-session-id': anon_id})
    resp = (
        _sb.table("work_archive")
        .select("id,prompt,payload,created_at,user_email,anon_session_id")
        .eq("id", archive_id)
        .limit(1)
        .execute()
    )
    data = getattr(resp, "data", None) or []
    return data[0] if data else None


@st.cache_data(ttl=60, show_spinner=False)
def _cached_db_fetch_followups(_sb, anon_id: str, archive_id: str) -> List[dict]:
    _sb.postgrest.headers.update({'x-session-id': anon_id})
    resp = (
        _sb.table("work_followups")
        .select("turn,role,content,created_at")
        .eq("archive_id", archive_id)
        .order("turn", desc=False)
        .execute()
    )
    return getattr(resp, "data", None) or []


def _clear_history_cache() -> None:
    try:
        _cached_db_fetch_history.clear()
        _cached_db_fetch_payload.clear()
        _cached_db_fetch_followups.clear()
    except Exception:
        pass


# =========================================================
# 2) STYLES  (✅ 여기 CSS/디자인은 네가 준 그대로. 변경 없음)
# =========================================================
st.set_page_config(
    page_title="AI 행정관 Pro - Govable AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 후속 질문창 플로팅 스타일
st.markdown("""
<style>
    /* 채팅 입력창 컨테이너 스타일링 */
    [data-testid="stChatInput"] {
        position: fixed !important;
        bottom: 40px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        width: 700px !important;
        max-width: 90% !important;
        z-index: 9999 !important;
        background-color: white !important;
        border-radius: 15px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15) !important;
        border: 2px solid #4A90E2 !important;
        padding: 10px !important;
    }
    
    /* 입력창 내부 스타일 */
    [data-testid="stChatInput"] textarea {
        background-color: transparent !important;
    }
    
    /* 하단 여백 확보 */
    .main .block-container {
        padding-bottom: 150px !important;
    }
</style>
""", unsafe_allow_html=True)
st.markdown(
    """
<style>
    /* ====================== */
    /* Design Tokens */
    /* ====================== */
    :root {
        /* Colors - Primary Palette */
        --primary-50: #eff6ff;
        --primary-100: #dbeafe;
        --primary-200: #bfdbfe;
        --primary-500: #3b82f6;
        --primary-600: #2563eb;
        --primary-700: #1d4ed8;
        --primary-800: #1e40af;
        
        /* Colors - Neutral Palette */
        --neutral-50: #f9fafb;
        --neutral-100: #f3f4f6;
        --neutral-200: #e5e7eb;
        --neutral-300: #d1d5db;
        --neutral-400: #9ca3af;
        --neutral-500: #6b7280;
        --neutral-600: #4b5563;
        --neutral-700: #374151;
        --neutral-800: #1f2937;
        --neutral-900: #111827;
        
        /* Colors - Semantic */
        --success-500: #10b981;
        --success-600: #059669;
        --warning-500: #f59e0b;
        --error-500: #ef4444;
        --error-600: #dc2626;
        
        /* Spacing */
        --space-xs: 0.25rem;
        --space-sm: 0.5rem;
        --space-md: 1rem;
        --space-lg: 1.5rem;
        --space-xl: 2rem;
        --space-2xl: 3rem;
        
        /* Border Radius */
        --radius-sm: 0.375rem;
        --radius-md: 0.5rem;
        --radius-lg: 0.75rem;
        --radius-xl: 1rem;
        
        /* Shadows */
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        --shadow-2xl: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        
        /* Typography */
        --font-serif: 'Batang', 'Nanum Myeongjo', serif;
        --font-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans KR', sans-serif;
    }
    
    /* ====================== */
    /* Base Styles */
    /* ====================== */
    .stApp { 
        background: linear-gradient(135deg, var(--neutral-50) 0%, var(--primary-50) 100%);
        font-family: var(--font-sans);
    }
    
    /* ====================== */
    /* Document Paper Style */
    /* ====================== */
    .paper-sheet {
        background-color: white;
        width: 100%;
        max-width: 210mm;
        min-height: 297mm;
        padding: 28mm;
        margin: var(--space-xl) auto;
        box-shadow: var(--shadow-2xl);
        font-family: var(--font-serif);
        color: var(--neutral-900);
        line-height: 1.8;
        position: relative;
        border-radius: var(--radius-sm);
    }

    .doc-header { 
        text-align: center; 
        font-size: 24pt; 
        font-weight: 900; 
        margin-bottom: var(--space-2xl); 
        letter-spacing: 3px;
        color: var(--neutral-900);
        padding-bottom: var(--space-lg);
        border-bottom: 3px double var(--neutral-800);
    }
    
    .doc-info { 
        display: flex; 
        justify-content: space-between; 
        font-size: 11pt; 
        background: var(--neutral-50);
        padding: var(--space-lg);
        border-radius: var(--radius-md);
        margin-bottom: var(--space-xl);
        gap: var(--space-md);
        flex-wrap: wrap;
        border-left: 4px solid var(--primary-600);
    }
    
    .doc-info span {
        font-weight: 600;
        color: var(--neutral-700);
    }
    
    .doc-body { 
        font-size: 12pt; 
        text-align: justify; 
        white-space: normal;
        color: var(--neutral-800);
    }
    
    .doc-footer { 
        text-align: center; 
        font-size: 22pt; 
        font-weight: bold; 
        margin-top: 100px; 
        letter-spacing: 6px;
        color: var(--neutral-900);
    }
    
    .stamp { 
        position: absolute; 
        bottom: 85px; 
        right: 80px; 
        border: 4px solid #dc2626; 
        color: #dc2626; 
        padding: 10px 18px; 
        font-size: 14pt; 
        font-weight: 900; 
        transform: rotate(-15deg); 
        opacity: 0.9; 
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.95);
        box-shadow: 
            0 8px 24px rgba(220, 38, 38, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.5);
        animation: stampPulse 2s ease-in-out infinite;
    }

    /* ====================== */
    /* Lawbot Button */
    /* ====================== */
    .lawbot-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 16px;
        padding: 0.9rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 
            0 8px 32px rgba(102, 126, 234, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
        display: inline-block;
        width: 100%;
        text-align: center;
        text-decoration: none !important;
    }
    
    .lawbot-btn::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .lawbot-btn:hover::before {
        width: 400px;
        height: 400px;
    }
    
    .lawbot-btn:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 
            0 12px 48px rgba(102, 126, 234, 0.6),
            0 0 40px rgba(118, 75, 162, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
        border-color: rgba(255, 255, 255, 0.5);
        color: white !important;
    }
    
    .lawbot-sub { 
        font-size: 0.8rem; 
        opacity: 0.9; 
        margin-top: var(--space-sm); 
        display: block; 
        color: rgba(255,255,255,0.95) !important; 
        font-weight: 500;
        letter-spacing: 0.2px;
    }

    /* ====================== */
    /* Sidebar Styles */
    /* ====================== */
    div[data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid var(--neutral-200);
        min-width: 60px !important;
        max-width: 800px !important;
        resize: horizontal !important;
        overflow: auto !important;
        transition: width 0.1s ease;
    }
    
    /* 사이드바 내부 영역 */
    section[data-testid="stSidebar"] > div {
        min-width: 60px !important;
    }
    
    /* 사이드바 접기 버튼 숨기기 */
    button[data-testid="stSidebarCollapseButton"],
    div[data-testid="stSidebarCollapsedControl"],
    button[data-testid="baseButton-headerNoPadding"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }
    
    /* 사이드바 항상 표시 강제 */
    section[data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        transform: none !important;
        left: 0 !important;
        min-width: 60px !important;
    }
    
    /* 접힌 상태에서도 최소 너비 유지 (한 글자 이상) */
    section[data-testid="stSidebar"][aria-expanded="false"] {
        width: 60px !important;
        min-width: 60px !important;
        transform: translateX(0) !important;
    }
    
    /* 사이드바 리사이즈 핸들 스타일 (우측 드래그 영역) */
    div[data-testid="stSidebar"]::after {
        content: '⋮';
        position: absolute;
        right: 0;
        top: 50%;
        transform: translateY(-50%);
        width: 12px;
        height: 60px;
        cursor: ew-resize;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.1));
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        color: var(--neutral-400);
        border-radius: 0 4px 4px 0;
    }
    
    div[data-testid="stSidebar"]:hover::after {
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.3));
        color: var(--primary-600);
    }



    div[data-testid="stSidebar"] button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        font-weight: 700;
        border-radius: 10px;
    }


    
    div[data-testid="stSidebar"] button[kind="secondary"] {
        width: 100%;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: var(--space-md) !important;
        border-radius: var(--radius-lg) !important;
        border: 1px solid var(--neutral-200) !important;
        background: white !important;
        color: var(--neutral-800) !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        margin-bottom: var(--space-xs) !important;
    }
    
    div[data-testid="stSidebar"] button[kind="secondary"]:hover { 
        background: var(--neutral-50) !important;
        border-color: var(--primary-300) !important;
        transform: translateX(2px);
    }

    /* ====================== */
    /* Form Elements */
    /* ====================== */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: var(--radius-lg) !important;
        border: 2px solid var(--neutral-200) !important;
        padding: var(--space-md) !important;
        font-family: var(--font-sans) !important;
        transition: all 0.2s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary-500) !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    /* ====================== */
    /* Buttons */
    /* ====================== */
    .stButton > button {
        border-radius: var(--radius-lg) !important;
        padding: var(--space-md) var(--space-xl) !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        border: none !important;
    }
    
    /* Default Primary Button (Red - for Main Area) */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%) !important;
        color: white !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 16px !important;
        padding: 0.9rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 
            0 8px 32px rgba(239, 68, 68, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stButton > button[kind="primary"]::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton > button[kind="primary"]:hover::before {
        width: 400px;
        height: 400px;
    }
    
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-4px) scale(1.02) !important;
        box-shadow: 
            0 12px 48px rgba(239, 68, 68, 0.6),
            0 0 40px rgba(185, 28, 28, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
        border-color: rgba(255, 255, 255, 0.5) !important;
    }
    
    .stButton > button[kind="primary"]:active {
        transform: scale(0.98) !important;
    }

    /* Sidebar Primary Button (White Glassmorphism - for New Chat) */
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: rgba(255, 255, 255, 0.7) !important;
        color: #1f2937 !important;
        border: 1px solid rgba(255, 255, 255, 0.8) !important;
        border-radius: 16px !important;
        padding: 0.9rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        box-shadow: 
            0 4px 6px -1px rgba(0, 0, 0, 0.1), 
            0 2px 4px -1px rgba(0, 0, 0, 0.06),
            inset 0 1px 0 rgba(255, 255, 255, 0.5) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button[kind="primary"]::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(0, 0, 0, 0.05);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover::before {
        width: 400px;
        height: 400px;
    }
    
    section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) scale(1.01) !important;
        background: rgba(255, 255, 255, 0.9) !important;
        box-shadow: 
            0 10px 15px -3px rgba(0, 0, 0, 0.1), 
            0 4px 6px -2px rgba(0, 0, 0, 0.05),
            inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
        border-color: white !important;
        color: black !important;
    }
    


    /* ====================== */
    /* Expanders */
    /* ====================== */
    .streamlit-expanderHeader {
        background: var(--neutral-50) !important;
        border-radius: var(--radius-lg) !important;
        padding: var(--space-md) !important;
        font-weight: 600 !important;
        border: 1px solid var(--neutral-200) !important;
    }
    
    /* ====================== */
    /* Info/Warning Boxes */
    /* ====================== */
    .stAlert {
        border-radius: var(--radius-lg) !important;
        border: none !important;
        padding: var(--space-lg) !important;
    }
    
    /* ====================== */
    /* Chat Messages */
    /* ====================== */
    .stChatMessage {
        border-radius: var(--radius-lg) !important;
        padding: var(--space-lg) !important;
        margin-bottom: var(--space-md) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    /* ====================== */
    /* Chat Input - Enhanced Visibility */
    /* ====================== */
    .stChatInputContainer {
        background: linear-gradient(135deg, var(--primary-50) 0%, white 100%) !important;
        border: 2px solid var(--primary-500) !important;
        border-radius: var(--radius-xl) !important;
        padding: var(--space-md) !important;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.1), var(--shadow-lg) !important;
        margin-top: var(--space-lg) !important;
        position: relative !important;
    }
    
    .stChatInputContainer::before {
        content: '💬 여기에 후속 질문을 입력하세요';
        position: absolute;
        top: -1.75rem;
        left: 0;
        font-size: 0.9rem;
        font-weight: 700;
        color: var(--primary-700);
        background: white;
        padding: 0.25rem 0.75rem;
        border-radius: var(--radius-md);
        border: 2px solid var(--primary-200);
        box-shadow: var(--shadow-sm);
    }
    
    .stChatInputContainer textarea {
        border: 2px solid var(--primary-300) !important;
        border-radius: var(--radius-lg) !important;
        background: white !important;
        font-size: 1rem !important;
        padding: var(--space-md) !important;
        transition: all 0.3s ease !important;
    }
    
    .stChatInputContainer textarea:focus {
        border-color: var(--primary-600) !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
        transform: scale(1.01);
    }
    
    .stChatInputContainer textarea::placeholder {
        color: var(--primary-400) !important;
        font-weight: 500 !important;
    }

    /* ====================== */
    /* Headers & Text */
    /* ====================== */
    h1, h2, h3 {
        color: var(--neutral-900) !important;
        font-weight: 700 !important;
    }
        /* ====================== */
    /* Hide Default Elements */
    /* ====================== */
    header [data-testid="stToolbar"] { display: none !important; }
    header [data-testid="stDecoration"] { display: none !important; }
    header { height: 0px !important; }
    footer { display: none !important; }
    div[data-testid="stStatusWidget"] { display: none !important; }
    


    .agent-log { 
        font-family: 'Inter', 'Consolas', monospace; 
        font-size: 0.9rem; 
        padding: 14px 20px; 
        border-radius: 16px; 
        margin-bottom: 12px; 
        backdrop-filter: blur(20px) saturate(180%);
        border: 2px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .agent-log::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.5s;
    }
    
    .agent-log:hover::before {
        left: 100%;
    }
    
    .agent-log:hover {
        transform: translateX(8px) scale(1.02);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
    }
    
    .log-legal { 
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.25), rgba(102, 126, 234, 0.15)); 
        color: #3730a3; 
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.2);
    }
    
    .log-legal:hover {
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        border-left-color: #5a67d8;
    }
    
    .log-search { 
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.25), rgba(79, 172, 254, 0.15)); 
        color: #0c4a6e; 
        border-left: 5px solid #4facfe;
        box-shadow: 0 4px 20px rgba(79, 172, 254, 0.2);
    }
    
    .log-search:hover {
        box-shadow: 0 8px 32px rgba(79, 172, 254, 0.3);
        border-left-color: #0ea5e9;
    }
    
    .log-strat { 
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.25), rgba(168, 85, 247, 0.15)); 
        color: #581c87; 
        border-left: 5px solid #a855f7;
        box-shadow: 0 4px 20px rgba(168, 85, 247, 0.2);
    }
    
    .log-strat:hover {
        box-shadow: 0 8px 32px rgba(168, 85, 247, 0.3);
        border-left-color: #9333ea;
    }
    
    .log-calc { 
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.25), rgba(34, 197, 94, 0.15)); 
        color: #14532d; 
        border-left: 5px solid #22c55e;
        box-shadow: 0 4px 20px rgba(34, 197, 94, 0.2);
    }
    
    .log-calc:hover {
        box-shadow: 0 8px 32px rgba(34, 197, 94, 0.3);
        border-left-color: #16a34a;
    }
    
    .log-draft { 
        background: linear-gradient(135deg, rgba(251, 113, 133, 0.25), rgba(251, 113, 133, 0.15)); 
        color: #881337; 
        border-left: 5px solid #fb7185;
        box-shadow: 0 4px 20px rgba(251, 113, 133, 0.2);
    }
    
    .log-draft:hover {
        box-shadow: 0 8px 32px rgba(251, 113, 133, 0.3);
        border-left-color: #f43f5e;
    }
    
    .log-sys { 
        background: linear-gradient(135deg, rgba(148, 163, 184, 0.25), rgba(148, 163, 184, 0.15)); 
        color: #1e293b; 
        border-left: 5px solid #94a3b8;
        box-shadow: 0 4px 20px rgba(148, 163, 184, 0.2);
    }
    
    .log-sys:hover {
        box-shadow: 0 8px 32px rgba(148, 163, 184, 0.3);
        border-left-color: #64748b;
    }

    /* ====================== */
    /* Spinner & Active Log Animation */
    /* ====================== */
    @keyframes spin { 
        0% { transform: rotate(0deg); } 
        100% { transform: rotate(360deg); } 
    }
    
    @keyframes pulse-active { 
        0% { border-color: rgba(59, 130, 246, 0.3); box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.2); } 
        50% { border-color: rgba(59, 130, 246, 0.8); box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.1); } 
        100% { border-color: rgba(59, 130, 246, 0.3); box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.2); } 
    }

    .spinner-icon {
        display: inline-block;
        animation: spin 1.2s linear infinite;
        margin-right: 8px;
        font-size: 1.1rem;
    }

    .log-active {
        animation: pulse-active 2s infinite;
        background: rgba(255, 255, 255, 0.95) !important;
        border-width: 2px !important;
        transform: scale(1.01);
    }
""",
    unsafe_allow_html=True,
)

# =========================================================
# 3) SERVICES
# =========================================================
def get_secret(path1: str, path2: str = "") -> Optional[str]:
    try:
        if path2:
            v = st.secrets.get(path1, {}).get(path2)
            return v if v is not None else os.environ.get(path2)
        v = st.secrets.get(path1)
        return v if v is not None else os.environ.get(path1)
    except Exception:
        return os.environ.get(path2 or path1)

def get_general_secret(key: str) -> Optional[str]:
    try:
        return (st.secrets.get("general", {}) or {}).get(key) or st.secrets.get(key)
    except Exception:
        # secrets.toml이 없는 경우
        return None

def get_supabase():
    if "sb" in st.session_state and st.session_state.sb is not None:
        return st.session_state.sb
    if not create_client:
        st.session_state.sb = None
        return None

    url = get_secret("supabase", "SUPABASE_URL") or get_secret("SUPABASE_URL")
    key = (
        get_secret("supabase", "SUPABASE_ANON_KEY")
        or get_secret("supabase", "SUPABASE_KEY")
        or get_secret("SUPABASE_ANON_KEY")
        or get_secret("SUPABASE_KEY")
    )
    if not (url and key):
        st.session_state.sb = None
        return None

    st.session_state.sb = create_client(url, key)
    return st.session_state.sb

def get_auth_user(sb):
    try:
        u = sb.auth.get_user()
        if isinstance(u, dict):
            return u.get("user") or u
        if hasattr(u, "user"):
            return u.user
        return u
    except Exception:
        return None

def _refresh_admin_flag(sb, email: str):
    """로그인 직후 app_admins 테이블로 관리자 여부 동기화"""
    st.session_state.is_admin_db = False
    if not sb or not email:
        return
    try:
        r = sb.table("app_admins").select("user_email").eq("user_email", email.strip()).limit(1).execute()
        st.session_state.is_admin_db = bool(getattr(r, "data", None) or [])
    except Exception:
        st.session_state.is_admin_db = False

def touch_session(sb):
    if not sb:
        return
    anon_id = ensure_anon_session_id()
    user_email = st.session_state.get("user_email") if st.session_state.get("logged_in") else None
    user_id = None
    user = get_auth_user(sb)
    if user and isinstance(user, dict):
        user_id = user.get("id")

    payload = {
        "session_id": anon_id,
        "last_seen": datetime.utcnow().isoformat() + "Z",
        "user_id": user_id,
        "user_email": user_email,
        "meta": {"app_ver": APP_VERSION},
    }
    try:
        sb.table("app_sessions").upsert(payload, on_conflict="session_id").execute()
    except Exception:
        pass

def log_event(sb, event_type: str, archive_id: Optional[str] = None, meta: Optional[dict] = None):
    if not sb:
        return
    
    try:
        # 1. 익명 ID 가져오기
        anon_id = str(ensure_anon_session_id())

        # 2. [핵심 수정] 출입증(헤더) 제출 (이게 없어서 에러가 났던 것)
        sb.postgrest.headers.update({'x-session-id': anon_id})

        # 3. 로그인 정보 확인 (하이브리드 체크)
        user = get_auth_user(sb)
        server_email = None
        server_user_id = None

        if user:
            if isinstance(user, dict):
                server_user_id = user.get("id")
                server_email = user.get("email")
            else:
                server_user_id = getattr(user, "id", None)
                server_email = getattr(user, "email", None)
        
        # 서버 조회 실패 시 세션 정보 사용
        final_email = server_email if server_email else st.session_state.get("user_email")
        final_user_id = server_user_id 

        row = {
            "event_type": event_type,
            "archive_id": archive_id,
            "user_id": final_user_id,
            "user_email": final_email,
            "anon_session_id": anon_id,
            "meta": meta or {},
        }
        
        sb.table("app_events").insert(row).execute()
        
    except Exception:
        pass


def log_api_call(
    sb,
    api_type: str,
    model_name: Optional[str] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    latency_ms: int = 0,
    success: bool = True,
    error_message: Optional[str] = None,
    request_summary: Optional[str] = None,
    response_summary: Optional[str] = None,
    archive_id: Optional[str] = None,
):
    """
    개별 API 호출 기록 (법령API, 네이버검색, LLM 등)
    """
    if not sb:
        return

    try:
        # 1. 익명 ID 가져오기
        anon_id = str(ensure_anon_session_id())

        # 2. [핵심 수정] 여기도 출입증(헤더) 제출 필수!
        sb.postgrest.headers.update({'x-session-id': anon_id})

        # 3. 로그인 정보 확인
        user = get_auth_user(sb)
        server_email = None
        if user:
            if isinstance(user, dict):
                server_email = user.get("email")
            else:
                server_email = getattr(user, "email", None)
        
        final_email = server_email if server_email else st.session_state.get("user_email")
        
        if not archive_id:
            archive_id = st.session_state.get("current_archive_id")
        
        row = {
            "archive_id": archive_id,
            "user_email": final_email,
            "anon_session_id": anon_id,
            "api_type": api_type,
            "model_name": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "latency_ms": latency_ms,
            "success": success,
            "error_message": error_message[:500] if error_message else None,
            "request_summary": request_summary[:200] if request_summary else None,
            "response_summary": response_summary[:200] if response_summary else None,
        }

        sb.table("api_call_logs").insert(row).execute()

    except Exception:
        pass


def log_document_revision(
    sb,
    original_text: str,
    revised_doc: dict,
    changelog: list,
    summary: str,
    model_used: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    execution_time: float = 0.0
):
    """
    기안/공고문 수정 내역 기록
    """
    if not sb:
        return
    
    try:
        anon_id = str(ensure_anon_session_id())
        sb.postgrest.headers.update({'x-session-id': anon_id})
        
        user = get_auth_user(sb)
        server_email = None
        if user:
            if isinstance(user, dict):
                server_email = user.get("email")
            else:
                server_email = getattr(user, "email", None)
        
        final_email = server_email if server_email else st.session_state.get("user_email")
        
        row = {
            "user_email": final_email,
            "anon_session_id": anon_id,
            "original_text": original_text[:1000],  # 첫 1000자만
            "revised_doc": revised_doc,
            "changelog": changelog,
            "summary": summary,
            "model_used": model_used,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "execution_time": execution_time,
        }
        
        sb.table("document_revisions").insert(row).execute()
    except Exception:
        pass


def log_lawbot_query(
    sb,
    query: str,
    results_count: int = 0,
    selected_laws: list = None,
    search_type: str = "law",
    execution_time: float = 0.0
):
    """
    당직봇 검색 내역 기록
    """
    if not sb:
        return
    
    try:
        anon_id = str(ensure_anon_session_id())
        sb.postgrest.headers.update({'x-session-id': anon_id})
        
        user = get_auth_user(sb)
        server_email = None
        if user:
            if isinstance(user, dict):
                server_email = user.get("email")
            else:
                server_email = getattr(user, "email", None)
        
        final_email = server_email if server_email else st.session_state.get("user_email")
        
        row = {
            "user_email": final_email,
            "anon_session_id": anon_id,
            "query": query[:500],  # 첫 500자만
            "results_count": results_count,
            "selected_laws": selected_laws or [],
            "search_type": search_type,
            "execution_time": execution_time,
        }
        
        sb.table("lawbot_queries").insert(row).execute()
    except Exception:
        pass


# StreamlitLLMService 는 govable_ai.core.llm_service.LLMService 로 단일화되었다.
# 인스턴스(`llm_service`)는 위쪽에서 _LazyLLMService 로 이미 초기화되어 있다.

class SearchService:
    """✅ 뉴스 중심 경량 검색"""
    def __init__(self):
        try:
            g = st.secrets.get("general", {})
            self.client_id = g.get("NAVER_CLIENT_ID")
            self.client_secret = g.get("NAVER_CLIENT_SECRET")
        except Exception:
            # secrets.toml이 없는 경우
            self.client_id = None
            self.client_secret = None
        self.news_url = "https://openapi.naver.com/v1/search/news.json"

    def _headers(self):
        return {"X-Naver-Client-Id": self.client_id, "X-Naver-Client-Secret": self.client_secret}

    def _clean_html(self, s: str) -> str:
        if not s:
            return ""
        s = re.sub(r"<[^>]+>", "", s)
        s = re.sub(r"&quot;", '"', s)
        s = re.sub(r"&lt;", "<", s)
        s = re.sub(r"&gt;", ">", s)
        s = re.sub(r"&amp;", "&", s)
        return s.strip()

    def _extract_keywords_llm(self, situation: str) -> str:
        # 1. LLM에게 핵심 명사 위주 2~3개만 요청
        prompt = f"""
상황: "{situation}"
위 상황에서 뉴스 검색에 사용할 가장 핵심적인 키워드 2~3개만 공백으로 구분하여 출력하시오.
조사, 서술어 제외. 명사 위주.
예: "공직선거법 시의원 포럼", "불법주정차 단속 과태료"
"""
        try:
            res = llm_service.generate_text(prompt, preferred_model=MODEL_WORK_INSTRUCTION).strip()
            # 2. 특수문자 제거 (마크다운, 괄호 등)
            res = re.sub(r'[#|\[\](){}"\'`]', "", res)
            res = re.sub(r'\s+', ' ', res).strip()
            return res
        except Exception:
            # 폴백: 앞부분 20자에서 특수문자 제거 후 리턴
            safe_fallback = re.sub(r'[#|\[\](){}"\'`]', "", situation[:20])
            return safe_fallback

    def search_news(self, query: str, top_k: int = 3) -> str:
        sb = get_supabase()
        start_time = time.time()
        
        if not self.client_id or not self.client_secret:
            return "⚠️ 네이버 API 키가 없습니다."
        if not query:
            return "⚠️ 검색어가 비었습니다."

        try:
            items = _cached_naver_news(
                query=query,
                display=10,
                sort="sim",
                _client_id=self.client_id,
                _client_secret=self.client_secret,
            )
            
            latency = int((time.time() - start_time) * 1000)
            log_api_call(sb, "naver_search", None, 0, 0, latency, True, None, query[:100], f"{len(items)} results")

            if not items:
                return f"🔍 `{query}` 관련 최신 사례가 없습니다."

            lines = [f"##### 📰 최신 뉴스 사례 (검색어: {query})", "---"]
            for it in items[:top_k]:
                title = self._clean_html(it.get("title", ""))
                desc = self._clean_html(it.get("description", ""))
                link = it.get("link", "#")
                lines.append(f"- **[{title}]({link})**\n  : {desc[:150]}...")
            return "\n".join(lines)
        except Exception as e:
            latency = int((time.time() - start_time) * 1000)
            log_api_call(sb, "naver_search", None, 0, 0, latency, False, str(e), query[:100])
            return f"검색 중 오류: {str(e)}"

    def search_precedents(self, situation: str, top_k: int = 3) -> str:
        keywords = self._extract_keywords_llm(situation)
        return self.search_news(keywords, top_k=top_k)


search_service = SearchService()


class LawOfficialService:
    """
    국가법령정보센터(law.go.kr) 공식 API 연동

    ✅ 후속질문에서 발생한 '링크는 줬는데 법령이 없다' 오류 원인:
    - lawService.do?ID=... 조합이 환경/값에 따라 불일치하는 경우가 있음(특히 000213 같은 값)
    - 해결: 검색 결과의 MST(법령일련번호)를 기반으로 링크를 생성(가장 안정적)
      => https://www.law.go.kr/DRF/lawService.do?OC=...&target=law&MST=<mst>&type=HTML
    - efYd(시행일) 파라미터는 넣지 않아서 "현행 아님" 문제를 최대한 회피
    """
    def __init__(self):
        self.api_id = get_general_secret("LAW_API_ID")
        self.base_url = "https://www.law.go.kr/DRF/lawSearch.do"
        self.service_url = "https://www.law.go.kr/DRF/lawService.do"

    def _make_current_link(self, mst_id: str) -> Optional[str]:
        if not self.api_id or not mst_id:
            return None
        # ✅ efYd 파라미터 미포함(현행 아닙니다 이슈 회피)
        return f"https://www.law.go.kr/DRF/lawService.do?OC={self.api_id}&target=law&MST={mst_id}&type=HTML"

    def ai_search(self, query: str, top_k: int = 6) -> List[dict]:
        if not requests or not self.api_id or not query:
            return []
        try:
            params = {"OC": self.api_id, "target": "aiSearch", "type": "XML", "query": query, "display": top_k}
            r = requests.get(self.base_url, params=params, timeout=8)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            out = []
            for node in root.findall(".//law"):
                name = (node.findtext("법령명") or node.findtext("lawName") or "").strip()
                if name:
                    out.append({"law_name": name})
            if not out:
                for tag in ["lawName", "법령명", "lawNm"]:
                    for node in root.findall(f".//{tag}"):
                        nm = (node.text or "").strip()
                        if nm:
                            out.append({"law_name": nm})
            seen = set()
            uniq = []
            for x in out:
                nm = x["law_name"]
                if nm not in seen:
                    seen.add(nm)
                    uniq.append(x)
            return uniq[:top_k]
        except Exception:
            return []

    def get_law_text(self, law_name, article_num=None, return_link: bool = False):
        sb = get_supabase()
        start_time = time.time()
        
        if not self.api_id:
            msg = "⚠️ API ID(OC)가 설정되지 않았습니다."
            return (msg, None) if return_link else msg

        # 1) 법령 검색 -> MST 확보
        mst_id = ""
        try:
            mst_id = _cached_law_search_first_mst(self.api_id, law_name) or ""
            if not mst_id:
                latency = int((time.time() - start_time) * 1000)
                log_api_call(sb, "law_api", None, 0, 0, latency, True, None, law_name[:50], "No results")
                msg = f"🔍 '{law_name}'에 대한 검색 결과가 없습니다."
                return (msg, None) if return_link else msg
            latency = int((time.time() - start_time) * 1000)
            log_api_call(sb, "law_api", None, 0, 0, latency, True, None, law_name[:50], f"MST: {mst_id}")
        except Exception as e:
            latency = int((time.time() - start_time) * 1000)
            log_api_call(sb, "law_api", None, 0, 0, latency, False, str(e), law_name[:50])
            msg = f"API 검색 중 오류: {e}"
            return (msg, None) if return_link else msg

        current_link = self._make_current_link(mst_id)

        # 2) 상세 조문 가져오기 (MST 기반)
        try:
            if not mst_id:
                msg = f"✅ '{law_name}'이(가) 확인되었습니다.\n(법령일련번호(MST) 추출 실패)\n🔗 현행 원문: {current_link or '-'}"
                return (msg, current_link) if return_link else msg

            xml_text = _cached_law_detail_xml(self.api_id, mst_id)
            if not xml_text:
                msg = f"상세 법령 파싱 실패: 상세 조회 응답이 비어 있습니다."
                return (msg, current_link) if return_link else msg
            root_detail = ET.fromstring(xml_text)

            # 조문번호 지정된 경우: 해당 조문만
            if article_num:
                want = re.sub(r"\D", "", str(article_num))
                for article in root_detail.findall(".//조문단위"):
                    jo_num_tag = article.find("조문번호")
                    jo_content_tag = article.find("조문내용")
                    if jo_num_tag is None or jo_content_tag is None:
                        continue
                    
                    current_num = re.sub(r"\D", "", (jo_num_tag.text or "").strip())
                    if want == current_num:
                        target_text = f"[{law_name} 제{current_num}조 전문]\n" + _escape((jo_content_tag.text or "").strip())
                        for hang in article.findall(".//항"):
                            hang_content = hang.find("항내용")
                            if hang_content is not None:
                                target_text += f"\n  - {(hang_content.text or '').strip()}"
                        return (target_text, current_link) if return_link else target_text

            # 못 찾았거나 조문번호 미지정
            msg = f"✅ '{law_name}'이(가) 확인되었습니다.\n(상세 조문 자동 추출 실패 또는 조문번호 미지정)\n🔗 현행 원문: {current_link or '-'}"
            return (msg, current_link) if return_link else msg

        except Exception as e:
            msg = f"상세 법령 파싱 실패: {e}"
            return (msg, current_link) if return_link else msg


law_api_service = LawOfficialService()


# =========================================================
# 4) AGENTS (BOOSTED)
# =========================================================
class CaseAnalyzer:
    @staticmethod
    def analyze(situation: str) -> dict:
        s = mask_sensitive(situation)
        prompt = f"""
너는 '민원/업무 케이스 분석관'이다.
한국어로 응답하되, 법률 용어나 고유명사 등 필요한 경우 영어는 사용 가능하다. 단, 베트남어/중국어/일본어 등 기타 외국어는 사용하지 마라.

[입력]
{s}

[출력 JSON]
{{
  "case_type": "예: 무단방치/번호판훼손/불법주정차/건설기계/기타",
  "core_issue": ["핵심 쟁점 3~6개 (한국어만)"],
  "required_facts": ["추가로 필요한 사실확인 질문 5개"],
  "required_evidence": ["필요 증빙 5개"],
  "risk_flags": ["절차상 리스크 3개(예: 통지 누락, 증거 부족...)"],
  "recommended_next_action": ["즉시 다음 행동 3개"]
}}
JSON만 출력. 반드시 한국어로.
"""
        data = llm_service.generate_json(prompt, preferred_model=MODEL_WORK_INSTRUCTION)
        if isinstance(data, dict) and data.get("case_type"):
            return data
        t = "기타"
        if "무단방치" in situation:
            t = "무단방치"
        if "번호판" in situation:
            t = "번호판훼손"
        return {
            "case_type": t,
            "core_issue": ["사실관계 확정", "증빙 확보", "절차적 정당성 확보"],
            "required_facts": ["장소/시간?", "증빙(사진/영상)?", "소유자 특정 가능?", "반복/상습 여부?", "요청사항(처분/계도/회신)?" ],
            "required_evidence": ["현장 사진", "위치/시간 기록", "신고내용 원문", "소유자 확인 자료", "조치/통지 기록"],
            "risk_flags": ["통지/의견제출 기회 누락", "증거 부족", "법적 근거 불명확"],
            "recommended_next_action": ["증빙 정리", "소유자/점유자 확인", "절차 플로우 확정"],
        }


class ProcedureAgent:
    @staticmethod
    def plan(situation: str, legal_basis_summary: str, analysis: dict) -> dict:
        prompt = f"""
너는 '행정 절차 플래너'이다.

[상황]
{situation}

[분석]
{json.dumps(analysis, ensure_ascii=False)}

[법적 근거(요약)]
{legal_basis_summary}

[출력 JSON]
{{
  "timeline": [
    {{"step": 1, "name": "단계명", "goal": "목표", "actions": ["행동1","행동2"], "records": ["기록/증빙"], "legal_note": "근거/유의"}}
  ],
  "checklist": ["담당자가 체크할 항목 10개"],
  "templates": ["필요 서식/문서 이름 5개"]
}}
JSON만.
"""
        data = llm_service.generate_json(prompt, preferred_model=MODEL_WORK_INSTRUCTION)
        if isinstance(data, dict) and data.get("timeline"):
            return data
        return {
            "timeline": [
                {"step": 1, "name": "사실확인", "goal": "사실관계 확정", "actions": ["현장 확인", "증빙 확보"], "records": ["사진/위치/시간"], "legal_note": "기록이 절차 정당성 핵심"},
                {"step": 2, "name": "대상 특정", "goal": "소유자/점유자 특정", "actions": ["등록정보 조회", "연락/안내"], "records": ["조회 로그", "통화/안내 기록"], "legal_note": "통지/연락 시도 기록"},
                {"step": 3, "name": "통지/계고", "goal": "자진 조치 유도", "actions": ["계고/안내", "기한 부여"], "records": ["통지문", "발송/수령 증빙"], "legal_note": "행정절차상 통지 누락 주의"},
                {"step": 4, "name": "불이행 시 조치", "goal": "강제/처분 검토", "actions": ["불이행 확인", "처분/강제 조치"], "records": ["확인서", "처분문"], "legal_note": "처분 사유/근거 명확화"},
            ],
            "checklist": ["증빙 확보", "법령 근거 확인", "통지/의견제출 기회", "문서번호/기한", "기록 남김"],
            "templates": ["회신 공문", "계고/통지", "의견제출 안내", "공시송달 공고", "처분서"],
        }





class LegalAgents:
    @staticmethod
    @staticmethod
    def researcher(situation: str, analysis: dict) -> str:
        prompt_extract = f"""
상황: "{situation}"

위 민원 처리를 위해 법적 근거로 삼아야 할 핵심 대한민국 법령과 조문 번호를
**중요도 순으로 최대 3개까지** JSON 리스트로 추출하시오.

형식: [{{"law_name": "도로교통법", "article_num": 32}}, ...]
* 법령명은 정식 명칭 사용. 조문 번호 불명확하면 null.
"""
        search_targets = []
        try:
            extracted = llm_service.generate_json(prompt_extract, preferred_model=MODEL_WORK_INSTRUCTION)
            if isinstance(extracted, list):
                search_targets = extracted
            elif isinstance(extracted, dict):
                search_targets = [extracted]
        except Exception:
            search_targets = [{"law_name": "도로교통법", "article_num": None}]

        if not search_targets:
            search_targets = [{"law_name": "도로교통법", "article_num": None}]

        report_lines = []
        api_success_count = 0

        report_lines.append(f"##### 🔍 AI가 식별한 핵심 법령 ({len(search_targets)}건)")
        report_lines.append("---")

        for idx, item in enumerate(search_targets):
            law_name = item.get("law_name", "관련법령")
            article_num = item.get("article_num")

            law_text, current_link = law_api_service.get_law_text(law_name, article_num, return_link=True)

            error_keywords = ["검색 결과가 없습니다", "오류", "API ID", "실패"]
            is_success = not any(k in (law_text or "") for k in error_keywords)

            if is_success:
                api_success_count += 1
                # ✅ 법령명 클릭 -> 새창에서 현행 원문
                law_title = f"[{law_name}]({current_link})" if current_link else law_name
                header = f"✅ **{idx+1}. {law_title} 제{article_num}조 (확인됨)**"
                content = law_text
            else:
                header = f"⚠️ **{idx+1}. {law_name} 제{article_num}조 (API 조회 실패)**"
                content = "(국가법령정보센터에서 해당 조문을 찾지 못했습니다. 법령명이 정확한지 확인이 필요합니다.)"

            report_lines.append(f"{header}\n{content}\n")

        final_report = "\n".join(report_lines)

        if api_success_count == 0:
            prompt_fallback = f"""
Role: 행정 법률 전문가
Task: 아래 상황에 적용될 법령과 조항을 찾아 설명하시오.
상황: "{situation}"

* 경고: 현재 외부 법령 API 연결이 원활하지 않습니다.
반드시 상단에 [AI 추론 결과]임을 명시하고 환각 가능성을 경고하시오.
"""
            ai_fallback_text = llm_service.generate_text(prompt_fallback, preferred_model=MODEL_WORK_INSTRUCTION).strip()

            return f"""⚠️ **[시스템 경고: API 조회 실패]**
(국가법령정보센터 연결 실패로 AI 지식 기반 답변입니다. **환각 가능성** 있으니 법제처 확인 필수)

--------------------------------------------------
{ai_fallback_text}"""

        return final_report

    @staticmethod
    def strategist(situation: str, legal_basis_md: str, search_results: str) -> str:
        prompt = f"""
당신은 행정 업무 베테랑 '주무관'입니다.

[민원 상황]: {situation}
[확보된 법적 근거]:
{legal_basis_md}

[유사 사례/판례]: {search_results}

위 정보를 종합하여 민원 처리 방향(Strategy)을 수립하세요.
서론(인사말/공감/네 알겠습니다 등) 금지.

1. 처리 방향
2. 핵심 주의사항
3. 예상 반발 및 대응
"""
        return llm_service.generate_text(prompt, preferred_model=MODEL_WORK_INSTRUCTION)

    @staticmethod
    def clerk() -> dict:
        today = datetime.now()
        return {
            "today_str": today.strftime("%Y. %m. %d."),
            "doc_num": f"행정-{today.strftime('%Y')}-{int(time.time())%1000:03d}호",
        }

    @staticmethod
    def drafter(situation: str, legal_basis_md: str, meta: dict, strategy: str, procedure: dict, objections: List[dict]) -> dict:
        schema = """
{
  "title": "제목",
  "receiver": "수신",
  "body_paragraphs": ["문단1", "문단2", "..."],
  "department_head": "OOO과장"
}
""".strip()

        prompt = f"""
당신은 대한민국 행정기관의 공문서 작성 및 교정 전문가인 'AI 행정관 Pro'이다.
아래 정보를 바탕으로 '2025년 개정 공문서 작성 표준'에 맞춰 완결된 공문서를 JSON으로 작성하라.

[입력]
- 민원: {situation}
- 시행일자: {meta.get('today_str')} (반드시 'YYYY. M. D.' 형식 준수)
- 문서번호: {meta.get('doc_num')}

[법령 근거(필수 인용)]
{legal_basis_md}

[처리방향]
{strategy}

[절차 플랜(반영)]
{json.dumps(procedure, ensure_ascii=False)}

[예상 반발(반영)]
{json.dumps(objections, ensure_ascii=False)}

[작성 원칙 (2025 개정 표준)]
1. **핵심 원칙**:
   - 사실성: 육하원칙 준수, 오자/탈자/계수 착오 금지.
   - 용이성: 쉬운 용어 사용, 짧고 명확한 문장.
   - 명확성: 불분명한 단어 회피, 정확한 조사 사용.
   - 비고압성: '금지', '엄금' 대신 '안내', '부탁' 등 긍정적 표현 사용.
   - 수요자 중심: 주어를 '국민(수요자)' 관점으로 (예: '교부합니다' -> '수령하실 수 있습니다').

2. **형식 및 표기 규칙 (엄격 준수)**:
   - **항목 기호 순서**: 1. -> 가. -> 1) -> 가) -> (1) -> (가) -> ① -> ㉮
   - **띄어쓰기**:
     - 첫째 항목 기호는 제목 첫 글자와 같은 위치.
     - 하위 항목은 상위 항목보다 2타(한글 1자) 오른쪽 시작.
     - 항목 기호와 내용 사이 1타 띄움.
   - **날짜/시간**:
     - 날짜: '2025. 1. 8.' (마지막 '일' 뒤에도 마침표).
     - 시간: '09:00', '13:20' (쌍점 앞뒤 붙임).
   - **금액**: '금13,500원(금일만삼천오백원)' (붙여쓰기).
   - **끝 표시**: 본문/붙임 끝에 2타 띄우고 '끝.'

3. **교정 가이드**:
   - 불명확한 주어 구체화 (예: '우리 기관' -> 공식 명칭).
   - 외래어 순화 (예: '스크린도어' -> '안전문').
   - 중복 표현 제거 ('2월달' -> '2월', '기간 동안' -> '기간에').
   - 문장 종결: 평서형 '-다' 원칙 (내부 결재는 '-함', '-것' 허용).

[출력 JSON 스키마]
{schema}

JSON만 출력.
"""
        data = llm_service.generate_json(prompt, preferred_model=MODEL_WORK_INSTRUCTION)
        if isinstance(data, dict) and data.get("title") and data.get("body_paragraphs"):
            return data

        retry = f"""
방금 출력이 스키마를 만족하지 않았다.
아래 스키마를 정확히 만족하는 JSON만 다시 출력하라.

스키마:
{schema}

(다른 텍스트 금지)
"""
        data2 = llm_service.generate_json(prompt + "\n\n" + retry, preferred_model=MODEL_WORK_INSTRUCTION)
        if isinstance(data2, dict) and data2.get("title") and data2.get("body_paragraphs"):
            return data2

        return {
            "title": "민원 처리 결과 회신(안)",
            "receiver": "수신자 참조",
            "body_paragraphs": [
                "**1**. 경위",
                f"- 민원 요지: {mask_sensitive(situation)}",
                "",
                "**2**. 법적 근거",
                "- 관련 법령 및 조문 근거에 따라 절차를 진행합니다.",
                "",
                "**3**. 조치 내용",
                "- 사실 확인 및 필요 절차를 단계적으로 이행 예정입니다.",
                "",
                "**4**. 이의제기/문의",
                "- 추가 의견이 있는 경우 의견제출 절차로 제출 바랍니다."
            ],
            "department_head": "OOO과장"
        }


def build_lawbot_pack(situation: str, analysis: dict) -> dict:
    prompt = f"""
상황: "{mask_sensitive(situation)}"
분석: {json.dumps(analysis, ensure_ascii=False)}
국가법령정보센터 Lawbot 검색창에 넣을 핵심 키워드 3~7개를 JSON 배열로만 출력.
예: ["무단방치","자동차관리법","공시송달","직권말소"]
"""
    kws = llm_service.generate_json(prompt, preferred_model=MODEL_WORK_INSTRUCTION) or []
    if not isinstance(kws, list):
        kws = []
    kws = [str(x).strip() for x in kws if str(x).strip()]
    query_text = (situation[:60] + " " + " ".join(kws[:7])).strip()
    query_text = re.sub(r"\s+", " ", query_text)
    return {"core_keywords": kws[:10], "query_text": query_text[:180], "url": make_lawbot_url(query_text[:180])}


def run_workflow(user_input: str, log_placeholder, mode: str = "신속") -> dict:
    start_time = time.time()
    search_count = 0
    logs = []  # List of dicts: {'msg': str, 'style': str, 'status': 'active'|'done', 'elapsed': float}
    phase_start_time = time.time()

    def render_logs():
        log_html = ""
        for log in logs:
            # 스타일 결정
            style = log['style']
            css_class = "log-sys"
            if style == "legal": css_class = "log-legal"
            elif style == "search": css_class = "log-search"
            elif style == "strat": css_class = "log-strat"
            elif style == "calc": css_class = "log-calc"
            elif style == "draft": css_class = "log-draft"
            
            # 상태별 아이콘 및 클래스
            if log['status'] == 'active':
                icon = "<span class='spinner-icon'>⏳</span>"
                css_class += " log-active"
                elapsed_text = ""
            else:
                icon = "✅"
                elapsed = log.get('elapsed', 0)
                if elapsed > 0:
                    elapsed_text = f"<span style='float:right; font-size:0.85em; color:#6b7280; font-weight:normal;'>{elapsed:.1f}s</span>"
                else:
                    elapsed_text = ""
            
            log_html += f"<div class='agent-log {css_class}' style='display:flex; justify-content:space-between; align-items:center; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'><span>{icon} {_escape(log['msg'])}</span>{elapsed_text}</div>"

        log_placeholder.markdown(
            f"""
            <div style='background:white; padding:1rem; border-radius:12px; border:1px solid #e5e7eb;'>
                <div style='font-weight:bold; margin-bottom:1rem; color:#374151; font-size:1.1rem;'>🤖 AI 에이전트 작업 로그</div>
                {log_html}
            </div>
            """, 
            unsafe_allow_html=True
        )

    def add_log(msg, style="sys"):
        nonlocal phase_start_time
        # 이전 활성 로그가 있다면 완료 처리 및 소요시간 기록
        if logs and logs[-1]['status'] == 'active':
            logs[-1]['status'] = 'done'
            logs[-1]['elapsed'] = time.time() - phase_start_time
        
        # 새 페이즈 시작 시간 기록
        phase_start_time = time.time()
        
        # 새 로그를 active 상태로 추가
        logs.append({'msg': msg, 'style': style, 'status': 'active', 'elapsed': 0})
        render_logs()
        time.sleep(0.05)  # 짧은 딜레이

    # Phase 1) 케이스 분석
    add_log("Phase 1: 민원 내용 분석 및 쟁점 파악...", "sys")
    analysis = CaseAnalyzer.analyze(user_input)

    # Phase 2~3) 법령/사례 병렬 조회 (실패 시 순차 폴백)
    add_log("Phase 2~3: 법령 근거 + 유사 사례 병렬 조회...", "search")
    law_md = ""
    news = ""
    try:
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_law = ex.submit(LegalAgents.researcher, user_input, analysis)
            fut_news = ex.submit(search_service.search_precedents, user_input)
            law_md = fut_law.result(timeout=30)
            news = fut_news.result(timeout=20)
            search_count += 2
    except Exception:
        # Thread/timeout 이슈가 있으면 기존 순차 흐름으로 안전 폴백
        add_log("병렬 조회 일부 실패 → 순차 모드로 재시도...", "sys")
        if not law_md:
            law_md = LegalAgents.researcher(user_input, analysis)
            search_count += 1
        if not news:
            news = search_service.search_precedents(user_input)
            search_count += 1

    # Phase 4) 처리방향/주의사항/체크리스트 생성
    add_log("Phase 4: 행정 처리 방향 및 전략 수립...", "strat")
    strategy = LegalAgents.strategist(user_input, law_md, news)

    # Phase 5) 절차 플랜
    add_log("Phase 5: 단계별 절차 플랜 및 타임라인 산정...", "calc")
    procedure = ProcedureAgent.plan(user_input, law_md[:1500], analysis)

    # Phase 6) 공문 조판
    add_log("Phase 6: 최종 공문서 초안 작성 중...", "draft")
    meta = LegalAgents.clerk()
    doc = LegalAgents.drafter(user_input, law_md, meta, strategy, procedure, [])

    # Phase 7) Lawbot 검색팩 생성
    add_log("Phase 7: 추가 검색 키워드 추출 및 마무리...", "sys")
    lb = build_lawbot_pack(user_input, analysis)
    
    # 마지막 로그 완료 처리 및 최종 메시지
    if logs and logs[-1]['status'] == 'active':
        logs[-1]['status'] = 'done'
        logs[-1]['elapsed'] = time.time() - phase_start_time
    
    total_elapsed = time.time() - start_time
    logs.append({'msg': f"모든 분석 완료! (총 {total_elapsed:.1f}초)", 'style': 'sys', 'status': 'done', 'elapsed': 0})
    render_logs()
    time.sleep(0.3)

    execution_time = round(time.time() - start_time, 2)
    # 간단한 성능 메트릭(세션 내 누적)
    perf = st.session_state.get("perf_metrics", {})
    wf_count = int(perf.get("workflow_count", 0)) + 1
    prev_avg = float(perf.get("avg_workflow_time", 0.0))
    new_avg = ((prev_avg * (wf_count - 1)) + execution_time) / wf_count
    perf.update({
        "workflow_count": wf_count,
        "avg_workflow_time": round(new_avg, 2),
        "last_workflow_time": execution_time,
    })
    st.session_state["perf_metrics"] = perf

    full_res_text = str(analysis) + str(law_md) + str(news) + str(strategy) + str(doc)
    estimated_tokens = int(len(full_res_text) * 0.7)
    model_used = getattr(llm_service, "last_model_used", None)

    return {
        "situation": user_input,
        "analysis": analysis,
        "law_pack": {}, # Deprecated but kept for compatibility
        "law": law_md,
        "search": news,
        "strategy": strategy,
        "objections": [], # Merged into strategy
        "procedure": procedure,
        "meta": meta,
        "doc": doc,
        "lawbot_pack": lb,
        "followups": [],
        "app_mode": mode,
        "token_usage": estimated_tokens,
        "execution_time": execution_time,
        "search_count": search_count,
        "model_used": model_used
    }



def run_complaint_analyzer_workflow(user_input: str, log_placeholder) -> dict:
    """민원(또는 국민제안) 텍스트를 '주장 단위'로 분해하고,
    법령 인용을 공식 API로 검증한 뒤, 책임 가능한(단정 최소) 회신 초안을 생성한다.

    반환 dict는 기존 run_workflow와 유사한 키를 포함하여 UI/다운로드 호환성을 유지한다.
    """
    start_time = time.time()
    logs = []
    phase_start_time = time.time()

    def render_logs():
        log_html = ""
        for log in logs:
            style = log.get("style", "sys")
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

            if log.get("status") == "active":
                icon = "<span class='spinner-icon'>⏳</span>"
                css_class += " log-active"
                elapsed_text = ""
            else:
                icon = "✅"
                elapsed = float(log.get("elapsed") or 0)
                elapsed_text = (
                    f"<span style='float:right; font-size:0.85em; color:#6b7280; font-weight:normal;'>{elapsed:.1f}s</span>"
                )

            msg = _escape(log.get("msg", ""))
            log_html += (
                f"<div class='agent-log {css_class}' "
                "style='display:flex; justify-content:space-between; align-items:center; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>"
                f"<span>{icon} {msg}</span>{elapsed_text}</div>"
            )

        log_placeholder.markdown(
            f"""
            <div style='background:white; padding:1rem; border-radius:12px; border:1px solid #e5e7eb;'>
                <div style='font-weight:bold; margin-bottom:1rem; color:#374151; font-size:1.1rem;'>🧾 민원 분석기 로그</div>
                {log_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    def add_log(msg, style="sys"):
        nonlocal phase_start_time
        if logs and logs[-1].get("status") == "active":
            logs[-1]["status"] = "done"
            logs[-1]["elapsed"] = time.time() - phase_start_time
        phase_start_time = time.time()
        logs.append({"msg": msg, "style": style, "status": "active", "elapsed": 0})
        render_logs()
        time.sleep(0.03)

    def _mvc_completion(mvc: dict) -> Tuple[int, int]:
        keys = ["time", "place", "target", "request"]
        filled = sum(1 for k in keys if (mvc.get(k) or "").strip())
        ev = mvc.get("evidence")
        if isinstance(ev, list) and len(ev) > 0:
            filled += 1
        elif isinstance(ev, str) and ev.strip():
            filled += 1
        return filled, 5

    def _normalize_article(article_val):
        if article_val is None:
            return None
        s = str(article_val).strip()
        if not s:
            return None
        digits = re.sub(r"\D", "", s)
        return {"raw": s, "digits": (digits or None)}

    # -------------------------
    # Phase 1) 주장 분해
    # -------------------------
    add_log("Phase 1: 민원 텍스트에서 주장/요건 요소를 분해...", "sys")
    s_masked = mask_sensitive(user_input or "")
    claim_prompt = f"""
너는 '민원 입력 품질 분석관'이다.
아래 민원 텍스트를 **주장 단위로 쪼개고**, 사실요건(MVC) 충족 여부를 구조화하라.
- 환각/추정 가능성이 있는 문장은 LEGAL/FACT로 구분하되, '단정'하지 마라.
- 법령/조문이 등장하면 citations에 넣되, **확실하지 않으면 null/빈값으로 남겨라.**
- 출력은 JSON만.

[민원 텍스트]
{s_masked}

[출력 JSON 스키마]
{{
  "mvc": {{
    "time": "언제(모르면 빈문자)",
    "place": "어디(모르면 빈문자)",
    "target": "대상(기관/사람/차량/시설 등, 모르면 빈문자)",
    "request": "민원인이 원하는 것(모르면 빈문자)",
    "evidence": ["사진/영상/문서/링크 등(없으면 빈배열)"]
  }},
  "claims": [
    {{
      "id": "C1",
      "type": "FACT|LEGAL|REQUEST|OPINION",
      "text": "주장 내용",
      "citations": [{{"law_name": "정식 법령명", "article": "조문(예: 26 또는 57-2, 없으면 빈문자)"}}],
      "notes": "모순/추정/감정적 수사 등 메모(없으면 빈문자)"
    }}
  ],
  "possible_hallucination_signals": ["환각 가능 신호(없으면 빈배열)"]
}}
"""
    parsed = llm_service.generate_json(claim_prompt) or {}
    if not isinstance(parsed, dict):
        parsed = {}
    mvc = parsed.get("mvc") if isinstance(parsed.get("mvc"), dict) else {}
    claims = parsed.get("claims") if isinstance(parsed.get("claims"), list) else []
    halluc_signals = parsed.get("possible_hallucination_signals")
    if not isinstance(halluc_signals, list):
        halluc_signals = []

    if not claims:
        claims = [{
            "id": "C1",
            "type": "FACT",
            "text": s_masked[:500],
            "citations": [],
            "notes": "자동 주장 분해 실패(원문 요약)"
        }]

    # -------------------------
    # Phase 2) 헛소리/요건 점검 (규칙 기반)
    # -------------------------
    add_log("Phase 2: 요건 충족/환각 신호를 점검...", "calc")
    filled, total = _mvc_completion(mvc)
    verifiability_score = round(filled / max(total, 1), 2)

    citation_items = []
    for c in claims:
        cits = c.get("citations") or []
        if isinstance(cits, dict):
            cits = [cits]
        if isinstance(cits, list):
            for it in cits:
                if not isinstance(it, dict):
                    continue
                law_name = (it.get("law_name") or "").strip()
                art = (it.get("article") or "").strip()
                if law_name:
                    citation_items.append({
                        "law_name": law_name,
                        "article": _normalize_article(art) if art else None,
                        "claim_id": c.get("id") or ""
                    })

    noise_grade = "GREEN"
    grade_reasons = []
    if verifiability_score <= 0.4:
        noise_grade = "YELLOW"
        grade_reasons.append("필수 사실요소(언제/어디/대상/요청/증거) 중 다수가 누락됨")
    if len(halluc_signals) >= 3 and noise_grade != "RED":
        noise_grade = "YELLOW"
        grade_reasons.append("환각/추정 신호가 다수 감지됨(보완요구 권장)")
    
    # Phase 2 완료 상태 메시지
    if not citation_items:
        add_log("  ↳ 요건 점검 완료 (명시적 법령 인용 없음 → Phase 3 건너뜀)", "calc")

    # -------------------------
    # Phase 3) 법령 인용 검증 (공식 API)
    # -------------------------
    add_log(f"Phase 3: 법령/조문 인용을 공식 API로 검증... ({len(citation_items)}건)", "legal")
    verified_citations = []
    invalid_count = 0

    if not citation_items:
        add_log("  ↳ 검증 대상 법령 인용 없음 → 건너뜀", "legal")
    else:
        _law_cache = {}
        error_keywords = ["검색 결과가 없습니다", "API ID", "오류", "실패", "찾지 못했습니다", "No results"]
        partial_keywords = ["자동 추출 실패", "조문번호 미지정", "법령일련번호(MST) 추출 실패"]

        for it in citation_items[:12]:
            law_name = it["law_name"]
            art = it.get("article")
            digits = art.get("digits") if isinstance(art, dict) else None

            cache_key = (law_name, digits or "")
            if cache_key in _law_cache:
                law_text, link, status = _law_cache[cache_key]
            else:
                article_num = digits if digits else None
                law_text, link = law_api_service.get_law_text(law_name, article_num, return_link=True)
                txt = (law_text or "")
                if any(k in txt for k in error_keywords):
                    status = "INVALID"
                elif any(k in txt for k in partial_keywords):
                    status = "PARTIAL"
                else:
                    status = "VALID"
                _law_cache[cache_key] = (law_text, link, status)

            if status == "INVALID":
                invalid_count += 1

            verified_citations.append({
                "claim_id": it.get("claim_id"),
                "law_name": law_name,
                "article_raw": (art.get("raw") if isinstance(art, dict) else None),
                "article_digits": digits,
                "status": status,
                "link": link,
                "excerpt": (law_text or "")[:900]
            })
        add_log(f"  ↳ 법령 검증 완료: {len(verified_citations)}건 (유효 {len(verified_citations) - invalid_count}, 미확인 {invalid_count})", "legal")

    if invalid_count >= 2 and noise_grade == "GREEN":
        noise_grade = "YELLOW"
        grade_reasons.append("법령 인용 중 확인 불가 항목이 다수 존재함")
    if invalid_count >= 4:
        noise_grade = "RED"
        grade_reasons.append("법령/조문 인용이 다수 확인 불가(허위/환각 가능성 높음)")

    law_lines = ["##### ⚖️ 법령 인용 검증 결과", "---"]
    if verified_citations:
        for v in verified_citations:
            nm = v["law_name"]
            link = v.get("link")
            status = v.get("status")
            art_raw = v.get("article_raw") or ""
            title = f"[{nm}]({link})" if link else nm
            badge = "✅" if status == "VALID" else ("🟨" if status == "PARTIAL" else "❌")
            law_lines.append(f"- {badge} **{title}** {('(' + art_raw + ')' ) if art_raw else ''}  \n  - 상태: {status}")
    else:
        law_lines.append("- (민원 텍스트에서 명시적 법령 인용이 없거나 추출하지 못했습니다.)")
    law_md = "\n".join(law_lines)

    # -------------------------
    # Phase 4) 주장별 안전 판정
    # -------------------------
    add_log("Phase 4: 주장별 '안전한 결론'을 산출...", "strat")
    verdicts = []
    for c in claims[:12]:
        cid = c.get("id") or ""
        ctype = (c.get("type") or "FACT").strip().upper()
        ctext = (c.get("text") or "").strip()
        cnotes = (c.get("notes") or "").strip()

        rel = [v for v in verified_citations if v.get("claim_id") == cid]
        rel_text = ""
        for v in rel[:2]:
            rel_text += f"- {v.get('law_name')} ({v.get('article_raw') or ''}) [{v.get('status')}]\n"
            rel_text += f"  EXCERPT: {v.get('excerpt','')[:400]}\n"

        judge_prompt = f"""
너는 '민원 주장 검증 보조관'이다.
중요: 너는 사실을 새로 만들면 안 된다. 아래 근거가 부족하면 반드시 INSUFFICIENT로 판단한다.
REFUTED(반박)은 근거가 명확할 때만 선택하며, 불확실하면 INSUFFICIENT로 둔다.

[주장]
- id: {cid}
- type: {ctype}
- text: {ctext}
- notes: {cnotes}

[가용 근거(법령 발췌/검증 상태)]
{rel_text if rel_text else "(관련 근거 없음)"}

[출력 JSON]
{{
  "verdict": "SUPPORTED|INSUFFICIENT|REFUTED",
  "confidence": 0.0,
  "safe_statement": "공무원이 책임질 수 있는 안전한 문장(단정 최소)",
  "needed": ["추가 제출/확인 항목 3~7개"]
}}
"""
        vj = llm_service.generate_json(judge_prompt) or {}
        if not isinstance(vj, dict):
            vj = {}
        verdict = (vj.get("verdict") or "INSUFFICIENT").strip().upper()
        if verdict not in ["SUPPORTED", "INSUFFICIENT", "REFUTED"]:
            verdict = "INSUFFICIENT"
        if noise_grade in ["YELLOW", "RED"] and verdict == "REFUTED":
            verdict = "INSUFFICIENT"

        needed = vj.get("needed")
        if not isinstance(needed, list):
            needed = []
        safe_stmt = (vj.get("safe_statement") or "").strip() or "제출된 자료 범위 내에서는 해당 주장에 대해 단정하기 어렵습니다."

        verdicts.append({
            "claim_id": cid,
            "type": ctype,
            "text": ctext,
            "verdict": verdict,
            "confidence": float(vj.get("confidence") or 0.5),
            "safe_statement": safe_stmt,
            "needed": needed[:10]
        })

    # -------------------------
    # Phase 5) 회신 초안(공문) 조립
    # -------------------------
    add_log("Phase 5: 회신 초안을 조립...", "draft")

    required_facts = []
    required_evidence = []
    if not (mvc.get("time") or "").strip():
        required_facts.append("발생 일시(연월일·시간)")
    if not (mvc.get("place") or "").strip():
        required_facts.append("발생 장소(주소/시설명/위치)")
    if not (mvc.get("target") or "").strip():
        required_facts.append("대상 특정(차량/시설/업체/담당부서 등)")
    if not (mvc.get("request") or "").strip():
        required_facts.append("요청사항(원하는 조치/결과)")
    ev = mvc.get("evidence") if isinstance(mvc.get("evidence"), (list, str)) else []
    if (isinstance(ev, list) and len(ev) == 0) or (isinstance(ev, str) and not ev.strip()):
        required_evidence.append("사진/영상/문서/링크 등 객관적 자료(가능한 범위)")

    grade_to_title = {
        "GREEN": "민원 검토 결과 안내(초안)",
        "YELLOW": "민원 처리 관련 추가자료 요청(보완요구) (초안)",
        "RED": "민원 내용 확인 및 절차 안내(요건/관할 검토) (초안)",
    }
    title = grade_to_title.get(noise_grade, "민원 검토 결과 안내(초안)")

    # VERIFIED / UNVERIFIED / NEEDED
    verified_lines = []
    unverified_lines = []
    needed_lines = []

    for k, label in [("time", "발생 일시"), ("place", "발생 장소"), ("target", "대상"), ("request", "요청사항")]:
        v = (mvc.get(k) or "").strip()
        if v:
            verified_lines.append(f"- {label}: {v}")
    if isinstance(ev, list) and ev:
        verified_lines.append(f"- 제출 자료: {', '.join([str(x) for x in ev[:5]])}")
    elif isinstance(ev, str) and ev.strip():
        verified_lines.append(f"- 제출 자료: {ev.strip()}")

    for vd in verdicts:
        if vd["verdict"] == "SUPPORTED":
            verified_lines.append(f"- (주장 {vd['claim_id']}) {vd['safe_statement']}")
        else:
            unverified_lines.append(f"- (주장 {vd['claim_id']}) {vd['safe_statement']}")
        for n in vd.get("needed", [])[:3]:
            needed_lines.append(f"- {n}")

    def _dedup(lines):
        seen = set()
        out = []
        for x in lines:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    verified_lines = _dedup(verified_lines) or ["- (기관이 확인 가능한 범위의 사실이 부족합니다.)"]
    unverified_lines = _dedup(unverified_lines) or ["- (미확인 주장 없음)"]
    needed_lines = _dedup(needed_lines + [f"- {x}" for x in required_facts] + [f"- {x}" for x in required_evidence]) or ["- (추가 제출 요청 없음)"]

    next_step = "제출된 자료 범위 내에서만 판단이 가능하며, 필요 시 추가 확인 후 처리합니다."
    if noise_grade == "YELLOW":
        next_step = "추가자료 제출 시 재검토 예정이며, 미제출 시 현 단계에서 사실확정이 어렵습니다."
    elif noise_grade == "RED":
        next_step = "제출된 자료만으로 특정/판단이 어려워 요건·관할 기준으로 정형 안내드립니다. 추가자료 제출 시 재검토합니다."

    body_paragraphs = [
        "**1. 확인된 사실(제출 자료 기준)**",
        *verified_lines,
        "",
        "**2. 미확인 주장(현 단계에서 단정 불가)**",
        *unverified_lines,
        "",
        "**3. 확인을 위한 추가 자료/사실 요청**",
        *needed_lines,
        "",
        "**4. 절차 및 안내**",
        f"- {next_step}",
        "- 본 회신(초안)은 민원인이 제출한 내용 및 기관이 확인 가능한 범위에 한하여 작성됩니다.",
    ]

    meta = LegalAgents.clerk()
    doc = {
        "title": title,
        "receiver": "민원인 귀하",
        "body_paragraphs": body_paragraphs,
        "department_head": "행정기관장",
    }

    strategy_lines = [
        f"- 처리 등급: **{noise_grade}** (검증가능성 {verifiability_score*100:.0f}%)",
        *(f"- 사유: {r}" for r in (grade_reasons or [])),
        "",
        "#### 운영 권고",
        "- **단정형 반박** 대신, `확인된 사실/미확인 주장/보완요구/절차` 구조로 회신",
        "- 법령 인용은 **공식 API로 확인된 범위만** 사용하고, 불확실한 인용은 '미확인' 처리",
        "- 동일·유사 민원은 사건ID로 **병합**하여 반복 대응 비용을 낮출 것",
    ]
    strategy = "\n".join(strategy_lines)

    analysis = {
        "case_type": "민원 분석",
        "core_issue": ["입력 품질(요건) 점검", "법령 인용 검증", "안전한 회신 조립"],
        "required_facts": _dedup(required_facts)[:10],
        "required_evidence": _dedup(required_evidence)[:10],
        "risk_flags": _dedup((halluc_signals or []) + (grade_reasons or []))[:10],
        "recommended_next_action": ["보완요구 또는 정형 안내 후 재검토"][:10],
        "summary": f"민원 분석기 결과: 등급 {noise_grade}, 검증가능성 {verifiability_score*100:.0f}%",
    }

    if logs and logs[-1].get("status") == "active":
        logs[-1]["status"] = "done"
        logs[-1]["elapsed"] = time.time() - phase_start_time
    total_elapsed = time.time() - start_time
    logs.append({"msg": f"완료 (총 {total_elapsed:.1f}초)", "style": "sys", "status": "done", "elapsed": 0})
    render_logs()
    time.sleep(0.2)

    full_res_text = str(parsed) + str(verified_citations) + str(verdicts) + str(doc)
    estimated_tokens = int(len(full_res_text) * 0.7)
    model_used = getattr(llm_service, "last_model_used", None)

    return {
        "situation": user_input,
        "analysis": analysis,
        "law_pack": {},
        "law": law_md,
        "search": "",
        "strategy": strategy,
        "objections": [],
        "procedure": {"timeline": [], "checklist": [], "templates": []},
        "meta": meta,
        "doc": doc,
        "lawbot_pack": build_lawbot_pack(user_input, analysis) if "build_lawbot_pack" in globals() else {},
        "followups": [],
        "app_mode": "complaint_analyzer",
        "token_usage": estimated_tokens,
        "execution_time": round(time.time() - start_time, 2),
        "search_count": 0,
        "model_used": model_used,
        "complaint_pack": {
            "mvc": mvc,
            "claims": claims,
            "noise_grade": noise_grade,
            "verifiability_score": verifiability_score,
            "hallucination_signals": halluc_signals,
            "citations": verified_citations,
            "verdicts": verdicts,
            "grade_reasons": grade_reasons,
        },
    }


# =========================================================
# 5) DB OPS (HYBRID CHECK VERSION)
# =========================================================
def db_insert_archive(sb, prompt: str, payload: dict) -> Optional[str]:
    archive_id = str(uuid.uuid4())
    anon_id = str(ensure_anon_session_id())

    # ---------------------------------------------------------
    # [최종 수정] 서버(sb)와 메모장(session)을 모두 뒤져서 이메일 찾아냄
    # ---------------------------------------------------------
    # 1. 서버(Supabase)에게 먼저 물어봄
    user = get_auth_user(sb)
    server_email = None
    server_user_id = None

    if user:
        if isinstance(user, dict):
             server_user_id = user.get("id")
             server_email = user.get("email")
        else:
             server_user_id = getattr(user, "id", None)
             server_email = getattr(user, "email", None)

    # 2. 메모장(Session State)도 확인 (로그인 직후 서버가 느릴 때 대비)
    session_email = st.session_state.get("user_email")
    
    # 3. [판결] 둘 중 하나라도 이메일이 있으면 그것을 사용
    # (서버에서 가져온 게 있으면 우선 사용, 없으면 세션 정보 사용)
    final_email = server_email if server_email else session_email
    final_user_id = server_user_id # ID는 없어도 RLS 작동엔 문제 없음
    
    # ---------------------------------------------------------

    row = {
        "id": archive_id,
        "prompt": prompt,
        "payload": payload,
        "anon_session_id": anon_id,
        "user_id": final_user_id,
        
        # ★ [핵심] 찾아낸 최종 이메일을 넣음
        "user_email": (final_email.strip() if final_email else None),
        
        "client_meta": {"app_ver": APP_VERSION},
        "app_mode": payload.get("app_mode", st.session_state.get("app_mode", "신속")),
        "search_count": int(payload.get("search_count") or 0),
        "execution_time": float(payload.get("execution_time") or 0.0),
        "token_usage": int(payload.get("token_usage") or 0),
        "model_used": payload.get("model_used"),
    }

    try:
        # 헤더 전송
        sb.postgrest.headers.update({'x-session-id': anon_id})
        sb.table("work_archive").insert(row).execute()
        _clear_history_cache()
        return archive_id
    except Exception as e:
        st.warning(f"ℹ️ DB 저장 실패: {e}")
        return None


def db_fetch_history(sb, limit: int = 80) -> List[dict]:
    anon_id = str(ensure_anon_session_id())
    try:
        return _cached_db_fetch_history(sb, anon_id, limit=limit)
    except Exception:
        return []

def db_fetch_payload(sb, archive_id: str) -> Optional[dict]:
    anon_id = str(ensure_anon_session_id())
    try:
        return _cached_db_fetch_payload(sb, anon_id, archive_id)
    except Exception:
        return None
    

def db_fetch_followups(sb, archive_id: str) -> List[dict]:
    anon_id = str(ensure_anon_session_id())
    try:
        return _cached_db_fetch_followups(sb, anon_id, archive_id)
    except Exception:
        return []

def db_insert_followup(sb, archive_id: str, turn: int, role: str, content: str):
    anon_id = str(ensure_anon_session_id())
    
    # [수정] 후속 질문도 동일하게 양쪽 확인
    user = get_auth_user(sb)
    server_email = None
    server_user_id = None

    if user:
        if isinstance(user, dict):
             server_user_id = user.get("id")
             server_email = user.get("email")
        else:
             server_user_id = getattr(user, "id", None)
             server_email = getattr(user, "email", None)
    
    session_email = st.session_state.get("user_email")
    final_email = server_email if server_email else session_email
    final_user_id = server_user_id

    row = {
        "archive_id": archive_id,
        "turn": turn,
        "role": role,
        "content": content,
        "user_id": final_user_id,
        "user_email": (final_email.strip() if final_email else None),
        "anon_session_id": anon_id,
    }
    try:
        sb.postgrest.headers.update({'x-session-id': anon_id})
        sb.table("work_followups").insert(row).execute()
        _clear_history_cache()
    except Exception:
        pass


# =========================================================
# 6) SIDEBAR AUTH UI
# =========================================================
def sidebar_auth(sb):
    st.sidebar.markdown("## 🔐 로그인")

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_email" not in st.session_state:
        st.session_state.user_email = ""
    if "admin_mode" not in st.session_state:
        st.session_state.admin_mode = False
    if "is_admin_db" not in st.session_state:
        st.session_state.is_admin_db = False

    if st.session_state.logged_in:
        email = st.session_state.user_email
        st.sidebar.success(f"✅ {email}")

        # ✅ admin toggle: 하드코딩 또는 DB에서 admin이면 노출
        if is_admin_user(email):
            st.sidebar.toggle("관리자모드 켜기", key="admin_mode")

        if st.sidebar.button("로그아웃", use_container_width=True):
            try:
                sb.auth.sign_out()
            except Exception:
                pass
            st.session_state.logged_in = False
            st.session_state.user_email = ""
            st.session_state.admin_mode = False
            st.session_state.is_admin_db = False
            log_event(sb, "logout")
            st.rerun()
        return

    menu = st.sidebar.radio("메뉴", ["로그인", "회원가입", "비밀번호 찾기"], horizontal=True)

    if "signup_stage" not in st.session_state:
        st.session_state.signup_stage = 1
    if "reset_stage" not in st.session_state:
        st.session_state.reset_stage = 1

    if menu == "로그인":
        email = st.sidebar.text_input("메일", placeholder="name@korea.kr", key="login_email")
        pw = st.sidebar.text_input("비밀번호", type="password", key="login_pw")
        if st.sidebar.button("로그인", use_container_width=True):
            try:
                sb.auth.sign_in_with_password({"email": email, "password": pw})
                st.session_state.logged_in = True
                st.session_state.user_email = (email or "").strip()
                _refresh_admin_flag(sb, st.session_state.user_email)
                log_event(sb, "login_success")
                st.rerun()
            except Exception:
                st.sidebar.error("로그인 실패: 메일/비밀번호 확인")

    elif menu == "회원가입":
        if st.session_state.signup_stage == 1:
            email = st.sidebar.text_input("메일(@korea.kr)", placeholder="name@korea.kr", key="su_email")
            if st.sidebar.button("코리아 메일로 인증번호 발송", use_container_width=True):
                if not (email or "").endswith("@korea.kr"):
                    st.sidebar.error("❌ @korea.kr 메일만 가입 가능")
                else:
                    try:
                        sb.auth.sign_in_with_otp({"email": email, "options": {"should_create_user": True}})
                        st.session_state.pending_email = email.strip()
                        st.session_state.signup_stage = 2
                        log_event(sb, "signup_otp_sent", meta={"email": email.strip()})
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"발송 실패: {e}")
        else:
            email = st.session_state.get("pending_email", "")
            st.sidebar.caption(f"발송 대상: {email}")
            code = st.sidebar.text_input("인증번호(OTP/토큰)", key="su_code")
            new_pw = st.sidebar.text_input("비밀번호 설정", type="password", key="su_pw")
            new_pw2 = st.sidebar.text_input("비밀번호 확인", type="password", key="su_pw2")
            if st.sidebar.button("인증 + 비밀번호 설정 완료", use_container_width=True):
                if not new_pw or new_pw != new_pw2:
                    st.sidebar.error("비밀번호가 일치하지 않습니다.")
                else:
                    ok = False
                    for t in ["signup", "magiclink"]:
                        try:
                            sb.auth.verify_otp({"email": email, "token": code, "type": t})
                            ok = True
                            break
                        except Exception:
                            pass
                    if not ok:
                        st.sidebar.error("인증번호 검증 실패")
                        return
                    try:
                        sb.auth.update_user({"password": new_pw})
                    except Exception as e:
                        st.sidebar.error(f"비밀번호 설정 실패: {e}")
                        return

                    st.session_state.logged_in = True
                    st.session_state.user_email = email.strip()
                    _refresh_admin_flag(sb, st.session_state.user_email)
                    st.session_state.signup_stage = 1
                    log_event(sb, "signup_done")
                    st.rerun()

    else:  # reset
        if st.session_state.reset_stage == 1:
            email = st.sidebar.text_input("메일", placeholder="name@korea.kr", key="rp_email")
            if st.sidebar.button("메일로 인증번호 발송", use_container_width=True):
                try:
                    sb.auth.sign_in_with_otp({"email": email, "options": {"should_create_user": False}})
                    st.session_state.reset_email = email.strip()
                    st.session_state.reset_stage = 2
                    log_event(sb, "reset_otp_sent", meta={"email": email.strip()})
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"발송 실패: {e}")
        else:
            email = st.session_state.get("reset_email", "")
            st.sidebar.caption(f"대상: {email}")
            code = st.sidebar.text_input("인증번호(OTP/토큰)", key="rp_code")
            new_pw = st.sidebar.text_input("새 비밀번호", type="password", key="rp_pw")
            new_pw2 = st.sidebar.text_input("새 비밀번호 확인", type="password", key="rp_pw2")
            if st.sidebar.button("인증 + 새 비밀번호 설정", use_container_width=True):
                if not new_pw or new_pw != new_pw2:
                    st.sidebar.error("비밀번호가 일치하지 않습니다.")
                    return
                ok = False
                for t in ["magiclink", "signup"]:
                    try:
                        sb.auth.verify_otp({"email": email, "token": code, "type": t})
                        ok = True
                        break
                    except Exception:
                        pass
                if not ok:
                    st.sidebar.error("인증번호 검증 실패")
                    return
                try:
                    sb.auth.update_user({"password": new_pw})
                except Exception as e:
                    st.sidebar.error(f"비밀번호 변경 실패: {e}")
                    return

                st.session_state.logged_in = True
                st.session_state.user_email = email.strip()
                _refresh_admin_flag(sb, st.session_state.user_email)
                st.session_state.reset_stage = 1
                log_event(sb, "reset_done")
                st.rerun()


# =========================================================
# 7) HISTORY (프롬프트만, 클릭 즉시 복원)
# =========================================================
def restore_archive(sb, row_id: str):
    row = db_fetch_payload(sb, row_id)
    if not row:
        st.sidebar.error("복원 실패(권한/RLS 또는 데이터 없음)")
        return
    payload = row.get("payload") or {}
    followups = db_fetch_followups(sb, row_id)
    msgs = [{"role": f.get("role"), "content": f.get("content")} for f in followups]
    payload["followups"] = msgs
    st.session_state["workflow_result"] = payload
    st.session_state["current_archive_id"] = row_id
    st.session_state["followup_messages"] = msgs
    st.session_state["selected_history_id"] = row_id
    log_event(sb, "restore_archive", archive_id=row_id)
    st.rerun()

def render_history_list(sb):
    email = st.session_state.get("user_email", "")
    admin_all = is_admin_user(email) and st.session_state.get("admin_mode", False)

    # 비로그인은 select 불가(RLS)
    if not st.session_state.get("logged_in") and not admin_all:

        return

    # 새 채팅 버튼 (로그인 유저용)
    c1, c2 = st.sidebar.columns(2)
    if c1.button("➕ 새 채팅", use_container_width=True, type="primary"):
        st.session_state.pop("workflow_result", None)
        st.session_state.pop("current_archive_id", None)
        st.session_state.pop("followup_messages", None)
        st.session_state.pop("selected_history_id", None)
        st.rerun()
    if c2.button("↻ 새로고침", use_container_width=True):
        _clear_history_cache()
        st.rerun()

    hist = db_fetch_history(sb, limit=120)
    if not hist:
        st.sidebar.caption("저장된 기록이 없습니다.")
        return

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🕘 기록")
    q = st.sidebar.text_input("검색", placeholder="프롬프트 검색", label_visibility="collapsed", key="hist_q")
    if q:
        ql = q.strip().lower()
        hist = [r for r in hist if ql in (r.get("prompt", "").lower())]

    if "selected_history_id" not in st.session_state:
        st.session_state.selected_history_id = None

    for row in hist:
        rid = row["id"]
        label = shorten_one_line(row.get("prompt", ""), 28) or "（프롬프트 없음）"
        prefix = "● " if st.session_state.selected_history_id == rid else "  "
        if st.sidebar.button(prefix + label, key=f"hist_{rid}", use_container_width=True, type="secondary"):
            restore_archive(sb, rid)


# =========================================================
# 8) ADMIN DASHBOARD (FINAL FIX: Direct Count)
# =========================================================
def admin_fetch_work_archive(sb, limit: int = 5000) -> List[dict]:
    try:
        resp = (
            sb.table("work_archive")
            .select("id,created_at,user_email,anon_session_id,prompt,app_mode,search_count,execution_time,token_usage,model_used")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return getattr(resp, "data", None) or []
    except Exception as e:
        st.error(f"관리자 조회 실패(work_archive): {e}")
        return []

def admin_fetch_sessions(sb, minutes: int = 10) -> List[dict]:
    """실시간 접속자 (최근 10분)"""
    try:
        cutoff = (datetime.utcnow() - timedelta(minutes=minutes)).isoformat() + "Z"
        resp = (
            sb.table("app_sessions")
            .select("session_id,first_seen,last_seen,user_email,user_id,meta")
            .gte("last_seen", cutoff)
            .order("last_seen", desc=True)
            .execute()
        )
        return getattr(resp, "data", None) or []
    except Exception as e:
        st.error(f"관리자 조회 실패(app_sessions): {e}")
        return []

def admin_fetch_events(sb, limit: int = 300) -> List[dict]:
    try:
        resp = (
            sb.table("app_events")
            .select("created_at,event_type,user_email,anon_session_id,archive_id,meta")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return getattr(resp, "data", None) or []
    except Exception as e:
        st.error(f"관리자 조회 실패(app_events): {e}")
        return []

# ─────────────────────────────────────────────────────────
# ★ [핵심] 뷰(View) 대신 직접 세는 함수들 (가장 정확함)
# ─────────────────────────────────────────────────────────
def admin_get_total_visits(sb) -> int:
    """누적 방문수 (전체 행 개수)"""
    try:
        res = sb.table("app_sessions").select("*", count="exact", head=True).execute()
        return res.count if res.count is not None else 0
    except: return 0

def admin_get_today_visitors(sb) -> int:
    """오늘 방문자 (한국시간 00:00 이후 생성된 세션 수)"""
    try:
        # 한국 시간 기준 '오늘 0시' 계산
        now_kst = datetime.utcnow() + timedelta(hours=9)
        today_start_kst = now_kst.replace(hour=0, minute=0, second=0, microsecond=0)
        # DB 쿼리용 UTC 변환 (한국 0시 = 전날 15시 UTC)
        today_start_utc = today_start_kst - timedelta(hours=9)
        cutoff = today_start_utc.isoformat() + "Z"
        
        # created_at이 오늘 0시 이후인 것만 카운트
        res = sb.table("app_sessions")\
            .select("*", count="exact", head=True)\
            .gte("created_at", cutoff)\
            .execute()
        return res.count if res.count is not None else 0
    except: return 0

def render_master_dashboard(sb, llm_service=None):
    st.markdown("## 🏛️ 관리자 운영 마스터 콘솔")

    if not is_admin_user(st.session_state.get("user_email", "")):
        st.warning("관리자만 접근 가능합니다.")
        return

    if not st.session_state.get("admin_mode", False):
        st.info("사이드바에서 **관리자모드 켜기**를 활성화하세요.")
        return

    # [NEW] 데이터 관리 (임베딩 생성)
    with st.expander("🛠️ 데이터베이스 관리 (임베딩 생성)", expanded=False):
        st.info("당직 매뉴얼 데이터에 벡터 임베딩이 없는 경우 검색이 되지 않습니다. 아래 버튼을 눌러 임베딩을 생성하세요.")
        
        col_db1, col_db2 = st.columns(2)
        with col_db1:
            if st.button("🔄 매뉴얼 임베딩 생성(재처리)", use_container_width=True):
                if not llm_service:
                    st.error("LLM 서비스가 연결되지 않았습니다.")
                else:
                    try:
                        # 1. 임베딩 없는 데이터 조회
                        res = sb.table("duty_manual_kb").select("*").is_("embedding", "null").execute()
                        rows = res.data
                        
                        if not rows:
                            st.success("모든 데이터에 임베딩이 이미 존재합니다.")
                        else:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            success_count = 0
                            
                            for idx, row in enumerate(rows):
                                content = row.get("content", "")
                                if content:
                                    emb = llm_service.embed_text(content)
                                    if emb:
                                        # 업데이트
                                        sb.table("duty_manual_kb").update({"embedding": emb}).eq("id", row["id"]).execute()
                                        success_count += 1
                                
                                progress = (idx + 1) / len(rows)
                                progress_bar.progress(progress)
                                status_text.text(f"처리 중... ({idx+1}/{len(rows)})")
                            
                            st.success(f"완료! {success_count}건의 임베딩을 생성했습니다.")
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"작업 중 오류 발생: {e}")

    # 1. 데이터 로드
    with st.spinner("📊 데이터 분석 중..."):
        data = admin_fetch_work_archive(sb, limit=5000)
        sessions = admin_fetch_sessions(sb, minutes=10) # 실시간
        events = admin_fetch_events(sb, limit=200)
        
        # ★ 직접 카운트 함수 호출
        total_visits = admin_get_total_visits(sb)
        today_visitors = admin_get_today_visitors(sb)

        # 뷰 데이터 (그래프용 - 에러나도 무시)
        try:
            res_peak = sb.table("view_analytics_peak_hours").select("*").execute()
            df_peak = pd.DataFrame(res_peak.data) if res_peak.data else pd.DataFrame()
        except: df_peak = pd.DataFrame()

        try:
            res_dur = sb.table("view_analytics_duration").select("*").execute()
            dur_data = res_dur.data[0] if res_dur.data else {"avg_duration_min": 0, "max_duration_min": 0}
        except: dur_data = {"avg_duration_min": 0, "max_duration_min": 0}

    if not pd:
        st.error("pandas 모듈 없음")
        return

    # 2. DataFrame 가공
    df = pd.DataFrame(data)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
        df["date"] = df["created_at"].dt.date
        df["hour"] = df["created_at"].dt.hour
        df["weekday"] = df["created_at"].dt.day_name()
        df["user_email"] = df["user_email"].fillna("(anon)")
        df["model_used"] = df["model_used"].fillna("(unknown)")
        df["token_usage"] = pd.to_numeric(df["token_usage"], errors="coerce").fillna(0).astype(int)
        df["execution_time"] = pd.to_numeric(df["execution_time"], errors="coerce").fillna(0)
        
        def calc_cost(row):
            model = row["model_used"]
            tokens = row["token_usage"]
            rate = MODEL_PRICING.get(model, MODEL_PRICING.get("(unknown)", 0.10))
            return (tokens / 1_000_000) * rate
        df["cost_usd"] = df.apply(calc_cost, axis=1)

        user_run_counts = df["user_email"].value_counts()
        heavy_threshold = user_run_counts.quantile(HEAVY_USER_PERCENTILE / 100) if len(user_run_counts) > 1 else 999999
        heavy_users = set(user_run_counts[user_run_counts >= heavy_threshold].index)
    else:
        df["cost_usd"] = 0.0
        heavy_users = set()

    # 2. 기능별 사용 통계
    st.subheader("🎯 기능별 사용 현황")
    try:
        res_features = sb.table("analytics_feature_usage").select("*").execute()
        if res_features.data:
            df_features = pd.DataFrame(res_features.data)
            
            col_f1, col_f2= st.columns(2)
            with col_f1:
                st.markdown("**사용량**")
                st.dataframe(
                    df_features[["feature_name", "usage_count", "unique_users"]],
                    use_container_width=True,
                    hide_index=True
                )
            
            with col_f2:
                st.markdown("**평균 실행 시간 & 토큰**")
                st.dataframe(
                    df_features[["feature_name", "avg_execution_time", "total_tokens"]],
                    use_container_width=True,
                    hide_index=True
                )
    except Exception as e:
        st.error(f"기능별 통계 로드 실패: {e}")
    
    st.markdown("---")
    
    # 2.5. 모델별 비용 분석
    st.subheader("💰 모델별 비용 분석")
    try:
        res_costs = sb.table("analytics_model_costs").select("*").execute()
        if res_costs.data:
            df_costs = pd.DataFrame(res_costs.data)
            
            # 비용이 높은 순서로 정렬
            df_costs = df_costs.sort_values("cost_usd", ascending=False)
            
            col_c1, col_c2, col_c3 = st.columns(3)
            total_cost = df_costs["cost_usd"].sum()
            total_calls = df_costs["call_count"].sum()
            total_tokens = df_costs["total_tokens"].sum()
            
            col_c1.metric("💵 총 비용 (USD)", f"${total_cost:.4f}")
            col_c2.metric("📞 총 호출 수", f"{int(total_calls):,}")
            col_c3.metric("🎫 총 토큰", f"{int(total_tokens):,}")
            
            st.markdown("**모델별 세부 정보**")
            st.dataframe(
                df_costs[[
                    "model_name", 
                    "call_count", 
                    "total_input_tokens", 
                    "total_output_tokens",
                    "cost_usd",
                    "avg_latency_ms",
                    "unique_users"
                ]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "cost_usd": st.column_config.NumberColumn("비용 (USD)", format="$%.6f"),
                    "call_count": st.column_config.NumberColumn("호출 수", format="%d"),
                    "total_input_tokens": st.column_config.NumberColumn("입력 토큰", format="%d"),
                    "total_output_tokens": st.column_config.NumberColumn("출력 토큰", format="%d"),
                    "avg_latency_ms": st.column_config.NumberColumn("평균 지연(ms)", format="%d"),
                    "unique_users": st.column_config.NumberColumn("사용자 수", format="%d"),
                }
            )
            
            # 일별 비용 추이
            try:
                res_daily_cost = sb.table("analytics_total_cost_summary").select("*").limit(30).execute()
                if res_daily_cost.data:
                    df_daily_cost = pd.DataFrame(res_daily_cost.data)
                    df_daily_cost["date"] = pd.to_datetime(df_daily_cost["date"])
                    df_daily_cost = df_daily_cost.sort_values("date")
                    
                    st.markdown("**일별 비용 추이 (최근 30일)**")
                    st.line_chart(df_daily_cost.set_index("date")["daily_cost_usd"])
            except:
                pass
                
    except Exception as e:
        st.error(f"모델 비용 통계 로드 실패: {e}")
    
    st.markdown("---")
    
    # 3. 사용자 활동 지표
    st.subheader("👥 사용자 활동 분석 (Engagement)")
    
    m0, m1, m2, m3, m4 = st.columns(5)
    m0.metric("🏆 누적 방문수", f"{total_visits:,}회")
    m1.metric("오늘 방문자 (DAU)", f"{today_visitors}명")  # <--- 이제 정확히 나옵니다!
    m2.metric("현재 실시간", f"{len(sessions)}명")
    m3.metric("평균 체류", f"{dur_data.get('avg_duration_min', 0)}분")
    m4.metric("최대 집중", f"{dur_data.get('max_duration_min', 0)}분")

    st.divider()

    # 그래프 배치 (일별 추이 그래프는 로직 단순화)
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("##### 📉 최근 7일 접속 추이")
        # View 대신 직접 집계 (더 안정적)
        try:
            # 최근 7일치 가져오기 로직 (간단 구현)
            res_7d = sb.table("app_sessions").select("created_at").gte("created_at", (datetime.utcnow() - timedelta(days=7)).isoformat()+"Z").execute()
            df_7d = pd.DataFrame(res_7d.data)
            if not df_7d.empty:
                df_7d['created_at'] = pd.to_datetime(df_7d['created_at']).dt.tz_convert('Asia/Seoul')
                df_7d['date'] = df_7d['created_at'].dt.date
                dau_chart = df_7d.groupby('date').size()
                st.line_chart(dau_chart, height=250)
            else:
                st.info("데이터가 부족합니다.")
        except:
            st.info("차트 데이터 로드 실패")

    with col_g2:
        st.markdown("##### ⏰ 시간대별 접속 분포")
        if not df_peak.empty:
            st.bar_chart(df_peak.set_index('hour')[['visit_count']], height=250)
        else:
            st.info("데이터가 부족합니다.")

    st.divider()

    # 4. 운영 성과 및 비용 분석
    st.subheader("💰 운영 성과 및 비용 분석")
    
    # 필터
    filter_cols = st.columns([2, 2, 2, 1])
    with filter_cols[0]:
        all_users = ["(전체)"] + sorted(df["user_email"].unique().tolist()) if not df.empty else ["(전체)"]
        selected_user = st.selectbox("👤 사용자", all_users, index=0)
    with filter_cols[1]:
        min_date = df["date"].min() if not df.empty else datetime.now().date()
        max_date = df["date"].max() if not df.empty else datetime.now().date()
        date_range = st.date_input("📅 날짜 범위", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    with filter_cols[2]:
        all_models = sorted(df["model_used"].unique().tolist()) if not df.empty else []
        selected_models = st.multiselect("🤖 모델", all_models, default=all_models)
    with filter_cols[3]:
        st.write("")
        apply_filter = st.button("적용", use_container_width=True, type="primary")

    # 필터링
    filtered_df = df.copy()
    if not filtered_df.empty:
        if selected_user != "(전체)":
            filtered_df = filtered_df[filtered_df["user_email"] == selected_user]
        if isinstance(date_range, tuple) and len(date_range) == 2:
            filtered_df = filtered_df[(filtered_df["date"] >= date_range[0]) & (filtered_df["date"] <= date_range[1])]
        if selected_models:
            filtered_df = filtered_df[filtered_df["model_used"].isin(selected_models)]

    st.divider()

    # KPI
    total_cost = filtered_df["cost_usd"].sum() if not filtered_df.empty else 0
    total_tokens = filtered_df["token_usage"].sum() if not filtered_df.empty else 0
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("💰 기간 내 비용", f"${total_cost:.4f}")
    kpi2.metric("🧾 사용 토큰", f"{total_tokens:,}")
    kpi3.metric("📦 실행 횟수", f"{len(filtered_df):,}건")

    st.divider()

    # 상세 차트
    chart_tabs = st.tabs(["📈 비용/토큰", "🤖 모델 분석", "🔥 시간대/히트맵", "👤 사용자 랭킹"])
    
    with chart_tabs[0]:
        c1, c2 = st.columns(2)
        if not filtered_df.empty:
            c1.line_chart(filtered_df.groupby("date")["token_usage"].sum())
            c2.area_chart(filtered_df.groupby("date")["cost_usd"].sum())
    
    with chart_tabs[1]:
        c1, c2 = st.columns(2)
        if not filtered_df.empty:
            c1.bar_chart(filtered_df["model_used"].value_counts())
            c2.bar_chart(filtered_df.groupby("model_used")["cost_usd"].sum())

    with chart_tabs[2]:
        if not filtered_df.empty:
            try:
                import plotly.express as px
                heatmap_data = filtered_df.groupby(["weekday", "hour"])["execution_time"].mean().unstack(fill_value=0)
                # 정렬
                days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                days = [d for d in days if d in heatmap_data.index]
                if days: heatmap_data = heatmap_data.reindex(days)
                
                fig = px.imshow(heatmap_data, labels=dict(x="시간", y="요일", color="지연(s)"), aspect="auto", color_continuous_scale="RdYlGn_r")
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.bar_chart(filtered_df.groupby("hour")["execution_time"].mean())

    with chart_tabs[3]:
        c1, c2 = st.columns(2)
        if not filtered_df.empty:
            c1.bar_chart(filtered_df["user_email"].value_counts().head(10))
            c2.bar_chart(filtered_df.groupby("user_email")["cost_usd"].sum().sort_values(ascending=False).head(10))

    st.divider()

    # 로그 테이블 & 관리
    st.subheader("📋 상세 감사 로그")
    
    if not filtered_df.empty:
        # 최근 100개만 조회
        disp = filtered_df.sort_values("created_at", ascending=False).head(100)
        
        # [수정 1] 작성일시를 '한국 시간'으로 변환하고 초 단위까지 표시
        disp["created_at"] = disp["created_at"].dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # [수정 2] 소요 시간(초) 포맷팅 (예: 1.2s)
        disp["execution_time"] = disp["execution_time"].apply(lambda x: f"{x:.1f}s")
        
        # [수정 3] 비용 포맷팅
        disp["cost_usd"] = disp["cost_usd"].apply(lambda x: f"${x:.6f}")
        
        # 스타일링 함수 (헤비유저/지연 건 강조)
        def style_rows(row):
            s = [""] * len(row)
            user = row["user_email"]
            # 문자열 "1.2s"에서 "s" 떼고 숫자로 변환해 비교
            try: exec_time = float(row["execution_time"].replace("s", ""))
            except: exec_time = 0
            
            if user in heavy_users: 
                s = ["background-color: #fef3c7"] * len(row) # 노란색
            if exec_time > LONG_LATENCY_THRESHOLD: 
                s = ["background-color: #fee2e2; color: #991b1b; font-weight: bold"] * len(row) # 빨간색
            return s
            
        # ★ [핵심] 컬럼 목록에 'execution_time'을 다시 넣었습니다!
        st.dataframe(
            disp[["created_at", "user_email", "prompt", "model_used", "cost_usd", "execution_time"]]
            .style.apply(style_rows, axis=1),
            use_container_width=True
        )
        
        # 상세 내용 보기 (프롬프트 전체)
        with st.expander("🔍 프롬프트 원문 보기"):
            sel_id = st.selectbox("로그 선택", disp["id"].tolist())
            if sel_id:
                txt = filtered_df[filtered_df["id"] == sel_id]["prompt"].values[0]
                st.text_area("전체 내용", txt, height=150)
    else:
        st.info("검색 조건에 맞는 로그가 없습니다.")

    # 삭제/다운로드
    col1, col2 = st.columns(2)
    with col1:
        if not filtered_df.empty:
            st.download_button("💾 CSV 다운로드", filtered_df.to_csv(index=False).encode("utf-8-sig"), "log.csv", "text/csv")
    with col2:
        if not filtered_df.empty:
            did = st.selectbox("삭제할 ID", filtered_df["id"].head(10).tolist())
            if st.button("❌ 삭제"):
                sb.table("work_archive").delete().eq("id", did).execute()
                st.success("삭제됨")
                st.rerun()

    # 원본 데이터 탭
    st.divider()
    t1, t2 = st.tabs(["세션(Sessions)", "이벤트(Events)"])
    with t1: st.dataframe(sessions, use_container_width=True)
    with t2: st.dataframe(events, use_container_width=True)

def render_lawbot_button(url: str):
    st.markdown(
        f"""
<a href="{_escape(url)}" target="_blank" class="lawbot-btn">
  <div style="font-size: 1.5rem; font-weight: 800; margin-bottom: 0.4rem; color: #FFD700; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">
    🤖 법령 AI (Lawbot) 실행 — 법령·규칙·서식 더 찾기(🖱️✨클릭)
  </div>
  <div style="font-size: 1rem; opacity: 0.95; font-weight: 500; color: rgba(255, 255, 255, 0.9);">
    클릭하면 검색창에 키워드가 들어간 상태로 새창이 열립니다
  </div>
</a>
""",
        unsafe_allow_html=True,
    )
# =========================================================
# 9) FOLLOWUP (깨진 부분 복원)
# =========================================================
def _followup_agent_answer(res: dict, user_q: str) -> Tuple[str, Optional[dict]]:
    """
    반환:
      - assistant_markdown: 사용자에게 보여줄 답변(마크다운)
      - updated_doc(optional): 공문 JSON 갱신이 필요하면 새 doc dict, 아니면 None
    """
    situation = res.get("situation", "")
    analysis = res.get("analysis", {})
    law_md = strip_html(res.get("law", ""))
    strategy = res.get("strategy", "")
    procedure = res.get("procedure", {})
    objections = res.get("objections", [])
    doc = res.get("doc", {})
    meta = res.get("meta", {})

    # 컨텍스트 과다 방지(문법 깨져있던 부분 여기서 안전하게 처리)
    ctx = f"""
[원 케이스]
{_short_for_context(mask_sensitive(situation), 1800)}

[케이스 분석]
{_short_for_context(json.dumps(analysis, ensure_ascii=False), 1400)}

[법령 근거(요약)]
{_short_for_context(law_md, 1600)}

[처리 가이드]
{_short_for_context(strategy, 1600)}

[절차 플랜]
{_short_for_context(json.dumps(procedure, ensure_ascii=False), 1200)}

[예상 반발]
{_short_for_context(json.dumps(objections, ensure_ascii=False), 800)}

[현재 공문(JSON)]
{_short_for_context(json.dumps(doc, ensure_ascii=False), 1400)}

[사용자 질문]
{mask_sensitive(user_q)}
""".strip()

    # 질문이 문서 수정/재작성 성격인지 간단 판별
    need_doc = bool(re.search(r"(공문|문서|회신|수정|고쳐|다시|재작성|문안|문구|제목|수신|본문)", user_q))

    if need_doc:
        prompt = f"""
너는 행정 실무 베테랑이다. 아래 컨텍스트를 기반으로 사용자의 질문에 답하고,
필요하면 공문(JSON)도 함께 수정하라.

[출력 형식 - 반드시 JSON 하나로만]
{{
  "answer_md": "사용자에게 보여줄 마크다운 답변(간결, 실무형)",
  "doc_update": {{
    "title": "제목",
    "receiver": "수신",
    "body_paragraphs": ["문단1","문단2"],
    "department_head": "OOO과장"
  }}
}}

- doc_update는 '공문 수정이 필요할 때만' 넣고, 아니면 null
- 다른 텍스트 금지. JSON만.
"""
        out = llm_service.generate_json(ctx + "\n\n" + prompt)
        if isinstance(out, dict):
            answer_md = (out.get("answer_md") or "").strip() or "처리 방향을 정리했습니다."
            doc_update = out.get("doc_update", None)
            if isinstance(doc_update, dict) and doc_update.get("title") and doc_update.get("body_paragraphs"):
                return answer_md, doc_update
            return answer_md, None
        return "후속 답변 생성 중 오류가 발생했습니다. 질문을 조금 더 구체화해 주세요.", None

    # 일반 질의응답
    prompt2 = f"""
너는 행정 실무 베테랑이다. 아래 컨텍스트를 기반으로 사용자 질문에 실무적으로 답하라.
- 서론/공감 금지, 바로 답
- 절차/증빙/기한 관점으로 정리
- 길게 늘어지지 말 것

마크다운으로만 출력.
"""
    ans = llm_service.generate_text(ctx + "\n\n" + prompt2)
    return (ans or "").strip() or "답변 생성 실패", None
def main():
    sb = get_supabase()
    ensure_anon_session_id()

    if sb:
        touch_session(sb)
        if "boot_logged" not in st.session_state:
            st.session_state.boot_logged = True
            log_event(sb, "app_open", meta={"ver": APP_VERSION})

        sidebar_auth(sb)

        # [NEW] 당직메뉴얼 버튼 추가
        st.sidebar.markdown("---")
        render_revision_sidebar_button() # [NEW] 기안/공고문 수정 버튼
        # [NEW] 민원 분석기 버튼
        if st.sidebar.button("🧾 민원 분석기", use_container_width=True):
            st.session_state["app_mode"] = "complaint_analyzer"
            st.session_state["workflow_result"] = None
            st.session_state["main_task_input"] = ""
            st.rerun()
        # [NEW] AI 민원 검증 버튼
        if st.sidebar.button("🔍 AI 민원 검증", use_container_width=True):
            st.session_state["app_mode"] = "hallucination_check"
            st.session_state["workflow_result"] = None
            st.session_state["main_task_input"] = ""
            st.rerun()
        # [NEW] 토목직 특화 AI 버튼
        if st.sidebar.button("👷 토목직 특화 AI", use_container_width=True):
            st.session_state["app_mode"] = "civil_engineering"
            st.session_state["workflow_result"] = None
            st.session_state["main_task_input"] = ""
            st.rerun()
        # [NEW] 업무지시로 돌아가기 버튼
        if st.session_state.get("app_mode") in ["revision", "complaint_analyzer", "hallucination_check", "civil_engineering"]:
            if st.sidebar.button("⬅️ 업무지시로 돌아가기", use_container_width=True):
                st.session_state["app_mode"] = None
                st.session_state["workflow_result"] = None
                st.rerun()
        render_duty_manual_button(sb, llm_service)
        render_history_list(sb)
    else:
        st.sidebar.error("Supabase 연결 정보(secrets)가 없습니다.")
        st.sidebar.caption("SUPABASE_URL / SUPABASE_ANON_KEY 필요")

    is_admin_tab = (
        sb
        and st.session_state.get("logged_in")
        and is_admin_user(st.session_state.get("user_email", ""))
        and st.session_state.get("admin_mode", False)
    )

    if is_admin_tab:
        tabs = st.tabs(["🧠 업무 처리", "🏛️ 마스터 대시보드"])
        with tabs[1]:
            render_master_dashboard(sb, llm_service)
        with tabs[0]:
            pass

    st.markdown(
        """
        <div style='text-align: center; padding: 2rem 0 3rem 0;'>
            <h1 style='font-size: 2.5rem; font-weight: 800; margin-bottom: 0.5rem; 
                       background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       background-clip: text;'>
                🏢 AI 행정관 Pro
            </h1>
            <p style='font-size: 1.1rem; color: #4b5563; font-weight: 500; margin-bottom: 0.75rem;'>
                충주시청 스마트 행정 솔루션
            </p>
            <p style='font-size: 0.9rem; color: #6b7280;'>
                문의 <a href='mailto:kim0395kk@korea.kr' style='color: #2563eb; text-decoration: none;'>kim0395kk@korea.kr</a> | Govable AI 에이전트
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ai_ok = "✅ AI" if llm_service.is_available() else "❌ AI"
    law_ok = "✅ LAW" if bool(get_general_secret("LAW_API_ID")) else "❌ LAW"
    nv_ok = "✅ NEWS" if bool(get_general_secret("NAVER_CLIENT_ID")) else "❌ NEWS"
    db_ok = "✅ DB" if sb else "❌ DB"

    st.markdown(
        f"""
        <div style='text-align: center; padding: 0.75rem 1.5rem; background: white; 
                    border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    border-left: 4px solid #2563eb;'>
            <span style='font-size: 0.9rem; color: #374151; font-weight: 600;'>
                시스템 상태: {ai_ok} · {law_ok} · {nv_ok} · {db_ok}
            </span>
            <span style='font-size: 0.85rem; color: #9ca3af; margin-left: 1rem;'>
                v{APP_VERSION}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1, 1.15], gap="large")

    with col_right:
        st.write("")  # Force column to render
        
        # 애니메이션 및 결과가 표시될 메인 플레이스홀더
        right_panel_placeholder = st.empty()

        if "workflow_result" not in st.session_state or not st.session_state.workflow_result:
            # 초기 상태: 문서 미리보기 안내 (모드별 메시지)
            with right_panel_placeholder.container():
                if st.session_state.get("app_mode") == "revision":
                    # 기안/공고문 수정 모드
                    st.markdown(
                        """
                        <div style='text-align: center; padding: 6rem 2rem; 
                                    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                                    border-radius: 16px; 
                                    border: 2px dashed #f59e0b; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                            <div style='font-size: 4rem; margin-bottom: 1rem;'>✨</div>
                            <h3 style='color: #92400e; font-size: 1.5rem; font-weight: 700; margin-bottom: 0.75rem;'>
                                수정된 문서 미리보기
                            </h3>
                            <p style='color: #b45309; font-size: 1rem; line-height: 1.6; font-weight: 500;'>
                                왼쪽에서 [수정안 생성] 버튼을 누르면<br>
                                <strong>✅ 수정된 공문서가 여기에 표시됩니다</strong>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                elif st.session_state.get("app_mode") == "civil_engineering":
                    # 토목직 특화 AI 모드
                    st.markdown(
                        """
                        <div style='text-align: center; padding: 6rem 2rem; 
                                    background: linear-gradient(135deg, #fef9c3 0%, #fde68a 100%); 
                                    border-radius: 16px; 
                                    border: 2px dashed #d97706; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                            <div style='font-size: 4rem; margin-bottom: 1rem; opacity: 0.7;'>👷</div>
                            <h3 style='color: #92400e; margin-bottom: 0.5rem; font-weight: 700;'>토목직 특화 AI 어시스턴트</h3>
                            <p style='color: #78350f; margin: 0; line-height: 1.8;'>
                                20개 내부 매뉴얼 · 지침 · 규정을 학습한 AI입니다.<br>
                                왼쪽 <strong>규정/매뉴얼 검색</strong> 탭에서 질문하거나<br>
                                <strong>공문 초안 작성</strong> 탭에서 문서를 생성하세요.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                elif st.session_state.get("app_mode") == "complaint_analyzer":
                    # 민원 분석기 모드
                    st.markdown(
                        """
                        <div style='text-align: center; padding: 6rem 2rem; 
                                    background: linear-gradient(135deg, #ecfeff 0%, #cffafe 100%); 
                                    border-radius: 16px; 
                                    border: 2px dashed #06b6d4; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                            <div style='font-size: 4rem; margin-bottom: 1rem; opacity: 0.7;'>🧾</div>
                            <h3 style='color: #0e7490; margin-bottom: 0.5rem; font-weight: 700;'>민원 분석 결과가 여기에 표시됩니다</h3>
                            <p style='color: #155e75; margin: 0; line-height: 1.5;'>
                                왼쪽에서 민원 원문을 입력하고 <strong>민원 분석 시작</strong>을 누르세요.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    # 일반 업무 지시 모드
                    st.markdown(
                        """
                        <div style='text-align: center; padding: 6rem 2rem; 
                                    background: white; border-radius: 16px; 
                                    border: 2px dashed #d1d5db; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                            <div style='font-size: 4rem; margin-bottom: 1rem; opacity: 0.5;'>📄</div>
                            <h3 style='color: #6b7280; font-size: 1.5rem; font-weight: 700; margin-bottom: 0.75rem;'>
                                Document Preview
                            </h3>
                            <p style='color: #9ca3af; font-size: 1rem; line-height: 1.6;'>
                                왼쪽에서 업무를 지시하면<br>완성된 공문서가 여기에 나타납니다.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            # 결과가 있을 때 렌더링(여기서 처리하도록 이동)
            with right_panel_placeholder.container():
                res = st.session_state.workflow_result
                
                # [REVISION MODE] 수정된 문서 렌더링
                if st.session_state.get("app_mode") == "revision":
                    revised_doc = res.get("revised_doc")
                    if revised_doc:
                        st.markdown("### 📄 수정된 공문서")
                        # 간단한 렌더링
                        st.info(f"**제목**: {revised_doc.get('title', 'N/A')}")
                        st.info(f"**수신**: {revised_doc.get('receiver', 'N/A')}")
                        st.markdown("**본문**:")
                        for p in revised_doc.get('body_paragraphs', []):
                            st.markdown(f"- {p}")
                    else:
                        st.warning("수정된 문서 내용이 없습니다.")

    with col_left:
        # ---------------------------------------------------------
        # [MODE] 기안/공고문 수정 모드
        # ---------------------------------------------------------
        if st.session_state.get("app_mode") == "revision":
            render_header("📝 기안/공고문 수정")
            
            # 사용 안내
            with st.expander("💡 사용법", expanded=False):
                st.markdown("""
                1. **원문 붙여넣기**: 수정할 기안문이나 공고문을 아래 '원문' 칸에 붙여넣으세요.
                2. **수정 요청 작성**: '수정 요청사항' 칸에 원하는 변경 내용을 작성하세요.
                   - 예: "일시를 내일로 변경", "제목을 더 부드럽게", "오타 수정" 등
                3. **생성 버튼 클릭**: 아래 '수정안 생성' 버튼을 누르면 오른쪽에 결과가 나타납니다.
                """)
            
            st.markdown("### 📄 원문")
            original_text = st.text_area(
                "원문 (기존 공문이나 공고문을 붙여넣으세요)",
                value=st.session_state.get("revision_org_text", ""),
                height=200,
                placeholder="여기에 수정할 문서의 원문을 붙여넣으세요.\n\n예시:\n제 목: 2025년 시민참여 예산 설명회 개최 안내\n수 신: 각 부서장\n발 신: 기획예산과\n\n시민참여 예산 설명회를 아래와 같이 개최하오니...",
                key="revision_org_text",
                label_visibility="collapsed",
            )
            
            st.markdown("### ✏️ 수정 요청사항 (선택)")
            revision_request = st.text_area(
                "수정 요청사항 (비워두면 '공문서 작성 표준'에 맞게 자동 교정합니다)",
                value=st.session_state.get("revision_req_text", ""),
                height=150,
                placeholder="비워두시면 '2025 개정 공문서 작성 표준'에 맞춰 오탈자, 띄어쓰기, 표현을 자동으로 교정합니다.\n\n특정 요청이 있다면 적어주세요:\n- 일시를 2025. 1. 28.로 변경해주세요\n- 제목을 좀 더 부드럽게 바꿔주세요",
                key="revision_req_text",
                label_visibility="collapsed",
            )
            
            if st.button("✨ 수정안 생성", type="primary", use_container_width=True):
                if not original_text:
                    st.warning("⚠️ 원문을 입력해주세요.")
                else:
                    # 두 입력을 합쳐서 전달
                    combined_input = f"[원문]\n{original_text}\n\n[수정 요청]\n{revision_request}"
                    
                    # 프리미엄 애니메이션과 함께 워크플로우 실행
                    user_email = st.session_state.get("user_email")
                    
                    # 오른쪽 패널에 애니메이션 표시
                    try:
                        res = render_revision_animation(
                            right_panel_placeholder,
                            run_revision_workflow,
                            combined_input,
                            llm_service,
                            sb,
                            user_email
                        )
                        
                        # res가 None일 수 있으므로 먼저 체크
                        if res is None:
                            st.error("❌ 문서 수정 기능을 사용할 수 없습니다. premium_animations 모듈을 확인해주세요.")
                        elif "error" in res:
                            st.error(res["error"])
                        else:
                            st.session_state.workflow_result = res
                            
                            # revision_id를 세션 상태에 저장
                            if "revision_id" in res:
                                st.session_state.current_revision_id = res["revision_id"]
                                st.toast("💾 수정 내역이 저장되었습니다", icon="✅")
                            
                            # 결과를 바로 표시 (rerun 제거로 깜빡임 방지)
                    except Exception as e:
                        st.error(f"처리 중 오류 발생: {str(e)}")

            # 결과가 있으면 왼쪽에 변경 로그 표시
            if "workflow_result" in st.session_state:
                res = st.session_state.workflow_result
                if res and "changelog" in res:
                    st.markdown("---")
                    render_header("🔍 변경 로그")
                    # [FIX] Use markdown list for compact spacing
                    logs = res.get("changelog", [])
                    if logs:
                        md_text = ""
                        for log in logs:
                            md_text += f"- ✅ {log}\n"
                        st.markdown(md_text)
                    
                    if res.get("summary"):
                        st.caption(res.get("summary"))

        # ---------------------------------------------------------
        # [MODE] 토목직 특화 AI (Civil Engineering AI)
        # ---------------------------------------------------------
        elif st.session_state.get("app_mode") == "civil_engineering":
            render_header("👷 토목직 특화 AI")
            
            # 탭 구성: [1. 실무 규정 질의] [2. 전문 공문 작성]
            # [UI 개선] 탭 이름에 아이콘과 명확한 행동 동사 사용
            ce_tab1, ce_tab2 = st.tabs(["🔍 규정/매뉴얼 검색", "✍️ 공문 초안 작성"])
            
            # --- Tab 1: 실무 규정 질의 (Tech Q&A) ---
            with ce_tab1:
                # [UI 개선] 탭 상단에 친절한 안내 문구 추가 (파란색 박스)
                st.info("🚧 **도로폭, 양생 온도 등 궁금한 실무 규정을 물어보세요.**\n\n내부 매뉴얼과 지침을 찾아보고 정확한 근거와 함께 답변해 드립니다.")
                
                # 채팅 기록 초기화
                if "civil_chat_history" not in st.session_state:
                    st.session_state.civil_chat_history = []
                
                # 채팅 기록 표시
                for msg in st.session_state.civil_chat_history:
                    with st.chat_message(msg["role"]):
                        if msg["role"] == "assistant":
                            render_civil_response_panel(msg)
                        else:
                            st.markdown(msg["content"])
                
                # 사용자 입력
                if prompt := st.chat_input("규정이나 지침에 대해 물어보세요 (예: 겨울철 콘크리트 양생 온도 기준?)"):
                    # 사용자 메시지 표시
                    st.session_state.civil_chat_history.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # AI 답변 생성
                    with st.chat_message("assistant"):
                        with st.spinner("규정을 찾아보고 있습니다..."):
                            # RAG 시스템 로드 (캐싱됨)
                            if load_rag_system:
                                rag = load_rag_system()
                                if rag:
                                    # RAG 답변 생성
                                    response = rag.answer_question(prompt, llm_service)
                                    answer_text = response.get("answer", "죄송합니다. 답변을 생성하지 못했습니다.")
                                    answer_summary = response.get("summary", "")
                                    sources = response.get("sources", [])
                                    parsed_facts = response.get("parsed_facts", [])
                                    fact_rows = response.get("fact_rows", [])
                                    quality = response.get("quality", {}) or {}
                                    confidence = float(response.get("confidence") or 0.0)
                                    retrieval_meta = response.get("retrieval_meta", {}) or {}
                                    
                                    # 답변 출력
                                    render_civil_response_panel(
                                        {
                                            "content": answer_text,
                                            "summary": answer_summary,
                                            "sources": sources,
                                            "fact_rows": fact_rows,
                                            "quality": quality,
                                            "confidence": confidence,
                                            "retrieval_meta": retrieval_meta,
                                        }
                                    )
                                    
                                    # 기록 저장
                                    st.session_state.civil_chat_history.append({
                                        "role": "assistant", 
                                        "content": answer_text,
                                        "sources": sources,
                                        "parsed_facts": parsed_facts,
                                        "fact_rows": fact_rows,
                                        "summary": answer_summary,
                                        "quality": quality,
                                        "confidence": confidence,
                                        "retrieval_meta": retrieval_meta,
                                    })
                                else:
                                    st.error("RAG 시스템을 로드할 수 없습니다.")
                            else:
                                st.error("RAG 모듈이 설치되지 않았습니다.")

            # --- Tab 2: 전문 공문 작성 (Tech Drafting) ---
            with ce_tab2:
                # [UI 개선] 작성 가이드 추가
                st.info("📝 **복잡한 공문서, 핵심 내용만 입력하면 전문가 수준으로 초안을 만들어 드립니다.**\n\n공사 감독 조서, 인허가 공문 등 필요한 양식을 선택하고 내용을 적어주세요.")
                
                # [UI Update] 1단 레이아웃 (Full Width)
                st.markdown("### ✍️ 작성 요청")
                
                draft_topic = st.text_input("공문 주제", placeholder="예: 수해복구 공사 착공신고서 접수 처리")
                
                # 높이 150 -> 300으로 확대
                draft_details = st.text_area("세부 사항", height=300, 
                                           placeholder="""예시:
1. 업체명: (주)대한건설
2. 공사기간: 2024.3.1 ~ 2024.6.30
3. 특이사항: 안전관리계획서 제출 완료됨.
4. 요청사항: 법령 근거를 포함해서 정중하게 작성해줘.""")
                
                if st.button("✨ 공문 초안 작성", type="primary", use_container_width=True):
                    if not draft_topic:
                        st.warning("공문 주제를 입력해주세요.")
                    else:
                        with st.spinner("규정 및 매뉴얼을 참조하여 공문을 작성 중입니다..."):
                            if load_rag_system:
                                rag = load_rag_system()
                                if rag:
                                    # 프롬프트 구성
                                    draft_prompt = f"""
다음 주제로 토목 공사 관련 공문 초안을 작성해줘.
관련 법령이나 매뉴얼(착공계 처리 요령 등)을 참고해서 필수 기재 사항을 포함해줘.

[주제]: {draft_topic}
[세부사항]: {draft_details}

형식:
1. 제목
2. 본문 (개조식)
3. 붙임 서류 목록
"""
                                    
                                    # RAG 답변 생성
                                    response = rag.answer_question(draft_prompt, llm_service)
                                    st.session_state.civil_draft_result = response
                                else:
                                    st.error("RAG 시스템 로드 실패")
                            else:
                                st.error("RAG 모듈 없음")
                
                # 결과 표시 (하단 배치)
                if "civil_draft_result" in st.session_state:
                    res = st.session_state.civil_draft_result
                    
                    st.markdown("---")
                    st.markdown("### 📄 공문 초안")
                    
                    # 결과 카드 스타일
                    st.info(res.get("answer"))
                    
                    # 복사 버튼 (텍스트 영역으로 제공)
                    with st.expander("📝 텍스트로 복사하기"):
                        st.code(res.get("answer"), language="text")
                    
                    # 근거 자료
                    if res.get("sources"):
                        with st.expander("📚 참고한 규정 및 매뉴얼", expanded=False):
                            for src in res.get("sources"):
                                st.caption(f"- {src}")

        # ---------------------------------------------------------
        # [MODE] 민원 분석기
        # ---------------------------------------------------------
        elif st.session_state.get("app_mode") == "complaint_analyzer":
            render_header("🧾 민원 분석기")

            with st.expander("💡 이 도구가 하는 일", expanded=False):
                st.markdown("""
- 민원 텍스트를 **주장(Claim) 단위로 분해**
- **환각/허위 법령 인용** 신호 점검
- 국가법령정보센터 **공식 API**로 법령/조문 존재 여부를 확인(가능 범위)
- 공무원이 책임질 수 있도록 **단정 최소** 형태의 회신 초안을 생성
                """)

            complaint_input = st.text_area(
                "민원 원문",
                value=st.session_state.get("complaint_input", ""),
                height=240,
                placeholder="민원 원문을 붙여넣으세요. (개인정보는 제거 권장)\n예: 언제/어디/대상/요청/증거가 포함되면 정확도가 크게 올라갑니다.",
                key="complaint_input",
                label_visibility="collapsed",
            )

            st.markdown(
                """
                <div style='background: #fef3c7; border-left: 4px solid #f59e0b; 
                            padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
                    <p style='margin: 0; color: #92400e; font-size: 0.9rem; font-weight: 500;'>
                        ⚠️ 민감정보(성명·연락처·주소·주민번호·차량번호 등) 입력 금지 / 또는 마스킹 후 입력
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("🧾 민원 분석 시작", type="primary", use_container_width=True):
                if not complaint_input:
                    st.warning("민원 원문을 입력해주세요.")
                else:
                    res = run_complaint_analyzer_workflow(complaint_input, right_panel_placeholder)

                    archive_id = None
                    if sb:
                        archive_id = db_insert_archive(sb, complaint_input, res)
                        if archive_id:
                            st.session_state.current_archive_id = archive_id
                            log_event(sb, "complaint_analyzer_run", archive_id=archive_id, meta={"prompt_len": len(complaint_input)})

                    res["archive_id"] = archive_id
                    st.session_state.workflow_result = res
                    st.session_state.followup_messages = []
                    st.rerun()

            # 결과 렌더 (좌측)
            if st.session_state.get("workflow_result") and st.session_state.get("app_mode") == "complaint_analyzer":
                res = st.session_state.workflow_result
                render_service_health(sb, llm_service)
                render_trust_panel(res)
                pack = res.get("complaint_pack") or {}
                grade = pack.get("noise_grade") or "GREEN"
                vscore = pack.get("verifiability_score")
                try:
                    vscore_pct = int(float(vscore) * 100)
                except Exception:
                    vscore_pct = None

                st.markdown(
                    f"""
                    <div style='background: #ecfeff; padding: 1rem; border-radius: 8px; border-left: 4px solid #06b6d4; margin-bottom: 1rem;'>
                        <p style='margin: 0 0 0.25rem 0; color: #0e7490; font-weight: 700;'>등급: {grade}</p>
                        <p style='margin: 0; color: #155e75;'>검증가능성: {str(vscore_pct) + '%' if vscore_pct is not None else '-'}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander("🧩 MVC(언제/어디/대상/요청/증거) 추출", expanded=False):
                    st.json(pack.get("mvc") or {})

                with st.expander("🧱 주장(Claim) 단위 판정", expanded=True):
                    verdicts = pack.get("verdicts") or []
                    if not verdicts:
                        st.info("주장 판정 결과가 없습니다.")
                    for v in verdicts:
                        st.markdown(f"**{v.get('claim_id','')}** · {v.get('verdict','INSUFFICIENT')}  \n{v.get('safe_statement','')}")
                        need = v.get("needed") or []
                        if need:
                            st.caption("추가 필요:")
                            for n in need[:7]:
                                st.write("- ", n)
                        st.divider()

                st.markdown(res.get("law") or "")
                st.markdown("#### 🔧 처리/회신 전략")
                st.markdown(res.get("strategy") or "")


        # ---------------------------------------------------------
        # [MODE] 환각 검증 모드
        # ---------------------------------------------------------
        elif st.session_state.get("app_mode") == "hallucination_check":
            render_header("🔍 AI 생성 민원 검증 시스템")
            
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
통계청 자료에 따르면 정확히 47.3829%가...""",
                    key="hallucination_petition_input"
                )
            
            with col2:
                uploaded_file = st.file_uploader(
                    "또는 파일 업로드",
                    type=['txt', 'docx', 'pdf'],
                    help="민원 문서를 업로드하세요",
                    key="hallucination_file_upload"
                )
                
                if uploaded_file:
                    try:
                        import io
                        if uploaded_file.type == "text/plain":
                            petition_input = uploaded_file.read().decode('utf-8')
                            st.session_state.hallucination_petition_input = petition_input
                        elif uploaded_file.type == "application/pdf":
                            st.info("PDF 파일 파싱 중...")
                            # TODO: PDF 파싱 로직 추가
                        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
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
                import time
                
                # 실시간 진행 로그 (st.empty로 교체 방식)
                log_placeholder = st.empty()
                progress_bar = st.progress(0)
                log_messages = []
                
                def add_log(icon, message, elapsed=None):
                    """실시간 로그 메시지 추가 (덮어쓰기 방식)"""
                    time_str = f"[{elapsed:.1f}초]" if elapsed is not None else ""
                    log_messages.append(f"{icon} {time_str} {message}")
                    log_placeholder.markdown(
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
                    
                    time.sleep(1)  # 완료 로그를 잠시 보여줌
                    log_placeholder.empty()  # 로그 제거
                    progress_bar.empty()
                    
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


        # ---------------------------------------------------------
        # [MODE] 기본 모드 (업무 지시)
        # ---------------------------------------------------------
        else:
            render_header("🗣️ 업무 지시")
            render_autotune_hint()

            user_input = st.text_area(
                "업무 내용",
                value=st.session_state.get("main_task_input", ""),
                height=190,
                placeholder="예시\n- 상황: (무슨 일 / 어디 / 언제 / 증거 유무...)\n- 쟁점: (요건/절차/근거...)\n- 요청: (원하는 결과물: 회신/사전통지/처분 등)",
                key="main_task_input",
                label_visibility="collapsed",
            )

            st.markdown(
                """
                <div style='background: #fef3c7; border-left: 4px solid #f59e0b; 
                            padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
                    <p style='margin: 0; color: #92400e; font-size: 0.9rem; font-weight: 500;'>
                        ⚠️ 민감정보(성명·연락처·주소·차량번호 등) 입력 금지
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("⚡ 스마트 분석 시작", type="primary", use_container_width=True):
                if not user_input:
                    st.warning("내용을 입력해주세요.")
                else:
                    tuned_input = user_input + _autotune_instruction_text()
                    # 진행 상황은 run_workflow 내부에서 애니메이션으로 표시됨 (오른쪽 패널)
                    res = run_workflow(tuned_input, right_panel_placeholder)
                    res["app_mode"] = st.session_state.get("app_mode", "신속")

                    archive_id = None
                    if sb:
                        archive_id = db_insert_archive(sb, user_input, res)
                        if archive_id:
                            st.session_state.current_archive_id = archive_id
                            log_event(sb, "workflow_run", archive_id=archive_id, meta={"prompt_len": len(user_input)})

                    res["archive_id"] = archive_id
                    st.session_state.workflow_result = res
                    st.session_state.followup_messages = []
                    st.rerun()

            if st.session_state.get("workflow_result"):
                res = st.session_state.workflow_result
                render_service_health(sb, llm_service)
                
                # [SAFETY] 결과가 문자열인 경우(에러 메시지 등) 처리
                if isinstance(res, str):
                    try:
                        import json as _json
                        res = _json.loads(res)
                    except:
                        # JSON 파싱도 실패하면 텍스트를 분석 결과로 포장
                        res = {
                            "analysis": {
                                "case_type": "일반 민원", 
                                "core_issue": ["분석 결과가 텍스트 형식입니다."], 
                                "summary": res,
                                "required_facts": [],
                                "required_evidence": [],
                                "risk_flags": [],
                                "recommended_next_action": []
                            },
                            "law": "",
                            "strategy": res,  # 처리가이드에 텍스트 표시
                            "procedure": {"timeline": [], "checklist": [], "templates": []}
                        }
                    # 변환된 결과를 다시 세션에 저장 (선택적)
                    # st.session_state.workflow_result = res

                if res:  # None 체크
                    pack = res.get("lawbot_pack") or {}
                if pack.get("url"):
                    render_lawbot_button(pack["url"])
                render_trust_panel(res)

                render_header("🧠 케이스 분석")

                a = res.get("analysis", {})
                st.markdown(
                    f"""
                    <div style='background: #eff6ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #2563eb; margin-bottom: 1rem;'>
                        <p style='margin: 0 0 0.5rem 0; color: #1e40af; font-weight: 600;'>유형: {a.get('case_type','')}</p>
                        <p style='margin: 0; color: #1e40af;'>쟁점: {", ".join(a.get("core_issue", []))}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander("📋 누락정보/증빙/리스크/다음행동 보기", expanded=False):
                    st.markdown("**추가 확인 질문**")
                    for x in a.get("required_facts", []):
                        st.write("- ", x)
                    st.markdown("**필요 증빙**")
                    for x in a.get("required_evidence", []):
                        st.write("- ", x)
                    st.markdown("**절차 리스크**")
                    for x in a.get("risk_flags", []):
                        st.write("- ", x)
                    st.markdown("**권장 다음 행동**")
                    for x in a.get("recommended_next_action", []):
                        st.write("- ", x)

                # 법령 근거 + 뉴스/사례 2단 레이아웃
                law_col, news_col = st.columns(2, gap="medium")
                
                with law_col:
                    render_header("📜 핵심 법령 근거")
                    law_content = res.get("law", "")
                    # 스크롤 가능한 컨테이너 (st.container + height)
                    with st.container(height=400):
                        st.markdown(law_content)
                
                with news_col:
                    render_header("📰 뉴스/사례")
                    news_content = res.get("search", "")
                    # 스크롤 가능한 컨테이너 (st.container + height)
                    with st.container(height=400):
                        st.markdown(news_content)
                
                # 원문 링크 섹션
                law_pack = res.get("law_pack", {})
                items = law_pack.get("items", [])
                if items:
                    # 원문 링크들을 그리드로 표시
                    link_cols = st.columns(3)
                    for idx, item in enumerate(items[:9]):  # 최대 9개
                        law_name = item.get("law_name", "법령")
                        link = item.get("current_link", "")
                        if link:
                            with link_cols[idx % 3]:
                                st.markdown(
                                    f"""
                                    <a href='{link}' target='_blank' style='display: block; 
                                        background: linear-gradient(135deg, #ffffff 0%, #fefce8 100%); 
                                        padding: 1rem 1.25rem; border-radius: 12px;
                                        text-decoration: none; color: #92400e; font-weight: 700;
                                        font-size: 1.1rem;
                                        border: 2px solid #fcd34d; margin-bottom: 0.75rem;
                                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                        transition: all 0.2s ease;'>
                                        <span style='font-size: 1.3rem; margin-right: 0.5rem;'>📄</span>
                                        {law_name}
                                    </a>
                                    <style>
                                        a:hover {{
                                            transform: translateY(-2px);
                                            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
                                        }}
                                    </style>
                                    """,
                                    unsafe_allow_html=True
                                )

                render_header("🧭 처리 가이드")
                st.markdown(res.get("strategy", ""))

                render_header("🗺️ 절차 플랜")
                proc = res.get("procedure", {})
                with st.expander("타임라인", expanded=True):
                    for step in proc.get("timeline", []):
                        st.markdown(f"**{step.get('step')}. {step.get('name')}** — {step.get('goal')}")
                        for x in step.get("actions", []):
                            st.write("- 행동:", x)
                        for x in step.get("records", []):
                            st.write("- 기록:", x)
                        if step.get("legal_note"):
                            st.caption(f"법/유의: {step['legal_note']}")
                        st.write("")
                with st.expander("체크리스트/서식", expanded=False):
                    st.markdown("**체크리스트**")
                    for x in proc.get("checklist", []):
                        st.write("- ", x)
                    st.markdown("**필요 서식/문서**")
                    for x in proc.get("templates", []):
                        st.write("- ", x)

    # ---------------------------------------------------------
    # RIGHT PANEL RENDER (결과물)
    # ---------------------------------------------------------
    if "workflow_result" in st.session_state and st.session_state.workflow_result:
        # 오른쪽 패널에 결과 렌더링
        with right_panel_placeholder.container():
            res = st.session_state.workflow_result
            
            # [REVISION MODE] 수정된 문서 렌더링
            if st.session_state.get("app_mode") == "revision":
                revised_doc = res.get("revised_doc")
                if res.get("warning"):
                    st.warning(res.get("warning"))
                if res.get("model_used"):
                    st.caption(f"실사용 모델: {res.get('model_used')}")
                if revised_doc:
                    render_header("📄 수정된 공문서")
                    html = f"""
        <div class="paper-sheet">
          <div class="stamp">직인생략</div>
          <div class="doc-header">{_escape(revised_doc.get('title') or '공 문 서')}</div>
          <div class="doc-info">
            <span>문서번호: (수정본)</span>
            <span>시행일자: {safe_now_utc_iso()[:10]}</span>
            <span>수신: {_escape(revised_doc.get('receiver') or '수신자 참조')}</span>
          </div>
          <hr style="border: 1px solid black; margin-bottom: 30px;">
          <div class="doc-body">
        """
                    paragraphs = revised_doc.get("body_paragraphs", [])
                    if isinstance(paragraphs, str):
                        paragraphs = [paragraphs]
                    for p in paragraphs:
                        html += f"<p style='margin-bottom: 14px;'>{md_bold_to_html_safe(p)}</p>"
                    html += f"""
          </div>
          <div class="doc-footer">{_escape(revised_doc.get('department_head') or '행정기관장')}</div>
        </div>
        """
                    st.markdown(html, unsafe_allow_html=True)
                    
                    # 수정된 공문서 HWPX 다운로드
                    st.divider()
                    try:
                        from datetime import datetime
                        hwpx_bytes = generate_official_docx(revised_doc)
                        today_str = datetime.now().strftime("%Y%m%d")
                        title = revised_doc.get('title', '수정문서')[:20]
                        filename = f"[수정공문]_{today_str}_{title}.docx"
                        
                        st.download_button(
                            label="📥 수정된 공문서(DOCX) 다운로드",
                            data=hwpx_bytes,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True,
                            type="primary"
                        )
                    except Exception as e:
                        st.error(f"HWPX 생성 오류: {str(e)}")
                else:
                    st.info("수정된 문서 내용이 없습니다.")

            # [NORMAL MODE] 기존 공문 렌더링
            else:
                doc = res.get("doc")
                meta = res.get("meta") or {}
                archive_id = res.get("archive_id") or st.session_state.get("current_archive_id")

                if st.session_state.get("app_mode") == "complaint_analyzer":
                    render_header("📄 민원 회신(초안)")
                else:
                    render_header("📄 공문서")

                if not doc:
                    st.warning("공문 생성 결과(doc)가 비어 있습니다.")
                else:
                    html = f"""
        <div class="paper-sheet">
          <div class="stamp">직인생략</div>
          <div class="doc-header">{_escape(doc.get('title', '공 문 서'))}</div>
          <div class="doc-info">
            <span>문서번호: {_escape(meta.get('doc_num',''))}</span>
            <span>시행일자: {_escape(meta.get('today_str',''))}</span>
            <span>수신: {_escape(doc.get('receiver', '수신자 참조'))}</span>
          </div>
          <hr style="border: 1px solid black; margin-bottom: 30px;">
          <div class="doc-body">
        """
                    paragraphs = doc.get("body_paragraphs", [])
                    if isinstance(paragraphs, str):
                        paragraphs = [paragraphs]
                    for p in paragraphs:
                        html += f"<p style='margin-bottom: 14px;'>{md_bold_to_html_safe(p)}</p>"
                    html += f"""
          </div>
          <div class="doc-footer">{_escape(doc.get('department_head', '행정기관장'))}</div>
        </div>
        """
                    st.markdown(html, unsafe_allow_html=True)
                
                # DOCX 다운로드 버튼
                st.divider()
                
                # 날짜 문자열 미리 생성 (스코프 문제 해결)
                from datetime import datetime
                today_str = datetime.now().strftime("%Y%m%d")
                
                col1, col2 = st.columns(2)
                
                # 왼쪽: 처리가이드
                with col1:
                    try:
                        # 데이터 타입 안전 처리
                        guide_data = res
                        if isinstance(guide_data, str):
                            try:
                                import _json
                                guide_data = _json.loads(guide_data)
                            except:
                                guide_data = {"analysis": {"summary": str(guide_data)}}
                        
                        guide_bytes = generate_guide_docx(guide_data)
                        filename = f"[보고서]_{today_str}_처리가이드.docx"
                        
                        st.download_button(
                            label="📊 처리가이드(DOCX) 다운로드",
                            data=guide_bytes,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"보고서 생성 오류: {str(e)}")
                
                # 오른쪽: 공문서
                with col2:
                    try:
                        # 데이터 타입 안전 처리
                        doc_data = doc
                        if isinstance(doc_data, str):
                            doc_data = {"title": "문서", "body_paragraphs": [str(doc_data)]}
                            
                        docx_bytes = generate_official_docx(doc_data)
                        title_safe = doc_data.get('title', '문서')[:20].replace('/', '_').replace('\\', '_')
                        filename = f"[공문]_{today_str}_{title_safe}.docx"
                        
                        st.download_button(
                            label="📥 공문서(DOCX) 다운로드",
                            data=docx_bytes,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"공문서 생성 오류: {str(e)}")
                
                # 데이터 브릿지: 이 초안을 기안문 수정으로 보내기
                st.divider()
                if st.button("📝 이 초안을 기안문 수정으로 보내기", type="primary", use_container_width=True, key="send_to_revision"):
                    # 데이터 추출 및 포맷팅
                    title = doc.get("title", "")
                    body_paragraphs = doc.get("body_paragraphs", [])
                    if isinstance(body_paragraphs, str):
                        body_paragraphs = [body_paragraphs]
                    
                    # 온나라 시스템 기안 서식 형식으로 변환
                    formatted_text = f"제목: {title}\n\n"
                    formatted_text += "\n".join(body_paragraphs)
                    
                    # 세션 상태에 주입
                    st.session_state.revision_org_text = formatted_text
                    st.session_state.revision_req_text = ""  # 수정 요청사항 초기화
                    
                    # 모드 전환
                    st.toast("🚀 초안 데이터를 수정 모드로 전송 중...")
                    st.session_state.app_mode = "revision"
                    st.session_state.workflow_result = None  # 기존 결과 초기화
                    st.rerun()

                render_header("💬 후속 질문")
                conf_score, _, _ = _compute_result_confidence(res or {})
                render_feedback_panel(
                    sb,
                    archive_id,
                    mode=str(st.session_state.get("app_mode") or "default"),
                    confidence_score=conf_score,
                )

                if not archive_id:
                    st.info("저장된 archive_id가 없습니다. (DB 저장 실패 가능)")
                else:
                    # DB 저장 성공 표시
                    st.success("✅ 업무 지시 내용이 DB에 안전하게 저장되었습니다.")

                if "followup_messages" not in st.session_state:
                    st.session_state.followup_messages = res.get("followups", []) or []

                used = len([m for m in st.session_state.followup_messages if m.get("role") == "user"])
                remain = max(0, MAX_FOLLOWUP_Q - used)
                
                pack = res.get("lawbot_pack") or {}
                if pack.get("url"):
                    render_lawbot_button(pack["url"])

                for m in st.session_state.followup_messages:
                    with st.chat_message(m["role"]):
                        st.markdown(m["content"])

                if remain == 0:
                    st.markdown(
                        """
                        <div style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
                                    padding: 1rem; border-radius: 12px; border-left: 4px solid #ef4444;
                                    text-align: center; margin: 1.5rem 0;'>
                            <p style='margin: 0; color: #991b1b; font-weight: 600; font-size: 1rem;'>
                                ⚠️ 후속 질문 한도(5회)를 모두 사용했습니다.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div style='background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); 
                                    padding: 1.25rem; border-radius: 12px; 
                                    border: 2px solid #3b82f6;
                                    margin: 1.5rem 0 1rem 0;
                                    box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.2);
                                    animation: pulse-border 2s ease-in-out infinite;'>
                            <div style='display: flex; align-items: center; gap: 1rem;'>
                                <div style='font-size: 2.5rem; line-height: 1;'>💬</div>
                                <div style='flex: 1;'>
                                    <p style='margin: 0 0 0.5rem 0; color: #1e40af; font-weight: 700; font-size: 1.1rem;'>
                                        👇 아래 입력창에 후속 질문을 입력하세요 (남은 횟수: {remain}회)
                                    </p>
                                    <p style='margin: 0; color: #3b82f6; font-size: 0.9rem;'>
                                        분석 결과에 대해 추가로 궁금한 점을 물어보세요
                                    </p>
                                </div>
                            </div>
                        </div>
                        <style>
                            @keyframes pulse-border {{
                                0%, 100% {{ border-color: #3b82f6; }}
                                50% {{ border-color: #60a5fa; }}
                            }}
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )

                # 후속 질문 자동 제안/히스토리
                suggested_q = [
                    "이 케이스에서 당장 필요한 증빙 3가지만 추려줘",
                    "담당자 체크리스트를 우선순위 순으로 다시 정리해줘",
                    "민원인 반발 가능성이 높은 포인트와 대응 문구를 알려줘",
                    "내일 결재용으로 5줄 요약본 만들어줘",
                ]
                analysis_issues = (res.get("analysis", {}) or {}).get("core_issue", [])[:2]
                for core in analysis_issues:
                    if core:
                        suggested_q.append(f"'{core}' 관련해서 반드시 확인할 사실관계는?")
                # 중복 제거
                suggested_q = list(dict.fromkeys([q0 for q0 in suggested_q if q0]))[:6]

                recent_user_q = [m.get("content", "") for m in st.session_state.followup_messages if m.get("role") == "user"][-8:]
                if suggested_q:
                    st.caption("빠른 질문")
                    qcols = st.columns(2)
                    for i, sq in enumerate(suggested_q):
                        if qcols[i % 2].button(f"💡 {sq[:28]}{'...' if len(sq) > 28 else ''}", key=f"fup_suggest_{i}", use_container_width=True):
                            st.session_state["followup_prefill"] = sq
                            st.rerun()
                if recent_user_q:
                    picked_hist = st.selectbox(
                        "과거 질문 다시 쓰기",
                        options=["(선택 안 함)"] + recent_user_q[::-1],
                        index=0,
                        key="followup_history_pick",
                    )
                    if picked_hist != "(선택 안 함)" and st.button("↩️ 질문칸에 넣기", key="followup_history_apply"):
                        st.session_state["followup_prefill"] = picked_hist
                        st.rerun()

                q = st.chat_input("💭 후속 질문을 입력하세요... (Enter로 전송)")
                prefill_applied = False
                if not q and st.session_state.get("followup_prefill"):
                    prefill = st.session_state.pop("followup_prefill")
                    q = prefill
                    prefill_applied = True
                if q:
                    turn = used + 1
                    st.session_state.followup_messages.append({"role": "user", "content": q})
                    db_insert_followup(sb, archive_id, turn=turn * 2 - 1, role="user", content=q)
                    log_event(sb, "followup_user", archive_id=archive_id, meta={"turn": turn, "prefill": prefill_applied})

                    # This part needs to be inside the container to be rendered by the placeholder
                    with st.chat_message("user"):
                        st.markdown(q)

                    case_context = f"""
[케이스]
상황: {res.get('situation','')}

케이스 분석:
{json.dumps(res.get("analysis", {}), ensure_ascii=False)}

법령(요약):
{strip_html(res.get('law',''))[:2500]}

절차 플랜:
{json.dumps(res.get("procedure", {}), ensure_ascii=False)[:2000]}

반발/대응:
{json.dumps(res.get("objections", []), ensure_ascii=False)[:1500]}

처리방향:
{res.get('strategy','')[:2200]}
"""
                    prompt = f"""
너는 '케이스 고정 행정 후속 Q&A'이다.
{case_context}

[사용자 질문]
{q}

[규칙]
- 위 컨텍스트 범위에서만 답한다.
- 절차/증빙/기록 포인트를 우선 제시한다.
- 모르면 모른다고 말하고, 추가 법령 근거는 Lawbot으로 찾게 안내한다.
- 서론 없이 실무형으로.
"""
                    with st.chat_message("assistant"):
                        with st.spinner("후속 답변 생성 중..."):
                            ans = llm_service.generate_text(prompt, preferred_model=MODEL_WORK_INSTRUCTION)
                            st.markdown(ans)

                    st.session_state.followup_messages.append({"role": "assistant", "content": ans})
                    db_insert_followup(sb, archive_id, turn=turn * 2, role="assistant", content=ans)
                    log_event(sb, "followup_assistant", archive_id=archive_id, meta={"turn": turn})

                    st.rerun()

if __name__ == "__main__":
    main()
