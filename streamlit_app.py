# streamlit_app.py
# -*- coding: utf-8 -*-
import json
import re
import time
import uuid
import urllib.parse
import xml.etree.ElementTree as ET
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
    import vertexai
    from vertexai.generative_models import GenerativeModel
except Exception:
    pass

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


# =========================================================
# 2) STYLES  (✅ 여기 CSS/디자인은 네가 준 그대로. 변경 없음)
# =========================================================
st.set_page_config(layout="wide", page_title="AI 행정관 Pro - Govable AI", page_icon="⚖️",initial_sidebar_state="expanded",)
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
    
    h1 { font-size: 2.5rem !important; }
    h2 { font-size: 1.75rem !important; margin-top: var(--space-xl) !important; }
    h3 { font-size: 1.25rem !important; margin-top: var(--space-lg) !important; }

    /* ====================== */
    /* Hide Default Elements */
    /* ====================== */
    header [data-testid="stToolbar"] { display: none !important; }
    header [data-testid="stDecoration"] { display: none !important; }
    header { height: 0px !important; }
    footer { display: none !important; }
    div[data-testid="stStatusWidget"] { display: none !important; }

    /* ====================== */
    /* Premium Agent Logs */
    /* ====================== */
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
            return st.secrets.get(path1, {}).get(path2)
        return st.secrets.get(path1)
    except Exception:
        return None

def get_general_secret(key: str) -> Optional[str]:
    return (st.secrets.get("general", {}) or {}).get(key) or st.secrets.get(key)

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

class LLMService:
    """Vertex AI 기반 LLM 서비스 (Gemini API 및 Groq 폴백 지원)"""
    
    def __init__(self):
        self.groq_key = get_general_secret("GROQ_API_KEY")
        self.gemini_key = get_general_secret("GEMINI_API_KEY")
        
        self.vertex_initialized = False
        self.model = None
        self.model_name = "gemini-2.5-flash"
        
        # Gemini API 모델 우선순위
        self.gemini_models = [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
        ]
        
        # Vertex AI 초기화
        self._init_vertex_ai()
        
        # Gemini API 초기화 (폴백용)
        self.gemini_api_ready = False
        
        # DEBUG: Check secrets
        # st.sidebar.write(f"DEBUG: gemini_key present? {bool(self.gemini_key)}")
        # st.sidebar.write(f"DEBUG: genai module? {bool(genai)}")
        
        if self.gemini_key and genai:
            try:
                genai.configure(api_key=self.gemini_key)
                self.gemini_api_ready = True
            except Exception as e:
                self.gemini_api_ready = False
                st.sidebar.error(f"Gemini Init Error: {e}")
        else:
            if not self.gemini_key:
                st.sidebar.warning("Gemini API Key missing")
                # Try to read directly to see if helper is broken
                direct_key = st.secrets.get("general", {}).get("GEMINI_API_KEY")
                if direct_key:
                     st.sidebar.info(f"Key found directly but helper failed? {bool(direct_key)}")
                     self.gemini_key = direct_key # Force fix
            if not genai:
                st.sidebar.error("google.generativeai module not found")
        
        # Groq 클라이언트 (최후 폴백용)
        self.groq_client = None
        if self.groq_key and Groq:
            try:
                self.groq_client = Groq(api_key=self.groq_key)
            except Exception:
                self.groq_client = None

    def _get_vertex_config(self) -> dict:
        """secrets.toml에서 Vertex AI 설정 읽기"""
        try:
            vertex_config = st.secrets.get("vertex_ai", {})
            return {
                "project_id": vertex_config.get("PROJECT_ID", ""),
                "location": vertex_config.get("LOCATION", "asia-northeast3"),
                "model_name": vertex_config.get("MODEL_NAME", "gemini-2.5-flash"),
                "service_account_json": vertex_config.get("SERVICE_ACCOUNT_JSON", ""),
            }
        except Exception:
            return {}

    def _init_vertex_ai(self):
        """Vertex AI 초기화"""
        if not vertexai or not GenerativeModel:
            return
        
        config = self._get_vertex_config()
        project_id = config.get("project_id", "")
        location = config.get("location", "asia-northeast3")
        self.model_name = config.get("model_name", "gemini-2.5-flash")
        service_account_json = config.get("service_account_json", "")
        
        if not project_id:
            return
        
        try:
            # 서비스 계정 JSON이 있으면 사용, 없으면 ADC 사용
            if service_account_json:
                import os
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_json
            
            vertexai.init(project=project_id, location=location)
            self.model = GenerativeModel(self.model_name)
            self.vertex_initialized = True
        except Exception as e:
            st.session_state["vertex_init_error"] = str(e)
            self.vertex_initialized = False

    def is_available(self) -> bool:
        return self.vertex_initialized or self.gemini_api_ready or (self.groq_client is not None)

    def _try_vertex_text(self, prompt: str) -> Tuple[str, str]:
        """Vertex AI로 텍스트 생성"""
        if not self.vertex_initialized or not self.model:
            raise Exception("Vertex AI not initialized")
        
        try:
            response = self.model.generate_content(prompt)
            return (response.text or "").strip(), self.model_name
        except Exception as e:
            raise Exception(f"Vertex AI error: {e}")

    def _try_gemini_api_text(self, prompt: str) -> Tuple[str, str]:
        """Gemini API (google.generativeai)로 텍스트 생성 (폴백)"""
        if not self.gemini_api_ready:
            reason = "Unknown"
            if not self.gemini_key:
                reason = "API Key Missing"
            elif not genai:
                reason = "Module Missing (Restart Terminal)"
            else:
                reason = "Init Failed (Check Sidebar)"
            raise Exception(f"Gemini API not ready: {reason}")
            
        last_error = None
        for m_name in self.gemini_models:
            try:
                model = genai.GenerativeModel(m_name)
                response = model.generate_content(prompt)
                return (response.text or "").strip(), m_name
            except Exception as e:
                last_error = e
                continue
        
        raise Exception(f"All Gemini API models failed. Last error: {last_error}")

    def _generate_groq(self, prompt: str) -> str:
        """Groq 폴백"""
        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return (completion.choices[0].message.content or "").strip()
        except Exception:
            return "System Error"

    def generate_text(self, prompt: str) -> str:
        """텍스트 생성 (Vertex AI -> Gemini API -> Groq 순서로 시도)"""
        sb = get_supabase()
        start_time = time.time()
        input_tokens = estimate_tokens(prompt)
        
        # 1. Vertex AI 시도
        try:
            text, used = self._try_vertex_text(prompt)
            if text:
                latency = int((time.time() - start_time) * 1000)
                output_tokens = estimate_tokens(text)
                st.session_state["last_model_used"] = f"{used} (Vertex AI)"
                log_api_call(sb, "llm_vertex", used, input_tokens, output_tokens, latency, True, None, prompt[:100], text[:100])
                return text
        except Exception:
            pass

        # 2. Gemini API 시도 (폴백 1)
        try:
            text, used = self._try_gemini_api_text(prompt)
            if text:
                latency = int((time.time() - start_time) * 1000)
                output_tokens = estimate_tokens(text)
                st.session_state["last_model_used"] = f"{used} (Gemini API)"
                log_api_call(sb, "llm_gemini", used, input_tokens, output_tokens, latency, True, None, prompt[:100], text[:100])
                return text
        except Exception as e:
            st.sidebar.error(f"Gemini API Error: {e}")
            pass

        # 3. Groq 시도 (폴백 2)
        if self.groq_client:
            out = self._generate_groq(prompt)
            latency = int((time.time() - start_time) * 1000)
            output_tokens = estimate_tokens(out)
            success = out != "System Error"
            if not success:
                 st.sidebar.error("Groq API Error: Check API Key or Quota")
                 log_api_call(sb, "llm_groq", "llama-3.3-70b-versatile", input_tokens, 0, latency, False, "System Error", prompt[:100])
            else:
                log_api_call(sb, "llm_groq", "llama-3.3-70b-versatile", input_tokens, output_tokens, latency, True, None, prompt[:100], out[:100])
            st.session_state["last_model_used"] = "llama-3.3-70b-versatile (Groq)"
            return out

        st.session_state["last_model_used"] = None
        return "시스템 오류: 모든 AI 모델 연결 실패"

    def generate_json(self, prompt: str) -> Optional[Any]:
        """JSON 생성"""
        strict = prompt + "\n\n반드시 JSON만 출력. 다른 텍스트 금지."
        text = self.generate_text(strict)
        return _safe_json_loads(text)

llm_service = LLMService()


class SearchService:
    """✅ 뉴스 중심 경량 검색"""
    def __init__(self):
        g = st.secrets.get("general", {})
        self.client_id = g.get("NAVER_CLIENT_ID")
        self.client_secret = g.get("NAVER_CLIENT_SECRET")
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
            res = llm_service.generate_text(prompt).strip()
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
            params = {"query": query, "display": 10, "sort": "sim"}
            res = requests.get(self.news_url, headers=self._headers(), params=params, timeout=8)
            res.raise_for_status()
            items = res.json().get("items", [])
            
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
            params = {"OC": self.api_id, "target": "law", "type": "XML", "query": law_name, "display": 1}
            res = requests.get(self.base_url, params=params, timeout=6)
            root = ET.fromstring(res.content)

            law_node = root.find(".//law")
            if law_node is None:
                latency = int((time.time() - start_time) * 1000)
                log_api_call(sb, "law_api", None, 0, 0, latency, True, None, law_name[:50], "No results")
                msg = f"🔍 '{law_name}'에 대한 검색 결과가 없습니다."
                return (msg, None) if return_link else msg

            mst_id = (law_node.findtext("법령일련번호") or "").strip()
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

            detail_params = {"OC": self.api_id, "target": "law", "type": "XML", "MST": mst_id}
            res_detail = requests.get(self.service_url, params=detail_params, timeout=10)
            root_detail = ET.fromstring(res_detail.content)

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
        data = llm_service.generate_json(prompt)
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
        data = llm_service.generate_json(prompt)
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
            extracted = llm_service.generate_json(prompt_extract)
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
            ai_fallback_text = llm_service.generate_text(prompt_fallback).strip()

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
        return llm_service.generate_text(prompt)

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
당신은 행정기관의 베테랑 서기이다. 아래 정보를 바탕으로 완결된 공문서를 JSON으로 작성하라.

[입력]
- 민원: {situation}
- 시행일자: {meta.get('today_str')}
- 문서번호: {meta.get('doc_num')}

[법령 근거(필수 인용)]
{legal_basis_md}

[처리방향]
{strategy}

[절차 플랜(반영)]
{json.dumps(procedure, ensure_ascii=False)}

[예상 반발(반영)]
{json.dumps(objections, ensure_ascii=False)}

[원칙]
- 본문에 법 조항/근거를 문장으로 인용할 것
- 구조: 경위 -> 법적 근거 -> 조치/안내 -> 이의제기/문의
- 개인정보는 OOO로 마스킹
- 문단 내에 **1** 같은 번호는 **볼드**로 표시해도 됨(마크다운 허용)

[출력 JSON 스키마]
{schema}

JSON만 출력.
"""
        data = llm_service.generate_json(prompt)
        if isinstance(data, dict) and data.get("title") and data.get("body_paragraphs"):
            return data

        retry = f"""
방금 출력이 스키마를 만족하지 않았다.
아래 스키마를 정확히 만족하는 JSON만 다시 출력하라.

스키마:
{schema}

(다른 텍스트 금지)
"""
        data2 = llm_service.generate_json(prompt + "\n\n" + retry)
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
    kws = llm_service.generate_json(prompt) or []
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

    # Phase 2) 법령 근거 강화
    add_log("Phase 2: 관련 법령 및 조문 정밀 조사...", "legal")
    law_md = LegalAgents.researcher(user_input, analysis)
    search_count += 1

    # Phase 3) 뉴스/사례 조회
    add_log("Phase 3: 유사 행정 심판/판례 검색...", "search")
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

    full_res_text = str(analysis) + str(law_md) + str(news) + str(strategy) + str(doc)
    estimated_tokens = int(len(full_res_text) * 0.7)
    model_used = st.session_state.get("last_model_used")

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
        return archive_id
    except Exception as e:
        st.warning(f"ℹ️ DB 저장 실패: {e}")
        return None


def db_fetch_history(sb, limit: int = 80) -> List[dict]:
    anon_id = str(ensure_anon_session_id())
    sb.postgrest.headers.update({'x-session-id': anon_id})

    try:
        q = (
            sb.table("work_archive")
            .select("id,prompt,created_at,user_email,anon_session_id")
            .order("created_at", desc=True)
            .limit(limit)
        )
        resp = q.execute()
        return getattr(resp, "data", None) or []
    except Exception:
        return []

def db_fetch_payload(sb, archive_id: str) -> Optional[dict]:
    anon_id = str(ensure_anon_session_id())
    sb.postgrest.headers.update({'x-session-id': anon_id})

    try:
        resp = (
            sb.table("work_archive")
            .select("id,prompt,payload,created_at,user_email,anon_session_id")
            .eq("id", archive_id)
            .limit(1)
            .execute()
        )
        data = getattr(resp, "data", None) or []
        if data:
            return data[0]
    except Exception:
        return None
    return None

def db_fetch_followups(sb, archive_id: str) -> List[dict]:
    anon_id = str(ensure_anon_session_id())
    sb.postgrest.headers.update({'x-session-id': anon_id})

    try:
        resp = (
            sb.table("work_followups")
            .select("turn,role,content,created_at")
            .eq("archive_id", archive_id)
            .order("turn", desc=False)
            .execute()
        )
        return getattr(resp, "data", None) or []
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
        st.sidebar.caption("로그인: 저장기능활성화")
        return

    # 새 채팅 버튼 (로그인 유저용)
    if st.sidebar.button("➕ 새 채팅 시작", use_container_width=True, type="primary"):
        st.session_state.pop("workflow_result", None)
        st.session_state.pop("current_archive_id", None)
        st.session_state.pop("followup_messages", None)
        st.session_state.pop("selected_history_id", None)
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
# 8) ADMIN DASHBOARD
# =========================================================
def admin_fetch_work_archive(sb, limit: int = 2000) -> List[dict]:
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

def admin_fetch_sessions(sb, minutes: int = 5) -> List[dict]:
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

def render_master_dashboard(sb):
    """Google-style BI 대시보드 with 드릴다운 필터, 비용 분석, 상세 감사로그"""
    st.markdown("## 🏛️ 관리자 운영 마스터 콘솔")

    if not is_admin_user(st.session_state.get("user_email", "")):
        st.warning("관리자만 접근 가능합니다.")
        return

    if not st.session_state.get("admin_mode", False):
        st.info("사이드바에서 **관리자모드 켜기**를 활성화하세요.")
        return

    # ─────────────────────────────────────────────────────────
    # 데이터 로드
    # ─────────────────────────────────────────────────────────
    with st.spinner("📊 데이터 로드 중..."):
        data = admin_fetch_work_archive(sb, limit=5000)
        sessions = admin_fetch_sessions(sb, minutes=5)
        events = admin_fetch_events(sb, limit=200)

    if not pd:
        st.error("pandas가 설치되어 있지 않습니다.")
        return

    if not data:
        st.info("표시할 데이터가 없습니다.")
        return

    # ─────────────────────────────────────────────────────────
    # DataFrame 준비 및 비용 계산
    # ─────────────────────────────────────────────────────────
    df = pd.DataFrame(data)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["date"] = df["created_at"].dt.date
    df["hour"] = df["created_at"].dt.hour
    df["weekday"] = df["created_at"].dt.day_name()
    df["user_email"] = df["user_email"].fillna("(anon)")
    df["app_mode"] = df["app_mode"].fillna("신속")
    df["model_used"] = df["model_used"].fillna("(unknown)")
    df["token_usage"] = pd.to_numeric(df["token_usage"], errors="coerce").fillna(0).astype(int)
    df["execution_time"] = pd.to_numeric(df["execution_time"], errors="coerce").fillna(0)
    df["search_count"] = pd.to_numeric(df["search_count"], errors="coerce").fillna(0).astype(int)
    df["prompt"] = df["prompt"].fillna("")

    # 비용 계산
    def calc_cost(row):
        model = row["model_used"]
        tokens = row["token_usage"]
        rate = MODEL_PRICING.get(model, MODEL_PRICING.get("(unknown)", 0.10))
        return (tokens / 1_000_000) * rate

    df["cost_usd"] = df.apply(calc_cost, axis=1)

    # Heavy user 임계값 계산
    user_run_counts = df["user_email"].value_counts()
    heavy_threshold = user_run_counts.quantile(HEAVY_USER_PERCENTILE / 100) if len(user_run_counts) > 1 else 999999
    heavy_users = set(user_run_counts[user_run_counts >= heavy_threshold].index)

    # ─────────────────────────────────────────────────────────
    # 🔍 드릴다운 필터 바
    # ─────────────────────────────────────────────────────────
    st.markdown("### 🔍 필터")
    filter_cols = st.columns([2, 2, 2, 1])

    with filter_cols[0]:
        all_users = ["(전체)"] + sorted(df["user_email"].unique().tolist())
        selected_user = st.selectbox("👤 사용자", all_users, index=0)

    with filter_cols[1]:
        min_date = df["date"].min() if not df.empty else datetime.now().date()
        max_date = df["date"].max() if not df.empty else datetime.now().date()
        date_range = st.date_input(
            "📅 날짜 범위",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

    with filter_cols[2]:
        all_models = sorted(df["model_used"].unique().tolist())
        selected_models = st.multiselect("🤖 모델", all_models, default=all_models)

    with filter_cols[3]:
        apply_filter = st.button("적용", use_container_width=True, type="primary")

    # 필터 적용
    filtered_df = df.copy()
    if selected_user != "(전체)":
        filtered_df = filtered_df[filtered_df["user_email"] == selected_user]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df["date"] >= date_range[0]) & (filtered_df["date"] <= date_range[1])
        ]
    if selected_models:
        filtered_df = filtered_df[filtered_df["model_used"].isin(selected_models)]

    st.divider()

    # ─────────────────────────────────────────────────────────
    # 📊 KPI 카드 (Google 스타일)
    # ─────────────────────────────────────────────────────────
    total_runs = len(filtered_df)
    total_tokens = int(filtered_df["token_usage"].sum())
    total_cost = filtered_df["cost_usd"].sum()
    cost_per_run = total_cost / total_runs if total_runs > 0 else 0.0
    avg_latency = filtered_df["execution_time"].mean() if total_runs > 0 else 0.0
    total_search = int(filtered_df["search_count"].sum())
    search_per_run = total_search / total_runs if total_runs > 0 else 0.0
    online_now = len(sessions)

    kpi_cols = st.columns(6)
    kpi_cols[0].metric("🟢 현재 접속", f"{online_now}")
    kpi_cols[1].metric("📦 총 실행", f"{total_runs:,}")
    kpi_cols[2].metric("🧾 총 토큰", f"{total_tokens:,}")
    kpi_cols[3].metric("💵 총 비용", f"${total_cost:.4f}")
    kpi_cols[4].metric("💰 회당 비용", f"${cost_per_run:.6f}")
    kpi_cols[5].metric("⏱️ 평균 지연", f"{avg_latency:.2f}s")

    st.divider()

    # ─────────────────────────────────────────────────────────
    # 📈 차트 섹션
    # ─────────────────────────────────────────────────────────
    chart_tabs = st.tabs(["📈 토큰/비용", "🤖 모델 분석", "🔥 지연 히트맵", "👤 사용자 분석"])

    with chart_tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 일별 토큰 사용량")
            tok_daily = filtered_df.groupby("date")["token_usage"].sum().sort_index()
            st.line_chart(tok_daily)
        with col2:
            st.subheader("💵 일별 비용 ($)")
            cost_daily = filtered_df.groupby("date")["cost_usd"].sum().sort_index()
            st.area_chart(cost_daily)

    with chart_tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🤖 모델별 사용 횟수")
            model_counts = filtered_df["model_used"].value_counts()
            st.bar_chart(model_counts)
        with col2:
            st.subheader("💵 모델별 비용")
            model_cost = filtered_df.groupby("model_used")["cost_usd"].sum().sort_values(ascending=False)
            st.bar_chart(model_cost)

    with chart_tabs[2]:
        st.subheader("🔥 시간대별 지연 히트맵")
        try:
            import plotly.express as px
            heatmap_data = filtered_df.groupby(["weekday", "hour"])["execution_time"].mean().unstack(fill_value=0)
            weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            heatmap_data = heatmap_data.reindex([w for w in weekday_order if w in heatmap_data.index])
            
            fig = px.imshow(
                heatmap_data,
                labels=dict(x="시간", y="요일", color="평균 지연(s)"),
                x=heatmap_data.columns,
                y=heatmap_data.index,
                color_continuous_scale="RdYlGn_r",
                aspect="auto",
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.info("plotly가 설치되어 있지 않아 히트맵을 표시할 수 없습니다. `pip install plotly`")
            # 대체: 간단한 테이블
            latency_by_hour = filtered_df.groupby("hour")["execution_time"].mean()
            st.bar_chart(latency_by_hour)

    with chart_tabs[3]:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("👤 사용자별 실행 Top 10")
            user_counts = filtered_df["user_email"].value_counts().head(10)
            st.bar_chart(user_counts)
        with col2:
            st.subheader("💵 사용자별 비용 Top 10")
            user_cost = filtered_df.groupby("user_email")["cost_usd"].sum().sort_values(ascending=False).head(10)
            st.bar_chart(user_cost)

    st.divider()

    # ─────────────────────────────────────────────────────────
    # 📋 상세 감사 로그 테이블
    # ─────────────────────────────────────────────────────────
    st.subheader("📋 상세 감사 로그 (Audit Trail)")

    # 표시용 DataFrame 준비
    display_df = filtered_df[["created_at", "user_email", "prompt", "model_used", "token_usage", "cost_usd", "execution_time", "id"]].copy()
    display_df = display_df.sort_values("created_at", ascending=False).head(100)
    display_df["created_at"] = display_df["created_at"].dt.strftime("%Y-%m-%d %H:%M")
    display_df["prompt_short"] = display_df["prompt"].apply(lambda x: x[:40] + "..." if len(x) > 40 else x)
    display_df["cost_usd"] = display_df["cost_usd"].apply(lambda x: f"${x:.6f}")
    display_df["execution_time"] = display_df["execution_time"].apply(lambda x: f"{x:.1f}s")

    # 하이라이트 조건
    def highlight_rows(row):
        styles = [""] * len(row)
        user = row["user_email"]
        exec_time = float(row["execution_time"].replace("s", ""))
        
        # Heavy user: 노란색 배경
        if user in heavy_users:
            styles = ["background-color: #fef3c7"] * len(row)
        
        # Long latency: 빨간색 배경
        if exec_time > LONG_LATENCY_THRESHOLD:
            styles = ["background-color: #fee2e2; color: #991b1b; font-weight: bold"] * len(row)
        
        return styles

    styled_df = display_df[["created_at", "user_email", "prompt_short", "model_used", "token_usage", "cost_usd", "execution_time"]].rename(columns={
        "created_at": "📅 일시",
        "user_email": "👤 사용자",
        "prompt_short": "📝 지시(요약)",
        "model_used": "🤖 모델",
        "token_usage": "🧾 토큰",
        "cost_usd": "💵 비용",
        "execution_time": "⏱️ 지연",
    })

    try:
        st.dataframe(
            styled_df.style.apply(highlight_rows, axis=1),
            use_container_width=True,
            height=400,
        )
    except Exception:
        st.dataframe(styled_df, use_container_width=True, height=400)

    st.caption("🟡 노란색: 과다 사용자 (상위 5%)  |  🔴 빨간색: 장시간 지연 (120초 초과)")

    # 프롬프트 상세 보기
    with st.expander("🔍 프롬프트 상세 보기 (클릭해서 전체 내용 확인)"):
        prompt_id = st.selectbox(
            "로그 선택 (최근 100개)",
            options=display_df["id"].tolist(),
            format_func=lambda x: f"{display_df[display_df['id'] == x]['created_at'].values[0]} - {display_df[display_df['id'] == x]['prompt_short'].values[0]}",
        )
        if prompt_id:
            full_prompt = filtered_df[filtered_df["id"] == prompt_id]["prompt"].values
            if len(full_prompt) > 0:
                st.text_area("전체 프롬프트", full_prompt[0], height=200, disabled=True)

    st.divider()

    # ─────────────────────────────────────────────────────────
    # 📥 데이터 내보내기 & 관리
    # ─────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📥 데이터 내보내기")
        export_df = filtered_df.copy()
        export_df["created_at"] = export_df["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
        csv = export_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "💾 CSV 다운로드 (필터 적용)",
            data=csv,
            file_name=f"work_archive_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        st.subheader("🗑️ 기록 삭제")
        if data:
            ids = [r["id"] for r in data if r.get("id")]
            sel = st.selectbox("삭제할 ID", options=ids[:20], index=0)
            confirm = st.checkbox("삭제 확인")
            if st.button("❌ 삭제", type="primary") and confirm:
                try:
                    sb.table("work_archive").delete().eq("id", sel).execute()
                    log_event(sb, "admin_delete_archive", archive_id=sel)
                    st.success("삭제 완료")
                    st.rerun()
                except Exception as e:
                    st.error(f"삭제 실패: {e}")

    st.divider()

    # ─────────────────────────────────────────────────────────
    # 🟢 세션 & 이벤트 로그
    # ─────────────────────────────────────────────────────────
    session_tabs = st.tabs(["🟢 현재 세션", "🧾 이벤트 로그"])

    with session_tabs[0]:
        if sessions:
            st.write(f"최근 5분 내 활성 세션: **{len(sessions)}**")
            st.dataframe(sessions, use_container_width=True)
        else:
            st.caption("최근 5분 내 활성 세션이 없습니다.")

    with session_tabs[1]:
        if events:
            st.dataframe(events, use_container_width=True, height=300)
        else:
            st.caption("이벤트 로그가 없습니다.")


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


# =========================================================
# 10) MAIN UI
# =========================================================
def main():
    sb = get_supabase()
    ensure_anon_session_id()

    if sb:
        touch_session(sb)
        if "boot_logged" not in st.session_state:
            st.session_state.boot_logged = True
            log_event(sb, "app_open", meta={"ver": APP_VERSION})

        sidebar_auth(sb)
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
            render_master_dashboard(sb)
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
        # 애니메이션 및 결과가 표시될 메인 플레이스홀더
        right_panel_placeholder = st.empty()

        if "workflow_result" not in st.session_state:
            # 초기 상태: 문서 미리보기 안내
            with right_panel_placeholder.container():
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
            # return # 여기서 return하면 안됨, placeholder 객체가 필요함
        
        # 결과가 있으면 아래에서 렌더링 (else 블록 아님, 흐름 제어 주의)

    with col_left:
        render_header("🗣️ 업무 지시")

        user_input = st.text_area(
            "업무 내용",
            height=190,
            placeholder="예시\n- 상황: (무슨 일 / 어디 / 언제 / 증거 유무...)\n- 쟁점: (요건/절차/근거...)\n- 요청: (원하는 결과물: 회신/사전통지/처분 등)",
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
                # 진행 상황은 run_workflow 내부에서 애니메이션으로 표시됨 (오른쪽 패널)
                res = run_workflow(user_input, right_panel_placeholder)
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

        if "workflow_result" in st.session_state:
            res = st.session_state.workflow_result
            pack = res.get("lawbot_pack") or {}
            if pack.get("url"):
                render_lawbot_button(pack["url"])

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

    # 결과 렌더링 (오른쪽 컬럼 다시 진입 필요? 아니면 위에서 처리?)
    # Streamlit 흐름상 col_right 컨텍스트가 닫혔으므로, 다시 열거나 위에서 처리해야 함.
    # 하지만 col_right는 위에서 이미 정의됨.
    # 구조를 약간 바꿔야 함. col_right 내용을 아래로 빼는 것이 좋음.
    
    if "workflow_result" in st.session_state:
        # 오른쪽 패널에 결과 렌더링
        with right_panel_placeholder.container():
            res = st.session_state.workflow_result
            doc = res.get("doc")
            meta = res.get("meta") or {}
            archive_id = res.get("archive_id") or st.session_state.get("current_archive_id")

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

            render_header("💬 후속 질문")

            if not archive_id:
                st.info("저장된 archive_id가 없습니다. (DB 저장 실패 가능)")
            else:
                # DB 저장 성공 표시 (기존 후속 질문 횟수 표시 대체)
                if archive_id:
                    st.success("✅ 업무 지시 내용이 DB에 안전하게 저장되었습니다.")
                else:
                    st.error("❌ DB 저장 실패 (Archive ID 없음)")

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

            q = st.chat_input("💭 후속 질문을 입력하세요... (Enter로 전송)")
            if q:
                turn = used + 1
                st.session_state.followup_messages.append({"role": "user", "content": q})
                db_insert_followup(sb, archive_id, turn=turn * 2 - 1, role="user", content=q)
                log_event(sb, "followup_user", archive_id=archive_id, meta={"turn": turn})

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
                        ans = llm_service.generate_text(prompt)
                        st.markdown(ans)

                st.session_state.followup_messages.append({"role": "assistant", "content": ans})
                db_insert_followup(sb, archive_id, turn=turn * 2, role="assistant", content=ans)
                log_event(sb, "followup_assistant", archive_id=archive_id, meta={"turn": turn})

                st.rerun()

if __name__ == "__main__":
    main()
