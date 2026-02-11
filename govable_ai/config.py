# -*- coding: utf-8 -*-
"""
Govable AI - 설정 및 상수 관리
UI 의존성 없음 (streamlit import 금지)
"""
import os
from typing import Any, Dict, Optional

# =========================================================
# 앱 상수
# =========================================================
APP_VERSION = "2026-01-20-modular"
MAX_FOLLOWUP_Q = 5
ADMIN_EMAIL = "kim0395kk@korea.kr"
LAW_BOT_SEARCH_URL = "https://www.law.go.kr/LSW/ais/searchList.do?query="

# Heavy user / Long latency 임계값
HEAVY_USER_PERCENTILE = 95  # 상위 5% = 과다 사용자
LONG_LATENCY_THRESHOLD = 120  # 초

# 모델별 토큰 가격 ($/1M tokens)
MODEL_PRICING: Dict[str, float] = {
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


# =========================================================
# Secrets / 환경변수 로딩
# =========================================================
_streamlit_secrets: Optional[Dict] = None


def _load_streamlit_secrets() -> Dict:
    """st.secrets를 한 번만 로드 (lazy)"""
    global _streamlit_secrets
    if _streamlit_secrets is not None:
        return _streamlit_secrets
    try:
        import streamlit as st
        _streamlit_secrets = dict(st.secrets) if hasattr(st, 'secrets') else {}
    except Exception:
        _streamlit_secrets = {}
    return _streamlit_secrets


def get_secret(path1: str, path2: str = "") -> Any:
    """
    Streamlit secrets 또는 환경변수에서 설정값을 가져옵니다.
    
    사용법:
        get_secret("general", "GEMINI_API_KEY")  # st.secrets["general"]["GEMINI_API_KEY"]
        get_secret("SUPABASE_URL")               # st.secrets["SUPABASE_URL"] or os.environ
    
    우선순위:
        1. st.secrets[path1][path2] (path2가 있을 때)
        2. st.secrets[path1] (path2가 없을 때)
        3. os.environ[path2 or path1]
    """
    secrets = _load_streamlit_secrets()
    
    # nested access: st.secrets[path1][path2]
    if path2:
        if path1 in secrets:
            nested = secrets[path1]
            if isinstance(nested, dict) and path2 in nested:
                return nested[path2]
        # fallback to env
        return os.environ.get(path2)
    
    # direct access: st.secrets[path1]
    if path1 in secrets:
        return secrets[path1]
    
    # fallback to env
    return os.environ.get(path1)


def get_vertex_config() -> Optional[Dict]:
    """
    Vertex AI 설정을 가져옵니다.
    
    Returns:
        dict with keys: project_id, location, model_id, credentials_json
        or None if not configured
    """
    secrets = _load_streamlit_secrets()
    vertex = secrets.get("vertex_ai", {})
    
    if not vertex:
        return None
    
    return {
        "project_id": vertex.get("project_id"),
        "location": vertex.get("location", "us-central1"),
        "model_id": vertex.get("model_id", "gemini-2.5-flash"),
        "credentials_json": vertex.get("credentials_json"),
    }


def get_supabase_config() -> Optional[Dict[str, str]]:
    """
    Supabase 연결 설정을 가져옵니다.
    
    Returns:
        dict with keys: url, anon_key or None
    """
    # 1. Try "supabase" section first (preferred)
    url = get_secret("supabase", "SUPABASE_URL")
    key = get_secret("supabase", "SUPABASE_KEY") or get_secret("supabase", "SUPABASE_ANON_KEY")
    
    # 2. Fallback to top-level or env vars if not found in section
    if not url:
        url = get_secret("SUPABASE_URL")
    if not key:
        key = get_secret("SUPABASE_KEY") or get_secret("SUPABASE_ANON_KEY")
    
    if url and key:
        return {"url": url, "anon_key": key}
    return None
