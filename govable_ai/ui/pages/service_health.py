# -*- coding: utf-8 -*-
"""
서비스 상태 점검 페이지 (Service Health)

streamlit_app_legacy.py 의 ✅ AI / LAW / NEWS / DB 인디케이터 및
govable_ai/main.py 의 인라인 시스템 상태 배너를 단일 모듈로 추출.

- render_service_health_banner: 다른 페이지에서도 재사용 가능한 컴팩트 배너
- render_service_health_page: 서비스별 상세 카드, 새로고침, 설정 가이드 포함

각 서비스 객체는 .is_available() 메서드를 제공한다고 가정한다.
일부 서비스가 None 인 경우에도 안전하게 "❌ (설정 필요)" 로 표시한다.
"""
from __future__ import annotations

from typing import Any, Optional

import streamlit as st


# ---------------------------------------------------------------------------
# 내부 유틸
# ---------------------------------------------------------------------------

def _safe_is_available(service: Any) -> bool:
    """서비스 객체에서 .is_available() 을 안전하게 호출한다.

    서비스가 None 이거나 메서드가 없으면 False 반환.
    """
    if service is None:
        return False
    fn = getattr(service, "is_available", None)
    if not callable(fn):
        return False
    try:
        return bool(fn())
    except Exception:
        return False


def _last_error(service: Any) -> str:
    """서비스 객체가 노출할 수 있는 마지막 오류 메시지를 추출한다."""
    if service is None:
        return ""
    for attr in ("last_error", "last_error_message", "error", "_last_error"):
        val = getattr(service, attr, None)
        if val:
            return str(val)
    return ""


def _endpoint_hint(service: Any, default_hint: str) -> str:
    """서비스 객체에서 엔드포인트/모델 힌트를 추출한다."""
    if service is None:
        return default_hint
    for attr in ("endpoint", "base_url", "model", "model_name", "host"):
        val = getattr(service, attr, None)
        if val:
            return str(val)
    return default_hint


def _status_label(ok: bool, missing: bool = False) -> str:
    """상태 라벨을 반환. 서비스 자체가 누락된 경우는 별도 표시."""
    if missing:
        return "❌ (설정 필요)"
    return "✅ 정상" if ok else "❌ 사용 불가"


# ---------------------------------------------------------------------------
# 1) 재사용 가능한 컴팩트 배너
# ---------------------------------------------------------------------------

def render_service_health_banner(services: dict, version: str = "") -> None:
    """페이지 상단에 표시할 컴팩트 시스템 상태 배너.

    다른 페이지(특히 main.py 의 워크플로 페이지)에서도 동일한 룩앤필로
    호출할 수 있도록 설계되었다.

    Args:
        services: {"llm": ..., "law_api": ..., "search": ..., "db": ...}
                  각 값은 .is_available() 을 가진 서비스 객체 또는 None.
        version: 우측에 표시할 앱 버전 문자열 (예: "1.2.3"). 비어있으면 생략.
    """
    services = services or {}

    ai_ok = _safe_is_available(services.get("llm"))
    law_ok = _safe_is_available(services.get("law_api"))
    nv_ok = _safe_is_available(services.get("search"))
    db_ok = _safe_is_available(services.get("db"))

    ai_label = f"{'✅' if ai_ok else '❌'} AI"
    law_label = f"{'✅' if law_ok else '❌'} LAW"
    nv_label = f"{'✅' if nv_ok else '❌'} NEWS"
    db_label = f"{'✅' if db_ok else '❌'} DB"

    version_html = (
        f"<span style='font-size: 0.85rem; color: #9ca3af; margin-left: 1rem;'>"
        f"v{version}</span>"
        if version
        else ""
    )

    st.markdown(
        f"""
        <div style='text-align: center; padding: 0.75rem 1.5rem; background: white;
                    border-radius: 12px; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    border-left: 4px solid #2563eb;'>
            <span style='font-size: 0.9rem; color: #374151; font-weight: 600;'>
                시스템 상태: {ai_label} · {law_label} · {nv_label} · {db_label}
            </span>
            {version_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 2) 풀 페이지 (상세 진단 + 설정 가이드)
# ---------------------------------------------------------------------------

# 페이지에서 표시할 서비스 정의 (순서대로 카드 렌더링).
# key: services dict 의 키
# title/icon: UI 표시용
# default_endpoint: 서비스 객체에서 힌트를 못 찾을 때 사용할 기본 설명
# required_keys: 어떤 시크릿이 필요한지 사용자에게 안내
_SERVICE_DEFS = [
    {
        "key": "llm",
        "title": "LLM (Gemini / Groq)",
        "icon": "🤖",
        "default_endpoint": "Google Gemini · Groq Cloud",
        "required_keys": ["GEMINI_API_KEY", "GROQ_API_KEY"],
        "help": "둘 중 하나만 설정해도 동작하지만, 둘 다 설정하면 자동 폴백이 활성화됩니다.",
    },
    {
        "key": "law_api",
        "title": "법령 정보 API (국가법령정보센터)",
        "icon": "⚖️",
        "default_endpoint": "law.go.kr OpenAPI",
        "required_keys": ["LAW_API_ID"],
        "help": "법령정보센터에서 발급받은 OC(아이디)를 입력해 주세요.",
    },
    {
        "key": "search",
        "title": "뉴스/검색 (Naver Open API)",
        "icon": "📰",
        "default_endpoint": "openapi.naver.com",
        "required_keys": ["NAVER_CLIENT_ID", "NAVER_CLIENT_SECRET"],
        "help": "네이버 개발자센터에서 검색 API 애플리케이션을 등록한 뒤 두 키를 모두 설정해야 합니다.",
    },
    {
        "key": "db",
        "title": "데이터베이스 (Supabase)",
        "icon": "🗄️",
        "default_endpoint": "Supabase (PostgreSQL)",
        "required_keys": ["SUPABASE_URL", "SUPABASE_KEY"],
        "help": "Supabase 프로젝트의 URL 과 anon/service role 키가 필요합니다.",
    },
]


def _render_service_card(service: Any, definition: dict) -> None:
    """단일 서비스 상세 카드."""
    missing = service is None
    ok = False if missing else _safe_is_available(service)
    status = _status_label(ok, missing=missing)
    endpoint = _endpoint_hint(service, definition["default_endpoint"])
    err = _last_error(service)

    border_color = "#10b981" if ok else ("#9ca3af" if missing else "#ef4444")
    bg_color = "#ecfdf5" if ok else ("#f3f4f6" if missing else "#fef2f2")

    with st.container():
        st.markdown(
            f"""
            <div style='padding: 1rem 1.25rem; background: {bg_color};
                        border-radius: 10px; border-left: 4px solid {border_color};
                        margin-bottom: 0.75rem;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span style='font-size: 1rem; font-weight: 700; color: #111827;'>
                        {definition['icon']} {definition['title']}
                    </span>
                    <span style='font-size: 0.9rem; font-weight: 600; color: #374151;'>
                        {status}
                    </span>
                </div>
                <div style='font-size: 0.82rem; color: #6b7280; margin-top: 0.4rem;'>
                    엔드포인트/모델: <code>{endpoint}</code>
                </div>
                <div style='font-size: 0.82rem; color: #6b7280; margin-top: 0.2rem;'>
                    필요한 키: {', '.join(f'<code>{k}</code>' for k in definition['required_keys'])}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if err:
            st.caption(f"⚠️ 마지막 오류: {err}")
        if definition.get("help"):
            st.caption(f"💡 {definition['help']}")


def _render_secrets_guide() -> None:
    """secrets.toml 설정 예시 expander."""
    with st.expander("⚙️ 어떤 키를 어디에 설정해야 하나요?", expanded=False):
        st.markdown(
            "프로젝트 루트의 `.streamlit/secrets.toml` 파일에 아래와 같이 입력해 주세요. "
            "값은 따옴표로 감싸야 하며, 발급받은 실제 키로 교체하세요."
        )
        st.code(
            """# .streamlit/secrets.toml

[general]
# 둘 중 하나만 있어도 동작합니다 (둘 다 설정 시 자동 폴백).
GEMINI_API_KEY = "your-gemini-api-key"
GROQ_API_KEY   = "your-groq-api-key"

# 국가법령정보센터 OpenAPI 신청 시 발급받은 OC(아이디)
LAW_API_ID = "your-law-api-id"

# 네이버 개발자센터 → 검색 애플리케이션 등록 후 발급
NAVER_CLIENT_ID     = "your-naver-client-id"
NAVER_CLIENT_SECRET = "your-naver-client-secret"

[supabase]
SUPABASE_URL = "https://xxxxxxxx.supabase.co"
SUPABASE_KEY = "your-supabase-anon-or-service-role-key"
""",
            language="toml",
        )
        st.caption(
            "변경 후에는 Streamlit 앱을 재시작하거나 우측 상단 '🔄 새로고침' 버튼을 눌러 주세요."
        )


def render_service_health_page(
    llm_service: Any,
    db: Optional[Any] = None,
    services: Optional[dict] = None,
) -> None:
    """서비스 상태 점검 페이지 진입점.

    Args:
        llm_service: govable_ai.core.llm_service.LLMService 인스턴스 (또는 None)
        db: 데이터베이스 클라이언트(예: Supabase wrapper) — .is_available() 보유
        services: {"law_api": ..., "search": ...} 형태의 외부 서비스 모음.
                  키가 누락되면 해당 카드는 "❌ (설정 필요)" 로 표시된다.
    """
    services = dict(services or {})
    # llm/db 는 인자로 따로 받지만 배너/카드 처리를 단순화하기 위해 dict 에 합쳐 둔다.
    services.setdefault("llm", llm_service)
    services.setdefault("db", db)

    # ---- 헤더 + 새로고침 버튼 ----
    head_col, btn_col = st.columns([6, 1])
    with head_col:
        st.markdown("## 🩺 서비스 상태 점검")
        st.caption("각 외부 서비스의 연결 상태와 설정 누락 여부를 한 눈에 확인합니다.")
    with btn_col:
        if st.button("🔄 새로고침", use_container_width=True, key="svc_health_refresh"):
            st.rerun()

    # ---- 큰 배너 ----
    render_service_health_banner(services)

    # ---- 서비스별 상세 카드 ----
    st.markdown("### 🔍 서비스별 상세")
    for definition in _SERVICE_DEFS:
        svc = services.get(definition["key"])
        _render_service_card(svc, definition)

    # ---- 설정 가이드 ----
    st.markdown("---")
    _render_secrets_guide()
