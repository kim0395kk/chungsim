# -*- coding: utf-8 -*-
"""
작업 이력 페이지 (Archive History)

streamlit_app_legacy.py 의 archive 관련 흐름(약 2952~3092 행, 3245~3288 행)을 단일 페이지로 추출.
사이드바의 ``render_history_list`` 가 최근 80건 컴팩트 보기를 담당한다면,
이 페이지는 풀페이지 브라우저로서 다음 기능을 제공한다.

- 날짜 / 검색어 / app_mode 필터
- 본인 전용 / 전체(관리자) 토글
- 20건 페이지네이션
- 각 행 expander 펼침 → payload 미리보기 + "🔁 다시 열기" + 관리자 "🗑️ 삭제"

DB 의존성: ``govable_ai.core.db_client.SupabaseClient``
- 사용 메서드: ``is_available``, ``fetch_history``, ``fetch_payload``, ``admin_fetch_work_archive``
- ``fetch_archives_filtered`` 같은 서버측 필터 메서드는 현재 미존재 → 클라이언트(Python) 사이드 필터링.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Iterable, Optional

import streamlit as st

try:
    from govable_ai.ui.auth import is_admin_user
except Exception:  # pragma: no cover - import 실패 시 안전 폴백
    is_admin_user = None  # type: ignore[assignment]

try:
    from govable_ai.helpers import shorten_one_line
except Exception:  # pragma: no cover
    def shorten_one_line(text: str, max_len: int = 60) -> str:  # type: ignore[no-redef]
        s = (text or "").replace("\n", " ").strip()
        return s if len(s) <= max_len else s[: max_len - 1] + "…"


# 필터 옵션. legacy 와 동일한 app_mode 키 집합.
_APP_MODES: tuple[str, ...] = (
    "신속",
    "revision",
    "complaint_analyzer",
    "hallucination_check",
    "civil_engineering",
)

# 페이지네이션 — 20건 / 페이지.
_PAGE_SIZE: int = 20

# 세션 상태 키.
_SS_PAGE = "archive_history_page"
_SS_EXPAND = "archive_history_expanded_id"
_SS_DELETE_CONFIRM = "archive_history_delete_confirm_id"


# ---------------------------------------------------------------------------
# 내부 유틸
# ---------------------------------------------------------------------------
def _is_current_user_admin() -> bool:
    """현재 세션 사용자가 관리자 권한을 가지는지 판정."""
    if not st.session_state.get("logged_in"):
        return False
    if is_admin_user is None:
        return False
    email = st.session_state.get("user_email", "") or ""
    db_admin_flag = bool(st.session_state.get("is_admin_db", False))
    try:
        return bool(is_admin_user(email, db_admin_flag))
    except Exception:
        return False


def _parse_created_at(value: Any) -> Optional[datetime]:
    """``created_at`` 문자열을 datetime 으로 안전 파싱."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    s = str(value)
    # Supabase 는 보통 ISO-8601 (+ 'Z') 로 반환한다.
    s = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except Exception:
        # 마이크로초 누락 등 변형 대응.
        try:
            return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")
        except Exception:
            return None


def _format_dt(value: Any) -> str:
    """행 표시용 일시 포맷 (KST 가정 — DB 는 UTC 이지만 legacy 와 동일하게 그대로 표기)."""
    dt = _parse_created_at(value)
    if dt is None:
        return "(시각 없음)"
    return dt.strftime("%Y-%m-%d %H:%M")


def _row_app_mode(row: dict) -> str:
    """행에서 app_mode 안전 추출. 일부 legacy 행은 payload 안에만 들어있다."""
    mode = row.get("app_mode")
    if mode:
        return str(mode)
    payload = row.get("payload") or {}
    if isinstance(payload, dict):
        return str(payload.get("app_mode") or "신속")
    return "신속"


def _fetch_archive_rows(db: Any, admin_view_all: bool) -> list[dict]:
    """
    필터링/페이지네이션 이전의 베이스 archive 목록을 가져온다.

    - admin_view_all=True: ``admin_fetch_work_archive`` (전체)
    - 그 외: 본인 이메일로 필터링된 ``fetch_history`` 결과를 사용.

    NOTE: SupabaseClient 에 현재 ``fetch_history`` 는 user_email 필터 인자가 없고
    ``payload`` 컬럼도 미반환이다. 풀페이로드는 펼침 시 ``fetch_payload`` 로 lazy fetch.
    TODO(db_client): 향후 ``fetch_archives_filtered(user_email, date_from, date_to,
    app_modes, q, limit, offset)`` 를 추가하면 서버측 필터로 대체 가능.
    """
    if not db or not db.is_available():
        return []

    if admin_view_all:
        try:
            return db.admin_fetch_work_archive(limit=2000) or []
        except Exception as e:
            st.error(f"전체 아카이브 조회 실패: {e}")
            return []

    # 본인 전용 보기 — 사용자 이메일로 클라이언트 사이드 필터.
    try:
        rows = db.fetch_history(limit=500) or []
    except Exception as e:
        st.error(f"이력 조회 실패: {e}")
        return []

    email = (st.session_state.get("user_email") or "").strip().lower()
    if email:
        rows = [r for r in rows if (r.get("user_email") or "").strip().lower() == email]
    return rows


def _apply_filters(
    rows: Iterable[dict],
    *,
    date_from: Optional[date],
    date_to: Optional[date],
    keyword: str,
    app_modes: tuple[str, ...],
) -> list[dict]:
    """클라이언트 사이드 필터링."""
    kw = (keyword or "").strip().lower()
    selected_modes = set(app_modes or ())
    out: list[dict] = []

    for row in rows:
        # 날짜
        dt = _parse_created_at(row.get("created_at"))
        if date_from is not None and dt is not None and dt.date() < date_from:
            continue
        if date_to is not None and dt is not None and dt.date() > date_to:
            continue

        # 검색어
        if kw and kw not in (row.get("prompt") or "").lower():
            continue

        # app_mode
        if selected_modes and _row_app_mode(row) not in selected_modes:
            continue

        out.append(row)
    return out


def _restore_archive(db: Any, archive_id: str) -> None:
    """
    아카이브를 복원하여 워크플로우 페이지에서 다시 보기.

    - payload 가 행에 이미 들어있으면 그대로 사용
    - 아니면 ``fetch_payload`` 로 lazy fetch
    """
    payload: Optional[dict] = None
    try:
        full = db.fetch_payload(archive_id)
        if full:
            payload = full.get("payload") or {}
    except Exception as e:
        st.error(f"복원 실패: {e}")
        return

    if not payload:
        st.error("복원할 데이터가 없습니다. (권한/RLS 확인)")
        return

    st.session_state["workflow_result"] = payload
    st.session_state["current_archive_id"] = archive_id
    st.session_state["selected_history_id"] = archive_id
    st.session_state["current_page"] = "workflow"
    st.rerun()


def _delete_archive(db: Any, archive_id: str) -> bool:
    """관리자 전용 — 아카이브 단건 삭제. Raw client 로 직접 수행한다."""
    if not db or not db.is_available():
        return False
    client = getattr(db, "client", None)
    if client is None:
        return False
    try:
        client.table("work_archive").delete().eq("id", archive_id).execute()
        return True
    except Exception as e:
        st.error(f"삭제 실패: {e}")
        return False


def _render_payload_preview(payload: Any) -> None:
    """payload(미리보기) 를 사용자 친화적으로 렌더링."""
    if not isinstance(payload, dict) or not payload:
        st.caption("미리볼 payload 가 없습니다.")
        return

    app_mode = payload.get("app_mode") or "(unknown)"
    st.caption(f"모드: {app_mode}")

    # 흔히 있는 필드들을 우선 노출.
    analysis = payload.get("analysis")
    if analysis:
        with st.container():
            st.markdown("**📊 분석**")
            st.markdown(str(analysis)[:1500])

    doc = payload.get("doc") or payload.get("document") or payload.get("draft")
    if doc:
        with st.container():
            st.markdown("**📄 문서**")
            st.markdown(str(doc)[:1500])

    answer = payload.get("answer") or payload.get("content")
    if answer and not analysis and not doc:
        with st.container():
            st.markdown("**💬 답변**")
            st.markdown(str(answer)[:1500])

    # Raw payload 토글.
    with st.expander("원본 payload (JSON)", expanded=False):
        try:
            st.json(payload)
        except Exception:
            st.code(repr(payload), language="text")


# ---------------------------------------------------------------------------
# 메인 진입점
# ---------------------------------------------------------------------------
def render_archive_history_page(
    llm_service: Any,
    db: Optional[Any] = None,
    services: Optional[dict] = None,
) -> None:
    """작업 이력 풀페이지.

    Args:
        llm_service: LLM 서비스 인스턴스 (현재 페이지에서는 미사용, 시그니처 통일).
        db: Supabase DB 클라이언트. None 또는 미연결 시 안내 후 종료.
        services: 추가 서비스 맵 (현재 페이지에서는 미사용, 시그니처 통일).
    """
    st.markdown("## 🗂️ 작업 이력")

    # 1) DB 가용성 확인
    if db is None or not getattr(db, "is_available", lambda: False)():
        st.warning("🔒 로그인 후 작업 이력을 볼 수 있습니다.")
        st.caption("좌측 사이드바의 '🔐 로그인'에서 로그인 또는 회원가입을 진행해 주세요.")
        return

    # 2) 세션 상태 초기화
    if _SS_PAGE not in st.session_state:
        st.session_state[_SS_PAGE] = 0

    is_admin = _is_current_user_admin()

    # 3) 필터 영역
    st.markdown("### 🔍 필터")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        default_from = date.today() - timedelta(days=30)
        date_from_value = st.date_input(
            "시작일",
            value=default_from,
            key="archive_history_date_from",
        )
    with c2:
        date_to_value = st.date_input(
            "종료일",
            value=date.today(),
            key="archive_history_date_to",
        )
    with c3:
        keyword = st.text_input(
            "검색어",
            placeholder="프롬프트에서 검색",
            key="archive_history_keyword",
        )

    c4, c5 = st.columns([3, 1])
    with c4:
        selected_modes = st.multiselect(
            "app_mode 필터",
            options=list(_APP_MODES),
            default=[],
            key="archive_history_app_modes",
            help="비워두면 전체 모드 표시",
        )
    with c5:
        if is_admin:
            view_all = st.toggle(
                "전체 보기",
                value=False,
                key="archive_history_view_all",
                help="관리자: 모든 사용자 이력",
            )
        else:
            view_all = False
            st.caption("본인 전용")

    # date_input 이 단일 date 와 (start, end) 튜플을 모두 반환할 수 있어 정규화.
    if isinstance(date_from_value, tuple):
        date_from_value = date_from_value[0] if date_from_value else None
    if isinstance(date_to_value, tuple):
        date_to_value = date_to_value[-1] if date_to_value else None

    # 필터 변경 시 페이지를 0으로 리셋. 단순 비교로 충분.
    filter_signature = (
        str(date_from_value),
        str(date_to_value),
        keyword.strip().lower() if keyword else "",
        tuple(sorted(selected_modes or ())),
        bool(view_all),
    )
    if st.session_state.get("archive_history_filter_sig") != filter_signature:
        st.session_state[_SS_PAGE] = 0
        st.session_state["archive_history_filter_sig"] = filter_signature

    st.markdown("---")

    # 4) 데이터 조회 + 필터링
    base_rows = _fetch_archive_rows(db, admin_view_all=bool(view_all and is_admin))
    filtered = _apply_filters(
        base_rows,
        date_from=date_from_value if isinstance(date_from_value, date) else None,
        date_to=date_to_value if isinstance(date_to_value, date) else None,
        keyword=keyword,
        app_modes=tuple(selected_modes or ()),
    )

    total = len(filtered)
    if total == 0:
        st.info("조건에 맞는 작업 이력이 없습니다. 필터를 조정해 보세요.")
        return

    # 5) 페이지네이션
    max_page = max(0, (total - 1) // _PAGE_SIZE)
    current_page = max(0, min(int(st.session_state.get(_SS_PAGE, 0)), max_page))
    start = current_page * _PAGE_SIZE
    end = start + _PAGE_SIZE
    page_rows = filtered[start:end]

    pc1, pc2, pc3 = st.columns([1, 2, 1])
    with pc1:
        if st.button(
            "◀ 이전",
            disabled=(current_page <= 0),
            use_container_width=True,
            key="archive_history_prev",
        ):
            st.session_state[_SS_PAGE] = max(0, current_page - 1)
            st.rerun()
    with pc2:
        st.markdown(
            f"<div style='text-align:center;padding-top:0.4rem;'>"
            f"총 <b>{total}</b>건 · <b>{current_page + 1}</b>/{max_page + 1} 페이지"
            f"</div>",
            unsafe_allow_html=True,
        )
    with pc3:
        if st.button(
            "다음 ▶",
            disabled=(current_page >= max_page),
            use_container_width=True,
            key="archive_history_next",
        ):
            st.session_state[_SS_PAGE] = min(max_page, current_page + 1)
            st.rerun()

    st.markdown("")

    # 6) 행 렌더링
    for row in page_rows:
        archive_id = str(row.get("id") or "")
        if not archive_id:
            continue

        prompt_text = row.get("prompt") or "(프롬프트 없음)"
        short = shorten_one_line(prompt_text, 60) or "(프롬프트 없음)"
        when = _format_dt(row.get("created_at"))
        mode = _row_app_mode(row)
        owner_email = row.get("user_email") or "(anon)"

        header = f"🕘 {when}  ·  `{mode}`  ·  {short}"
        with st.expander(header, expanded=(st.session_state.get(_SS_EXPAND) == archive_id)):
            st.caption(f"작성자: {owner_email}  ·  archive_id: {archive_id}")

            with st.container():
                st.markdown("**📝 프롬프트**")
                st.markdown(f"> {prompt_text[:600]}")

            # payload 가 행에 들어있으면 그대로, 없으면 lazy fetch.
            payload = row.get("payload")
            if not isinstance(payload, dict) or not payload:
                with st.spinner("payload 불러오는 중..."):
                    full = None
                    try:
                        full = db.fetch_payload(archive_id)
                    except Exception as e:
                        st.error(f"payload 조회 실패: {e}")
                    if full:
                        payload = full.get("payload") or {}
                    else:
                        payload = {}

            _render_payload_preview(payload)

            # 액션 버튼들.
            bcol1, bcol2, bcol3 = st.columns([1, 1, 2])
            with bcol1:
                if st.button(
                    "🔁 다시 열기",
                    key=f"archive_reopen_{archive_id}",
                    use_container_width=True,
                    type="primary",
                ):
                    _restore_archive(db, archive_id)

            with bcol2:
                if is_admin:
                    confirm_id = st.session_state.get(_SS_DELETE_CONFIRM)
                    if confirm_id == archive_id:
                        if st.button(
                            "✅ 삭제 확정",
                            key=f"archive_delete_confirm_{archive_id}",
                            use_container_width=True,
                        ):
                            if _delete_archive(db, archive_id):
                                st.session_state[_SS_DELETE_CONFIRM] = None
                                st.success("삭제되었습니다.")
                                st.rerun()
                    else:
                        if st.button(
                            "🗑️ 삭제",
                            key=f"archive_delete_{archive_id}",
                            use_container_width=True,
                        ):
                            st.session_state[_SS_DELETE_CONFIRM] = archive_id
                            st.rerun()
            with bcol3:
                if is_admin and st.session_state.get(_SS_DELETE_CONFIRM) == archive_id:
                    if st.button(
                        "취소",
                        key=f"archive_delete_cancel_{archive_id}",
                        use_container_width=True,
                    ):
                        st.session_state[_SS_DELETE_CONFIRM] = None
                        st.rerun()
