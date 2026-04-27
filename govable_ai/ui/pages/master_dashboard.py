# -*- coding: utf-8 -*-
"""
관리자 마스터 대시보드 페이지 (Master Dashboard)

streamlit_app_legacy.py 의 admin 대시보드 영역(약 3364~3750 행)을 페이지 모듈로 추출.
실제 렌더링 본체는 govable_ai.ui.dashboard.render_master_dashboard 가 담당하며,
본 모듈은 다음과 같은 페이지 진입 책임만 수행한다.

- 표준 페이지 인터페이스(`render_master_dashboard_page(...)`) 노출
- 관리자 전용 게이팅 (로그인 + 관리자 여부 확인)
- DB 가용성 그레이스풀 폴백
- 진입 안내/오류 메시지의 한국어 정렬

NOTE: 본 페이지는 dashboard.py 의 함수를 "감싸기"만 한다. 차트/통계 로직 자체는
       dashboard.py 에서 관리하며, 본 파일에서는 임의로 보강하지 않는다.
"""
from __future__ import annotations

from typing import Any, Optional

import streamlit as st

try:
    from govable_ai.ui.dashboard import render_master_dashboard
except Exception:  # pragma: no cover - 의존성 누락 시 안전 처리
    render_master_dashboard = None  # type: ignore[assignment]

try:
    from govable_ai.ui.auth import is_admin_user
except Exception:  # pragma: no cover
    is_admin_user = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# TODO(legacy-parity): streamlit_app_legacy.py 3364~3544 행에 비해 현재
# govable_ai/ui/dashboard.py 의 render_master_dashboard 가 누락 중인 항목.
# 본 페이지에서는 보강하지 않으며(다른 작업으로 분리), 추후 dashboard.py 수정 시 참고.
#   - 누적/오늘 방문자 메트릭 (total_visits, today_visitors, DAU)
#   - 평균/최대 체류 시간 (analytics_session_duration)
#   - 최근 7일 접속 추이(line) / 시간대별 접속 분포(bar)
#   - 사용자/날짜/모델 멀티 필터 + KPI(기간 내 비용·토큰·실행수)
#   - 토큰/비용 상세 차트 탭 (📈 비용/토큰, 🤖 모델 분석, 🔥 시간대 히트맵, 👤 사용자 랭킹)
#   - 헤비유저 강조 + 긴 레이턴시 행 하이라이트가 결합된 감사 로그 표
#   - CSV 다운로드 / 단건 삭제 / 프롬프트 원문 보기
#   - 세션(Sessions)·이벤트(Events) 원본 데이터 탭
#   - analytics_model_costs / analytics_total_cost_summary 뷰 기반 모델 비용 분석
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


def render_master_dashboard_page(
    llm_service: Any,
    db: Optional[Any] = None,
    services: Optional[dict] = None,
) -> None:
    """관리자 마스터 대시보드 페이지 진입점.

    Args:
        llm_service: LLM 서비스 인스턴스 (임베딩 재생성 등에 사용).
        db: Supabase DB 클라이언트. None 또는 미연결 시 안내 후 종료.
        services: 추가 서비스 맵 (현재 페이지에서는 미사용, 시그니처 통일 목적).
    """
    st.markdown("## 🏛️ 마스터 대시보드")

    # 1) 로그인 + 관리자 여부 확인
    if not _is_current_user_admin():
        st.warning("🔒 관리자 전용 페이지입니다.")
        st.info(
            "💡 관리자 계정으로 로그인하신 뒤, 사이드바에서 '관리자모드 켜기'를 활성화해 주세요."
        )
        return

    # 2) DB 가용성 확인 (None / is_available() False 모두 방어)
    db_ok = False
    if db is not None:
        try:
            db_ok = bool(db.is_available())
        except Exception:
            db_ok = False

    if not db_ok:
        st.warning("⚠️ DB 연결이 필요합니다.")
        st.info(
            "💡 Supabase 연결 정보(SUPABASE_URL / SUPABASE_ANON_KEY)를 확인한 뒤 다시 시도해 주세요."
        )
        return

    # 3) 본체 렌더 위임
    if render_master_dashboard is None:
        st.error(
            "마스터 대시보드 모듈을 불러오지 못했습니다. "
            "govable_ai.ui.dashboard 의존성을 확인해 주세요."
        )
        return

    try:
        render_master_dashboard(db, llm_service)
    except Exception as e:  # pragma: no cover - 런타임 안전망
        st.error(f"대시보드 렌더링 중 오류가 발생했습니다: {e}")
