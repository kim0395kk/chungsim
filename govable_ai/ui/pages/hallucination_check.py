# -*- coding: utf-8 -*-
"""
환각 검사 (사실성 검증) 페이지

streamlit_app_legacy.py 의 hallucination_check 모드를 단일 페이지로 추출.
- 패턴 기반 + LLM 기반 환각 탐지
- 의심 구간 하이라이트, 검증 수행 내역, 상세 리포트
- 처리 체크리스트 / 응답 초안 보조 생성

의존성: hallucination_detection.py (project root) 와 LLM 서비스.
모듈이 누락되면 우아하게 비활성 메시지를 띄운다.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import streamlit as st

try:
    from hallucination_detection import (
        detect_hallucination_cached,
        generate_processing_checklist,
        generate_response_draft,
        get_text_hash,
        render_hallucination_report,
        render_highlighted_text,
        render_verification_log,
    )
except Exception:  # pragma: no cover - optional dependency
    detect_hallucination_cached = None  # type: ignore[assignment]
    generate_processing_checklist = None  # type: ignore[assignment]
    generate_response_draft = None  # type: ignore[assignment]
    get_text_hash = None  # type: ignore[assignment]
    render_hallucination_report = None  # type: ignore[assignment]
    render_highlighted_text = None  # type: ignore[assignment]
    render_verification_log = None  # type: ignore[assignment]


_RISK_LABELS = {
    "high": "높음 ⚠️",
    "medium": "중간 ⚡",
    "low": "낮음 ✅",
}


def _parse_context(raw: str) -> dict:
    """사용자가 입력한 JSON 컨텍스트를 안전하게 파싱."""
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        st.warning("관련 사실은 JSON 객체(dict) 형태여야 합니다. 빈 컨텍스트로 진행합니다.")
        return {}
    except json.JSONDecodeError as e:
        st.warning(f"관련 사실 JSON 파싱 실패: {e}. 빈 컨텍스트로 진행합니다.")
        return {}


def _render_top_metrics(detection_result: dict) -> None:
    """상단 위험도 / 점수 / 탐지 건수 메트릭."""
    risk_level = detection_result.get("risk_level", "low")
    overall_score = float(detection_result.get("overall_score", 0.0) or 0.0)
    total_issues = int(detection_result.get("total_issues_found", 0) or 0)

    c1, c2, c3 = st.columns(3)
    c1.metric("위험도", _RISK_LABELS.get(risk_level, risk_level))
    c2.metric("전체 점수", f"{overall_score:.2f}")
    c3.metric("총 탐지 건수", f"{total_issues}건")


def _persist_to_db(
    db: Any,
    text: str,
    detection_result: dict,
) -> None:
    """검증 결과를 가능하면 DB에 보관 (govable_ai/main.py 패턴)."""
    if db is None or not getattr(db, "is_available", lambda: False)():
        return

    anon_id = st.session_state.get("anon_session_id")
    payload = {
        "app_mode": "hallucination_check",
        "text": text,
        "detection": detection_result,
    }
    try:
        archive_id = db.insert_archive(
            prompt=text,
            payload=payload,
            anon_session_id=anon_id,
            user_id=st.session_state.get("user_id"),
            user_email=st.session_state.get("user_email"),
        )
        if archive_id:
            st.session_state["current_archive_id"] = archive_id
            db.log_event(
                "hallucination_run",
                anon_id,
                archive_id=archive_id,
                meta={
                    "text_len": len(text),
                    "risk_level": detection_result.get("risk_level"),
                    "total_issues_found": detection_result.get("total_issues_found", 0),
                },
            )
    except Exception as e:  # pragma: no cover - DB 실패는 페이지에 노출하지 않음
        st.caption(f"⚠️ 결과 보관 중 오류 (무시 가능): {e}")


def _render_results(
    text: str,
    detection_result: dict,
    llm_service: Any,
) -> None:
    """검증 결과 본문 렌더링."""
    st.divider()
    _render_top_metrics(detection_result)

    st.divider()

    # 1. 검증 수행 내역
    v_log = detection_result.get("verification_log") or {}
    if v_log and render_verification_log is not None:
        try:
            render_verification_log(detection_result, v_log)
            st.divider()
        except Exception as e:  # pragma: no cover - UI 보호
            st.warning(f"검증 수행 내역 렌더링 오류: {e}")

    # 2. 원문 하이라이트
    if render_highlighted_text is not None:
        try:
            render_highlighted_text(text, detection_result.get("suspicious_parts", []) or [])
            st.divider()
        except Exception as e:  # pragma: no cover - UI 보호
            st.warning(f"하이라이트 렌더링 오류: {e}")

    # 3. 상세 환각 탐지 리포트
    if render_hallucination_report is not None:
        st.subheader("🔍 환각 탐지 결과")
        try:
            render_hallucination_report(detection_result)
        except Exception as e:  # pragma: no cover - UI 보호
            st.warning(f"리포트 렌더링 오류: {e}")

    st.divider()

    # 4. 보조 도구 — 처리 체크리스트
    with st.expander("✅ 처리 체크리스트 생성", expanded=False):
        st.caption("탐지 결과를 바탕으로 단계별 업무 체크리스트를 생성합니다.")
        if st.button("체크리스트 생성", key="hallucination_checklist_btn"):
            if generate_processing_checklist is None:
                st.error("체크리스트 생성 모듈을 사용할 수 없습니다.")
            else:
                with st.spinner("체크리스트 생성 중..."):
                    try:
                        checklist = generate_processing_checklist(
                            {
                                "petition": text,
                                "detection": detection_result,
                                "priority": {},
                            },
                            llm_service,
                        )
                    except Exception as e:
                        st.error(f"체크리스트 생성 오류: {e}")
                        checklist = []

                if not checklist:
                    st.info("생성된 체크리스트가 없습니다.")
                else:
                    for step_data in checklist:
                        step_num = step_data.get("step", 0)
                        step_title = step_data.get("title", "단계")
                        step_deadline = step_data.get("deadline", "")
                        items = step_data.get("items", []) or []
                        st.markdown(
                            f"**Step {step_num}. {step_title}** "
                            f"_(기한: {step_deadline})_"
                        )
                        for item in items:
                            task_text = item.get("task", "") if isinstance(item, dict) else str(item)
                            st.markdown(f"- ☐ {task_text}")

    # 5. 보조 도구 — 응답 초안 생성
    with st.expander("📄 응답 초안 생성", expanded=False):
        st.caption("탐지 결과를 반영한 회신/응답 초안을 자동으로 작성합니다.")
        response_type = st.selectbox(
            "회신 유형",
            ["approval", "rejection", "partial", "request_info"],
            format_func=lambda x: {
                "approval": "✅ 승인/수용",
                "rejection": "❌ 불가/거부",
                "partial": "⚖️ 부분 수용",
                "request_info": "📝 보완 요청",
            }.get(x, x),
            key="hallucination_response_type",
        )
        if st.button("초안 생성", key="hallucination_draft_btn"):
            if generate_response_draft is None:
                st.error("초안 생성 모듈을 사용할 수 없습니다.")
            else:
                with st.spinner("응답 초안 작성 중... (약 10초)"):
                    try:
                        draft = generate_response_draft(
                            text,
                            {"detection": detection_result, "priority": {}},
                            response_type,
                            llm_service,
                        )
                    except Exception as e:
                        st.error(f"초안 생성 오류: {e}")
                        draft = ""

                if draft:
                    st.text_area(
                        "생성된 응답 초안 (수정 가능)",
                        draft,
                        height=360,
                        key="hallucination_draft_editor",
                    )


def render_hallucination_check_page(
    llm_service: Any,
    db: Optional[Any] = None,
    services: Optional[dict] = None,
) -> None:
    """환각 검사 (사실성 검증) 페이지 진입점.

    Args:
        llm_service: govable_ai.core.llm_service.LLMService 인스턴스
        db: Supabase 등 DB 게이트웨이 (선택)
        services: 추가 서비스 컨테이너 (선택, 미사용)
    """
    st.markdown("## 🔍 환각 검사 (사실성 검증)")

    if detect_hallucination_cached is None or get_text_hash is None:
        st.error(
            "hallucination_detection 모듈을 import 하지 못했습니다. "
            "프로젝트 루트의 hallucination_detection.py 파일을 확인해주세요."
        )
        return

    with st.expander("💡 이 도구가 하는 일", expanded=False):
        st.markdown(
            """
            AI가 생성한 문서(민원, 보고서 등)의 **환각(허위 정보)**을 자동으로 탐지합니다.

            - **패턴 기반 검증**: 잘못된 날짜, 가상 법조항, 의심스러운 통계, 금액 불일치 등 규칙 기반 탐지
            - **LLM 교차 검증**: AI가 본문을 다시 읽어 사실 관계를 점검
            - **위험도 라벨링**: 낮음 ✅ / 중간 ⚡ / 높음 ⚠️ 으로 자동 분류
            - **보조 도구**: 처리 체크리스트, 응답 초안 자동 생성

            > 결과는 **보조 자료**입니다. 최종 판단은 담당자가 직접 확인해주세요.
            """
        )

    text = st.text_area(
        "검증할 텍스트",
        height=400,
        placeholder=(
            "검증할 AI 생성 초안을 붙여넣으세요.\n"
            "예: 2024년 13월 32일자 ○○법 제999조에 따르면..."
        ),
        key="hallucination_text_input",
    )

    context_raw = st.text_input(
        "관련 사실 (선택)",
        placeholder='예: {"date": "2024-01-01", "law": "주민등록법..."}',
        help="JSON 객체 형태로 입력하면 검증 정확도가 향상됩니다.",
        key="hallucination_context_input",
    )

    col_btn, col_meta = st.columns([3, 1])
    with col_btn:
        run_btn = st.button(
            "🔬 검증 시작",
            type="primary",
            use_container_width=True,
            disabled=not (text or "").strip(),
        )
    with col_meta:
        if text:
            st.caption(f"📏 {len(text)}자")

    if run_btn and (text or "").strip():
        context = _parse_context(context_raw)
        with st.spinner("패턴 + AI 교차 검증 수행 중..."):
            try:
                text_hash = get_text_hash(text)
                detection_result = detect_hallucination_cached(
                    text_hash, text, context, llm_service
                )
            except Exception as e:
                st.error(f"검증 중 오류가 발생했습니다: {e}")
                return

        st.session_state["hallucination_last_result"] = {
            "text": text,
            "detection": detection_result,
        }
        # DB 보관은 검증 직후 한 번만 시도
        _persist_to_db(db, text, detection_result)

    # 직전 결과가 있으면 항상 다시 렌더 (체크리스트/초안 버튼 클릭 후에도 유지)
    last = st.session_state.get("hallucination_last_result")
    if last and last.get("detection"):
        _render_results(last["text"], last["detection"], llm_service)
