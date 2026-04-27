# -*- coding: utf-8 -*-
"""
민원 분석기 페이지 (Complaint Analyzer Page).

streamlit_app_legacy.py 의 ``app_mode == "complaint_analyzer"`` 분기(~L4094)를
``govable_ai/ui/pages`` 로 추출한 모듈. 워크플로우 본체는
:mod:`govable_ai.features.complaint_analyzer` 에 있고 이 파일은 UI 만 담당한다.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import streamlit as st

from govable_ai.features.complaint_analyzer import run_complaint_analyzer_workflow

logger = logging.getLogger(__name__)


def _resolve_law_api(services: Optional[dict]) -> Any:
    """services dict 에서 law_api 객체를 꺼낸다 (없으면 None)."""
    if not services:
        return None
    return services.get("law_api") or services.get("law")


def _verdict_badge(verdict: str) -> str:
    return {
        "SUPPORTED": "✅ SUPPORTED",
        "INSUFFICIENT": "🟨 INSUFFICIENT",
        "REFUTED": "❌ REFUTED",
    }.get((verdict or "").upper(), "🟨 INSUFFICIENT")


def _citation_status_badge(status: str) -> str:
    return {
        "VALID": "✅ VALID",
        "PARTIAL": "🟨 PARTIAL",
        "INVALID": "❌ INVALID",
    }.get((status or "").upper(), "❔")


def _render_draft_tab(res: dict) -> None:
    """탭 1 — 핵심 답변 (draft_doc)."""
    doc = res.get("draft_doc") or res.get("doc") or {}
    pack = res.get("complaint_pack") or {}
    grade = pack.get("noise_grade") or "GREEN"
    vscore = pack.get("verifiability_score")
    try:
        vscore_pct = int(float(vscore) * 100)
    except Exception:
        vscore_pct = None

    st.markdown(
        f"""
        <div style='background:#ecfeff; padding:1rem; border-radius:8px;
                    border-left:4px solid #06b6d4; margin-bottom:1rem;'>
            <p style='margin:0 0 0.25rem 0; color:#0e7490; font-weight:700;'>등급: {grade}</p>
            <p style='margin:0; color:#155e75;'>검증가능성: {str(vscore_pct) + '%' if vscore_pct is not None else '-'}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not doc:
        st.info("회신 초안이 비어있습니다.")
        return

    st.markdown(f"### {doc.get('title', '민원 검토 결과 안내(초안)')}")
    st.caption(f"수신: {doc.get('receiver', '민원인 귀하')}")

    body = doc.get("body_paragraphs") or []
    body_text = "\n".join(body) if isinstance(body, list) else str(body)
    st.markdown(body_text)

    st.markdown("---")
    st.caption(f"기관장: {doc.get('department_head', '행정기관장')}")

    with st.expander("📋 텍스트로 복사하기"):
        copy_text = (
            f"제목: {doc.get('title', '')}\n"
            f"수신: {doc.get('receiver', '')}\n\n"
            f"{body_text}\n\n"
            f"{doc.get('department_head', '')}"
        )
        st.code(copy_text, language="text")


def _render_claims_tab(res: dict) -> None:
    """탭 2 — 주장별 검증 (verified_citations + verdicts)."""
    pack = res.get("complaint_pack") or {}
    verdicts = res.get("verdicts") or pack.get("verdicts") or []
    verified_citations = res.get("verified_citations") or pack.get("citations") or []

    st.markdown("#### 🧱 주장별 판정")
    if not verdicts:
        st.info("주장 판정 결과가 없습니다.")
    else:
        verdict_rows = []
        for v in verdicts:
            verdict_rows.append({
                "ID": v.get("claim_id") or "",
                "유형": v.get("type") or "",
                "판정": _verdict_badge(v.get("verdict") or ""),
                "신뢰도": f"{float(v.get('confidence') or 0.0) * 100:.0f}%",
                "안전 진술": (v.get("safe_statement") or "")[:120],
            })
        st.table(verdict_rows)

        with st.expander("📌 주장별 추가 필요 자료", expanded=False):
            for v in verdicts:
                cid = v.get("claim_id") or ""
                needed = v.get("needed") or []
                if not needed:
                    continue
                st.markdown(f"**{cid}**")
                for n in needed[:7]:
                    st.write("- ", n)
                st.divider()

    st.markdown("#### ⚖️ 법령 인용 검증")
    if not verified_citations:
        st.caption("(민원 텍스트에서 명시적 법령 인용이 없거나 추출되지 않았습니다.)")
    else:
        cit_rows = []
        for c in verified_citations:
            cit_rows.append({
                "주장ID": c.get("claim_id") or "",
                "법령": c.get("law_name") or "",
                "조문": c.get("article_raw") or "",
                "상태": _citation_status_badge(c.get("status") or ""),
                "링크": c.get("link") or "",
            })
        st.table(cit_rows)


def _render_hallucination_tab(res: dict) -> None:
    """탭 3 — 환각 신호."""
    pack = res.get("complaint_pack") or {}
    signals = pack.get("hallucination_signals") or []
    grade_reasons = pack.get("grade_reasons") or []

    st.markdown("#### 🚨 환각/추정 신호")
    if signals:
        for s in signals:
            st.markdown(f"- {s}")
    else:
        st.success("자동 점검에서 명백한 환각 신호는 감지되지 않았습니다.")

    st.markdown("#### 📊 등급 부여 사유")
    if grade_reasons:
        for r in grade_reasons:
            st.markdown(f"- {r}")
    else:
        st.caption("특이 사유 없음.")


def _render_evidence_tab(res: dict) -> None:
    """탭 4 — 원문/근거."""
    pack = res.get("complaint_pack") or {}

    st.markdown("#### 📝 민원 원문")
    situation = res.get("situation") or ""
    if situation:
        st.text_area(
            "원문",
            value=situation,
            height=180,
            disabled=True,
            label_visibility="collapsed",
        )
    else:
        st.caption("원문이 없습니다.")

    with st.expander("🧩 MVC(언제/어디/대상/요청/증거) 추출", expanded=True):
        st.json(pack.get("mvc") or {})

    with st.expander("🧱 주장(Claim) 원본", expanded=False):
        st.json(pack.get("claims") or [])

    with st.expander("⚖️ 법령 검증 근거(요약)", expanded=False):
        st.markdown(res.get("law") or "_근거 없음_")

    with st.expander("🔧 처리/회신 전략", expanded=False):
        st.markdown(res.get("strategy") or "_전략 없음_")


def render_complaint_analyzer_page(
    llm_service: Any,
    db: Optional[Any] = None,
    services: Optional[dict] = None,
) -> None:
    """민원 분석기 페이지 진입점.

    Args:
        llm_service: govable_ai.core.llm_service.LLMService 인스턴스.
        db: govable_ai.core.db_client.SupabaseClient (선택).
        services: ``{"law_api": ..., ...}`` 의존성 딕셔너리(선택).
    """
    # 헤더 (civil_engineering.py 스타일에 맞춤)
    st.markdown("## 🧾 민원 분석기")

    with st.expander("💡 이 도구가 하는 일", expanded=False):
        st.markdown("""
- 민원 텍스트를 **주장(Claim) 단위로 분해**
- **환각/허위 법령 인용** 신호 점검
- 국가법령정보센터 **공식 API**로 법령/조문 존재 여부를 확인(가능 범위)
- 공무원이 책임질 수 있도록 **단정 최소** 형태의 회신 초안을 생성
        """)

    st.markdown(
        """
        <div style='background:#fef3c7; border-left:4px solid #f59e0b;
                    padding:1rem; border-radius:8px; margin:1rem 0;'>
            <p style='margin:0; color:#92400e; font-size:0.9rem; font-weight:500;'>
                ⚠️ 민감정보(성명·연락처·주소·주민번호·차량번호 등) 입력 금지 / 또는 마스킹 후 입력
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    complaint_input = st.text_area(
        "민원 원문",
        value=st.session_state.get("complaint_analyzer_input", ""),
        height=300,
        placeholder=(
            "민원 원문을 붙여넣으세요. (개인정보는 제거 권장)\n"
            "예: 언제/어디/대상/요청/증거가 포함되면 정확도가 크게 올라갑니다."
        ),
        key="complaint_analyzer_input",
    )

    law_api = _resolve_law_api(services)
    if law_api is None:
        st.caption("ℹ️ 법령 API 클라이언트가 주입되지 않아 인용 검증이 제한될 수 있습니다.")

    if st.button("🔍 분석 시작", type="primary", use_container_width=True):
        if not (complaint_input or "").strip():
            st.warning("민원 원문을 입력해주세요.")
        else:
            stage_box = st.empty()

            def _on_progress(stage: str, message: str) -> None:
                stage_box.info(f"⏳ [{stage}] {message}")

            with st.spinner("민원을 분석 중입니다..."):
                try:
                    res = run_complaint_analyzer_workflow(
                        complaint_input,
                        llm_service=llm_service,
                        law_api=law_api,
                        services=services,
                        on_progress=_on_progress,
                    )
                except Exception as e:
                    logger.exception("complaint analyzer workflow failed")
                    st.error(f"분석 중 오류가 발생했습니다: {e}")
                    res = None

            stage_box.empty()

            if res:
                # DB archive (govable_ai/main.py L414-429 패턴)
                archive_id = None
                if db is not None:
                    try:
                        if getattr(db, "is_available", lambda: False)():
                            anon_id = st.session_state.get("anon_session_id")
                            archive_id = db.insert_archive(
                                prompt=complaint_input,
                                payload=res,
                                anon_session_id=anon_id,
                                user_id=st.session_state.get("user_id"),
                                user_email=st.session_state.get("user_email"),
                            )
                            if archive_id:
                                st.session_state.current_archive_id = archive_id
                                try:
                                    db.log_event(
                                        "complaint_run",
                                        anon_id,
                                        archive_id=archive_id,
                                        meta={"prompt_len": len(complaint_input)},
                                    )
                                except Exception:
                                    logger.debug("db.log_event failed", exc_info=True)
                    except Exception as e:
                        logger.warning("DB archive 저장 실패: %s", e)
                        st.caption(f"⚠️ 결과 저장 실패 (분석 자체는 정상): {e}")

                res["archive_id"] = archive_id
                st.session_state["complaint_analyzer_result"] = res

    # 결과 렌더
    res = st.session_state.get("complaint_analyzer_result")
    if not res:
        return

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 핵심 답변",
        "🧱 주장별 검증",
        "🚨 환각 신호",
        "📑 원문/근거",
    ])
    with tab1:
        _render_draft_tab(res)
    with tab2:
        _render_claims_tab(res)
    with tab3:
        _render_hallucination_tab(res)
    with tab4:
        _render_evidence_tab(res)
