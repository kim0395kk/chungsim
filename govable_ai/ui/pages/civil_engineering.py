# -*- coding: utf-8 -*-
"""
토목직 특화 AI 페이지 (Civil Engineering RAG)

streamlit_app_legacy.py 에 흩어져 있던 civil_engineering 모드를 단일 페이지로 추출.
- Tab 1: 규정/매뉴얼 검색 (RAG 채팅)
- Tab 2: 전문 공문 작성 (RAG 컨텍스트 기반 초안)

의존성: civil_engineering 패키지(rag_system, dashboard) 와 LLM 서비스.
RAG 모듈이나 sentence-transformers/faiss 가 없으면 우아하게 비활성 메시지를 띄운다.
"""
from __future__ import annotations

from typing import Any, Optional

import streamlit as st

try:
    from civil_engineering.rag_system import load_rag_system
except Exception:  # pragma: no cover - optional dependency
    load_rag_system = None  # type: ignore[assignment]


def _render_response_panel(payload: dict) -> None:
    """RAG 답변 페이로드를 사용자 친화적으로 렌더링."""
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
        st.warning(
            "내부 문서 근거가 부족해 일반 지식 기반 답변이 포함되었습니다. "
            "최종 판단 전 원문 확인이 필요합니다."
        )
    if quality.get("location_only_risk"):
        st.warning(
            "위치 편중 응답 위험을 감지해 핵심 항목(면적/예산/기간 등)을 보강했습니다."
        )
    if confidence < 0.45:
        st.info(
            "💡 더 정확한 답변을 위해 대상/기간/사업명 등 구체 정보를 추가해 주세요."
        )

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


def _get_rag() -> Optional[Any]:
    """RAG 시스템을 로드한다. 의존성 누락 시 None."""
    if load_rag_system is None:
        return None
    try:
        return load_rag_system()
    except Exception as e:
        st.error(f"RAG 시스템 초기화 실패: {e}")
        return None


def _render_search_tab(llm_service: Any) -> None:
    """Tab 1 — 규정/매뉴얼 검색 (RAG 채팅)."""
    st.info(
        "🚧 **도로폭, 양생 온도 등 궁금한 실무 규정을 물어보세요.**\n\n"
        "내부 매뉴얼과 지침을 찾아보고 정확한 근거와 함께 답변해 드립니다."
    )

    if "civil_chat_history" not in st.session_state:
        st.session_state.civil_chat_history = []

    for msg in st.session_state.civil_chat_history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                _render_response_panel(msg)
            else:
                st.markdown(msg["content"])

    prompt = st.chat_input(
        "규정이나 지침에 대해 물어보세요 (예: 겨울철 콘크리트 양생 온도 기준?)"
    )
    if not prompt:
        return

    st.session_state.civil_chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("규정을 찾아보고 있습니다..."):
            rag = _get_rag()
            if rag is None:
                st.error("RAG 시스템을 사용할 수 없습니다. (의존성 또는 데이터 누락)")
                return

            try:
                response = rag.answer_question(prompt, llm_service)
            except Exception as e:
                st.error(f"답변 생성 중 오류가 발생했습니다: {e}")
                return

            answer_text = response.get("answer", "죄송합니다. 답변을 생성하지 못했습니다.")
            payload = {
                "content": answer_text,
                "summary": response.get("summary", ""),
                "sources": response.get("sources", []),
                "fact_rows": response.get("fact_rows", []),
                "quality": response.get("quality", {}) or {},
                "confidence": float(response.get("confidence") or 0.0),
                "retrieval_meta": response.get("retrieval_meta", {}) or {},
            }
            _render_response_panel(payload)

            st.session_state.civil_chat_history.append({
                "role": "assistant",
                **payload,
                "parsed_facts": response.get("parsed_facts", []),
            })


def _render_drafting_tab(llm_service: Any) -> None:
    """Tab 2 — 전문 공문 작성 (RAG 컨텍스트 기반)."""
    st.info(
        "📝 **복잡한 공문서, 핵심 내용만 입력하면 전문가 수준으로 초안을 만들어 드립니다.**\n\n"
        "공사 감독 조서, 인허가 공문 등 필요한 양식을 선택하고 내용을 적어주세요."
    )

    st.markdown("### ✍️ 작성 요청")

    draft_topic = st.text_input(
        "공문 주제",
        placeholder="예: 수해복구 공사 착공신고서 접수 처리",
    )
    draft_details = st.text_area(
        "세부 사항",
        height=300,
        placeholder=(
            "예시:\n"
            "1. 업체명: (주)대한건설\n"
            "2. 공사기간: 2024.3.1 ~ 2024.6.30\n"
            "3. 특이사항: 안전관리계획서 제출 완료됨.\n"
            "4. 요청사항: 법령 근거를 포함해서 정중하게 작성해줘."
        ),
    )

    if st.button("✨ 공문 초안 작성", type="primary", use_container_width=True):
        if not draft_topic:
            st.warning("공문 주제를 입력해주세요.")
        else:
            # 새 작성 요청 시 이전 결과를 즉시 비워 혼동을 막는다.
            st.session_state.pop("civil_draft_result", None)
            with st.spinner("규정 및 매뉴얼을 참조하여 공문을 작성 중입니다..."):
                rag = _get_rag()
                if rag is None:
                    st.error("RAG 시스템을 사용할 수 없습니다.")
                else:
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
                    try:
                        st.session_state.civil_draft_result = rag.answer_question(
                            draft_prompt, llm_service
                        )
                    except Exception as e:
                        st.error(f"공문 초안 생성 중 오류가 발생했습니다: {e}")

    if "civil_draft_result" in st.session_state:
        res = st.session_state.civil_draft_result
        st.markdown("---")
        st.markdown("### 📄 공문 초안")
        st.info(res.get("answer"))

        with st.expander("📝 텍스트로 복사하기"):
            st.code(res.get("answer"), language="text")

        if res.get("sources"):
            with st.expander("📚 참고한 규정 및 매뉴얼", expanded=False):
                for src in res.get("sources"):
                    st.caption(f"- {src}")


def render_civil_engineering_page(llm_service: Any) -> None:
    """토목직 특화 AI 페이지 진입점.

    Args:
        llm_service: govable_ai.core.llm_service.LLMService 인스턴스
    """
    st.markdown("## 👷 토목직 특화 AI")

    if load_rag_system is None:
        st.error(
            "civil_engineering 패키지를 import 하지 못했습니다. "
            "sentence-transformers/faiss 등 의존성을 확인해주세요."
        )
        return

    tab1, tab2 = st.tabs(["🔍 규정/매뉴얼 검색", "✍️ 공문 초안 작성"])
    with tab1:
        _render_search_tab(llm_service)
    with tab2:
        _render_drafting_tab(llm_service)
