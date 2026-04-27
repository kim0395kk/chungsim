# -*- coding: utf-8 -*-
"""
업무편람(당직 매뉴얼) 검색 페이지

streamlit_app_legacy.py 의 사이드바 버튼/다이얼로그(`render_duty_manual_button`)
형태였던 당직메뉴얼 봇을, 검색에 특화된 풀 페이지 형태로 재구성한다.

- 큰 검색창 + 검색 버튼
- 동의어(구어체→행정용어) 매핑 결과 표시
- LLM 키워드 추출 결과 표시
- 후보 매뉴얼(섹션/부서/연락처) 목록과 LLM 종합 답변

기존 `govable_ai/features/duty_manual.py` 의 공개 함수를 재사용하며,
페이지 자체는 비즈니스 로직을 중복하지 않는다.
"""
from __future__ import annotations

from typing import Any, Optional

import streamlit as st

try:
    from govable_ai.features.duty_manual import (
        SYNONYMS,
        normalize_query,
        llm_extract_keywords,
        retrieve_duty_context,
        call_llm,
    )
except Exception:  # pragma: no cover - 모듈 누락 시 graceful fallback
    SYNONYMS = {}  # type: ignore[assignment]
    normalize_query = None  # type: ignore[assignment]
    llm_extract_keywords = None  # type: ignore[assignment]
    retrieve_duty_context = None  # type: ignore[assignment]
    call_llm = None  # type: ignore[assignment]


def _resolve_db(db: Optional[Any], services: Optional[dict]) -> Optional[Any]:
    """Supabase 클라이언트(sb)를 services dict 또는 db 인자에서 추출."""
    if db is not None:
        return db
    if services:
        for key in ("sb", "supabase", "db"):
            client = services.get(key)
            if client is not None:
                return client
    return None


def _render_synonym_summary(query: str) -> str:
    """입력 질의에 적용된 동의어 매핑을 사용자에게 보여주고 정규화 결과 반환."""
    if not query:
        return query
    if normalize_query is None:
        return query

    try:
        normalized = normalize_query(query)
    except Exception as e:
        st.warning(f"동의어 정규화 실패: {e}")
        return query

    applied = []
    for src, dst in (SYNONYMS or {}).items():
        if src in query:
            applied.append((src, dst))

    if applied:
        with st.expander("🔁 적용된 동의어 매핑", expanded=False):
            for src, dst in applied:
                st.markdown(f"- `{src}` → **{dst}**")
            if normalized != query:
                st.caption(f"정규화된 질의: {normalized}")
    return normalized


def _render_keywords(llm_service: Any, query: str) -> list:
    """LLM 키워드 추출 결과를 표시."""
    if llm_extract_keywords is None or llm_service is None:
        return []
    try:
        keywords = llm_extract_keywords(llm_service, query) or []
    except Exception as e:
        st.warning(f"키워드 추출 중 오류: {e}")
        return []

    if keywords:
        with st.expander("🧠 LLM 추출 키워드", expanded=False):
            st.markdown(" · ".join(f"`{kw}`" for kw in keywords))
    return keywords


def _render_candidates(candidates: list) -> None:
    """검색된 매뉴얼 후보(섹션/부서/연락처/내용)를 카드처럼 나열."""
    st.markdown("### 📂 검색된 매뉴얼 섹션")
    if not candidates:
        st.info("일치하는 매뉴얼 섹션이 없습니다. 다른 표현으로 다시 검색해 보세요.")
        return

    for idx, item in enumerate(candidates, start=1):
        section = item.get("section_path") or "(섹션 없음)"
        dept = item.get("dept") or "-"
        contact = item.get("team_contact") or "-"
        content = item.get("content") or ""
        with st.container(border=True):
            st.markdown(f"**{idx}. {section}**")
            st.caption(f"담당부서: {dept}  ·  ☎ {contact}")
            st.write(content)


def _render_llm_summary(
    llm_service: Any,
    query: str,
    candidates: list,
) -> None:
    """Top 3 후보를 컨텍스트로 LLM 종합 답변 생성."""
    if call_llm is None or llm_service is None or not candidates:
        return

    top = candidates[:3]
    context_str = ""
    for idx, item in enumerate(top, start=1):
        context_str += (
            f"\n[후보 {idx}]\n"
            f"- 위치: {item.get('section_path')}\n"
            f"- 부서: {item.get('dept')} (☎ {item.get('team_contact')})\n"
            f"- 내용: {item.get('content')}\n"
        )

    sys_prompt = f"""
너는 충주시청 당직 근무자 도우미다.
사용자 질문에 대해 아래 [매뉴얼 후보]를 참고하여 답변하라.

[매뉴얼 후보]
{context_str}

[답변 규칙]
1. 후보들 중 사용자 질문과 가장 상황이 일치하는 하나를 골라 답변하라.
2. 질문이 모호하다면 사용자에게 상황을 되물어라.
3. 답변 시 담당 부서와 연락처를 가장 먼저 명시하라.
4. 매뉴얼에 없는 내용은 지어내지 말고 "내용 없음"이라고 하라.
"""
    rag_prompt = f"{sys_prompt}\n\n질문: {query}"

    st.markdown("### 🧾 종합 답변")
    with st.spinner("매뉴얼을 종합하여 답변을 작성 중입니다..."):
        try:
            answer = call_llm(llm_service, rag_prompt)
        except Exception as e:
            st.error(f"답변 생성 중 오류가 발생했습니다: {e}")
            return
    st.info(answer)


def _render_empty_state() -> None:
    """질의가 비었을 때의 친절한 안내."""
    st.info(
        "🔍 **검색어를 입력하면 충주시청 업무편람(당직 매뉴얼)에서 관련 섹션을 찾아드립니다.**\n\n"
        "예시:\n"
        "- 하수도에서 냄새가 나요\n"
        "- 도로에 고라니가 죽어있어요\n"
        "- 노숙자가 시청에 찾아왔어요\n"
        "- 불법주정차 신고 처리 방법"
    )


def render_duty_manual_page(
    llm_service: Any,
    db: Optional[Any] = None,
    services: Optional[dict] = None,
) -> None:
    """업무편람 검색 풀 페이지 진입점.

    Args:
        llm_service: LLMService 인스턴스(키워드 추출/종합답변용).
        db: Supabase 클라이언트(sb). services dict 가 있다면 거기서도 조회.
        services: 부가 서비스 모음(dict). `sb`/`supabase`/`db` 키를 인식.
    """
    st.markdown("## 📚 업무편람 검색")
    st.caption(
        "충주시청 당직 근무 매뉴얼(업무편람)에서 상황에 맞는 처리 절차와 담당 부서·연락처를 찾아드립니다."
    )

    # 의존성/연결 상태 확인
    if retrieve_duty_context is None:
        st.error(
            "duty_manual 기능 모듈을 import 하지 못했습니다. "
            "`govable_ai/features/duty_manual.py` 설치 상태를 확인해주세요."
        )
        return

    sb = _resolve_db(db, services)
    if sb is None:
        st.warning(
            "Supabase 연결이 없어 매뉴얼 검색이 비활성화되었습니다. "
            "관리자에게 SUPABASE_URL / SUPABASE_ANON_KEY 설정을 요청해주세요."
        )
        return

    # 검색 입력 영역
    with st.form(key="duty_manual_search_form", clear_on_submit=False):
        query = st.text_input(
            "무엇을 찾고 계신가요?",
            placeholder="예: 겨울철 도로 결빙 신고가 들어왔어요",
            key="duty_manual_query_input",
        )
        col_a, col_b = st.columns([1, 5])
        with col_a:
            submitted = st.form_submit_button(
                "🔎 검색", type="primary", use_container_width=True
            )
        with col_b:
            st.form_submit_button("초기화", use_container_width=False)

    # 빈 상태
    if not submitted or not (query or "").strip():
        _render_empty_state()
        return

    query = query.strip()
    st.markdown("---")
    st.markdown(f"#### 질의: `{query}`")

    # 1) 동의어 매핑 표시 (정규화 결과는 참고용으로만 노출 — 검색은 원문으로 수행)
    _render_synonym_summary(query)

    # 2) LLM 키워드 추출 결과 표시 (참고용)
    _render_keywords(llm_service, query)

    # 3) 매뉴얼 후보 검색
    try:
        candidates = retrieve_duty_context(sb, query, llm_service) or []
    except Exception as e:
        st.error(f"매뉴얼 검색 중 오류가 발생했습니다: {e}")
        return

    # 4) 후보 카드 렌더링
    _render_candidates(candidates)

    # 5) LLM 종합 답변
    # TODO: govable_ai.features.duty_manual 에 종합답변 함수가 별도로 노출되면
    #       해당 함수를 재사용하도록 교체. 현재는 page 측에서 prompt 를 조립한다.
    _render_llm_summary(llm_service, query, candidates)
