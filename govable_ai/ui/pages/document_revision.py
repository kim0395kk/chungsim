# -*- coding: utf-8 -*-
"""
기안/공고문 수정 페이지 (Document Revision).

streamlit_app_legacy.py 의 `app_mode == "revision"` 블록을 단일 페이지 모듈로 추출.
- 원문과 수정 요청사항을 입력받아 '2025 개정 공문서 작성 표준'에 맞춰 교정한다.
- 워크플로우 실행은 govable_ai.features.document_revision.run_revision_workflow 를 사용하고,
  진행 표시는 govable_ai.ui.premium_animations.render_revision_animation 으로 감싼다.
- 결과(수정본/변경 로그/요약)를 사용자 친화적으로 렌더링하고, DB 가 사용 가능하면 아카이브에 저장한다.
"""
from __future__ import annotations

from typing import Any, Optional

import streamlit as st

from govable_ai.features.document_revision import run_revision_workflow
from govable_ai.ui.premium_animations import render_revision_animation


def _render_revised_document(revised_doc: dict) -> None:
    """수정된 공문(revised_doc)을 보기 좋게 렌더링."""
    if not isinstance(revised_doc, dict):
        st.warning("수정된 문서 형식이 올바르지 않습니다.")
        return

    title = revised_doc.get("title") or ""
    receiver = revised_doc.get("receiver") or ""
    body_paragraphs = revised_doc.get("body_paragraphs") or []
    department_head = revised_doc.get("department_head") or ""

    if title:
        st.markdown(f"#### 📌 {title}")
    if receiver:
        st.caption(f"수 신: {receiver}")

    if body_paragraphs:
        body_text = "\n\n".join(str(p) for p in body_paragraphs)
        st.markdown(body_text)
    else:
        st.caption("본문 내용이 없습니다.")

    if department_head:
        st.markdown("---")
        st.markdown(f"**{department_head}**")

    # 텍스트 복사를 위한 expander
    with st.expander("📝 텍스트로 복사하기"):
        plain_lines = []
        if title:
            plain_lines.append(f"제 목: {title}")
        if receiver:
            plain_lines.append(f"수 신: {receiver}")
        plain_lines.append("")
        for p in body_paragraphs:
            plain_lines.append(str(p))
        if department_head:
            plain_lines.append("")
            plain_lines.append(department_head)
        st.code("\n".join(plain_lines), language="text")


def _render_revision_result(res: dict) -> None:
    """run_revision_workflow 결과 dict 를 화면에 렌더링."""
    if not isinstance(res, dict):
        st.error("결과 형식이 올바르지 않습니다.")
        return

    if res.get("warning"):
        st.warning(res["warning"])

    revised_doc = res.get("revised_doc") or {}
    changelog = res.get("changelog") or []
    summary = res.get("summary") or ""
    model_used = res.get("model_used")

    st.markdown("### 📄 수정된 문서")
    _render_revised_document(revised_doc)

    st.markdown("---")
    st.markdown("### 🔍 변경 로그")
    if changelog:
        md_text = ""
        for log in changelog:
            md_text += f"- ✅ {log}\n"
        st.markdown(md_text)
    else:
        st.caption("변경 사항이 없습니다.")

    if summary:
        st.markdown("### 📝 수정 요약")
        st.info(summary)

    if model_used:
        st.caption(f"사용 모델: `{model_used}`")


def render_document_revision_page(
    llm_service: Any,
    db: Optional[Any] = None,
    services: Optional[dict] = None,
) -> None:
    """기안/공고문 수정 페이지 진입점.

    Args:
        llm_service: govable_ai.core.llm_service.LLMService 인스턴스.
        db: govable_ai.core.db_client.DBClient (옵션). is_available() 시 결과 저장.
        services: 추가 서비스 dict (옵션, 향후 확장용).
    """
    st.markdown("## 📝 기안/공고문 수정")

    # 사용 안내
    with st.expander("💡 사용법", expanded=False):
        st.markdown(
            """
            1. **원문 붙여넣기**: 수정할 기안문이나 공고문을 '원문' 칸에 붙여넣으세요.
            2. **수정 요청 작성**: '수정 요청사항' 칸에 원하는 변경 내용을 작성하세요.
               - 예: "일시를 내일로 변경", "제목을 더 부드럽게", "오타 수정" 등
            3. **수정 시작 버튼 클릭**: 아래 버튼을 누르면 AI 가 '2025 개정 공문서 작성 표준'에 맞춰
               문서를 교정하고 변경 로그를 함께 보여드립니다.
            """
        )

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("### 📄 원문 입력")
        original_text = st.text_area(
            "원문",
            value=st.session_state.get("revision_org_text", ""),
            height=300,
            placeholder=(
                "여기에 수정할 문서의 원문을 붙여넣으세요.\n\n"
                "예시:\n"
                "제 목: 2025년 시민참여 예산 설명회 개최 안내\n"
                "수 신: 각 부서장\n"
                "발 신: 기획예산과\n\n"
                "시민참여 예산 설명회를 아래와 같이 개최하오니..."
            ),
            key="revision_org_text",
            label_visibility="collapsed",
        )

    with col_r:
        st.markdown("### ✏️ 수정 요청사항")
        revision_request = st.text_area(
            "수정 요청사항 (선택)",
            value=st.session_state.get("revision_req_text", ""),
            height=200,
            placeholder=(
                "비워두시면 '2025 개정 공문서 작성 표준'에 맞춰\n"
                "오탈자, 띄어쓰기, 표현을 자동으로 교정합니다.\n\n"
                "특정 요청 예시:\n"
                "- 일시를 2025. 1. 28.로 변경해주세요\n"
                "- 제목을 좀 더 부드럽게 바꿔주세요"
            ),
            key="revision_req_text",
            label_visibility="collapsed",
        )

    if st.button("✨ 수정 시작", type="primary", use_container_width=True):
        if not original_text:
            st.warning("⚠️ 원문을 입력해주세요.")
        else:
            combined_input = f"[원문]\n{original_text}\n\n[수정 요청]\n{revision_request}"
            user_email = st.session_state.get("user_email")

            placeholder = st.empty()
            res: Optional[dict] = None
            try:
                res = render_revision_animation(
                    placeholder,
                    run_revision_workflow,
                    combined_input,
                    llm_service,
                    sb=None,
                    user_email=user_email,
                )
            except Exception as e:
                st.error(f"처리 중 오류가 발생했습니다: {e}")
                res = None

            if res is None:
                st.error("❌ 문서 수정 결과를 받지 못했습니다. 잠시 후 다시 시도해 주세요.")
            elif isinstance(res, dict) and res.get("error"):
                st.error(res["error"])
            else:
                # DB 저장 (가능한 경우)
                archive_id = None
                if db is not None:
                    try:
                        if db.is_available():
                            archive_id = db.insert_archive(
                                prompt=combined_input,
                                payload=res,
                                anon_session_id=st.session_state.get("anon_session_id"),
                                user_id=st.session_state.get("user_id"),
                                user_email=user_email,
                            )
                            if archive_id:
                                st.session_state.current_archive_id = archive_id
                                db.log_event(
                                    "revision_run",
                                    st.session_state.get("anon_session_id"),
                                    archive_id=archive_id,
                                    meta={"prompt_len": len(combined_input)},
                                )
                                st.toast("💾 수정 내역이 저장되었습니다", icon="✅")
                    except Exception as e:
                        # 저장 실패는 사용자 흐름을 막지 않는다.
                        st.caption(f"(저장 중 경고: {e})")

                if isinstance(res, dict):
                    res["archive_id"] = archive_id
                st.session_state["revision_result"] = res

    # 결과 렌더링 / 빈 상태
    result = st.session_state.get("revision_result")
    if result:
        st.markdown("---")
        _render_revision_result(result)
    else:
        st.caption(
            "원문과 수정 요청사항을 입력하고 '수정 시작' 버튼을 누르면 결과가 이 영역에 표시됩니다."
        )
