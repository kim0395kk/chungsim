# -*- coding: utf-8 -*-
"""
프리미엄 애니메이션 컴포넌트.

향후 정교한 단계별 애니메이션을 추가할 수 있는 자리지만,
현재는 Streamlit 기본 spinner로 워크플로우를 감싸는 베이스라인만 제공한다.
"""
from typing import Any, Callable, Optional

import streamlit as st


def render_revision_animation(
    placeholder: Any,
    workflow_fn: Callable,
    combined_input: str,
    llm_service: Any,
    sb: Optional[Any] = None,
    user_email: Optional[str] = None,
) -> Any:
    """문서 수정 워크플로우를 spinner 안에서 실행하고 결과를 반환한다.

    Args:
        placeholder: 결과를 렌더링할 Streamlit placeholder (현재 미사용, 향후 단계 표시용)
        workflow_fn: (combined_input, llm_service) -> result 형태의 워크플로우 함수
        combined_input: 워크플로우에 전달할 입력 텍스트
        llm_service: LLM 서비스 인스턴스
        sb: Supabase 클라이언트 (옵션, 향후 텔레메트리용)
        user_email: 사용자 이메일 (옵션, 향후 텔레메트리용)

    Returns:
        workflow_fn 의 반환값
    """
    with st.spinner("📝 AI가 문서를 분석하고 수정하고 있습니다..."):
        return workflow_fn(combined_input, llm_service)
