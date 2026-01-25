# -*- coding: utf-8 -*-
import streamlit as st
import json
import re
from typing import Optional, Tuple, Dict, Any

def render_revision_sidebar_button():
    """
    사이드바에 '기안/공고문 수정' 버튼을 렌더링하고,
    클릭 시 세션 상태를 'revision' 모드로 변경합니다.
    """
    if st.sidebar.button("\U0001F4DD 기안, 공고문 수정", use_container_width=True):
        st.session_state["app_mode"] = "revision"
        st.session_state["workflow_result"] = None  # 기존 결과 초기화
        st.session_state["current_archive_id"] = None
        st.rerun()

def run_revision_workflow(user_input: str, llm_service) -> Dict[str, Any]:
    """
    사용자 입력(원문+지시사항)을 받아 수정된 공문과 변경 로그를 반환합니다.
    """
    if not user_input:
        return {}

    # 1. LLM에게 원문과 지시사항 분리 및 수정 요청
    prompt = f"""
당신은 대한민국 행정기관의 공문서 작성 및 교정 전문가인 'AI 행정관 Pro'이다.
사용자가 입력한 텍스트(원문+지시사항)를 분석하여 '2025년 개정 공문서 작성 표준'에 맞춰 문서를 수정하라.

[수정 기준]
[작성 원칙 (2025 개정 표준)]
1. **핵심 원칙**:
   - 사실성: 육하원칙 준수, 오자/탈자/계수 착오 금지.
   - 용이성: 쉬운 용어 사용, 짧고 명확한 문장.
   - 명확성: 불분명한 단어 회피, 정확한 조사 사용.
   - 비고압성: '금지', '엄금' 대신 '안내', '부탁' 등 긍정적 표현 사용.
   - 수요자 중심: 주어를 '국민(수요자)' 관점으로 (예: '교부합니다' -> '수령하실 수 있습니다').

2. **형식 및 표기 규칙 (엄격 준수)**:
   - **항목 기호 순서**: 1. -> 가. -> 1) -> 가) -> (1) -> (가) -> ① -> ㉮
   - **띄어쓰기**:
     - 첫째 항목 기호는 제목 첫 글자와 같은 위치.
     - 하위 항목은 상위 항목보다 2타(한글 1자) 오른쪽 시작.
     - 항목 기호와 내용 사이 1타 띄움.
   - **날짜/시간**:
     - 날짜: '2025. 1. 8.' (마지막 '일' 뒤에도 마침표).
     - 시간: '09:00', '13:20' (쌍점 앞뒤 붙임).
   - **금액**: '금13,500원(금일만삼천오백원)' (붙여쓰기).
   - **끝 표시**: 본문/붙임 끝에 2타 띄우고 '끝.'

3. **교정 가이드**:
   - 불명확한 주어 구체화 (예: '우리 기관' -> 공식 명칭).
   - 외래어 순화 (예: '스크린도어' -> '안전문').
   - 중복 표현 제거 ('2월달' -> '2월', '기간 동안' -> '기간에').
   - 문장 종결: 평서형 '-다' 원칙 (내부 결재는 '-함', '-것' 허용).

[사용자 입력]
{user_input}

[출력 형식 - 반드시 JSON 하나로만]
{{{{
  "revised_doc": {{{{
    "title": "수정된 제목",
    "receiver": "수정된 수신자",
    "body_paragraphs": ["수정된 본문 문단1", "수정된 본문 문단2", ...],
    "department_head": "발신 명의"
  }}}},
  "changelog": [
    "제목 변경: 'OOO' -> 'XXX'",
    "일시 수정: '10일' -> '2025. 1. 10.' (표기법 준수)",
    "오타 수정: '함니다' -> '합니다'",
    "표현 개선: '역전앞' -> '역전' (동어반복 삭제)"
  ],
  "summary": "수정 요약 멘트"
}}}}

- revised_doc 구조는 반드시 지켜라.
- changelog는 구체적으로 '무엇이 어떻게 바뀌었는지' 명시하라.
- JSON 이외의 텍스트는 출력하지 마라.
"""
    try:
        result = llm_service.generate_json(prompt, preferred_model="gemini-2.5-flash")
        if not result:
            return {"error": "AI 응답을 분석할 수 없습니다."}
        return result
    except Exception as e:
        return {"error": f"처리 중 오류 발생: {str(e)}"}
