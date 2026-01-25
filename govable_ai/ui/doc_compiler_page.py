# -*- coding: utf-8 -*-
"""
Govable AI - 공문서 컴파일러 페이지

이 모듈에서만 streamlit import 허용
"""
import difflib
from html import escape
from typing import Optional, Tuple, List

import streamlit as st

from govable_ai.ui.components import render_header
from govable_ai.skills.doc_compiler import DocumentCompiler


def _generate_diff_html(original: str, compiled: str) -> Tuple[str, str]:
    """
    원문과 수정본의 Diff를 생성하여 HTML로 반환
    
    Args:
        original: 원문
        compiled: 수정본
        
    Returns:
        (원문 HTML with highlights, 수정본 HTML with highlights)
    """
    # 줄 단위로 분리
    orig_lines = original.splitlines()
    comp_lines = compiled.splitlines()
    
    # SequenceMatcher로 비교
    matcher = difflib.SequenceMatcher(None, orig_lines, comp_lines)
    
    orig_html_parts = []
    comp_html_parts = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # 동일한 부분
            for line in orig_lines[i1:i2]:
                orig_html_parts.append(f'<div style="padding: 2px 8px; line-height: 1.6;">{escape(line)}</div>')
            for line in comp_lines[j1:j2]:
                comp_html_parts.append(f'<div style="padding: 2px 8px; line-height: 1.6;">{escape(line)}</div>')
        elif tag == 'delete':
            # 삭제된 부분 (원문에만 있음)
            for line in orig_lines[i1:i2]:
                orig_html_parts.append(f'<div style="background: #fee2e2; padding: 2px 8px; line-height: 1.6; border-left: 3px solid #ef4444;"><del>{escape(line)}</del></div>')
        elif tag == 'insert':
            # 추가된 부분 (수정본에만 있음)
            for line in comp_lines[j1:j2]:
                comp_html_parts.append(f'<div style="background: #dcfce7; padding: 2px 8px; line-height: 1.6; border-left: 3px solid #22c55e;"><ins style="text-decoration: none; font-weight: 600;">{escape(line)}</ins></div>')
        elif tag == 'replace':
            # 변경된 부분
            for line in orig_lines[i1:i2]:
                orig_html_parts.append(f'<div style="background: #fef3c7; padding: 2px 8px; line-height: 1.6; border-left: 3px solid #f59e0b;"><span style="text-decoration: line-through; opacity: 0.7;">{escape(line)}</span></div>')
            for line in comp_lines[j1:j2]:
                comp_html_parts.append(f'<div style="background: #dbeafe; padding: 2px 8px; line-height: 1.6; border-left: 3px solid #3b82f6;"><span style="font-weight: 600;">{escape(line)}</span></div>')
    
    return '\n'.join(orig_html_parts), '\n'.join(comp_html_parts)


def render_doc_compiler_page(llm_service) -> None:
    """
    공문서 컴파일러 페이지 렌더링
    
    Args:
        llm_service: LLM 서비스 인스턴스
    """
    # 페이지 헤더
    st.markdown(
        """
        <div style='text-align: center; padding: 2rem 0 2rem 0;'>
            <h1 style='font-size: 2.2rem; font-weight: 800; margin-bottom: 0.5rem; 
                       background: linear-gradient(135deg, #059669 0%, #10b981 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       background-clip: text;'>
                📋 공문서 컴파일러
            </h1>
            <p style='font-size: 1rem; color: #4b5563; font-weight: 500;'>
                거친 초안을 합격 문서로 정제합니다
            </p>
            <p style='font-size: 0.85rem; color: #9ca3af; margin-top: 0.5rem;'>
                행정업무운영 편람 · 공문서 작성 지침 기반 | 📊 Diff View 비교 지원
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # 컴파일러 인스턴스 생성
    compiler = DocumentCompiler(llm_service)
    
    # 2단 레이아웃
    col_left, col_right = st.columns([1, 1.15], gap="large")
    
    with col_left:
        render_header("📝 초안 입력")
        
        draft_input = st.text_area(
            "초안 내용",
            height=250,
            placeholder="""예시:
주민센터에서 내일 오후 2시에 주민 설명회를 개최하려고 합니다.
도로공사 관련 안내를 해드릴 예정입니다.
참석하시는 분들께 간단한 다과를 제공해 드리겠습니다.
많은 참석 부탁드립니다.""",
            label_visibility="collapsed",
        )
        
        # 안내 박스
        st.markdown(
            """
            <div style='background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); 
                        border-left: 4px solid #059669; 
                        padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
                <p style='margin: 0 0 0.5rem 0; color: #047857; font-weight: 700; font-size: 0.95rem;'>
                    ✨ 컴파일러가 자동으로 변환하는 항목
                </p>
                <p style='margin: 0; color: #065f46; font-size: 0.85rem; line-height: 1.6;'>
                    • 문체: "~합니다" → "~함", "~바랍니다" → "~바람"<br>
                    • 날짜: 2026. 1. 21. 형식<br>
                    • 시간: 14:00 형식<br>
                    • 항목 체계: 1. → 가. → 1) → 가) → (1) → (가)<br>
                    • 본문 끝: '끝.' 표기
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        if st.button("⚡ 컴파일 실행", type="primary", use_container_width=True):
            if not draft_input.strip():
                st.warning("초안 내용을 입력해주세요.")
            else:
                with st.spinner("공문서 컴파일 중..."):
                    result = compiler.compile(draft_input)
                    st.session_state.compiler_result = result
                st.rerun()
        
        # 결과가 있으면 교정 내역 표시
        if "compiler_result" in st.session_state:
            result = st.session_state.compiler_result
            corrections = result.get("corrections", {})
            
            render_header("🛠️ 규범 교정 내역")
            
            # 문체 교정
            style_items = corrections.get("style", [])
            if style_items:
                st.markdown(
                    f"""
                    <div style='background: #eff6ff; padding: 0.8rem 1rem; border-radius: 8px; 
                                margin-bottom: 0.5rem; border-left: 4px solid #3b82f6;'>
                        <p style='margin: 0 0 0.4rem 0; color: #1e40af; font-weight: 700; font-size: 0.9rem;'>
                            📝 문체 교정
                        </p>
                        <ul style='margin: 0; padding-left: 1.2rem; color: #1e40af; font-size: 0.85rem;'>
                            {"".join(f"<li>{item}</li>" for item in style_items)}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            # 형식 교정
            format_items = corrections.get("format", [])
            if format_items:
                st.markdown(
                    f"""
                    <div style='background: #fef3c7; padding: 0.8rem 1rem; border-radius: 8px; 
                                margin-bottom: 0.5rem; border-left: 4px solid #f59e0b;'>
                        <p style='margin: 0 0 0.4rem 0; color: #92400e; font-weight: 700; font-size: 0.9rem;'>
                            📐 형식 교정
                        </p>
                        <ul style='margin: 0; padding-left: 1.2rem; color: #92400e; font-size: 0.85rem;'>
                            {"".join(f"<li>{item}</li>" for item in format_items)}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            # 보안 교정
            security_items = corrections.get("security", [])
            if security_items:
                st.markdown(
                    f"""
                    <div style='background: #fee2e2; padding: 0.8rem 1rem; border-radius: 8px; 
                                margin-bottom: 0.5rem; border-left: 4px solid #ef4444;'>
                        <p style='margin: 0 0 0.4rem 0; color: #991b1b; font-weight: 700; font-size: 0.9rem;'>
                            🔒 보안 교정
                        </p>
                        <ul style='margin: 0; padding-left: 1.2rem; color: #991b1b; font-size: 0.85rem;'>
                            {"".join(f"<li>{item}</li>" for item in security_items)}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            # 교정 내역이 없는 경우
            if not style_items and not format_items and not security_items:
                st.info("교정 내역이 없습니다.")
    
    with col_right:
        if "compiler_result" not in st.session_state:
            # 대기 상태 UI
            st.markdown(
                """
                <div style='text-align: center; padding: 6rem 2rem; 
                            background: white; border-radius: 16px; 
                            border: 2px dashed #d1d5db; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                    <div style='font-size: 4rem; margin-bottom: 1rem; opacity: 0.5;'>📋</div>
                    <h3 style='color: #6b7280; font-size: 1.5rem; font-weight: 700; margin-bottom: 0.75rem;'>
                        컴파일 결과
                    </h3>
                    <p style='color: #9ca3af; font-size: 1rem; line-height: 1.6;'>
                        왼쪽에서 초안을 입력하고<br>'컴파일 실행'을 누르면<br>완성된 기안문이 여기에 나타납니다.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            result = st.session_state.compiler_result
            compiled_doc = result.get("compiled_doc", "")
            original_draft = result.get("original_draft", "")
            structure = result.get("structure", {})
            
            # 탭으로 결과 표시
            tab_result, tab_diff = st.tabs(["📋 컴파일 결과", "📊 비교 (Diff View)"])
            
            with tab_result:
                render_header("📋 컴파일된 기안문")
                
                # 공문서 스타일로 렌더링
                st.markdown(
                    f"""
                    <div class="paper-sheet" style="min-height: auto; padding: 24px;">
                        <div class="doc-header" style="font-size: 18pt; margin-bottom: 1.5rem; padding-bottom: 1rem;">
                            {structure.get("title", "기안문")}
                        </div>
                        <div class="doc-body" style="font-size: 11pt; white-space: pre-wrap; line-height: 1.8;">
{compiled_doc}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                # 버튼들
                st.markdown("---")
                
                col_btn1, col_btn2 = st.columns(2, gap="medium")
                
                with col_btn1:
                    st.download_button(
                        label="📥 텍스트로 다운로드",
                        data=compiled_doc,
                        file_name="기안문.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                
                with col_btn2:
                    if st.button("🔄 새 초안 작성", use_container_width=True):
                        if "compiler_result" in st.session_state:
                            del st.session_state.compiler_result
                        st.rerun()
                
                # 구조 분석 (접기)
                with st.expander("📊 문서 구조 분석", expanded=False):
                    st.markdown("**제목**")
                    st.write(structure.get("title", "-"))
                    st.markdown("**개요**")
                    st.write(structure.get("overview", "-"))
                    st.markdown("**근거**")
                    st.write(structure.get("basis", "-"))
                    st.markdown("**세부 내용**")
                    st.write(structure.get("details", "-"))
                    st.markdown("**행정 사항**")
                    st.write(structure.get("admin_notes", "-"))
            
            with tab_diff:
                render_header("📊 원문 vs 수정본 비교")
                
                # 범례
                st.markdown(
                    """
                    <div style='display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; padding: 0.75rem; background: #f9fafb; border-radius: 8px;'>
                        <span style='display: flex; align-items: center; gap: 0.3rem; font-size: 0.8rem;'>
                            <span style='display: inline-block; width: 12px; height: 12px; background: #fee2e2; border-left: 3px solid #ef4444;'></span> 삭제
                        </span>
                        <span style='display: flex; align-items: center; gap: 0.3rem; font-size: 0.8rem;'>
                            <span style='display: inline-block; width: 12px; height: 12px; background: #dcfce7; border-left: 3px solid #22c55e;'></span> 추가
                        </span>
                        <span style='display: flex; align-items: center; gap: 0.3rem; font-size: 0.8rem;'>
                            <span style='display: inline-block; width: 12px; height: 12px; background: #fef3c7; border-left: 3px solid #f59e0b;'></span> 원문(변경됨)
                        </span>
                        <span style='display: flex; align-items: center; gap: 0.3rem; font-size: 0.8rem;'>
                            <span style='display: inline-block; width: 12px; height: 12px; background: #dbeafe; border-left: 3px solid #3b82f6;'></span> 수정본(변경됨)
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                # Diff 생성
                orig_html, comp_html = _generate_diff_html(original_draft, compiled_doc)
                
                # 2열 비교
                diff_col1, diff_col2 = st.columns(2, gap="small")
                
                with diff_col1:
                    st.markdown(
                        f"""
                        <div style='background: #fef2f2; padding: 0.5rem 0.75rem; border-radius: 8px 8px 0 0; border-bottom: 2px solid #ef4444;'>
                            <strong style='color: #991b1b; font-size: 0.9rem;'>📄 원문 (입력된 초안)</strong>
                        </div>
                        <div style='background: white; padding: 12px; border: 1px solid #e5e7eb; border-top: none; border-radius: 0 0 8px 8px; max-height: 500px; overflow-y: auto; font-family: monospace; font-size: 0.85rem;'>
                            {orig_html if orig_html else '<em style="color: #9ca3af;">(원문 없음)</em>'}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                
                with diff_col2:
                    st.markdown(
                        f"""
                        <div style='background: #ecfdf5; padding: 0.5rem 0.75rem; border-radius: 8px 8px 0 0; border-bottom: 2px solid #22c55e;'>
                            <strong style='color: #166534; font-size: 0.9rem;'>✅ 수정본 (컴파일 결과)</strong>
                        </div>
                        <div style='background: white; padding: 12px; border: 1px solid #e5e7eb; border-top: none; border-radius: 0 0 8px 8px; max-height: 500px; overflow-y: auto; font-family: monospace; font-size: 0.85rem;'>
                            {comp_html if comp_html else '<em style="color: #9ca3af;">(수정본 없음)</em>'}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                
                # 변경 통계
                st.markdown("---")
                orig_lines = len(original_draft.splitlines()) if original_draft else 0
                comp_lines = len(compiled_doc.splitlines()) if compiled_doc else 0
                
                st.markdown(
                    f"""
                    <div style='display: flex; gap: 2rem; justify-content: center; padding: 1rem; background: #f9fafb; border-radius: 8px;'>
                        <div style='text-align: center;'>
                            <div style='font-size: 1.5rem; font-weight: 700; color: #6b7280;'>{orig_lines}</div>
                            <div style='font-size: 0.8rem; color: #9ca3af;'>원문 줄 수</div>
                        </div>
                        <div style='font-size: 1.5rem; color: #d1d5db;'>→</div>
                        <div style='text-align: center;'>
                            <div style='font-size: 1.5rem; font-weight: 700; color: #059669;'>{comp_lines}</div>
                            <div style='font-size: 0.8rem; color: #9ca3af;'>수정본 줄 수</div>
                        </div>
                        <div style='text-align: center;'>
                            <div style='font-size: 1.5rem; font-weight: 700; color: {"#22c55e" if comp_lines >= orig_lines else "#ef4444"};'>
                                {("+" if comp_lines >= orig_lines else "")}{comp_lines - orig_lines}
                            </div>
                            <div style='font-size: 0.8rem; color: #9ca3af;'>변경량</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
