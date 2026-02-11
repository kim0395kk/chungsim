# -*- coding: utf-8 -*-
"""
Govable AI - Streamlit 진입점

모든 의존성을 여기서 조립하고 UI를 렌더링합니다.
"""
import json
import time
import uuid
from html import escape as _escape
from typing import Optional

import streamlit as st

# =========================================================
# 패키지 임포트
# =========================================================
from govable_ai.config import (
    APP_VERSION,
    MAX_FOLLOWUP_Q,
    get_secret,
    get_vertex_config,
    get_supabase_config,
)
from govable_ai.helpers import (
    make_lawbot_url,
    mask_sensitive,
    strip_html,
    md_bold_to_html_safe,
)

# Core Services
from govable_ai.core.llm_service import LLMService
from govable_ai.core.law_api import LawOfficialService
from govable_ai.core.search_api import SearchService
from govable_ai.core.db_client import SupabaseClient
from govable_ai.core.doc_generator import generate_official_doc, generate_report_doc

# Skills (Agents)
from govable_ai.skills.analyzer import CaseAnalyzer
from govable_ai.skills.researcher import LegalResearcher
from govable_ai.skills.strategist import Strategist, ProcedurePlanner
from govable_ai.skills.drafter import DocumentDrafter

# UI Components
from govable_ai.ui.styles import apply_styles
from govable_ai.ui.components import render_header, render_lawbot_button, render_agent_logs
from govable_ai.ui.auth import sidebar_auth, render_history_list, is_admin_user
from govable_ai.ui.dashboard import render_master_dashboard
from govable_ai.ui.doc_compiler_page import render_doc_compiler_page


# =========================================================
# 서비스 초기화 (싱글톤 패턴)
# =========================================================
@st.cache_resource
def get_services():
    """서비스 인스턴스들을 초기화하고 캐싱"""
    # LLM Service
    llm = LLMService(
        vertex_config=get_vertex_config(),
        gemini_key=get_secret("general", "GEMINI_API_KEY"),
        groq_key=get_secret("general", "GROQ_API_KEY"),
    )
    
    # Law API Service
    law_api = LawOfficialService(
        api_id=get_secret("general", "LAW_API_ID")
    )
    
    # Search Service (with LLM for keyword extraction)
    search = SearchService(
        client_id=get_secret("general", "NAVER_CLIENT_ID"),
        client_secret=get_secret("general", "NAVER_CLIENT_SECRET"),
        llm_service=llm,
    )
    
    # DB Client
    db_config = get_supabase_config()
    db = SupabaseClient(
        url=db_config.get("url") if db_config else None,
        anon_key=db_config.get("anon_key") if db_config else None,
    )
    
    return {
        "llm": llm,
        "law_api": law_api,
        "search": search,
        "db": db,
    }


@st.cache_resource
def get_agents(_services: dict):
    """에이전트 인스턴스들을 초기화하고 캐싱"""
    llm = _services["llm"]
    law_api = _services["law_api"]
    
    return {
        "analyzer": CaseAnalyzer(llm),
        "researcher": LegalResearcher(llm, law_api),
        "strategist": Strategist(llm),
        "planner": ProcedurePlanner(llm),
        "drafter": DocumentDrafter(llm),
    }


# =========================================================
# 워크플로우 실행
# =========================================================
def run_workflow(user_input: str, log_placeholder, services: dict, agents: dict) -> dict:
    """
    전체 워크플로우 실행
    
    Args:
        user_input: 사용자 업무 지시
        log_placeholder: st.empty() 로그 플레이스홀더
        services: 서비스 인스턴스 딕셔너리
        agents: 에이전트 인스턴스 딕셔너리
    
    Returns:
        워크플로우 결과 딕셔너리
    """
    start_time = time.time()
    logs = []
    phase_start_time = time.time()
    
    def add_log(msg: str, style: str = "sys", status: str = "active"):
        nonlocal phase_start_time
        elapsed = time.time() - phase_start_time
        
        # 이전 로그를 완료 상태로 변경
        for log in logs:
            if log["status"] == "active":
                log["status"] = "done"
        
        logs.append({"msg": msg, "style": style, "status": status, "elapsed": elapsed})
        render_agent_logs(logs, log_placeholder)
        phase_start_time = time.time()
    
    llm = services["llm"]
    search = services["search"]
    
    analyzer = agents["analyzer"]
    researcher = agents["researcher"]
    strategist = agents["strategist"]
    planner = agents["planner"]
    drafter = agents["drafter"]
    
    # 1. 민원 분석
    add_log("🔍 민원/업무 케이스 분석 중...", style="legal")
    analysis = analyzer.analyze(user_input)
    add_log(f"✅ 케이스 유형: {analysis.get('case_type', '기타')}", style="legal", status="done")
    
    # 2. 법령 탐색
    add_log("📜 관련 법령 탐색 중...", style="legal")
    law_md = researcher.research(user_input, analysis)
    add_log("✅ 법령 근거 확보 완료", style="legal", status="done")
    
    # 3. 뉴스/사례 검색
    add_log("📰 관련 뉴스/사례 검색 중...", style="search")
    news_results = search.search_precedents(user_input, top_k=3)
    search_md = ""
    if news_results:
        for item in news_results:
            search_md += f"- [{item.get('title', '')}]({item.get('link', '')})\n"
            search_md += f"  {item.get('description', '')[:100]}...\n\n"
    else:
        search_md = "(관련 뉴스/사례 없음)"
    add_log("✅ 뉴스/사례 검색 완료", style="search", status="done")
    
    # 4. 전략 수립
    add_log("🧭 처리 전략 수립 중...", style="strat")
    strategy = strategist.plan_strategy(user_input, law_md, search_md)
    add_log("✅ 처리 전략 수립 완료", style="strat", status="done")
    
    # 5. 절차 계획
    add_log("🗺️ 절차 플랜 생성 중...", style="calc")
    procedure = planner.plan(user_input, law_md[:1000], analysis)
    add_log("✅ 절차 플랜 생성 완료", style="calc", status="done")
    
    # 6. 공문서 작성
    add_log("📝 공문서 초안 작성 중...", style="draft")
    meta = drafter.generate_meta()
    doc = drafter.draft(
        situation=user_input,
        legal_basis_md=law_md,
        meta=meta,
        strategy=strategy,
        procedure=procedure,
        objections=[],
    )
    add_log("✅ 공문서 초안 작성 완료", style="draft", status="done")
    
    # 7. Lawbot 키워드 추출
    lawbot_keywords = researcher.extract_law_keywords(user_input, analysis)
    query_text = (user_input[:60] + " " + " ".join(lawbot_keywords[:5])).strip()
    lawbot_url = make_lawbot_url(query_text[:180])
    
    # 완료 로그
    total_time = time.time() - start_time
    add_log(f"🎉 전체 워크플로우 완료 ({total_time:.1f}초)", style="sys", status="done")
    
    # 토큰 사용량 집계
    usage = llm.get_last_usage()
    
    return {
        "situation": user_input,
        "analysis": analysis,
        "law": law_md,
        "search": search_md,
        "strategy": strategy,
        "procedure": procedure,
        "meta": meta,
        "doc": doc,
        "lawbot_pack": {
            "core_keywords": lawbot_keywords,
            "query_text": query_text,
            "url": lawbot_url,
        },
        "execution_time": total_time,
        "token_usage": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        "model_used": usage.get("model_used", ""),
    }


# =========================================================
# 세션 관리
# =========================================================
def ensure_anon_session_id() -> str:
    """익명 세션 ID 보장"""
    if "anon_session_id" not in st.session_state:
        st.session_state.anon_session_id = str(uuid.uuid4())
    return st.session_state.anon_session_id


# =========================================================
# 메인 UI
# =========================================================
def main():
    """Streamlit 메인 엔트리포인트"""
    
    # 스타일 적용
    apply_styles()
    
    # 서비스 및 에이전트 초기화
    services = get_services()
    agents = get_agents(services)
    
    llm = services["llm"]
    db = services["db"]
    
    # 세션 관리
    anon_id = ensure_anon_session_id()
    
    if db.is_available():
        user = None
        user_id = None
        if st.session_state.get("logged_in"):
            user_id = st.session_state.get("user_id")
        db.touch_session(anon_id, user_id)
        
        if "boot_logged" not in st.session_state:
            st.session_state.boot_logged = True
            db.log_event("app_open", anon_id, meta={"ver": APP_VERSION})
    
    # 사이드바 인증
    sidebar_auth(db)
    
    # 페이지 네비게이션
    if "current_page" not in st.session_state:
        st.session_state.current_page = "workflow"
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📌 메뉴")
    
    col_nav1, col_nav2 = st.sidebar.columns(2)
    with col_nav1:
        if st.button("🧠 업무 처리", use_container_width=True, key="nav_workflow"):
            st.session_state.current_page = "workflow"
            st.rerun()
    with col_nav2:
        if st.button("📋 공문 컴파일", use_container_width=True, key="nav_compiler"):
            st.session_state.current_page = "compiler"
            st.rerun()
    
    # 현재 페이지 표시
    page_name = "업무 처리" if st.session_state.current_page == "workflow" else "공문서 컴파일러"
    st.sidebar.caption(f"📍 현재: {page_name}")
    
    render_history_list(db)
    
    # 공문서 컴파일러 페이지인 경우 별도 렌더링
    if st.session_state.current_page == "compiler":
        render_doc_compiler_page(llm)
        return  # 컴파일러 페이지만 표시하고 종료
    
    # 관리자 모드 체크
    is_admin_tab = (
        db.is_available()
        and st.session_state.get("logged_in")
        and is_admin_user(
            st.session_state.get("user_email", ""),
            st.session_state.get("is_admin_db", False)
        )
        and st.session_state.get("admin_mode", False)
    )
    
    if is_admin_tab:
        tabs = st.tabs(["🧠 업무 처리", "🏛️ 마스터 대시보드"])
        with tabs[1]:
            render_master_dashboard(db)
        with tabs[0]:
            pass  # 아래에서 렌더링
    
    # 헤더
    st.markdown(
        f"""
        <div style='text-align: center; padding: 2rem 0 3rem 0;'>
            <h1 style='font-size: 2.5rem; font-weight: 800; margin-bottom: 0.5rem; 
                       background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       background-clip: text;'>
                🏢 AI 행정관 Pro
            </h1>
            <p style='font-size: 1.1rem; color: #4b5563; font-weight: 500; margin-bottom: 0.75rem;'>
                충주시청 스마트 행정 솔루션
            </p>
            <p style='font-size: 0.9rem; color: #6b7280;'>
                문의 <a href='mailto:kim0395kk@korea.kr' style='color: #2563eb; text-decoration: none;'>kim0395kk@korea.kr</a> | Govable AI 에이전트
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # 시스템 상태 표시
    ai_ok = "✅ AI" if llm.is_available() else "❌ AI"
    law_ok = "✅ LAW" if services["law_api"].is_available() else "❌ LAW"
    nv_ok = "✅ NEWS" if services["search"].is_available() else "❌ NEWS"
    db_ok = "✅ DB" if db.is_available() else "❌ DB"
    
    st.markdown(
        f"""
        <div style='text-align: center; padding: 0.75rem 1.5rem; background: white; 
                    border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    border-left: 4px solid #2563eb;'>
            <span style='font-size: 0.9rem; color: #374151; font-weight: 600;'>
                시스템 상태: {ai_ok} · {law_ok} · {nv_ok} · {db_ok}
            </span>
            <span style='font-size: 0.85rem; color: #9ca3af; margin-left: 1rem;'>
                v{APP_VERSION}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # 2단 레이아웃
    col_left, col_right = st.columns([1, 1.15], gap="large")
    
    with col_right:
        right_panel_placeholder = st.empty()
        
        if "workflow_result" not in st.session_state:
            with right_panel_placeholder.container():
                st.markdown(
                    """
                    <div style='text-align: center; padding: 6rem 2rem; 
                                background: white; border-radius: 16px; 
                                border: 2px dashed #d1d5db; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                        <div style='font-size: 4rem; margin-bottom: 1rem; opacity: 0.5;'>📄</div>
                        <h3 style='color: #6b7280; font-size: 1.5rem; font-weight: 700; margin-bottom: 0.75rem;'>
                            Document Preview
                        </h3>
                        <p style='color: #9ca3af; font-size: 1rem; line-height: 1.6;'>
                            왼쪽에서 업무를 지시하면<br>완성된 공문서가 여기에 나타납니다.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    
    with col_left:
        render_header("🗣️ 업무 지시")
        
        user_input = st.text_area(
            "업무 내용",
            height=190,
            placeholder="예시\n- 상황: (무슨 일 / 어디 / 언제 / 증거 유무...)\n- 쟁점: (요건/절차/근거...)\n- 요청: (원하는 결과물: 회신/사전통지/처분 등)",
            label_visibility="collapsed",
        )
        
        st.markdown(
            """
            <div style='background: #fef3c7; border-left: 4px solid #f59e0b; 
                        padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
                <p style='margin: 0; color: #92400e; font-size: 0.9rem; font-weight: 500;'>
                    ⚠️ 민감정보(성명·연락처·주소·차량번호 등) 입력 금지
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        if st.button("⚡ 스마트 분석 시작", type="primary", use_container_width=True):
            if not user_input:
                st.warning("내용을 입력해주세요.")
            else:
                res = run_workflow(user_input, right_panel_placeholder, services, agents)
                res["app_mode"] = st.session_state.get("app_mode", "신속")
                
                archive_id = None
                if db.is_available():
                    archive_id = db.insert_archive(
                        prompt=user_input,
                        payload=res,
                        anon_session_id=anon_id,
                        user_id=st.session_state.get("user_id"),
                        user_email=st.session_state.get("user_email"),
                    )
                    if archive_id:
                        st.session_state.current_archive_id = archive_id
                        db.log_event(
                            "workflow_run",
                            anon_id,
                            archive_id=archive_id,
                            meta={"prompt_len": len(user_input)},
                        )
                
                res["archive_id"] = archive_id
                st.session_state.workflow_result = res
                st.session_state.followup_messages = []
                st.rerun()
        
        # 결과가 있으면 분석 결과 표시
        if "workflow_result" in st.session_state:
            res = st.session_state.workflow_result
            pack = res.get("lawbot_pack") or {}
            if pack.get("url"):
                render_lawbot_button(pack["url"])
            
            render_header("🧠 케이스 분석")
            
            a = res.get("analysis", {})
            st.markdown(
                f"""
                <div style='background: #eff6ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #2563eb; margin-bottom: 1rem;'>
                    <p style='margin: 0 0 0.5rem 0; color: #1e40af; font-weight: 600;'>유형: {a.get('case_type','')}</p>
                    <p style='margin: 0; color: #1e40af;'>쟁점: {", ".join(a.get("core_issue", []))}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            with st.expander("📋 누락정보/증빙/리스크/다음행동 보기", expanded=False):
                st.markdown("**추가 확인 질문**")
                for x in a.get("required_facts", []):
                    st.write("- ", x)
                st.markdown("**필요 증빙**")
                for x in a.get("required_evidence", []):
                    st.write("- ", x)
                st.markdown("**절차 리스크**")
                for x in a.get("risk_flags", []):
                    st.write("- ", x)
                st.markdown("**권장 다음 행동**")
                for x in a.get("recommended_next_action", []):
                    st.write("- ", x)
            
            # 법령 근거 + 뉴스/사례
            law_col, news_col = st.columns(2, gap="medium")
            
            with law_col:
                render_header("📜 핵심 법령 근거")
                with st.container(height=400):
                    st.markdown(res.get("law", ""))
            
            with news_col:
                render_header("📰 뉴스/사례")
                with st.container(height=400):
                    st.markdown(res.get("search", ""))
            
            render_header("🧭 처리 가이드")
            st.markdown(res.get("strategy", ""))
            
            render_header("🗺️ 절차 플랜")
            proc = res.get("procedure", {})
            with st.expander("타임라인", expanded=True):
                for step in proc.get("timeline", []):
                    st.markdown(f"**{step.get('step')}. {step.get('name')}** — {step.get('goal')}")
                    for x in step.get("actions", []):
                        st.write("- 행동:", x)
                    for x in step.get("records", []):
                        st.write("- 기록:", x)
                    if step.get("legal_note"):
                        st.caption(f"법/유의: {step['legal_note']}")
                    st.write("")
            
            with st.expander("체크리스트/서식", expanded=False):
                st.markdown("**체크리스트**")
                for x in proc.get("checklist", []):
                    st.write("- ", x)
                st.markdown("**필요 서식/문서**")
                for x in proc.get("templates", []):
                    st.write("- ", x)
    
    # 오른쪽 패널 결과 렌더링
    if "workflow_result" in st.session_state:
        with right_panel_placeholder.container():
            res = st.session_state.workflow_result
            doc = res.get("doc")
            meta = res.get("meta") or {}
            archive_id = res.get("archive_id") or st.session_state.get("current_archive_id")
            
            render_header("📄 공문서")
            
            if not doc:
                st.warning("공문 생성 결과(doc)가 비어 있습니다.")
            else:
                html = f"""
<div class="paper-sheet">
  <div class="stamp">직인생략</div>
  <div class="doc-header">{_escape(doc.get('title', '공 문 서'))}</div>
  <div class="doc-info">
    <span>문서번호: {_escape(meta.get('doc_num',''))}</span>
    <span>시행일자: {_escape(meta.get('today_str',''))}</span>
    <span>수신: {_escape(doc.get('receiver', '수신자 참조'))}</span>
  </div>
  <hr style="border: 1px solid black; margin-bottom: 30px;">
  <div class="doc-body">
"""
                paragraphs = doc.get("body_paragraphs", [])
                if isinstance(paragraphs, str):
                    paragraphs = [paragraphs]
                for p in paragraphs:
                    html += f"<p style='margin-bottom: 14px;'>{md_bold_to_html_safe(p)}</p>"
                html += f"""
  </div>
  <div class="doc-footer">{_escape(doc.get('department_head', '행정기관장'))}</div>
</div>
"""
                st.markdown(html, unsafe_allow_html=True)
            
            # =========================================================
            # 💾 HWPX 다운로드 센터
            # =========================================================
            st.markdown("---")
            render_header("💾 문서 다운로드 센터")
            
            dl_col1, dl_col2 = st.columns(2, gap="medium")
            
            with dl_col1:
                st.markdown(
                    """
                    <div style='background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); 
                                padding: 1rem; border-radius: 12px; border: 2px solid #3b82f6;
                                text-align: center; margin-bottom: 0.5rem;'>
                        <p style='margin: 0; color: #1e40af; font-weight: 700; font-size: 1rem;'>
                            📤 대외 발송용
                        </p>
                        <p style='margin: 0.25rem 0 0 0; color: #3b82f6; font-size: 0.85rem;'>
                            대민/대기관 발송 공문
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("📄 공문서(.hwpx) 생성", use_container_width=True, key="gen_official"):
                    with st.spinner("공문서 렌더링 중..."):
                        try:
                            file_path = generate_official_doc(doc, meta)
                            with open(file_path, "rb") as f:
                                file_data = f.read()
                            st.download_button(
                                label="📥 공문서 다운로드",
                                data=file_data,
                                file_name=f"공문_{meta.get('doc_num', 'doc')}.hwpx".replace("/", "-").replace(":", "-"),
                                mime="application/hwp+zip",
                                use_container_width=True,
                            )
                        except Exception as e:
                            st.error(f"생성 실패: {e}")
            
            with dl_col2:
                st.markdown(
                    """
                    <div style='background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); 
                                padding: 1rem; border-radius: 12px; border: 2px solid #10b981;
                                text-align: center; margin-bottom: 0.5rem;'>
                        <p style='margin: 0; color: #047857; font-weight: 700; font-size: 1rem;'>
                            📑 내부 결재용
                        </p>
                        <p style='margin: 0.25rem 0 0 0; color: #10b981; font-size: 0.85rem;'>
                            결과보고서/계획서
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("📊 보고서(.hwpx) 생성", use_container_width=True, key="gen_report"):
                    with st.spinner("보고서 렌더링 중..."):
                        try:
                            file_path = generate_report_doc(
                                analysis_data=res.get("analysis", {}),
                                procedure_data=res.get("procedure", {}),
                                strategy_text=res.get("strategy", ""),
                                legal_text=res.get("law", ""),
                            )
                            with open(file_path, "rb") as f:
                                file_data = f.read()
                            case_type = res.get("analysis", {}).get("case_type", "민원")
                            st.download_button(
                                label="📥 보고서 다운로드",
                                data=file_data,
                                file_name=f"보고서_{case_type}.hwpx".replace("/", "-"),
                                mime="application/hwp+zip",
                                use_container_width=True,
                            )
                        except Exception as e:
                            st.error(f"생성 실패: {e}")
            
            st.caption("💡 HWPX 템플릿이 없으면 텍스트 파일(.txt)로 생성됩니다. 템플릿 추가: `govable_ai/assets/templates/`")
            
            render_header("💬 후속 질문")
            
            if archive_id:
                st.success("✅ 업무 지시 내용이 DB에 안전하게 저장되었습니다.")
            else:
                st.info("저장된 archive_id가 없습니다. (DB 저장 실패 가능)")
            
            if "followup_messages" not in st.session_state:
                st.session_state.followup_messages = res.get("followups", []) or []
            
            used = len([m for m in st.session_state.followup_messages if m.get("role") == "user"])
            remain = max(0, MAX_FOLLOWUP_Q - used)
            
            pack = res.get("lawbot_pack") or {}
            if pack.get("url"):
                render_lawbot_button(pack["url"])
            
            for m in st.session_state.followup_messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
            
            if remain == 0:
                st.markdown(
                    """
                    <div style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
                                padding: 1rem; border-radius: 12px; border-left: 4px solid #ef4444;
                                text-align: center; margin: 1.5rem 0;'>
                        <p style='margin: 0; color: #991b1b; font-weight: 600; font-size: 1rem;'>
                            ⚠️ 후속 질문 한도(5회)를 모두 사용했습니다.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style='background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); 
                                padding: 1.25rem; border-radius: 12px; 
                                border: 2px solid #3b82f6;
                                margin: 1.5rem 0 1rem 0;'>
                        <div style='display: flex; align-items: center; gap: 1rem;'>
                            <div style='font-size: 2.5rem; line-height: 1;'>💬</div>
                            <div style='flex: 1;'>
                                <p style='margin: 0 0 0.5rem 0; color: #1e40af; font-weight: 700; font-size: 1.1rem;'>
                                    👇 아래 입력창에 후속 질문을 입력하세요 (남은 횟수: {remain}회)
                                </p>
                                <p style='margin: 0; color: #3b82f6; font-size: 0.9rem;'>
                                    분석 결과에 대해 추가로 궁금한 점을 물어보세요
                                </p>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            q = st.chat_input("💭 후속 질문을 입력하세요... (Enter로 전송)")
            if q and remain > 0:
                turn = used + 1
                st.session_state.followup_messages.append({"role": "user", "content": q})
                
                if db.is_available() and archive_id:
                    db.insert_followup(
                        archive_id, turn * 2 - 1, "user", q,
                        anon_id, st.session_state.get("user_id"), st.session_state.get("user_email")
                    )
                    db.log_event("followup_user", anon_id, archive_id=archive_id, meta={"turn": turn})
                
                with st.chat_message("user"):
                    st.markdown(q)
                
                case_context = f"""
[케이스]
상황: {res.get('situation','')}

케이스 분석:
{json.dumps(res.get("analysis", {}), ensure_ascii=False)}

법령(요약):
{strip_html(res.get('law',''))[:2500]}

절차 플랜:
{json.dumps(res.get("procedure", {}), ensure_ascii=False)[:2000]}

처리방향:
{res.get('strategy','')[:2200]}
"""
                prompt = f"""
너는 '케이스 고정 행정 후속 Q&A'이다.
{case_context}

[사용자 질문]
{q}

[규칙]
- 위 컨텍스트 범위에서만 답한다.
- 절차/증빙/기록 포인트를 우선 제시한다.
- 모르면 모른다고 말하고, 추가 법령 근거는 Lawbot으로 찾게 안내한다.
- 서론 없이 실무형으로.
"""
                with st.chat_message("assistant"):
                    with st.spinner("후속 답변 생성 중..."):
                        ans = llm.generate_text(prompt)
                        st.markdown(ans)
                
                st.session_state.followup_messages.append({"role": "assistant", "content": ans})
                
                if db.is_available() and archive_id:
                    db.insert_followup(
                        archive_id, turn * 2, "assistant", ans,
                        anon_id, st.session_state.get("user_id"), st.session_state.get("user_email")
                    )
                    db.log_event("followup_assistant", anon_id, archive_id=archive_id, meta={"turn": turn})
                
                st.rerun()


if __name__ == "__main__":
    main()
