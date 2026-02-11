import streamlit as st
from datetime import datetime

# =========================================================
# DUTY MANUAL BOT (Refactored)
# =========================================================

def call_llm(llm_service, prompt: str) -> str:
    """기존 LLMService를 래핑하여 안전하게 호출"""
    try:
        return llm_service.generate_text(prompt)
    except Exception as e:
        return f"오류 발생: {str(e)}"

# 동의어 매핑 (구어체 → 행정용어)
SYNONYMS = {
    "노숙자": "노숙인",
    "행려자": "노숙인",
    "거지": "노숙인",
    "고라니": "야생동물",
    "사슴": "야생동물",
    "멧돼지": "야생동물",
    "로드킬": "동물사체",
    "죽은동물": "동물사체",
    "악취": "냄새",
    "냄새나": "악취",
    "싱크홀": "도로침하",
    "구멍": "파손",
}

def normalize_query(q: str) -> str:
    """동의어 치환으로 검색어 정규화"""
    result = q
    for src, dst in SYNONYMS.items():
        result = result.replace(src, dst)
    return result

def llm_extract_keywords(llm_service, query: str) -> list:
    """LLM을 사용하여 검색 키워드 추출"""
    if not llm_service:
        return []
    
    prompt = f"""너는 검색 키워드 추출 전문가다.
사용자 질문에서 **당직 매뉴얼 검색에 적합한 핵심 키워드**를 추출하라.

[규칙]
1. 구어체를 행정 용어로 변환 (예: 노숙자→노숙인, 고라니→야생동물)
2. 키워드는 2~4개, 쉼표로 구분
3. 조사/불용어 제거 (가, 이, 를, 은, 는 등)
4. 키워드만 출력, 설명 금지

[예시]
질문: "노숙자가 찾아왔는데 어떻게 해요?"
키워드: 노숙인, 귀향여비, 숙박조치

질문: "고라니가 죽어있어요"
키워드: 야생동물, 사체, 로드킬

질문: "하수구에서 냄새나요"
키워드: 하수도, 악취, 역류

질문: "{query}"
키워드:"""
    
    try:
        result = llm_service.generate_text(prompt).strip()
        # 쉼표로 분리하고 정리
        keywords = [kw.strip() for kw in result.split(",") if kw.strip()]
        return keywords[:5]  # 최대 5개
    except Exception:
        return []

def retrieve_duty_context(sb, query: str, llm_service=None) -> list:
    """v5 검색: LLM 키워드 추출 + 다단계 검색"""
    if not sb:
        return []
    
    # 1. LLM으로 키워드 추출
    keywords = []
    if llm_service:
        keywords = llm_extract_keywords(llm_service, query)
    
    # 2. 각 키워드로 검색
    for kw in keywords:
        try:
            r = sb.table("duty_manual_kb").select("*").ilike("content", f"%{kw}%").limit(8).execute()
            data = getattr(r, "data", [])
            if data:
                return data
        except Exception:
            pass
        
        try:
            r = sb.table("duty_manual_kb").select("*").ilike("section_path", f"%{kw}%").limit(8).execute()
            data = getattr(r, "data", [])
            if data:
                return data
        except Exception:
            pass
    
    # 3. 원문으로 직접 검색 (폴백)
    simple_terms = query.replace("?", "").replace("!", "").split()
    for term in simple_terms:
        if len(term) >= 2:
            try:
                r = sb.table("duty_manual_kb").select("*").ilike("content", f"%{term}%").limit(8).execute()
                data = getattr(r, "data", [])
                if data:
                    return data
            except Exception:
                pass
    
    return []

def _render_duty_chat_ui(sb, llm_service):
    """실제 채팅 UI 구현 (Dialog/Fallback 공용)"""
    st.caption("충주시청 당직 근무 매뉴얼 기반 Q&A")
    
    # 세션 상태 초기화
    if "duty_messages" not in st.session_state:
        st.session_state.duty_messages = [
            {"role": "assistant", "content": "당직 근무 중 궁금한 점을 물어보세요.\n(예: '하수도 역류', '로드킬', '불법주정차')"}
        ]
    


def _render_duty_chat_ui(sb, llm_service):
    """실제 채팅 UI 구현 (v2: Hybrid Search + Single Pass Generation)"""
    st.caption("충주시청 당직 근무 매뉴얼 기반 Q&A (v2 Hybrid)")
    
    # 세션 상태 초기화
    if "duty_messages" not in st.session_state:
        st.session_state.duty_messages = [
            {"role": "assistant", "content": "당직 근무 중 궁금한 점을 물어보세요.\n(예: '하수도 역류', '로드킬', '불법주정차')"}
        ]

    # 채팅 기록 표시
    for msg in st.session_state.duty_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 1분 보고서 생성 버튼
    if len(st.session_state.duty_messages) > 1:
        if st.button("📝 1분 보고서 생성", key="btn_duty_report", use_container_width=True):
            with st.spinner("보고서 작성 중..."):
                prompt = build_1min_report_prompt(st.session_state.duty_messages)
                report = call_llm(llm_service, prompt)
                st.session_state.duty_messages.append({"role": "assistant", "content": report})
                st.rerun()

    # 입력창
    if query := st.chat_input("질문을 입력하세요..."):
        # 사용자 메시지 추가
        st.session_state.duty_messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # 답변 생성
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            status_placeholder.caption("🔍 매뉴얼 검색 중... (Hybrid)")
            
            # 1. 검색 (LLM 최적화 없이 바로 검색)
            candidates = retrieve_duty_context(sb, query, llm_service)
            
            # [DEBUG] 후보군 확인
            with st.expander("🔍 [DEBUG] 검색된 후보군 (Top 5)", expanded=False):
                if not candidates:
                    st.write("검색 결과 없음")
                for idx, c in enumerate(candidates):
                    st.write(f"**{idx+1}. {c.get('section_path')}** (Score: {c.get('score', 0):.4f})")
                    st.caption(c.get('content')[:100])

            status_placeholder.caption("✍️ 답변 작성 중...")
            
            if not candidates:
                fail_msg = "죄송합니다. 관련 내용을 매뉴얼에서 찾지 못했습니다."
                st.markdown(fail_msg)
                answer = fail_msg
            else:
                # 2. Top 3 컨텍스트 구성
                top_candidates = candidates[:3]
                context_str = ""
                for idx, item in enumerate(top_candidates):
                    context_str += f"""
[후보 {idx+1}]
- 위치: {item.get('section_path')}
- 부서: {item.get('dept')} (☎ {item.get('team_contact')})
- 내용: {item.get('content')}
"""

                # 3. 답변 생성 (Single Pass)
                sys_prompt = f"""
너는 충주시청 당직 근무자 도우미다.
사용자 질문에 대해 아래 [매뉴얼 후보]를 참고하여 답변하라.

[매뉴얼 후보]
{context_str}

[답변 규칙]
1. 후보들 중 **사용자 질문과 가장 상황이 일치하는 하나**를 골라 답변하라.
2. 만약 질문이 모호하여(예: '소음'인데 공사장인지 생활소음인지 불분명) 하나를 특정할 수 없다면, **사용자에게 상황을 되물어라.** (예: "공사장 소음인가요, 아니면 생활 소음인가요?")
3. 답변 시 **담당 부서와 연락처**를 가장 먼저 명시하라.
4. 매뉴얼에 없는 내용은 지어내지 말고 "내용 없음"이라고 하라.
"""
                rag_prompt = f"{sys_prompt}\n\n질문: {query}"
                answer = call_llm(llm_service, rag_prompt)
                st.markdown(answer)
            
            status_placeholder.empty()
                
        st.session_state.duty_messages.append({"role": "assistant", "content": answer})
        # st.rerun()
    # 대화 초기화 버튼
    if st.button("🔄 대화 초기화", key="btn_duty_clear"):
        st.session_state.duty_messages = []
        st.rerun()

def render_duty_manual_button(sb, llm_service):
    """사이드바에 버튼 렌더링 (토글 방식)"""
    # [보안] 비로그인 시: 버튼은 보이지만 비활성화 (옅은 색 + 안내)
    if not st.session_state.get("logged_in"):
        st.sidebar.button(
            "📘 당직메뉴얼", 
            disabled=True, 
            key="btn_duty_login_req",
            help="로그인 후 이용 가능한 서비스입니다."
        )
        st.sidebar.caption("🔒 로그인 후 이용 가능")
        return

    # 1. 봇 사용 여부 토글 (관리자 모드처럼)
    use_bot = st.sidebar.checkbox("📘 당직 봇 사용", value=True, key="chk_use_duty_bot")
    
    if not use_bot:
        # 사용 안 함으로 설정 시 세션 상태도 닫힘으로 변경 (선택 사항)
        if st.session_state.get("show_duty_bot"):
            st.session_state.show_duty_bot = False
        return

    # 세션 상태에 토글 변수 초기화
    if "show_duty_bot" not in st.session_state:
        st.session_state.show_duty_bot = False

    # 버튼 클릭 시 상태 토글
    if st.sidebar.button("📘 당직메뉴얼", use_container_width=True):
        st.session_state.show_duty_bot = not st.session_state.show_duty_bot
        st.rerun()

    # 상태가 True일 때만 다이얼로그/UI 렌더링
    if st.session_state.show_duty_bot:
        duty_manual_chat_dialog(sb, llm_service)

# st.dialog가 있는지 확인 (Streamlit 1.34+)
if hasattr(st, "dialog"):
    @st.dialog("📘 당직메뉴얼 챗봇")
    def duty_manual_chat_dialog(sb, llm_service):
        _render_duty_chat_ui(sb, llm_service)
else:
    # Fallback
    def duty_manual_chat_dialog(sb, llm_service):
        # Fallback에서는 닫기 버튼을 제공해야 함 (Expander는 자동이지만 여기선 커스텀 처리)
        with st.expander("📘 당직메뉴얼 챗봇", expanded=True):
            _render_duty_chat_ui(sb, llm_service)
            if st.button("닫기", key="btn_close_duty_bot"):
                st.session_state.show_duty_bot = False
                st.rerun()
