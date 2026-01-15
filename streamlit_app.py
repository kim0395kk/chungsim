# streamlit_app.py
# -*- coding: utf-8 -*-

import json
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from html import escape as _escape
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# ---------------------------
# Optional deps (앱 전체가 죽지 않도록)
# ---------------------------
try:
    import requests
except Exception:
    requests = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from groq import Groq
except Exception:
    Groq = None

try:
    from supabase import create_client
except Exception:
    create_client = None

# ==========================================
# 0) Settings
# ==========================================
MAX_FOLLOWUP_Q = 5
LAW_BOT_SEARCH_URL = "https://www.law.go.kr/LSW/ais/searchList.do?query="
ADMIN_EMAIL = "kim0395kk@korea.kr"

# ✅ 사이드바 초기 상태(접힘)
st.set_page_config(
    layout="wide",
    page_title="AI Bureau: The Legal Glass",
    page_icon="⚖️",
    initial_sidebar_state="collapsed",
)


def get_secret(path1: str, path2: Optional[str] = None, default=None):
    """
    st.secrets를 안전하게 읽기:
    - get_secret("supabase","SUPABASE_URL")
    - get_secret("general","LAW_API_ID")
    """
    try:
        if path2 is None:
            return st.secrets.get(path1, default)
        return st.secrets.get(path1, {}).get(path2, default)
    except Exception:
        return default


def make_lawbot_url(query: str) -> str:
    return LAW_BOT_SEARCH_URL + urllib.parse.quote((query or "").strip())


def _safe_json_loads(text: str) -> Optional[Any]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        m = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception:
        return None
    return None


def safe_inline_md_to_html(s: str) -> str:
    """
    공문 내부에서 **볼드** 같은 최소 마크다운만 안전하게 HTML로 변환
    - HTML injection 방지 위해 먼저 escape 후, **...**만 <b>로 치환
    """
    if s is None:
        s = ""
    s = _escape(str(s))
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)  # ✅ **1** 볼드 처리
    s = s.replace("\n", "<br>")
    return s


# ==========================================
# 1) Styles
# ==========================================
st.markdown(
    """
<style>
    .stApp { background-color: #f3f4f6; }

    /* A4 조판 */
    .paper-sheet {
        background-color: white;
        width: 100%;
        max-width: 210mm;
        min-height: 297mm;
        padding: 25mm;
        margin: auto;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        font-family: 'Batang', serif;
        color: #111;
        line-height: 1.6;
        position: relative;
        border-radius: 12px;
    }

    .doc-header { text-align: center; font-size: 22pt; font-weight: 900; margin-bottom: 30px; letter-spacing: 2px; }
    .doc-info { display: flex; justify-content: space-between; font-size: 11pt; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; gap:10px; flex-wrap:wrap; }
    .doc-body { font-size: 12pt; text-align: justify; white-space: normal; }
    .doc-footer { text-align: center; font-size: 20pt; font-weight: bold; margin-top: 80px; letter-spacing: 5px; }
    .stamp { position: absolute; bottom: 85px; right: 80px; border: 3px solid #cc0000; color: #cc0000; padding: 5px 10px; font-size: 14pt; font-weight: bold; transform: rotate(-15deg); opacity: 0.8; border-radius: 5px; }

    .agent-log { font-family: 'Consolas', monospace; font-size: 0.85rem; padding: 6px 12px; border-radius: 6px; margin-bottom: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    .log-legal { background-color: #eff6ff; color: #1e40af; border-left: 4px solid #3b82f6; }
    .log-search { background-color: #fff7ed; color: #c2410c; border-left: 4px solid #f97316; }
    .log-strat { background-color: #f5f3ff; color: #6d28d9; border-left: 4px solid #8b5cf6; }
    .log-calc { background-color: #f0fdf4; color: #166534; border-left: 4px solid #22c55e; }
    .log-draft { background-color: #fef2f2; color: #991b1b; border-left: 4px solid #ef4444; }
    .log-sys { background-color: #f3f4f6; color: #4b5563; border-left: 4px solid #9ca3af; }

    /* ✅ 법령AI 버튼: 파란 배경 + 화이트 */
    .lawai-btn {
        display: inline-flex;
        gap: 8px;
        align-items: center;
        padding: 10px 14px;
        border-radius: 12px;
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        color: white !important;
        text-decoration: none !important;
        font-weight: 900;
        border: 1px solid rgba(255,255,255,0.25);
        box-shadow: 0 8px 22px rgba(37,99,235,0.28);
    }
    .lawai-btn:hover { filter: brightness(1.05); transform: translateY(-1px); }

    /* Streamlit Cloud 상단 Fork/GitHub 숨김 */
    header [data-testid="stToolbar"] { display: none !important; }
    header [data-testid="stDecoration"] { display: none !important; }
    header { height: 0px !important; }
    footer { display: none !important; }
    div[data-testid="stStatusWidget"] { display: none !important; }

    /* ✅ 사이드바 숨김 토글용 (JS 없이 CSS로 숨김/표시) */
    .hide-sidebar [data-testid="stSidebar"] { display: none !important; }
    .hide-sidebar [data-testid="stSidebarNav"] { display: none !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 2) Auth / Supabase helpers
# ==========================================
def _sb_make_client():
    if not create_client:
        return None

    sb_url = get_secret("supabase", "SUPABASE_URL")
    sb_key = get_secret("supabase", "SUPABASE_ANON_KEY") or get_secret("supabase", "SUPABASE_KEY")
    if not sb_url or not sb_key:
        return None
    try:
        return create_client(sb_url, sb_key)
    except Exception:
        return None


def _sb_apply_session(sb):
    """
    Streamlit rerun마다 세션을 다시 주입 (access/refresh 토큰)
    """
    try:
        access = st.session_state.get("sb_access_token", "")
        refresh = st.session_state.get("sb_refresh_token", "")
        if access and refresh and hasattr(sb, "auth") and hasattr(sb.auth, "set_session"):
            sb.auth.set_session(access, refresh)
    except Exception:
        pass

    # postgrest에 auth 토큰 먹이기(버전별 대응)
    try:
        access = st.session_state.get("sb_access_token", "")
        if access and hasattr(sb, "postgrest") and hasattr(sb.postgrest, "auth"):
            sb.postgrest.auth(access)
    except Exception:
        pass


def _auth_set_logged_in(sb, email: str):
    """
    로그인 성공 후: 세션/유저정보 저장
    """
    try:
        sess = None
        if hasattr(sb, "auth") and hasattr(sb.auth, "get_session"):
            sess = sb.auth.get_session()
        if sess and getattr(sess, "access_token", None):
            st.session_state["sb_access_token"] = sess.access_token
            st.session_state["sb_refresh_token"] = sess.refresh_token
    except Exception:
        pass

    # 유저정보
    st.session_state["logged_in"] = True
    st.session_state["user_email"] = (email or "").strip().lower()

    # user_id 가져오기
    try:
        if hasattr(sb, "auth") and hasattr(sb.auth, "get_user"):
            u = sb.auth.get_user()
            uid = None
            if u and getattr(u, "user", None):
                uid = getattr(u.user, "id", None)
            st.session_state["user_id"] = uid
    except Exception:
        st.session_state["user_id"] = None


def _auth_logout(sb):
    try:
        if sb and hasattr(sb, "auth") and hasattr(sb.auth, "sign_out"):
            sb.auth.sign_out()
    except Exception:
        pass

    for k in ["logged_in", "user_email", "user_id", "sb_access_token", "sb_refresh_token",
              "signup_stage", "pending_email"]:
        if k in st.session_state:
            del st.session_state[k]


def is_admin_user() -> bool:
    return (st.session_state.get("user_email", "").lower() == ADMIN_EMAIL.lower())


# ==========================================
# 3) AI / Services
# ==========================================
class LLMService:
    def __init__(self):
        self.gemini_key = get_secret("general", "GEMINI_API_KEY")
        self.groq_key = get_secret("general", "GROQ_API_KEY")

        self.gemini_models = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]

        if self.gemini_key and genai:
            try:
                genai.configure(api_key=self.gemini_key)
            except Exception:
                pass

        self.groq_client = None
        if self.groq_key and Groq:
            try:
                self.groq_client = Groq(api_key=self.groq_key)
            except Exception:
                self.groq_client = None

    def is_available(self) -> bool:
        return bool((self.gemini_key and genai) or (self.groq_client is not None))

    def _try_gemini_text(self, prompt: str) -> Tuple[str, str]:
        if not (self.gemini_key and genai):
            raise Exception("Gemini not configured")

        last_err = None
        for model_name in self.gemini_models:
            try:
                model = genai.GenerativeModel(model_name)
                res = model.generate_content(prompt)
                return (res.text or "").strip(), model_name
            except Exception as e:
                last_err = e
                continue
        raise Exception(f"All Gemini models failed: {last_err}")

    def generate_text(self, prompt: str) -> str:
        try:
            text, _ = self._try_gemini_text(prompt)
            if text:
                return text
        except Exception:
            pass

        if self.groq_client:
            try:
                completion = self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                return (completion.choices[0].message.content or "").strip()
            except Exception:
                return "System Error"

        return "시스템 오류: AI 모델 연결 실패"

    def generate_json(self, prompt: str) -> Optional[Any]:
        text = self.generate_text(prompt + "\n\n반드시 JSON만 출력. 다른 텍스트 금지.")
        return _safe_json_loads(text)


class SearchService:
    def __init__(self):
        self.client_id = get_secret("general", "NAVER_CLIENT_ID")
        self.client_secret = get_secret("general", "NAVER_CLIENT_SECRET")
        self.news_url = "https://openapi.naver.com/v1/search/news.json"

    def _headers(self):
        return {"X-Naver-Client-Id": self.client_id, "X-Naver-Client-Secret": self.client_secret}

    def _clean_html(self, s: str) -> str:
        if not s:
            return ""
        s = re.sub(r"<[^>]+>", "", s)
        s = s.replace("&quot;", '"').replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
        return s.strip()

    def _extract_keywords_llm(self, situation: str) -> str:
        prompt = f"상황: '{situation}'\n뉴스 검색을 위한 핵심 키워드 2~3개만 콤마로 구분해 출력."
        try:
            res = (llm_service.generate_text(prompt) or "").strip()
            res = re.sub(r'[".?]', "", res)
            return res
        except Exception:
            return situation[:20]

    def search_news(self, query: str, top_k: int = 3) -> str:
        if not requests:
            return "⚠️ requests 모듈이 없습니다."
        if not self.client_id or not self.client_secret:
            return "⚠️ 네이버 API 키가 없습니다."
        if not query:
            return "⚠️ 검색어가 비었습니다."

        try:
            params = {"query": query, "display": 10, "sort": "date"}
            res = requests.get(self.news_url, headers=self._headers(), params=params, timeout=8)
            res.raise_for_status()
            items = res.json().get("items", [])

            if not items:
                return f"🔍 `{query}` 관련 최신 사례가 없습니다."

            lines = [f"📰 **최신 뉴스 사례 (검색어: {query})**", "---"]
            for it in items[:top_k]:
                title = self._clean_html(it.get("title", ""))
                desc = self._clean_html(it.get("description", ""))
                link = it.get("link", "#")
                pub = self._clean_html(it.get("pubDate", ""))
                pub_txt = f" ({pub})" if pub else ""
                lines.append(f"- **[{title}]({link})**{pub_txt}\n  : {desc[:150]}...")
            return "\n".join(lines)
        except Exception as e:
            return f"검색 중 오류: {str(e)}"

    def search_precedents(self, situation: str, top_k: int = 3) -> str:
        keywords = self._extract_keywords_llm(situation)
        return self.search_news(keywords, top_k=top_k)


class LawOfficialService:
    def __init__(self):
        self.api_id = get_secret("general", "LAW_API_ID")
        self.base_url = "http://www.law.go.kr/DRF/lawSearch.do"
        self.service_url = "http://www.law.go.kr/DRF/lawService.do"

    def _make_current_link(self, mst_id: str) -> Optional[str]:
        if not self.api_id or not mst_id:
            return None
        return f"https://www.law.go.kr/DRF/lawService.do?OC={self.api_id}&target=law&MST={mst_id}&type=HTML"

    def get_law_text(self, law_name: str, article_num: Optional[int] = None, return_link: bool = False):
        if not requests:
            msg = "⚠️ requests 모듈이 없습니다."
            return (msg, None) if return_link else msg

        if not self.api_id:
            msg = "⚠️ API ID(OC)가 설정되지 않았습니다."
            return (msg, None) if return_link else msg

        mst_id = ""
        try:
            params = {"OC": self.api_id, "target": "law", "type": "XML", "query": law_name, "display": 1}
            res = requests.get(self.base_url, params=params, timeout=6)
            root = ET.fromstring(res.content)

            law_node = root.find(".//law")
            if law_node is None:
                msg = f"🔍 '{law_name}'에 대한 검색 결과가 없습니다."
                return (msg, None) if return_link else msg

            mst_id = (law_node.findtext("법령일련번호") or "").strip()
        except Exception as e:
            msg = f"API 검색 중 오류: {e}"
            return (msg, None) if return_link else msg

        current_link = self._make_current_link(mst_id)

        try:
            if not mst_id:
                msg = f"✅ '{law_name}' 확인\n(법령일련번호(MST) 추출 실패)\n🔗 현행 원문: {current_link or '-'}"
                return (msg, current_link) if return_link else msg

            detail_params = {"OC": self.api_id, "target": "law", "type": "XML", "MST": mst_id}
            res_detail = requests.get(self.service_url, params=detail_params, timeout=10)
            root_detail = ET.fromstring(res_detail.content)

            if article_num:
                for article in root_detail.findall(".//조문단위"):
                    jo_num_tag = article.find("조문번호")
                    jo_content_tag = article.find("조문내용")
                    if jo_num_tag is None or jo_content_tag is None:
                        continue

                    current_num = (jo_num_tag.text or "").strip()
                    if str(article_num) == current_num:
                        target_text = f"[{law_name} 제{current_num}조 전문]\n" + _escape((jo_content_tag.text or "").strip())
                        for hang in article.findall(".//항"):
                            hang_content = hang.find("항내용")
                            if hang_content is not None:
                                target_text += f"\n  - {(hang_content.text or '').strip()}"
                        return (target_text, current_link) if return_link else target_text

            msg = f"✅ '{law_name}' 확인\n(상세 조문 자동 추출 실패 또는 조문번호 미지정)\n🔗 현행 원문: {current_link or '-'}"
            return (msg, current_link) if return_link else msg

        except Exception as e:
            msg = f"상세 법령 파싱 실패: {e}"
            return (msg, current_link) if return_link else msg


# ==========================================
# 4) Global Instances
# ==========================================
llm_service = LLMService()
search_service = SearchService()
law_api_service = LawOfficialService()


# ==========================================
# 5) Agents / Workflow
# ==========================================
class LegalAgents:
    @staticmethod
    def researcher(situation: str) -> str:
        prompt_extract = f"""
상황: "{situation}"

위 민원 처리를 위해 법적 근거로 삼아야 할 핵심 대한민국 법령과 조문 번호를
중요도 순으로 최대 3개까지 JSON 리스트로 추출하시오.

형식: [{{"law_name": "도로교통법", "article_num": 32}}, ...]
법령명은 정식 명칭. 조문번호 불명확하면 null.
"""
        search_targets: List[Dict[str, Any]] = []
        extracted = llm_service.generate_json(prompt_extract)

        if isinstance(extracted, list):
            search_targets = extracted
        elif isinstance(extracted, dict):
            search_targets = [extracted]

        if not search_targets:
            search_targets = [{"law_name": "도로교통법", "article_num": None}]

        report_lines = []
        api_success_count = 0

        report_lines.append(f"🔍 **AI가 식별한 핵심 법령 ({len(search_targets)}건)**")
        report_lines.append("---")

        for idx, item in enumerate(search_targets):
            law_name = (item.get("law_name") or "관련법령").strip()
            article_num = item.get("article_num", None)

            if isinstance(article_num, str):
                m = re.search(r"\d+", article_num)
                article_num = int(m.group(0)) if m else None
            if isinstance(article_num, (int, float)):
                article_num = int(article_num)
            else:
                article_num = None

            law_text, current_link = law_api_service.get_law_text(law_name, article_num, return_link=True)

            error_keywords = ["검색 결과가 없습니다", "오류", "API ID", "실패", "requests 모듈"]
            is_success = not any(k in (law_text or "") for k in error_keywords)

            if is_success:
                api_success_count += 1
                law_title = f"[{law_name}]({current_link})" if current_link else law_name
                header = f"✅ **{idx+1}. {law_title} {('제'+str(article_num)+'조') if article_num else ''} (확인됨)**"
                content = law_text
            else:
                header = f"⚠️ **{idx+1}. {law_name} {('제'+str(article_num)+'조') if article_num else ''} (API 조회 실패)**"
                content = "(국가법령정보센터에서 해당 조문을 찾지 못했습니다. 법령명이 정확한지 확인 필요)"

            report_lines.append(f"{header}\n{content}\n")

        if api_success_count == 0:
            prompt_fallback = f"""
Role: 행정 법률 전문가
Task: 아래 상황에 적용될 법령과 조항을 찾아 설명하시오.
상황: "{situation}"

* 경고: 현재 외부 법령 API 연결이 원활하지 않습니다.
반드시 상단에 [AI 추론 결과]임을 명시하고 환각 가능성을 경고하시오.
"""
            ai_fallback_text = (llm_service.generate_text(prompt_fallback) or "").strip()
            return f"""⚠️ **[시스템 경고: API 조회 실패]**
(국가법령정보센터 연결 실패로 AI 지식 기반 답변입니다. **환각 가능성** 있으니 법제처 확인 필수)

--------------------------------------------------
{ai_fallback_text}"""

        return "\n".join(report_lines)

    @staticmethod
    def strategist(situation: str, legal_basis: str, search_results: str) -> str:
        prompt = f"""
당신은 행정 업무 베테랑 '주무관'입니다.

[민원 상황]: {situation}
[확보된 법적 근거]:
{legal_basis}

[유사 사례/기사]:
{search_results}

위 정보를 종합하여 민원 처리 방향(Strategy)을 수립하세요.
서론(인사말/공감) 금지.

1. 처리 방향
2. 핵심 주의사항
3. 예상 반발 및 대응
"""
        return llm_service.generate_text(prompt)

    @staticmethod
    def clerk(situation: str, legal_basis: str) -> dict:
        today = datetime.now()
        prompt = f"""
오늘: {today.strftime('%Y-%m-%d')}
상황: {situation}
법령: {legal_basis}
이행/의견제출 기간은 며칠인가?
숫자만 출력. 모르겠으면 15.
"""
        try:
            res = (llm_service.generate_text(prompt) or "").strip()
            m = re.search(r"\d{1,3}", res)
            days = int(m.group(0)) if m else 15
            days = max(1, min(days, 180))
        except Exception:
            days = 15

        deadline = today + timedelta(days=days)
        return {
            "today_str": today.strftime("%Y. %m. %d."),
            "deadline_str": deadline.strftime("%Y. %m. %d."),
            "days_added": days,
            "doc_num": f"행정-{today.strftime('%Y')}-{int(time.time())%1000:03d}호",
        }

    @staticmethod
    def drafter_json(situation: str, legal_basis: str, meta_info: dict, strategy: str) -> Optional[dict]:
        prompt = f"""
당신은 행정기관의 베테랑 서기입니다. 아래 정보를 바탕으로 완결된 공문서를 JSON으로 작성하세요.

[입력]
- 민원: {situation}
- 법적 근거: {legal_basis}
- 시행일자: {meta_info['today_str']}
- 기한: {meta_info['deadline_str']} ({meta_info['days_added']}일)

[전략]
{strategy}

[원칙]
1) 본문에 법 조항 인용 필수
2) 구조: 경위 -> 법적 근거 -> 처분 내용 -> 이의제기 절차
3) 개인정보 마스킹('OOO')
4) 반드시 아래 JSON 스키마를 지킬 것

[JSON 스키마]
{{
  "title": "string",
  "receiver": "string",
  "body_paragraphs": ["string", "..."],
  "department_head": "string"
}}
"""
        return llm_service.generate_json(prompt)

    @staticmethod
    def drafter_fallback_text(situation: str, legal_basis: str, meta_info: dict, strategy: str) -> dict:
        prompt = f"""
아래 입력으로 공문서를 '텍스트'로 작성하라.
단, 섹션 표시는 반드시 아래 마커를 그대로 쓰고, 문단은 줄바꿈으로 구분.

[TITLE]
...
[RECEIVER]
...
[BODY]
...
[HEAD]
...

[입력]
- 민원: {situation}
- 법적 근거: {legal_basis}
- 시행일자: {meta_info['today_str']}
- 기한: {meta_info['deadline_str']} ({meta_info['days_added']}일)
[전략]
{strategy}
"""
        t = (llm_service.generate_text(prompt) or "").strip()
        def pick(tag: str) -> str:
            m = re.search(rf"\[{tag}\]\s*(.*?)(?=\n\[[A-Z]+\]|\Z)", t, re.DOTALL)
            return (m.group(1).strip() if m else "").strip()

        title = pick("TITLE") or "공 문 서"
        receiver = pick("RECEIVER") or "수신자 참조"
        body = pick("BODY") or "(본문 생성 실패)"
        head = pick("HEAD") or "행정기관장"

        body_paragraphs = [p.strip() for p in re.split(r"\n{1,}", body) if p.strip()]
        return {"title": title, "receiver": receiver, "body_paragraphs": body_paragraphs, "department_head": head}


def repair_doc_data(doc: Any) -> Optional[dict]:
    if not isinstance(doc, dict):
        return None

    title = str(doc.get("title") or "공 문 서").strip()
    receiver = str(doc.get("receiver") or "수신자 참조").strip()
    head = str(doc.get("department_head") or "행정기관장").strip()

    body = doc.get("body_paragraphs", [])
    if isinstance(body, str):
        body = [body]
    if not isinstance(body, list):
        body = []

    body = [str(p).strip() for p in body if str(p).strip()]
    if not body:
        body = ["(본문 생성 실패)"]

    return {"title": title, "receiver": receiver, "body_paragraphs": body, "department_head": head}


def build_lawbot_pack(situation: str, legal_text: str) -> dict:
    """
    ✅ 법령AI(Lawbot) 검색용 키워드 + 링크
    """
    s = (situation or "").strip()
    prompt = f"""
상황: "{s}"
국가법령정보센터 법령AI(검색)에 넣을 핵심 키워드 3~6개를 JSON 배열로만 출력.
예: ["무단방치", "자동차관리법", "공시송달", "직권말소"]
"""
    kws = llm_service.generate_json(prompt) or []
    if not isinstance(kws, list):
        kws = []
    kws = [str(x).strip() for x in kws if str(x).strip()]

    query_text = (s[:60] + " " + " ".join(kws[:6])).strip()
    query_text = re.sub(r"\s+", " ", query_text)

    return {
        "core_keywords": kws[:10],
        "query_text": query_text[:180],
        "url": make_lawbot_url(query_text[:180]),
    }


def run_workflow(user_input: str) -> dict:
    log_placeholder = st.empty()
    logs: List[str] = []

    def add_log(msg: str, style: str = "sys"):
        logs.append(f"<div class='agent-log log-{style}'>{_escape(msg)}</div>")
        log_placeholder.markdown("".join(logs), unsafe_allow_html=True)
        time.sleep(0.12)

    add_log("🔍 Phase 1: 법령 및 유사 사례 리서치 중...", "legal")
    legal_basis = LegalAgents.researcher(user_input)
    add_log("📜 법적 근거 발견 완료", "legal")

    add_log("🟩 네이버 검색 엔진 가동...", "search")
    try:
        search_results = search_service.search_precedents(user_input)
    except Exception:
        search_results = "검색 모듈 미연결 (건너뜀)"

    add_log("🧠 Phase 2: AI 주무관이 업무 처리 방향 수립...", "strat")
    strategy = LegalAgents.strategist(user_input, legal_basis, search_results)

    add_log("📅 Phase 3: 기한 산정 및 공문서 작성...", "calc")
    meta_info = LegalAgents.clerk(user_input, legal_basis)

    add_log("✍️ 최종 공문서 조판 중...", "draft")
    doc_data = None

    # 1) JSON 시도 2회
    for _ in range(2):
        try:
            cand = LegalAgents.drafter_json(user_input, legal_basis, meta_info, strategy)
            doc_data = repair_doc_data(cand)
            if doc_data:
                break
        except Exception:
            doc_data = None

    # 2) 텍스트 fallback
    if not doc_data:
        doc_data = LegalAgents.drafter_fallback_text(user_input, legal_basis, meta_info, strategy)
        doc_data = repair_doc_data(doc_data)

    time.sleep(0.2)
    log_placeholder.empty()

    lawbot_pack = build_lawbot_pack(user_input, legal_basis)

    return {
        "situation": user_input,
        "doc": doc_data,
        "meta": meta_info,
        "law": legal_basis,
        "search": search_results,
        "strategy": strategy,
        "lawbot_pack": lawbot_pack,
    }


# ==========================================
# 6) DB (work_archive)
# ==========================================
def db_insert_archive(sb, payload: dict) -> Tuple[bool, str, Optional[str]]:
    if not sb:
        return False, "DB 미연결 (supabase client 없음)", None
    if not st.session_state.get("logged_in"):
        return False, "로그인 필요 (DB 저장 불가)", None

    uid = st.session_state.get("user_id")
    email = st.session_state.get("user_email")

    meta = payload.get("meta") or {}
    doc = payload.get("doc") or {}

    # 복원용 payload는 그대로 저장
    data = {
        "case_id": meta.get("doc_num", ""),
        "user_id": uid,
        "user_email": email,

        "prompt": payload.get("situation", ""),
        "law": payload.get("law", ""),
        "news": payload.get("search", ""),
        "guide": payload.get("strategy", ""),
        "official_doc": json.dumps(doc, ensure_ascii=False),

        "payload": payload,
    }

    try:
        resp = sb.table("work_archive").insert(data).execute()
        inserted_id = None
        try:
            if hasattr(resp, "data") and resp.data and isinstance(resp.data, list):
                inserted_id = resp.data[0].get("id")
        except Exception:
            inserted_id = None

        return True, "DB 저장 성공", inserted_id
    except Exception as e:
        return False, f"DB 저장 실패: {e}", None


def db_update_archive_payload(sb, row_id: str, payload: dict) -> Tuple[bool, str]:
    if not sb or not row_id:
        return False, "DB 미연결/ID 없음"
    try:
        sb.table("work_archive").update({"payload": payload}).eq("id", row_id).execute()
        return True, "DB 업데이트 성공"
    except Exception as e:
        return False, f"DB 업데이트 실패: {e}"


def db_list_archives(sb, limit: int = 50) -> List[dict]:
    if not sb or not st.session_state.get("logged_in"):
        return []

    try:
        q = sb.table("work_archive").select("id, created_at, case_id, user_email, prompt, payload").order("created_at", desc=True).limit(limit)
        # 일반 유저는 자기것만(정책상 어차피 제한되지만, 쿼리도 좁힘)
        if not is_admin_user():
            uid = st.session_state.get("user_id")
            q = q.eq("user_id", uid)
        resp = q.execute()
        return resp.data or []
    except Exception:
        return []


def db_delete_archive(sb, row_id: str) -> Tuple[bool, str]:
    if not sb or not row_id:
        return False, "DB 미연결/ID 없음"
    try:
        sb.table("work_archive").delete().eq("id", row_id).execute()
        return True, "삭제 완료"
    except Exception as e:
        return False, f"삭제 실패: {e}"


def db_admin_upsert_raw(sb, row_id: Optional[str], payload: dict, user_id: Optional[str], user_email: Optional[str], case_id: str) -> Tuple[bool, str]:
    """
    관리자 전용: 임의 삽입/수정
    """
    if not sb:
        return False, "DB 미연결"
    if not is_admin_user():
        return False, "관리자만 가능"

    data = {
        "case_id": case_id,
        "user_id": user_id,
        "user_email": user_email,
        "prompt": payload.get("situation", ""),
        "law": payload.get("law", ""),
        "news": payload.get("search", ""),
        "guide": payload.get("strategy", ""),
        "official_doc": json.dumps(payload.get("doc") or {}, ensure_ascii=False),
        "payload": payload,
    }

    try:
        if row_id:
            sb.table("work_archive").update(data).eq("id", row_id).execute()
            return True, "관리자 수정 완료"
        else:
            sb.table("work_archive").insert(data).execute()
            return True, "관리자 삽입 완료"
    except Exception as e:
        return False, f"관리자 upsert 실패: {e}"


# ==========================================
# 7) Follow-up Chat (Nested expander 금지)
# ==========================================
def _strip_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return text


def needs_tool_call(user_msg: str) -> dict:
    t = (user_msg or "").lower()
    law_triggers = ["근거", "조문", "법령", "몇 조", "원문", "현행", "추가 조항", "다른 조문", "전문", "절차법", "행정절차"]
    news_triggers = ["뉴스", "사례", "판례", "기사", "보도", "최근", "유사", "선례"]
    return {"need_law": any(k in t for k in law_triggers), "need_news": any(k in t for k in news_triggers)}


def plan_tool_calls_llm(user_msg: str, situation: str, known_law_text: str) -> dict:
    prompt = f"""
너는 행정업무 보조 에이전트다. 사용자의 후속 질문을 보고, 추가 조회가 필요하면 계획을 JSON으로 만든다.

[민원 상황]
{situation}

[이미 확보된 적용 법령 텍스트]
{known_law_text[:2500]}

[사용자 질문]
{user_msg}

[출력 JSON]
{{
  "need_law": true/false,
  "law_name": "정식 법령명(필요시)",
  "article_num": 0 또는 정수(모르면 0),
  "need_news": true/false,
  "news_query": "2~4단어 키워드",
  "reason": "왜 필요한지"
}}
반드시 JSON만 출력.
"""
    plan = llm_service.generate_json(prompt) or {}
    if not isinstance(plan, dict):
        return {"need_law": False, "law_name": "", "article_num": 0, "need_news": False, "news_query": "", "reason": "parse failed"}

    plan["need_law"] = bool(plan.get("need_law"))
    plan["need_news"] = bool(plan.get("need_news"))
    plan["law_name"] = str(plan.get("law_name") or "").strip()
    try:
        plan["article_num"] = int(plan.get("article_num") or 0)
    except Exception:
        plan["article_num"] = 0
    plan["news_query"] = str(plan.get("news_query") or "").strip()
    plan["reason"] = str(plan.get("reason") or "").strip()
    return plan


def answer_followup(case_context: str, extra_context: str, chat_history: list, user_msg: str) -> str:
    hist = chat_history[-8:]
    hist_txt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in hist])

    prompt = f"""
너는 '케이스 고정 행정 후속 Q&A 챗봇'이다.

{case_context}

[추가 조회 결과(있으면)]
{extra_context if extra_context else "(없음)"}

[대화 히스토리(최근)]
{hist_txt if hist_txt else "(없음)"}

[사용자 질문]
{user_msg}

[답변 규칙]
- 케이스 컨텍스트/추가 조회 결과 범위에서만 답한다.
- 모르면 모른다고 하고, 필요한 추가 조회 종류(법령/뉴스)를 구체적으로 말한다.
- 서론 없이 실무형으로.
"""
    return llm_service.generate_text(prompt)


def build_case_context(res: dict) -> str:
    situation = res.get("situation", "")
    law_txt = _strip_html(res.get("law", ""))
    news_txt = _strip_html(res.get("search", ""))
    strategy = res.get("strategy", "")
    doc = res.get("doc") or {}

    body_paras = doc.get("body_paragraphs", [])
    if isinstance(body_paras, str):
        body_paras = [body_paras]
    body = "\n".join([f"- {p}" for p in body_paras])

    ctx = f"""
[케이스 컨텍스트]
1) 민원 상황(원문)
{situation}

2) 적용 법령/조문(이미 확인된 내용)
{law_txt}

3) 관련 뉴스/사례(이미 조회된 내용)
{news_txt}

4) 업무 처리 방향(Strategy)
{strategy}

5) 생성된 공문서(요약)
- 제목: {doc.get('title','')}
- 수신: {doc.get('receiver','')}
- 본문:
{body}
- 발신: {doc.get('department_head','')}

[규칙]
- 기본 답변은 위 컨텍스트 범위에서만 작성.
- 컨텍스트에 없는 법령/사례를 단정하지 말 것.
- 사용자가 “근거 더 / 다른 조문 / 뉴스 더” 요청하면 그때만 추가 조회.
"""
    return ctx.strip()


def render_followup_chat(sb, res: dict, archive_row_id: Optional[str]):
    # 세션 초기화
    st.session_state.setdefault("followup_count", 0)
    st.session_state.setdefault("followup_messages", [])
    st.session_state.setdefault("followup_extra_context", "")

    remain = max(0, MAX_FOLLOWUP_Q - st.session_state["followup_count"])
    st.info(f"후속 질문 가능 횟수: **{remain}/{MAX_FOLLOWUP_Q}**")

    # ✅ 법령AI 버튼(강조)
    pack = res.get("lawbot_pack", {}) or {}
    qb = (pack.get("query_text") or "").strip()
    if qb:
        st.markdown(
            f"""<a class="lawai-btn" href="{make_lawbot_url(qb)}" target="_blank">
            🤖 법령 AI · Lawbot 실행 (법령·규칙·서식 찾기)
            </a>""",
            unsafe_allow_html=True
        )

    if remain == 0:
        st.warning("후속 질문 한도(5회)를 모두 사용했습니다.")
        return

    # 대화 렌더
    for m in st.session_state["followup_messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_q = st.chat_input("공문 결과를 바탕으로 후속 질문 (최대 5회)")
    if not user_q:
        return

    st.session_state["followup_messages"].append({"role": "user", "content": user_q})
    st.session_state["followup_count"] += 1

    with st.chat_message("user"):
        st.markdown(user_q)

    case_context = build_case_context(res)
    extra_ctx = st.session_state.get("followup_extra_context", "")

    tool_need = needs_tool_call(user_q)
    if tool_need["need_law"] or tool_need["need_news"]:
        plan = plan_tool_calls_llm(user_q, res.get("situation", ""), _strip_html(res.get("law", "")))

        # 법령봇 빠른검색 링크(후속)
        if plan.get("need_law") and plan.get("law_name"):
            q2 = f"{plan.get('law_name','')} 제{int(plan.get('article_num') or 0)}조 {user_q}".strip()
            q2 = re.sub(r"\s+", " ", q2)[:180]
            extra_ctx += f"\n\n[법령AI 빠른검색]\n- 키워드: {q2}\n- 링크: {make_lawbot_url(q2)}"

        if plan.get("need_law") and plan.get("law_name"):
            art = int(plan.get("article_num") or 0)
            art = art if art > 0 else None
            law_text, law_link = law_api_service.get_law_text(plan["law_name"], art, return_link=True)

            extra_ctx += f"\n\n[추가 법령 조회]\n- 요청: {plan['law_name']} / 제{art if art else '?'}조\n{_strip_html(law_text)}"
            if law_link:
                extra_ctx += f"\n(현행 원문 링크: {law_link})"

        if plan.get("need_news") and plan.get("news_query"):
            news_txt = search_service.search_news(plan["news_query"])
            extra_ctx += f"\n\n[추가 뉴스 조회]\n- 검색어: {plan['news_query']}\n{_strip_html(news_txt)}"

        st.session_state["followup_extra_context"] = extra_ctx

    with st.chat_message("assistant"):
        with st.spinner("후속 답변 생성 중..."):
            ans = answer_followup(
                case_context=case_context,
                extra_context=st.session_state.get("followup_extra_context", ""),
                chat_history=st.session_state["followup_messages"],
                user_msg=user_q,
            )
            st.markdown(ans)

    st.session_state["followup_messages"].append({"role": "assistant", "content": ans})

    # ✅ payload에 후속 대화까지 반영 후 DB 업데이트
    if archive_row_id:
        res2 = dict(res)
        res2["followup"] = {
            "count": st.session_state["followup_count"],
            "messages": st.session_state["followup_messages"],
            "extra_context": st.session_state.get("followup_extra_context", ""),
        }
        ok, msg = db_update_archive_payload(sb, archive_row_id, res2)
        if not ok:
            st.caption(msg)


# ==========================================
# 8) Sidebar UI: Toggle + Auth + History
# ==========================================
def apply_sidebar_visibility_css():
    if "sidebar_open" not in st.session_state:
        st.session_state["sidebar_open"] = False  # 기본 접힘
    if not st.session_state["sidebar_open"]:
        st.markdown("<div class='hide-sidebar'></div>", unsafe_allow_html=True)


def sidebar_toggle_button():
    # 메인 화면 상단에 토글
    colA, colB = st.columns([1, 12])
    with colA:
        if st.button("☰", help="사이드바 접기/펼치기"):
            st.session_state["sidebar_open"] = not st.session_state.get("sidebar_open", False)
            st.rerun()
    with colB:
        st.caption("메뉴(로그인/히스토리) 토글")


def render_sidebar_auth(sb):
    st.sidebar.title("🔐 로그인 / 히스토리")

    # 세션 초기화
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("user_email", "")
    st.session_state.setdefault("user_id", None)
    st.session_state.setdefault("signup_stage", "idle")  # idle|otp_sent|otp_verified|set_pw
    st.session_state.setdefault("pending_email", "")

    tabs = st.sidebar.tabs(["로그인", "회원가입", "히스토리"])

    # -------------------
    # 로그인
    # -------------------
    with tabs[0]:
        st.sidebar.subheader("로그인")
        email = st.sidebar.text_input("아이디(이메일)", key="login_email")
        pw = st.sidebar.text_input("비밀번호", type="password", key="login_pw")

        if st.sidebar.button("로그인", use_container_width=True):
            if not sb:
                st.sidebar.error("Supabase 연결 실패 (secrets 확인)")
            elif not email or not pw:
                st.sidebar.error("이메일/비밀번호를 입력하세요")
            else:
                try:
                    res = sb.auth.sign_in_with_password({"email": email, "password": pw})
                    _auth_set_logged_in(sb, email)
                    st.sidebar.success("로그인 성공")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error("아이디/비밀번호 확인 필요")
                    st.sidebar.caption(str(e))

    # -------------------
    # 회원가입(OTP → 비번설정)
    # -------------------
    with tabs[1]:
        st.sidebar.subheader("회원가입")
        st.sidebar.caption("✅ @korea.kr 이메일만 가입 허용")

        stage = st.session_state.get("signup_stage", "idle")
        email = st.sidebar.text_input("메일 주소", key="su_email", value=st.session_state.get("pending_email", ""))

        if stage == "idle":
            if st.sidebar.button("인증번호 발송", use_container_width=True):
                if not sb:
                    st.sidebar.error("Supabase 연결 실패 (secrets 확인)")
                elif not email.endswith("@korea.kr"):
                    st.sidebar.error("❌ @korea.kr 메일만 가입 가능")
                else:
                    # Email OTP 발송 (should_create_user 옵션은 버전별로 다를 수 있어 try)
                    try:
                        # 일부 버전: options / should_create_user 지원
                        sb.auth.sign_in_with_otp({"email": email, "options": {"should_create_user": True}})
                    except Exception:
                        # fallback: 최소 형태
                        sb.auth.sign_in_with_otp({"email": email})

                    st.session_state["pending_email"] = email
                    st.session_state["signup_stage"] = "otp_sent"
                    st.sidebar.success("메일로 인증번호를 보냈습니다.")
                    st.rerun()

        elif stage == "otp_sent":
            st.sidebar.info("메일로 받은 인증번호(OTP)를 입력하세요.")
            otp = st.sidebar.text_input("인증번호(OTP)", key="su_otp")

            if st.sidebar.button("인증 확인", use_container_width=True):
                try:
                    # verify_otp (type='email')
                    sb.auth.verify_otp({"email": email, "token": otp, "type": "email"})
                    # 세션 저장(OTP 인증 성공 = 로그인 세션 생김)
                    _auth_set_logged_in(sb, email)
                    st.session_state["signup_stage"] = "set_pw"
                    st.sidebar.success("인증 성공. 이제 비밀번호를 설정하세요.")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error("인증 실패 (OTP 확인)")
                    st.sidebar.caption(str(e))

        elif stage == "set_pw":
            st.sidebar.success("이제 비밀번호를 설정하면, 앞으로 이메일+비번으로 로그인됩니다.")
            pw1 = st.sidebar.text_input("비밀번호", type="password", key="su_pw1")
            pw2 = st.sidebar.text_input("비밀번호 확인", type="password", key="su_pw2")

            if st.sidebar.button("비밀번호 설정", use_container_width=True):
                if not pw1 or len(pw1) < 8:
                    st.sidebar.error("비밀번호는 8자 이상 권장")
                elif pw1 != pw2:
                    st.sidebar.error("비밀번호가 일치하지 않습니다")
                else:
                    try:
                        # OTP로 생성된 세션에서 password 업데이트
                        sb.auth.update_user({"password": pw1})
                        st.sidebar.success("비밀번호 설정 완료! 이제 이메일+비번 로그인")
                        st.session_state["signup_stage"] = "idle"
                        st.session_state["pending_email"] = ""
                        # 로그아웃 후 재로그인 유도(선택)
                        # 여기선 그대로 로그인 유지해도 됨
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error("비밀번호 설정 실패")
                        st.sidebar.caption(str(e))

        st.sidebar.divider()

        if st.sidebar.button("회원가입 단계 초기화", use_container_width=True):
            st.session_state["signup_stage"] = "idle"
            st.session_state["pending_email"] = ""
            st.sidebar.success("초기화 완료")
            st.rerun()

    # -------------------
    # 히스토리
    # -------------------
    with tabs[2]:
        if not st.session_state.get("logged_in"):
            st.sidebar.info("로그인 후 히스토리 사용 가능")
        else:
            email = st.session_state.get("user_email", "")
            st.sidebar.write(f"✅ 접속 중: {email}")
            if st.sidebar.button("로그아웃", use_container_width=True):
                _auth_logout(sb)
                st.rerun()

            st.sidebar.divider()

            rows = db_list_archives(sb, limit=60)
            if not rows:
                st.sidebar.caption("저장된 기록이 없습니다.")
                return

            # 목록
            labels = []
            id_map = {}
            for r in rows:
                created = (r.get("created_at") or "")[:19].replace("T", " ")
                case_id = r.get("case_id") or "-"
                who = r.get("user_email") or "-"
                title = (r.get("prompt") or "").strip().replace("\n", " ")
                title = title[:22] + ("…" if len(title) > 22 else "")
                lab = f"{created} | {case_id} | {who} | {title}"
                labels.append(lab)
                id_map[lab] = r.get("id")

            pick = st.sidebar.selectbox("기록 선택", labels)
            row_id = id_map.get(pick)
            picked_row = next((x for x in rows if x.get("id") == row_id), None)

            if picked_row:
                payload = picked_row.get("payload") or {}

                # ✅ 복원(짠!)
                if st.sidebar.button("⚡ 짠! 이 기록 복원", use_container_width=True):
                    st.session_state["workflow_result"] = payload
                    st.session_state["archive_row_id"] = row_id
                    # followup reset
                    st.session_state["followup_count"] = 0
                    st.session_state["followup_messages"] = []
                    st.session_state["followup_extra_context"] = ""
                    st.sidebar.success("복원 완료")
                    st.rerun()

                # 삭제
                if st.sidebar.button("🗑️ 삭제", use_container_width=True):
                    ok, msg = db_delete_archive(sb, row_id)
                    if ok:
                        st.sidebar.success(msg)
                        if st.session_state.get("archive_row_id") == row_id:
                            st.session_state["archive_row_id"] = None
                        st.rerun()
                    else:
                        st.sidebar.error(msg)

                # 관리자 편집(수정/삽입)
                if is_admin_user():
                    st.sidebar.divider()
                    st.sidebar.subheader("🛡️ 관리자 편집")
                    raw = st.sidebar.text_area("payload(JSON)", value=json.dumps(payload, ensure_ascii=False, indent=2), height=240)
                    target_user_id = st.sidebar.text_input("user_id(선택)", value=str(picked_row.get("user_id") or ""))
                    target_user_email = st.sidebar.text_input("user_email(선택)", value=str(picked_row.get("user_email") or ""))
                    target_case_id = st.sidebar.text_input("case_id", value=str(picked_row.get("case_id") or ""))

                    col1, col2 = st.sidebar.columns(2)
                    with col1:
                        if st.button("수정 저장", use_container_width=True):
                            try:
                                new_payload = json.loads(raw)
                                ok, msg = db_admin_upsert_raw(
                                    sb,
                                    row_id=row_id,
                                    payload=new_payload,
                                    user_id=(target_user_id.strip() or None),
                                    user_email=(target_user_email.strip() or None),
                                    case_id=(target_case_id.strip() or ""),
                                )
                                st.sidebar.success(msg) if ok else st.sidebar.error(msg)
                                st.rerun()
                            except Exception as e:
                                st.sidebar.error(f"JSON 파싱 실패: {e}")

                    with col2:
                        if st.button("새로 삽입", use_container_width=True):
                            try:
                                new_payload = json.loads(raw)
                                ok, msg = db_admin_upsert_raw(
                                    sb,
                                    row_id=None,
                                    payload=new_payload,
                                    user_id=(target_user_id.strip() or None),
                                    user_email=(target_user_email.strip() or None),
                                    case_id=(target_case_id.strip() or ""),
                                )
                                st.sidebar.success(msg) if ok else st.sidebar.error(msg)
                                st.rerun()
                            except Exception as e:
                                st.sidebar.error(f"JSON 파싱 실패: {e}")


# ==========================================
# 9) Main UI
# ==========================================
def main():
    apply_sidebar_visibility_css()
    sidebar_toggle_button()

    sb = _sb_make_client()
    if sb:
        _sb_apply_session(sb)

    render_sidebar_auth(sb)

    # 페이지 레이아웃
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.title("🏢 AI 행정관 Pro 충주시청")
        st.caption("문의 kim0395kk@korea.kr \n 세계최초 행정 Govable AI 에이젼트")
        st.markdown("---")

        # 상태표시
        ai_ok = "✅AI" if llm_service.is_available() else "❌AI"
        law_ok = "✅LAW" if bool(get_secret("general", "LAW_API_ID")) else "❌LAW"
        nv_ok = "✅NEWS" if bool(get_secret("general", "NAVER_CLIENT_ID")) else "❌NEWS"
        sb_ok = "✅SUPABASE" if bool(get_secret("supabase", "SUPABASE_URL") and (get_secret("supabase", "SUPABASE_ANON_KEY") or get_secret("supabase", "SUPABASE_KEY"))) else "❌SUPABASE"
        st.caption(f"상태: {ai_ok}  |  {law_ok}  |  {nv_ok}  |  {sb_ok}")

        if not st.session_state.get("logged_in"):
            st.warning("로그인 후 사용 가능합니다. (사이드바 ☰ 메뉴 → 로그인/회원가입)")
            st.stop()

        st.markdown("### 🗣️ 업무 지시")
        user_input = st.text_area(
            "업무 내용",
            height=150,
            placeholder="예시\n- 상황: (무슨 일 / 어디 / 언제 / 증거 유무...)\n- 의도: (쟁점: 요건/절차/근거...)\n- 요청: (원하는 결과물: 공문 종류/회신/사전통지 등)",
            label_visibility="collapsed",
        )

        st.warning("⚠️ 비공개 문서 부분복사/내부검토 민감정보(성명·연락처·주소·차량번호 등) 입력 금지")

        if st.button("⚡ 스마트 분석 시작", type="primary", use_container_width=True):
            if not user_input:
                st.warning("내용을 입력해주세요.")
            else:
                with st.spinner("AI 에이전트 팀이 협업 중입니다..."):
                    res = run_workflow(user_input)
                    st.session_state["workflow_result"] = res

                    # DB 저장
                    ok, msg, row_id = db_insert_archive(sb, res)
                    st.session_state["save_msg"] = msg
                    st.session_state["archive_row_id"] = row_id

        if "workflow_result" in st.session_state:
            res = st.session_state["workflow_result"]
            st.markdown("---")

            msg = st.session_state.get("save_msg", "")
            if msg:
                if "성공" in msg:
                    st.success(f"✅ {msg}")
                else:
                    st.info(f"ℹ️ {msg}")

            # ✅ 법령AI 버튼(강조)
            pack = res.get("lawbot_pack", {}) or {}
            qb = (pack.get("query_text") or "").strip()
            if qb:
                st.markdown(
                    f"""<a class="lawai-btn" href="{make_lawbot_url(qb)}" target="_blank">
                    🤖 법령 AI · Lawbot 실행 (법령·규칙·서식 찾기)
                    </a>""",
                    unsafe_allow_html=True
                )

            with st.expander("✅ [검토] 법령 및 유사 사례 확인", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**📜 적용 법령 (법령명 클릭 시 현행 원문 새창)**")
                    raw_law = res.get("law", "")

                    cleaned = raw_law.replace("&lt;", "<").replace("&gt;", ">")
                    cleaned = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", cleaned)
                    cleaned = re.sub(
                        r'\[([^\]]+)\]\(([^)]+)\)',
                        r'<a href="\2" target="_blank" style="color:#2563eb; text-decoration:none; font-weight:800;">\1</a>',
                        cleaned,
                    )
                    cleaned = cleaned.replace("---", "<br><br>").replace("\n", "<br>")

                    st.markdown(
                        f"""
                        <div style="height: 300px; overflow-y: auto; padding: 15px; border-radius: 8px;
                            border: 1px solid #e5e7eb; background: #f8fafc; font-family: 'Pretendard', sans-serif;
                            font-size: 0.9rem; line-height: 1.6; color: #334155;">
                        {cleaned}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col2:
                    st.markdown("**🟩 관련 뉴스/사례**")
                    raw_news = res.get("search", "")

                    news_body = raw_news.replace("# ", "").replace("## ", "")
                    news_body = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", news_body)
                    news_html = re.sub(
                        r"\[([^\]]+)\]\(([^)]+)\)",
                        r'<a href="\2" target="_blank" style="color:#2563eb; text-decoration:none; font-weight:700;">\1</a>',
                        news_body
                    )
                    news_html = news_html.replace("\n", "<br>")

                    st.markdown(
                        f"""
                        <div style="height: 300px; overflow-y: auto; padding: 15px; border-radius: 8px;
                            border: 1px solid #dbeafe; background: #eff6ff; font-family: 'Pretendard', sans-serif;
                            font-size: 0.9rem; line-height: 1.6; color: #1e3a8a;">
                        {news_html}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            with st.expander("🧭 [방향] 업무 처리 가이드라인", expanded=True):
                st.markdown(res.get("strategy", ""))

    with col_right:
        if "workflow_result" in st.session_state:
            res = st.session_state["workflow_result"]
            doc = res.get("doc")
            meta = res.get("meta", {})

            if doc:
                # ✅ **볼드 처리** 포함 HTML 렌더
                html_content = f"""
<div class="paper-sheet">
  <div class="stamp">직인생략</div>
  <div class="doc-header">{safe_inline_md_to_html(doc.get('title', '공 문 서'))}</div>
  <div class="doc-info">
    <span>문서번호: {safe_inline_md_to_html(meta.get('doc_num',''))}</span>
    <span>시행일자: {safe_inline_md_to_html(meta.get('today_str',''))}</span>
    <span>수신: {safe_inline_md_to_html(doc.get('receiver', '수신자 참조'))}</span>
  </div>
  <hr style="border: 1px solid black; margin-bottom: 30px;">
  <div class="doc-body">
"""
                paragraphs = doc.get("body_paragraphs", [])
                if isinstance(paragraphs, str):
                    paragraphs = [paragraphs]

                for p in paragraphs:
                    html_content += f"<p style='margin-bottom: 15px;'>{safe_inline_md_to_html(p)}</p>"

                html_content += f"""
  </div>
  <div class="doc-footer">{safe_inline_md_to_html(doc.get('department_head', '행정기관장'))}</div>
</div>
"""
                st.markdown(html_content, unsafe_allow_html=True)

                st.markdown("---")
                # ✅ expander 1번만 (내부에서 expander 쓰지 않음)
                with st.expander("💬 [후속 질문] 케이스 고정 챗봇 (최대 5회)", expanded=True):
                    render_followup_chat(
                        sb=sb,
                        res=res,
                        archive_row_id=st.session_state.get("archive_row_id"),
                    )
            else:
                st.warning("공문 생성 결과(doc)가 비어 있습니다. (모델 출력 실패 가능)")
        else:
            st.markdown(
                """<div style='text-align: center; padding: 100px; color: #aaa; background: white;
border-radius: 10px; border: 2px dashed #ddd;'>
<h3>📄 Document Preview</h3><p>왼쪽에서 업무를 지시하면<br>완성된 공문서가 여기에 나타납니다.</p></div>""",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
