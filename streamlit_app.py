# streamlit_app.py
# -*- coding: utf-8 -*-
import json
import re
import time
import uuid
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

# =========================================================
# 0) SETTINGS
# =========================================================
APP_VERSION = "2026-01-15-full"
MAX_FOLLOWUP_Q = 5

ADMIN_EMAIL = "kim0395kk@korea.kr"

LAW_BOT_SEARCH_URL = "https://www.law.go.kr/LSW/ais/searchList.do?query="


# =========================================================
# 1) HELPERS
# =========================================================
def make_lawbot_url(query: str) -> str:
    return LAW_BOT_SEARCH_URL + urllib.parse.quote((query or "").strip())


def shorten_one_line(text: str, max_len: int = 28) -> str:
    s = re.sub(r"\s+", " ", (text or "").strip())
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


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


def strip_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return text


def ensure_anon_session_id() -> str:
    if "anon_session_id" not in st.session_state:
        st.session_state.anon_session_id = str(uuid.uuid4())
    return st.session_state.anon_session_id


def is_admin_user(email: str) -> bool:
    return (email or "").strip().lower() == ADMIN_EMAIL.lower()


def md_bold_to_html_safe(text: str) -> str:
    """
    ✅ 공문서 내부 **볼드**가 HTML에서 실제 <b>로 보이게.
    - 사용자 입력/AI 출력은 모두 escape 처리
    - **...** 패턴만 <b>로 변환
    """
    s = text or ""
    out = []
    pos = 0
    for m in re.finditer(r"\*\*(.+?)\*\*", s):
        out.append(_escape(s[pos:m.start()]))
        out.append(f"<b>{_escape(m.group(1))}</b>")
        pos = m.end()
    out.append(_escape(s[pos:]))
    html = "".join(out)
    html = html.replace("\n", "<br>")
    return html


# =========================================================
# 2) STYLES
# =========================================================
st.set_page_config(layout="wide", page_title="AI Bureau: The Legal Glass", page_icon="⚖️")

st.markdown(
    """
<style>
    .stApp { background-color: #f3f4f6; }

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
    }

    .doc-header { text-align: center; font-size: 22pt; font-weight: 900; margin-bottom: 30px; letter-spacing: 2px; }
    .doc-info { display: flex; justify-content: space-between; font-size: 11pt; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; gap:10px; flex-wrap:wrap; }
    .doc-body { font-size: 12pt; text-align: justify; white-space: normal; }
    .doc-footer { text-align: center; font-size: 20pt; font-weight: bold; margin-top: 80px; letter-spacing: 5px; }
    .stamp { position: absolute; bottom: 85px; right: 80px; border: 3px solid #cc0000; color: #cc0000; padding: 5px 10px; font-size: 14pt; font-weight: bold; transform: rotate(-15deg); opacity: 0.8; border-radius: 5px; }

    /* Lawbot 버튼(파란 배경 + 화이트 강조) */
    .lawbot-btn {
        display: inline-block;
        width: 100%;
        padding: 12px 14px;
        border-radius: 12px;
        text-decoration: none !important;
        font-weight: 900;
        letter-spacing: 0.2px;
        text-align: center;
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 55%, #60a5fa 100%);
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.25);
        box-shadow: 0 10px 22px rgba(37, 99, 235, 0.25);
    }
    .lawbot-btn:hover {
        filter: brightness(1.03);
        transform: translateY(-1px);
    }
    .lawbot-sub {
        font-size: 0.82rem;
        opacity: 0.92;
        margin-top: 6px;
        display: block;
        color: rgba(255,255,255,0.92) !important;
        font-weight: 700;
    }

    /* Sidebar history: ChatGPT 느낌 */
    div[data-testid="stSidebar"] button[kind="secondary"]{
        width:100%;
        text-align:left !important;
        justify-content:flex-start !important;
        padding: 0.55rem 0.65rem !important;
        border-radius: 10px !important;
        border: 1px solid #e5e7eb !important;
        background: white !important;
        color: #111827 !important;
        font-weight: 650 !important;
    }
    div[data-testid="stSidebar"] button[kind="secondary"]:hover{
        background: #f3f4f6 !important;
    }

    /* Streamlit Cloud 상단 Fork/GitHub 숨김 */
    header [data-testid="stToolbar"] { display: none !important; }
    header [data-testid="stDecoration"] { display: none !important; }
    header { height: 0px !important; }
    footer { display: none !important; }
    div[data-testid="stStatusWidget"] { display: none !important; }
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# 3) SERVICES
# =========================================================
def get_secret(path1: str, path2: str = "") -> Optional[str]:
    """
    secrets.toml이
      [supabase]
      SUPABASE_URL=...
    혹은 최상단에 있는 경우까지 모두 대응
    """
    try:
        if path2:
            return st.secrets.get(path1, {}).get(path2)
        return st.secrets.get(path1)
    except Exception:
        return None


def get_supabase():
    if "sb" in st.session_state and st.session_state.sb is not None:
        return st.session_state.sb

    if not create_client:
        st.session_state.sb = None
        return None

    url = get_secret("supabase", "SUPABASE_URL") or get_secret("SUPABASE_URL")
    key = (
        get_secret("supabase", "SUPABASE_ANON_KEY")
        or get_secret("supabase", "SUPABASE_KEY")
        or get_secret("SUPABASE_ANON_KEY")
        or get_secret("SUPABASE_KEY")
    )

    if not (url and key):
        st.session_state.sb = None
        return None

    st.session_state.sb = create_client(url, key)
    return st.session_state.sb


def get_auth_user(sb):
    try:
        u = sb.auth.get_user()
        # supabase-py 응답 구조가 다를 수 있어 방어적으로 처리
        if isinstance(u, dict):
            user = u.get("user") or u
            return user
        if hasattr(u, "user"):
            return u.user
        return u
    except Exception:
        return None


def touch_session(sb):
    """
    ✅ 접속 세션 heartbeat (동시접속자 추정용)
    - anon_session_id 기준 upsert
    """
    if not sb:
        return

    anon_id = ensure_anon_session_id()
    user_email = st.session_state.get("user_email") if st.session_state.get("logged_in") else None
    user_id = None
    user = get_auth_user(sb)
    if user and isinstance(user, dict):
        user_id = user.get("id")

    payload = {
        "session_id": anon_id,
        "last_seen": datetime.utcnow().isoformat() + "Z",
        "user_id": user_id,
        "user_email": user_email,
        "meta": {"app_ver": APP_VERSION},
    }
    try:
        sb.table("app_sessions").upsert(payload, on_conflict="session_id").execute()
    except Exception:
        pass


def log_event(sb, event_type: str, archive_id: Optional[str] = None, meta: Optional[dict] = None):
    if not sb:
        return
    anon_id = ensure_anon_session_id()
    user_email = st.session_state.get("user_email") if st.session_state.get("logged_in") else None
    user_id = None
    user = get_auth_user(sb)
    if user and isinstance(user, dict):
        user_id = user.get("id")

    row = {
        "event_type": event_type,
        "archive_id": archive_id,
        "user_id": user_id,
        "user_email": user_email,
        "anon_session_id": anon_id,
        "meta": meta or {},
    }
    try:
        sb.table("app_events").insert(row).execute()
    except Exception:
        pass


class LLMService:
    """
    1) Gemini
    2) Groq
    """
    def __init__(self):
        g = st.secrets.get("general", {})
        self.gemini_key = g.get("GEMINI_API_KEY")
        self.groq_key = g.get("GROQ_API_KEY")

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

    def _generate_groq(self, prompt: str) -> str:
        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return (completion.choices[0].message.content or "").strip()
        except Exception:
            return "System Error"

    def generate_text(self, prompt: str) -> str:
        try:
            text, _ = self._try_gemini_text(prompt)
            if text:
                return text
        except Exception:
            pass

        if self.groq_client:
            return self._generate_groq(prompt)

        return "시스템 오류: AI 모델 연결 실패"

    def generate_json(self, prompt: str) -> Optional[Any]:
        json_prompt = prompt + "\n\n반드시 JSON만 출력. 다른 텍스트 금지."
        text = self.generate_text(json_prompt)
        return _safe_json_loads(text)


class SearchService:
    def __init__(self):
        g = st.secrets.get("general", {})
        self.client_id = g.get("NAVER_CLIENT_ID")
        self.client_secret = g.get("NAVER_CLIENT_SECRET")
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
        self.api_id = st.secrets.get("general", {}).get("LAW_API_ID")
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
                        target_text = f"[{law_name} 제{current_num}조 전문]\n" + (jo_content_tag.text or "").strip()
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


llm_service = LLMService()
search_service = SearchService()
law_api_service = LawOfficialService()


# =========================================================
# 4) WORKFLOW AGENTS
# =========================================================
class LegalAgents:
    @staticmethod
    def researcher(situation: str) -> str:
        prompt_extract = f"""
상황: "{situation}"

위 민원 처리를 위해 법적 근거로 삼아야 할 핵심 대한민국 법령과 조문 번호를
중요도 순으로 최대 3개까지 JSON 리스트로 추출하시오.

형식: [{{"law_name": "도로교통법", "article_num": 32}}, ...]
* 법령명은 정식 명칭 사용. 조문 번호 불명확하면 null.
"""
        extracted = llm_service.generate_json(prompt_extract)
        targets: List[Dict[str, Any]] = []

        if isinstance(extracted, list):
            targets = extracted
        elif isinstance(extracted, dict):
            targets = [extracted]

        if not targets:
            targets = [{"law_name": "행정절차법", "article_num": None}]

        out = [f"🔍 **AI가 식별한 핵심 법령 ({len(targets)}건)**", "---"]
        for i, item in enumerate(targets):
            law_name = (item.get("law_name") or "관련법령").strip()
            art = item.get("article_num", None)

            if isinstance(art, str):
                m = re.search(r"\d+", art)
                art = int(m.group(0)) if m else None
            if isinstance(art, (int, float)):
                art = int(art)
            else:
                art = None

            law_text, link = law_api_service.get_law_text(law_name, art, return_link=True)
            ok = link is not None and "오류" not in (law_text or "") and "없습니다" not in (law_text or "")

            if ok and link:
                title = f"[{law_name}]({link})"
                out.append(f"✅ **{i+1}. {title} {('제'+str(art)+'조') if art else ''} (확인됨)**\n{law_text}\n")
            else:
                out.append(f"⚠️ **{i+1}. {law_name} {('제'+str(art)+'조') if art else ''} (API 조회 불확실)**\n(법령명/조문 확인 필요)\n")

        return "\n".join(out)

    @staticmethod
    def strategist(situation: str, legal_basis: str, search_results: str) -> str:
        prompt = f"""
당신은 행정 업무 베테랑 '주무관'입니다.

[민원 상황]: {situation}
[확보된 법적 근거]:
{legal_basis}

[유사 사례/기사]:
{search_results}

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
    def drafter(situation: str, legal_basis: str, meta: dict, strategy: str) -> Optional[dict]:
        prompt = f"""
당신은 행정기관의 베테랑 서기입니다. 아래 정보를 바탕으로 완결된 공문서를 JSON으로 작성하세요.

[입력]
- 민원: {situation}
- 법적 근거: {legal_basis}
- 시행일자: {meta['today_str']}
- 기한: {meta['deadline_str']} ({meta['days_added']}일)

[전략]
{strategy}

[원칙]
1) 본문에 법 조항 인용 필수
2) 구조: 경위 -> 법적 근거 -> 처분 내용 -> 이의제기 절차
3) 개인정보 마스킹('OOO')

[출력 JSON 형식]
{{
  "title": "제목",
  "receiver": "수신",
  "body_paragraphs": ["문단1", "문단2", "..."],
  "department_head": "OOO과장"
}}
JSON만 출력.
"""
        data = llm_service.generate_json(prompt)
        if isinstance(data, dict) and data.get("title") and data.get("body_paragraphs"):
            return data
        return None


def build_lawbot_pack(situation: str) -> dict:
    prompt = f"""
상황: "{situation}"
국가법령정보센터 법령 AI(Lawbot) 검색창에 넣을 핵심 키워드 3~6개를 JSON 배열로만 출력.
예: ["무단방치", "자동차관리법", "공시송달", "직권말소"]
"""
    kws = llm_service.generate_json(prompt) or []
    if not isinstance(kws, list):
        kws = []
    kws = [str(x).strip() for x in kws if str(x).strip()]
    query_text = (situation[:60] + " " + " ".join(kws[:6])).strip()
    query_text = re.sub(r"\s+", " ", query_text)
    return {"core_keywords": kws[:10], "query_text": query_text[:180], "url": make_lawbot_url(query_text[:180])}


def run_workflow(user_input: str) -> dict:
    log = st.empty()
    logs: List[str] = []

    def add(msg: str):
        logs.append(f"- {msg}")
        log.markdown("\n".join(logs))

    add("Phase 1) 법령 조회")
    law = LegalAgents.researcher(user_input)
    add("Phase 2) 뉴스/사례 조회")
    news = search_service.search_precedents(user_input)
    add("Phase 3) 처리방향 수립")
    strategy = LegalAgents.strategist(user_input, law, news)
    add("Phase 4) 공문 조판")
    meta = LegalAgents.clerk(user_input, law)
    doc = LegalAgents.drafter(user_input, law, meta, strategy)

    log.empty()

    res = {
        "situation": user_input,
        "law": law,
        "search": news,
        "strategy": strategy,
        "meta": meta,
        "doc": doc,
        "lawbot_pack": build_lawbot_pack(user_input),
        "followups": [],  # 복원 시 여기에 합쳐서 넣음
    }
    return res


# =========================================================
# 5) DB OPS
# =========================================================
def db_insert_archive(sb, prompt: str, payload: dict) -> Optional[str]:
    anon_id = ensure_anon_session_id()
    user = get_auth_user(sb)
    user_id = user.get("id") if isinstance(user, dict) else None
    user_email = st.session_state.get("user_email") if st.session_state.get("logged_in") else None

    row = {
        "prompt": prompt,
        "payload": payload,
        "anon_session_id": anon_id,
        "user_id": user_id,
        "user_email": user_email,
        "client_meta": {"app_ver": APP_VERSION},
    }
    try:
        resp = sb.table("work_archive").insert(row).execute()
        # supabase-py: resp.data[0]["id"]
        if hasattr(resp, "data") and resp.data and isinstance(resp.data, list):
            return resp.data[0].get("id")
    except Exception as e:
        st.warning(f"ℹ️ DB 저장 실패: {e}")
    return None


def db_fetch_history(sb, scope: str = "me", limit: int = 80) -> List[dict]:
    try:
        q = sb.table("work_archive").select("id,prompt,created_at,user_email,anon_session_id").order("created_at", desc=True).limit(limit)
        if scope == "all":
            resp = q.execute()
        else:
            # RLS가 자동으로 내 것만 반환
            resp = q.execute()
        return resp.data or []
    except Exception:
        return []


def db_fetch_payload(sb, archive_id: str) -> Optional[dict]:
    try:
        resp = sb.table("work_archive").select("id,prompt,payload,created_at,user_email,anon_session_id").eq("id", archive_id).limit(1).execute()
        if resp.data:
            return resp.data[0]
    except Exception:
        return None
    return None


def db_fetch_followups(sb, archive_id: str) -> List[dict]:
    try:
        resp = (
            sb.table("work_followups")
            .select("turn,role,content,created_at")
            .eq("archive_id", archive_id)
            .order("turn", desc=False)
            .execute()
        )
        return resp.data or []
    except Exception:
        return []


def db_insert_followup(sb, archive_id: str, turn: int, role: str, content: str):
    anon_id = ensure_anon_session_id()
    user = get_auth_user(sb)
    user_id = user.get("id") if isinstance(user, dict) else None
    user_email = st.session_state.get("user_email") if st.session_state.get("logged_in") else None

    row = {
        "archive_id": archive_id,
        "turn": turn,
        "role": role,
        "content": content,
        "user_id": user_id,
        "user_email": user_email,
        "anon_session_id": anon_id,
    }
    try:
        sb.table("work_followups").insert(row).execute()
    except Exception:
        pass


# =========================================================
# 6) SIDEBAR AUTH UI (컴팩트)
# =========================================================
def sidebar_auth(sb):
    st.sidebar.markdown("## 🔐 로그인")

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_email" not in st.session_state:
        st.session_state.user_email = ""
    if "admin_mode" not in st.session_state:
        st.session_state.admin_mode = False

    # logged in view
    if st.session_state.logged_in:
        email = st.session_state.user_email
        st.sidebar.success(f"✅ {email}")

        if is_admin_user(email):
            st.sidebar.toggle("관리자모드 켜기", key="admin_mode")

        if st.sidebar.button("로그아웃", use_container_width=True):
            try:
                sb.auth.sign_out()
            except Exception:
                pass
            st.session_state.logged_in = False
            st.session_state.user_email = ""
            st.session_state.admin_mode = False
            log_event(sb, "logout")
            st.rerun()
        return

    # not logged in -> minimal menu
    menu = st.sidebar.radio("메뉴", ["로그인", "회원가입", "비밀번호 찾기"], horizontal=True)

    # state machines
    if "signup_stage" not in st.session_state:
        st.session_state.signup_stage = 1
    if "reset_stage" not in st.session_state:
        st.session_state.reset_stage = 1

    # LOGIN
    if menu == "로그인":
        email = st.sidebar.text_input("메일", placeholder="kim0395kk@korea.kr", key="login_email")
        pw = st.sidebar.text_input("비밀번호", type="password", key="login_pw")

        if st.sidebar.button("로그인", use_container_width=True):
            try:
                sb.auth.sign_in_with_password({"email": email, "password": pw})
                st.session_state.logged_in = True
                st.session_state.user_email = (email or "").strip()
                log_event(sb, "login_success")
                st.rerun()
            except Exception:
                st.sidebar.error("로그인 실패: 메일/비밀번호 확인")

    # SIGNUP (email -> otp -> set password)
    elif menu == "회원가입":
        if st.session_state.signup_stage == 1:
            email = st.sidebar.text_input("메일(@korea.kr)", placeholder="name@korea.kr", key="su_email")
            if st.sidebar.button("코리아 메일로 인증번호 발송", use_container_width=True):
                if not (email or "").endswith("@korea.kr"):
                    st.sidebar.error("❌ @korea.kr 메일만 가입 가능")
                else:
                    try:
                        # Email OTP (should_create_user=True)
                        sb.auth.sign_in_with_otp({"email": email, "options": {"should_create_user": True}})
                        st.session_state.pending_email = email.strip()
                        st.session_state.signup_stage = 2
                        log_event(sb, "signup_otp_sent", meta={"email": email.strip()})
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"발송 실패: {e}")

        elif st.session_state.signup_stage == 2:
            email = st.session_state.get("pending_email", "")
            st.sidebar.caption(f"발송 대상: {email}")
            code = st.sidebar.text_input("인증번호(OTP/토큰)", key="su_code")
            new_pw = st.sidebar.text_input("비밀번호 설정", type="password", key="su_pw")
            new_pw2 = st.sidebar.text_input("비밀번호 확인", type="password", key="su_pw2")

            if st.sidebar.button("인증 + 비밀번호 설정 완료", use_container_width=True):
                if not new_pw or new_pw != new_pw2:
                    st.sidebar.error("비밀번호가 일치하지 않습니다.")
                else:
                    ok = False
                    # verify otp: signup 먼저 시도, 실패 시 magiclink fallback
                    for t in ["signup", "magiclink"]:
                        try:
                            sb.auth.verify_otp({"email": email, "token": code, "type": t})
                            ok = True
                            break
                        except Exception:
                            pass

                    if not ok:
                        st.sidebar.error("인증번호 검증 실패")
                        return

                    try:
                        sb.auth.update_user({"password": new_pw})
                    except Exception as e:
                        st.sidebar.error(f"비밀번호 설정 실패: {e}")
                        return

                    st.session_state.logged_in = True
                    st.session_state.user_email = email.strip()
                    st.session_state.signup_stage = 1
                    log_event(sb, "signup_done")
                    st.rerun()

    # RESET PW (email -> otp -> set new password)
    else:
        if st.session_state.reset_stage == 1:
            email = st.sidebar.text_input("메일", placeholder="name@korea.kr", key="rp_email")
            if st.sidebar.button("메일로 인증번호 발송", use_container_width=True):
                try:
                    # OTP 로그인으로 세션 확보 후 update_user로 비번 변경하는 방식(코드/토큰 기반)
                    sb.auth.sign_in_with_otp({"email": email, "options": {"should_create_user": False}})
                    st.session_state.reset_email = email.strip()
                    st.session_state.reset_stage = 2
                    log_event(sb, "reset_otp_sent", meta={"email": email.strip()})
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"발송 실패: {e}")

        elif st.session_state.reset_stage == 2:
            email = st.session_state.get("reset_email", "")
            st.sidebar.caption(f"대상: {email}")
            code = st.sidebar.text_input("인증번호(OTP/토큰)", key="rp_code")
            new_pw = st.sidebar.text_input("새 비밀번호", type="password", key="rp_pw")
            new_pw2 = st.sidebar.text_input("새 비밀번호 확인", type="password", key="rp_pw2")

            if st.sidebar.button("인증 + 새 비밀번호 설정", use_container_width=True):
                if not new_pw or new_pw != new_pw2:
                    st.sidebar.error("비밀번호가 일치하지 않습니다.")
                    return

                ok = False
                for t in ["magiclink", "signup"]:
                    try:
                        sb.auth.verify_otp({"email": email, "token": code, "type": t})
                        ok = True
                        break
                    except Exception:
                        pass

                if not ok:
                    st.sidebar.error("인증번호 검증 실패")
                    return

                try:
                    sb.auth.update_user({"password": new_pw})
                except Exception as e:
                    st.sidebar.error(f"비밀번호 변경 실패: {e}")
                    return

                # 로그인 상태로 전환
                st.session_state.logged_in = True
                st.session_state.user_email = email.strip()
                st.session_state.reset_stage = 1
                log_event(sb, "reset_done")
                st.rerun()


# =========================================================
# 7) SIDEBAR HISTORY (프롬프트만, 클릭 즉시 복원)
# =========================================================
def restore_archive(sb, row_id: str):
    row = db_fetch_payload(sb, row_id)
    if not row:
        st.sidebar.error("복원 실패(권한/RLS 또는 데이터 없음)")
        return
    payload = row.get("payload") or {}
    followups = db_fetch_followups(sb, row_id)

    # followups -> chat messages
    msgs = [{"role": f.get("role"), "content": f.get("content")} for f in followups]
    payload["followups"] = msgs

    st.session_state["workflow_result"] = payload
    st.session_state["current_archive_id"] = row_id
    st.session_state["followup_messages"] = msgs
    st.session_state["selected_history_id"] = row_id
    log_event(sb, "restore_archive", archive_id=row_id)
    st.rerun()


def render_history_list(sb):
    email = st.session_state.get("user_email", "")
    admin_all = is_admin_user(email) and st.session_state.get("admin_mode", False)

    # 비로그인은 기록 표시 X (삽입만 됨)
    if not st.session_state.get("logged_in") and not admin_all:
        st.sidebar.caption("비로그인: 기록은 저장되지만 조회/복원은 불가")
        return

    scope = "all" if admin_all else "me"
    hist = db_fetch_history(sb, scope=scope, limit=120)
    if not hist:
        st.sidebar.caption("저장된 기록이 없습니다.")
        return

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🕘 기록")
    q = st.sidebar.text_input("검색", placeholder="프롬프트 검색", label_visibility="collapsed", key="hist_q")
    if q:
        ql = q.strip().lower()
        hist = [r for r in hist if ql in (r.get("prompt", "").lower())]

    if "selected_history_id" not in st.session_state:
        st.session_state.selected_history_id = None

    for row in hist:
        rid = row["id"]
        label = shorten_one_line(row.get("prompt", ""), 28) or "（프롬프트 없음）"
        prefix = "● " if st.session_state.selected_history_id == rid else "  "
        if st.sidebar.button(prefix + label, key=f"hist_{rid}", use_container_width=True, type="secondary"):
            restore_archive(sb, rid)


# =========================================================
# 8) ADMIN DASHBOARD
# =========================================================
def render_admin_dashboard(sb):
    st.subheader("🛡️ 관리자 대시보드")

    # sessions: last 5 minutes
    now = datetime.utcnow()
    five_min_ago = (now - timedelta(minutes=5)).isoformat() + "Z"
    today_00 = datetime(now.year, now.month, now.day).isoformat() + "Z"

    sessions = []
    events = []
    archives = []

    try:
        sessions = (
            sb.table("app_sessions")
            .select("session_id,user_email,first_seen,last_seen")
            .order("last_seen", desc=True)
            .limit(2000)
            .execute()
            .data
            or []
        )
    except Exception:
        sessions = []

    try:
        events = (
            sb.table("app_events")
            .select("created_at,event_type,user_email,anon_session_id,archive_id,meta")
            .order("created_at", desc=True)
            .limit(300)
            .execute()
            .data
            or []
        )
    except Exception:
        events = []

    try:
        archives = (
            sb.table("work_archive")
            .select("id,created_at,user_email,anon_session_id,prompt")
            .order("created_at", desc=True)
            .limit(3000)
            .execute()
            .data
            or []
        )
    except Exception:
        archives = []

    active = [s for s in sessions if (s.get("last_seen") or "") >= five_min_ago]
    visitors_today = {s.get("session_id") for s in sessions if (s.get("first_seen") or "") >= today_00}

    col1, col2, col3 = st.columns(3)
    col1.metric("동시 접속(5분)", f"{len(active)}")
    col2.metric("오늘 방문자(세션)", f"{len(visitors_today)}")
    col3.metric("최근 기록(3k)", f"{len(archives)}")

    # user usage
    st.markdown("### 👥 사용자별 사용량(최근 3,000건 기준)")
    stats: Dict[str, int] = {}
    for a in archives:
        who = a.get("user_email") or "(anonymous)"
        stats[who] = stats.get(who, 0) + 1
    top = sorted(stats.items(), key=lambda x: x[1], reverse=True)[:30]
    st.write(top)

    # recent events
    st.markdown("### 🧾 최근 이벤트 로그(300)")
    for ev in events[:60]:
        st.write(f"- {ev.get('created_at')} | {ev.get('event_type')} | {ev.get('user_email') or '(anon)'}")

    st.markdown("### 🧹 관리자 작업(선택 기록)")
    target = st.text_input("관리 대상 archive_id(uuid)", placeholder="복사해 넣기")
    if target:
        row = db_fetch_payload(sb, target)
        if not row:
            st.error("해당 id를 찾을 수 없습니다.")
        else:
            st.success("레코드 로드됨")
            new_prompt = st.text_area("프롬프트 수정", value=row.get("prompt", ""), height=120)
            c1, c2 = st.columns(2)
            if c1.button("프롬프트 업데이트"):
                sb.table("work_archive").update({"prompt": new_prompt}).eq("id", target).execute()
                log_event(sb, "admin_update_prompt", archive_id=target)
                st.rerun()
            if c2.button("레코드 삭제"):
                sb.table("work_archive").delete().eq("id", target).execute()
                log_event(sb, "admin_delete_archive", archive_id=target)
                st.rerun()


# =========================================================
# 9) UI RENDER
# =========================================================
def render_lawbot_button(url: str):
    st.markdown(
        f"""
<a class="lawbot-btn" href="{_escape(url)}" target="_blank">
  🤖 법령 AI (Lawbot) 실행 — 법령·규칙·서식 더 찾기
  <span class="lawbot-sub">클릭하면 검색창에 키워드가 들어간 상태로 새창이 열립니다</span>
</a>
""",
        unsafe_allow_html=True,
    )


def main():
    sb = get_supabase()
    ensure_anon_session_id()

    if sb:
        touch_session(sb)
        if "boot_logged" not in st.session_state:
            st.session_state.boot_logged = True
            log_event(sb, "app_open", meta={"ver": APP_VERSION})

    # sidebar
    if sb:
        sidebar_auth(sb)
        render_history_list(sb)
    else:
        st.sidebar.error("Supabase 연결 정보(secrets)가 없습니다.")
        st.sidebar.caption("SUPABASE_URL / SUPABASE_ANON_KEY 필요")

    # admin page switch
    email = st.session_state.get("user_email", "")
    is_admin = is_admin_user(email)
    admin_mode = bool(st.session_state.get("admin_mode", False))

    page = "업무"
    if is_admin:
        # 관리자에게만 보이기
        page = st.sidebar.selectbox("페이지", ["업무", "관리자 대시보드"], index=0)

    if is_admin and admin_mode and page == "관리자 대시보드":
        st.title("🏢 AI 행정관 Pro — 관리자")
        if not sb:
            st.error("Supabase 연결 필요")
            return
        render_admin_dashboard(sb)
        return

    # main content
    st.title("🏢 AI 행정관 Pro 충주시청")
    st.caption("문의 kim0395kk@korea.kr  |  세계최초 행정 Govable AI 에이전트")

    # 상태
    ai_ok = "✅AI" if llm_service.is_available() else "❌AI"
    law_ok = "✅LAW" if bool(st.secrets.get("general", {}).get("LAW_API_ID")) else "❌LAW"
    nv_ok = "✅NEWS" if bool(st.secrets.get("general", {}).get("NAVER_CLIENT_ID")) else "❌NEWS"
    db_ok = "✅DB" if sb else "❌DB"
    st.caption(f"상태: {ai_ok} | {law_ok} | {nv_ok} | {db_ok} | ver {APP_VERSION}")

    col_left, col_right = st.columns([1, 1.15], gap="large")

    with col_left:
        st.markdown("### 🗣️ 업무 지시")
        user_input = st.text_area(
            "업무 내용",
            height=160,
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

                    # ✅ DB에는 항상 insert (비로그인도 저장됨)
                    archive_id = None
                    if sb:
                        archive_id = db_insert_archive(sb, user_input, res)
                        if archive_id:
                            st.session_state.current_archive_id = archive_id
                            log_event(sb, "workflow_run", archive_id=archive_id, meta={"prompt_len": len(user_input)})

                    res["archive_id"] = archive_id
                    st.session_state.workflow_result = res
                    st.session_state.followup_messages = []  # reset

        # results left
        if "workflow_result" in st.session_state:
            res = st.session_state.workflow_result
            pack = res.get("lawbot_pack") or {}
            if pack.get("url"):
                render_lawbot_button(pack["url"])

            tabs = st.tabs(["📜 법령/사례", "🧭 처리 가이드"])
            with tabs[0]:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**📜 적용 법령 (법령명 클릭 시 현행 원문 새창)**")
                    raw = res.get("law", "")
                    cleaned = raw.replace("&lt;", "<").replace("&gt;", ">")
                    cleaned = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", cleaned)
                    cleaned = re.sub(
                        r'\[([^\]]+)\]\(([^)]+)\)',
                        r'<a href="\2" target="_blank" style="color:#2563eb; text-decoration:none; font-weight:800;">\1</a>',
                        cleaned,
                    )
                    cleaned = cleaned.replace("---", "<br><br>").replace("\n", "<br>")
                    st.markdown(
                        f"<div style='height:320px;overflow-y:auto;padding:14px;border-radius:10px;border:1px solid #e5e7eb;background:#f8fafc;'>{cleaned}</div>",
                        unsafe_allow_html=True,
                    )

                with c2:
                    st.markdown("**🟩 관련 뉴스/사례**")
                    raw_news = res.get("search", "")
                    news_body = raw_news.replace("# ", "").replace("## ", "")
                    news_body = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", news_body)
                    news_html = re.sub(
                        r"\[([^\]]+)\]\(([^)]+)\)",
                        r'<a href="\2" target="_blank" style="color:#2563eb; text-decoration:none; font-weight:700;">\1</a>',
                        news_body,
                    ).replace("\n", "<br>")
                    st.markdown(
                        f"<div style='height:320px;overflow-y:auto;padding:14px;border-radius:10px;border:1px solid #dbeafe;background:#eff6ff;'>{news_html}</div>",
                        unsafe_allow_html=True,
                    )

            with tabs[1]:
                st.markdown(res.get("strategy", ""))

    with col_right:
        if "workflow_result" not in st.session_state:
            st.markdown(
                "<div style='text-align:center;padding:120px;color:#aaa;background:white;border-radius:12px;border:2px dashed #ddd;'>"
                "<h3>📄 Document Preview</h3><p>왼쪽에서 업무를 지시하면<br>완성된 공문서가 여기에 나타납니다.</p></div>",
                unsafe_allow_html=True,
            )
            return

        res = st.session_state.workflow_result
        doc = res.get("doc")
        meta = res.get("meta") or {}
        archive_id = res.get("archive_id") or st.session_state.get("current_archive_id")

        tab_doc, tab_chat = st.tabs(["📄 공문서", "💬 후속 질문(복원됨)"])

        with tab_doc:
            if not doc:
                st.warning("공문 생성 결과(doc)가 비어 있습니다. (모델 JSON 출력 실패 가능)")
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

                # ✅ bold(**...**)가 실제 bold로 보이도록
                for p in paragraphs:
                    html += f"<p style='margin-bottom: 14px;'>{md_bold_to_html_safe(p)}</p>"

                html += f"""
  </div>
  <div class="doc-footer">{_escape(doc.get('department_head', '행정기관장'))}</div>
</div>
"""
                st.markdown(html, unsafe_allow_html=True)

        with tab_chat:
            if not archive_id:
                st.info("저장된 archive_id가 없습니다. (DB 연결/저장 실패)")
                return

            # 복원된 메시지 렌더
            if "followup_messages" not in st.session_state:
                st.session_state.followup_messages = res.get("followups", []) or []

            # 남은 횟수
            used = len([m for m in st.session_state.followup_messages if m.get("role") == "user"])
            remain = max(0, MAX_FOLLOWUP_Q - used)
            st.info(f"후속 질문 가능 횟수: **{remain}/{MAX_FOLLOWUP_Q}**")

            # Lawbot quick launch(항상)
            pack = res.get("lawbot_pack") or {}
            if pack.get("url"):
                render_lawbot_button(pack["url"])

            # chat history
            for m in st.session_state.followup_messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

            if remain == 0:
                st.warning("후속 질문 한도(5회)를 모두 사용했습니다.")
                return

            q = st.chat_input("후속 질문 (최대 5회)")
            if not q:
                return

            # user message
            st.session_state.followup_messages.append({"role": "user", "content": q})
            turn = len([m for m in st.session_state.followup_messages if m["role"] == "user"])
            db_insert_followup(sb, archive_id, turn=turn*2-1, role="user", content=q)
            log_event(sb, "followup_user", archive_id=archive_id, meta={"turn": turn})

            with st.chat_message("user"):
                st.markdown(q)

            # assistant answer (케이스 고정)
            case_context = f"""
[케이스]
상황: {res.get('situation','')}

법령:
{strip_html(res.get('law',''))[:2500]}

뉴스/사례:
{strip_html(res.get('search',''))[:1800]}

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
- 모르면 모른다고 하고, 필요한 추가 근거/뉴스는 Lawbot(법령 AI)로 더 찾도록 안내한다.
- 서론 없이 실무형으로.
"""
            with st.chat_message("assistant"):
                with st.spinner("후속 답변 생성 중..."):
                    ans = llm_service.generate_text(prompt)
                    st.markdown(ans)

            st.session_state.followup_messages.append({"role": "assistant", "content": ans})
            db_insert_followup(sb, archive_id, turn=turn*2, role="assistant", content=ans)
            log_event(sb, "followup_assistant", archive_id=archive_id, meta={"turn": turn})

            # (선택) work_archive payload 자체는 건드리지 않아도 됨 (followups는 테이블로 복원)
            st.rerun()


if __name__ == "__main__":
    main()
