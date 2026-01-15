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


def _strip_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return text


def _coerce_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def _normalize_receiver(receiver: str) -> str:
    r = (receiver or "").strip()
    r = re.sub(r"^\s*수신\s*[:：]\s*", "", r)
    r = re.sub(r"^\s*수신자\s*[:：]\s*", "", r)
    return r.strip() if r.strip() else "수신자 참조"


def normalize_doc(doc: Optional[dict], meta: dict, situation: str, legal_basis: str, strategy: str) -> dict:
    """
    ✅ 공문 조판이 '절대' 깨지지 않게 강제 보정.
    LLM JSON 실패(None/누락/형식오류)여도 최소 공문 구조 생성.
    """
    doc = doc if isinstance(doc, dict) else {}

    title = _coerce_str(doc.get("title")).strip() or "공 문 서"
    receiver = _normalize_receiver(_coerce_str(doc.get("receiver")).strip())

    body_paragraphs = doc.get("body_paragraphs")
    if isinstance(body_paragraphs, str):
        body_paragraphs = [body_paragraphs]
    if not isinstance(body_paragraphs, list):
        body_paragraphs = []

    # list 안에 None/숫자 등 섞여도 문자열화
    body_paragraphs = [(_coerce_str(p).strip()) for p in body_paragraphs if _coerce_str(p).strip()]

    department_head = _coerce_str(doc.get("department_head")).strip() or "행정기관장 OOO"

    # 본문이 비면 fallback 본문 자동 생성
    if not body_paragraphs:
        today_str = meta.get("today_str", datetime.now().strftime("%Y. %m. %d."))
        deadline_str = meta.get("deadline_str", (datetime.now() + timedelta(days=15)).strftime("%Y. %m. %d."))
        body_paragraphs = [
            "1. 귀하의 민원사항에 대하여 아래와 같이 검토 결과를 알려드립니다.",
            f"2. 본 건은 다음 법령을 근거로 처리합니다.\n{_strip_html(legal_basis)[:1200]}",
            "3. 처리 절차 및 주요 사항은 다음과 같습니다.",
            _strip_html(strategy)[:1200] if strategy else " - (처리 방향 요약이 비어 있습니다. 추가 확인이 필요합니다.)",
            f"4. 의견 제출 기한: {deadline_str}까지 (시행일: {today_str})",
            "5. 본 문서는 AI 초안이며, 최종 결재 전 담당자가 반드시 검토합니다.",
        ]

    return {
        "title": title,
        "receiver": receiver,
        "body_paragraphs": body_paragraphs,
        "department_head": department_head,
    }


# ==========================================
# 1) Configuration & Styles
# ==========================================
st.set_page_config(
    layout="wide",
    page_title="AI Bureau: The Legal Glass",
    page_icon="⚖️",
    initial_sidebar_state="collapsed",  # ✅ 사이드바 접고/펼 수 있게 (초기 접힘)
)

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
    .doc-body { font-size: 12pt; text-align: justify; white-space: pre-line; }
    .doc-footer { text-align: center; font-size: 20pt; font-weight: bold; margin-top: 80px; letter-spacing: 5px; }
    .stamp { position: absolute; bottom: 85px; right: 80px; border: 3px solid #cc0000; color: #cc0000; padding: 5px 10px; font-size: 14pt; font-weight: bold; transform: rotate(-15deg); opacity: 0.8; border-radius: 5px; }

    .agent-log { font-family: 'Consolas', monospace; font-size: 0.85rem; padding: 6px 12px; border-radius: 6px; margin-bottom: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    .log-legal { background-color: #eff6ff; color: #1e40af; border-left: 4px solid #3b82f6; }
    .log-search { background-color: #fff7ed; color: #c2410c; border-left: 4px solid #f97316; }
    .log-strat { background-color: #f5f3ff; color: #6d28d9; border-left: 4px solid #8b5cf6; }
    .log-calc { background-color: #f0fdf4; color: #166534; border-left: 4px solid #22c55e; }
    .log-draft { background-color: #fef2f2; color: #991b1b; border-left: 4px solid #ef4444; }
    .log-sys { background-color: #f3f4f6; color: #4b5563; border-left: 4px solid #9ca3af; }

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


# ==========================================
# 2) Supabase Auth + Archive
# ==========================================
def get_supabase():
    if not create_client:
        return None
    sb = st.secrets.get("supabase", {})
    url = sb.get("SUPABASE_URL")
    anon = sb.get("SUPABASE_ANON_KEY") or sb.get("SUPABASE_KEY")
    if not (url and anon):
        return None
    try:
        return create_client(url, anon)
    except Exception:
        return None


supabase = get_supabase()


def restore_supabase_session():
    if not supabase:
        return
    at = st.session_state.get("sb_access_token")
    rt = st.session_state.get("sb_refresh_token")
    if at and rt:
        try:
            supabase.auth.set_session(at, rt)
        except Exception:
            pass


def set_session_from_auth(res):
    session = getattr(res, "session", None)
    if session is None:
        session = (getattr(res, "data", None) or {}).get("session")

    if not session:
        return False

    access_token = getattr(session, "access_token", None) or (session.get("access_token") if isinstance(session, dict) else None)
    refresh_token = getattr(session, "refresh_token", None) or (session.get("refresh_token") if isinstance(session, dict) else None)
    if not access_token or not refresh_token:
        return False

    st.session_state["sb_access_token"] = access_token
    st.session_state["sb_refresh_token"] = refresh_token
    st.session_state["logged_in"] = True
    return True


def get_current_user_email() -> str:
    # 세션 상태 우선, 실패하면 supabase.auth.get_user()
    e = st.session_state.get("user_email") or ""
    if e:
        return e
    if not supabase:
        return ""
    try:
        u = supabase.auth.get_user()
        # 버전별 반환차이 대응
        user_obj = getattr(u, "user", None) or getattr(u, "data", None) or u
        email = getattr(user_obj, "email", None) or (user_obj.get("email") if isinstance(user_obj, dict) else None)
        return email or ""
    except Exception:
        return ""


def is_admin_email(email: str) -> bool:
    return (email or "").strip().lower() == ADMIN_EMAIL.lower()


def logout():
    if supabase:
        try:
            supabase.auth.sign_out()
        except Exception:
            pass
    for k in ["sb_access_token", "sb_refresh_token", "logged_in", "user_email", "signup_step", "pending_email"]:
        if k in st.session_state:
            del st.session_state[k]


class ArchiveService:
    def __init__(self, sb_client):
        self.sb = sb_client

    def is_ready(self) -> bool:
        return self.sb is not None

    def insert_case(self, payload: dict) -> Tuple[bool, str, Optional[str]]:
        """
        ✅ 로그인 상태에서만 저장 (RLS가 auth.uid 필요)
        """
        if not self.sb:
            return False, "DB 미연결", None
        try:
            resp = self.sb.table("work_archive").insert({
                "title": payload.get("title"),
                "situation": payload.get("situation"),
                "payload": payload,
                # user_id/user_email은 DB 트리거가 자동 세팅
            }).execute()
            inserted_id = None
            try:
                data = getattr(resp, "data", None) or []
                if isinstance(data, list) and data:
                    inserted_id = data[0].get("id")
            except Exception:
                inserted_id = None
            return True, "DB 저장 성공", inserted_id
        except Exception as e:
            return False, f"DB 저장 실패: {e}", None

    def update_case(self, case_id: str, payload: dict) -> Tuple[bool, str]:
        if not self.sb:
            return False, "DB 미연결"
        try:
            self.sb.table("work_archive").update({
                "title": payload.get("title"),
                "situation": payload.get("situation"),
                "payload": payload,
            }).eq("id", case_id).execute()
            return True, "DB 업데이트 성공"
        except Exception as e:
            return False, f"DB 업데이트 실패: {e}"

    def delete_case(self, case_id: str) -> Tuple[bool, str]:
        if not self.sb:
            return False, "DB 미연결"
        try:
            self.sb.table("work_archive").delete().eq("id", case_id).execute()
            return True, "삭제 완료"
        except Exception as e:
            return False, f"삭제 실패: {e}"

    def list_cases(self, limit: int = 80) -> List[dict]:
        if not self.sb:
            return []
        try:
            resp = (self.sb.table("work_archive")
                    .select("id, created_at, title, situation, user_email")
                    .order("created_at", desc=True)
                    .limit(limit)
                    .execute())
            return getattr(resp, "data", None) or []
        except Exception:
            return []

    def get_case(self, case_id: str) -> Optional[dict]:
        if not self.sb:
            return None
        try:
            resp = (self.sb.table("work_archive")
                    .select("*")
                    .eq("id", case_id)
                    .limit(1)
                    .execute())
            data = getattr(resp, "data", None) or []
            return data[0] if data else None
        except Exception:
            return None


archive = ArchiveService(supabase)

# restore session on every rerun
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user_email" not in st.session_state:
    st.session_state["user_email"] = ""
if "signup_step" not in st.session_state:
    st.session_state["signup_step"] = 0  # 0메일, 1OTP, 2비번설정
if "pending_email" not in st.session_state:
    st.session_state["pending_email"] = ""

restore_supabase_session()
if st.session_state.get("logged_in"):
    # 보강: 이메일 보정
    st.session_state["user_email"] = get_current_user_email() or st.session_state.get("user_email", "")


# ==========================================
# 3) Infrastructure Services (LLM/News/Law API)
# ==========================================
class LLMService:
    def __init__(self):
        g = st.secrets.get("general", {})
        self.gemini_key = g.get("GEMINI_API_KEY")
        self.groq_key = g.get("GROQ_API_KEY")

        self.gemini_models = [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
        ]

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
            return ""

    def generate_text(self, prompt: str) -> str:
        try:
            text, _ = self._try_gemini_text(prompt)
            if text:
                return text
        except Exception:
            pass

        if self.groq_client:
            t = self._generate_groq(prompt)
            if t:
                return t

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
        return {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
        }

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
                msg = f"✅ '{law_name}' 확인\n(MST 추출 실패)\n🔗 현행 원문: {current_link or '-'}"
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
# 5) Agents
# ==========================================
class LegalAgents:
    @staticmethod
    def researcher(situation: str) -> str:
        prompt_extract = f"""
상황: "{situation}"

위 민원 처리를 위해 법적 근거로 삼아야 할 핵심 대한민국 법령과 조문 번호를
**중요도 순으로 최대 3개까지** JSON 리스트로 추출하시오.

형식: [{{"law_name": "도로교통법", "article_num": 32}}, ...]
* 법령명은 정식 명칭 사용. 조문 번호 불명확하면 null.
"""
        extracted = llm_service.generate_json(prompt_extract)
        search_targets: List[Dict[str, Any]] = []

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
                content = "(국가법령정보센터에서 해당 조문을 찾지 못했습니다. 법령명이 정확한지 확인이 필요합니다.)"

            report_lines.append(f"{header}\n{content}\n")

        final_report = "\n".join(report_lines)

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

        return final_report

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
    def drafter(situation: str, legal_basis: str, meta_info: dict, strategy: str) -> Optional[dict]:
        prompt = f"""
당신은 행정기관의 베테랑 서기입니다. 아래 정보를 바탕으로 완결된 공문서를 작성하세요.

[입력]
- 민원: {situation}
- 법적 근거: {legal_basis}
- 시행일자: {meta_info['today_str']}
- 기한: {meta_info['deadline_str']} ({meta_info['days_added']}일)

[전략]
{strategy}

[출력은 JSON만]
{{
  "title": "문서 제목",
  "receiver": "수신자(예: OOO 시장)",
  "body_paragraphs": ["문단1", "문단2", "..."],
  "department_head": "발신(예: 교통행정과장 OOO)"
}}

[원칙]
1) 본문에 법 조항 인용 필수
2) 구조: 경위 -> 법적 근거 -> 처분/조치 내용 -> 이의제기 절차
3) 개인정보 마스킹('OOO')
4) receiver에는 '수신:' 같은 접두어 쓰지 말고 수신자만 작성
"""
        data = llm_service.generate_json(prompt)
        return data if isinstance(data, dict) else None


# ==========================================
# 6) Workflow + Lawbot pack
# ==========================================
def build_lawbot_pack(res: dict) -> dict:
    situation = (res.get("situation") or "").strip()
    prompt = f"""
상황: "{situation}"
국가법령정보센터 법령 AI(Lawbot/검색)에 넣을 핵심 키워드 3~6개를 JSON 배열로만 출력.
예: ["무단방치", "자동차관리법", "공시송달", "직권말소", "시행규칙", "서식"]
"""
    kws = llm_service.generate_json(prompt) or []
    if not isinstance(kws, list):
        kws = []
    kws = [str(x).strip() for x in kws if str(x).strip()]

    query_text = (situation[:60] + " " + " ".join(kws[:6])).strip()
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
    add_log("📜 법적 근거 확인 완료", "legal")

    add_log("🟩 네이버 검색 엔진 가동...", "search")
    try:
        search_results = search_service.search_precedents(user_input)
    except Exception:
        search_results = "검색 모듈 미연결 (건너뜀)"

    add_log("🧠 Phase 2: AI 주무관이 업무 처리 방향 수립...", "strat")
    strategy = LegalAgents.strategist(user_input, legal_basis, search_results)

    add_log("📅 Phase 3: 기한 산정 및 공문서 작성...", "calc")
    meta_info = LegalAgents.clerk(user_input, legal_basis)

    add_log("✍️ Phase 4: 공문서 조판...", "draft")
    doc_data = LegalAgents.drafter(user_input, legal_basis, meta_info, strategy)

    time.sleep(0.2)
    log_placeholder.empty()

    # ✅ 공문 조판 안정화(절대 깨지지 않음)
    fixed_doc = normalize_doc(doc_data, meta_info, user_input, legal_basis, strategy)

    res = {
        "title": (user_input[:60] if user_input else "케이스"),
        "situation": user_input,
        "doc": fixed_doc,
        "meta": meta_info,
        "law": legal_basis,
        "search": search_results,
        "strategy": strategy,
    }
    res["lawbot_pack"] = build_lawbot_pack(res)
    return res


# ==========================================
# 7) Follow-up Chat (NO nested expanders)
# ==========================================
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


def needs_tool_call(user_msg: str) -> dict:
    t = (user_msg or "").lower()
    law_triggers = ["근거", "조문", "법령", "몇 조", "원문", "현행", "추가 조항", "다른 조문", "전문", "절차법", "행정절차", "규칙", "서식", "시행규칙"]
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


def render_followup_chat(res: dict):
    # 세션 초기화
    st.session_state.setdefault("case_id", None)
    st.session_state.setdefault("followup_count", 0)
    st.session_state.setdefault("followup_messages", [])
    st.session_state.setdefault("followup_extra_context", "")
    st.session_state.setdefault("archive_case_id", None)

    current_case_id = (res.get("meta") or {}).get("doc_num", "") or "case"
    if st.session_state["case_id"] != current_case_id:
        st.session_state["case_id"] = current_case_id
        st.session_state["followup_count"] = 0
        st.session_state["followup_messages"] = []
        st.session_state["followup_extra_context"] = ""

    remain = max(0, MAX_FOLLOWUP_Q - st.session_state["followup_count"])
    st.info(f"후속 질문 가능 횟수: **{remain}/{MAX_FOLLOWUP_Q}**")

    # ✅ Lawbot 실행 버튼 (이름 변경)
    pack = res.get("lawbot_pack", {}) or {}
    qb = (pack.get("query_text") or "").strip()
    if qb:
        st.caption("법령/규칙/서식까지 더 파고들기:")
        st.link_button("🔎 법령 AI · Lawbot 실행 (법령/규칙/서식 찾기)", make_lawbot_url(qb), use_container_width=True)

    # ✅ Lawbot 결과 붙여넣기(중첩 expander 금지 → toggle + container)
    paste_mode = st.toggle("📎 Lawbot 결과 가져오기(복붙) — 법령/규칙/서식 발췌를 케이스에 반영", value=False)
    if paste_mode:
        lawbot_paste = st.text_area(
            "Lawbot에서 찾은 근거/서식 텍스트를 붙여넣기",
            height=140,
            placeholder="예) 시행규칙 별지서식, 지침 문구, 규정 조항 발췌 등",
        )
        if st.button("✅ 붙여넣은 근거를 케이스에 반영", use_container_width=True):
            if lawbot_paste.strip():
                extra_ctx = st.session_state.get("followup_extra_context", "")
                extra_ctx += "\n\n[Lawbot 수집 근거(사용자 복붙)]\n" + lawbot_paste.strip()
                st.session_state["followup_extra_context"] = extra_ctx
                st.success("반영 완료")
            else:
                st.warning("붙여넣기 내용이 비었습니다.")

    if remain == 0:
        st.warning("후속 질문 한도(5회)를 모두 사용했습니다. (추가 질문 불가)")
        return

    # 기존 대화 렌더
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

        if plan.get("need_law") and plan.get("law_name"):
            art = plan.get("article_num", 0)
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

    # ✅ 후속까지 payload에 반영 후 DB 업데이트(로그인 + 저장된 케이스 id가 있을 때)
    res["followup_payload"] = {
        "count": st.session_state["followup_count"],
        "messages": st.session_state["followup_messages"],
        "extra_context": st.session_state.get("followup_extra_context", ""),
    }

    case_id = st.session_state.get("archive_case_id")
    if st.session_state.get("logged_in") and case_id:
        ok, msg = archive.update_case(case_id, res)
        if not ok:
            st.caption(msg)


# ==========================================
# 8) Sidebar: 로그인/회원가입/히스토리
# ==========================================
def sidebar_auth_history():
    st.sidebar.title("🔐 로그인 / 회원가입 / 히스토리")

    if not supabase:
        st.sidebar.error("Supabase 설정이 없습니다. (secrets.toml 확인)")
        return

    logged_in = bool(st.session_state.get("logged_in"))
    user_email = (st.session_state.get("user_email") or "").strip()

    menu = st.sidebar.radio("메뉴", ["로그인", "회원가입", "히스토리"], index=0)

    # ---------- 회원가입(OTP→비번) ----------
    if menu == "회원가입":
        st.sidebar.subheader("🧾 회원가입 (@korea.kr만)")

        if st.session_state["signup_step"] == 0:
            email = st.sidebar.text_input("메일 주소", value=st.session_state.get("pending_email", ""))
            if st.sidebar.button("인증번호 발송", use_container_width=True):
                if not email.endswith("@korea.kr"):
                    st.sidebar.error("❌ @korea.kr 메일만 가입 허용")
                else:
                    try:
                        supabase.auth.sign_in_with_otp({
                            "email": email,
                            "options": {"should_create_user": True}
                        })
                        st.session_state["pending_email"] = email
                        st.session_state["signup_step"] = 1
                        st.sidebar.success("✅ 인증번호를 메일로 보냈습니다. 코드를 입력하세요.")
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"발송 실패: {e}")

        elif st.session_state["signup_step"] == 1:
            email = st.session_state.get("pending_email", "")
            st.sidebar.write(f"대상: **{email}**")
            otp = st.sidebar.text_input("인증번호(OTP)", placeholder="메일로 받은 숫자코드")
            c1, c2 = st.sidebar.columns(2)
            with c1:
                if st.button("확인", use_container_width=True):
                    try:
                        res = supabase.auth.verify_otp({
                            "email": email,
                            "token": otp,
                            "type": "email"
                        })
                        ok = set_session_from_auth(res)
                        if ok:
                            st.session_state["user_email"] = email
                            st.session_state["signup_step"] = 2
                            st.sidebar.success("✅ 인증 완료! 비밀번호를 설정하세요.")
                            st.rerun()
                        else:
                            st.sidebar.error("세션 생성 실패(OTP 확인은 됐으나 session이 없음).")
                    except Exception as e:
                        st.sidebar.error(f"인증 실패: {e}")
            with c2:
                if st.button("처음부터", use_container_width=True):
                    st.session_state["signup_step"] = 0
                    st.session_state["pending_email"] = ""
                    st.rerun()

        elif st.session_state["signup_step"] == 2:
            email = st.session_state.get("pending_email") or st.session_state.get("user_email")
            st.sidebar.write(f"대상: **{email}**")
            pw1 = st.sidebar.text_input("비밀번호", type="password")
            pw2 = st.sidebar.text_input("비밀번호 확인", type="password")
            if st.sidebar.button("비밀번호 설정 완료", use_container_width=True):
                if not pw1 or pw1 != pw2:
                    st.sidebar.error("비밀번호가 비었거나 서로 다릅니다.")
                else:
                    try:
                        supabase.auth.update_user({"password": pw1})
                        st.sidebar.success("✅ 비밀번호 설정 완료! 이제 로그인 메뉴에서 메일+비번으로 로그인하세요.")
                        st.session_state["signup_step"] = 0
                        st.session_state["pending_email"] = ""
                    except Exception as e:
                        st.sidebar.error(f"설정 실패: {e}")

    # ---------- 로그인 ----------
    if menu == "로그인":
        st.sidebar.subheader("🔑 로그인 (메일 + 비밀번호)")

        if logged_in:
            st.sidebar.success(f"접속 중: {user_email}")
            if is_admin_email(user_email):
                st.sidebar.warning("👑 관리자 권한: 전체 기록 CRUD 가능")
            if st.sidebar.button("로그아웃", use_container_width=True):
                logout()
                st.rerun()
        else:
            email = st.sidebar.text_input("아이디(이메일)")
            password = st.sidebar.text_input("비밀번호", type="password")
            if st.sidebar.button("로그인", use_container_width=True):
                try:
                    res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    ok = set_session_from_auth(res)
                    if ok:
                        st.session_state["user_email"] = email
                        st.sidebar.success("✅ 로그인 성공")
                        st.rerun()
                    else:
                        st.sidebar.error("로그인 세션 설정 실패")
                except Exception as e:
                    st.sidebar.error(f"로그인 실패: {e}")

    # ---------- 히스토리 ----------
    if menu == "히스토리":
        st.sidebar.subheader("📚 히스토리 (짠-복원)")

        if not logged_in:
            st.sidebar.info("로그인 후 이용 가능합니다.")
            return

        if is_admin_email(user_email):
            st.sidebar.warning("👑 관리자: 전체 기록이 보입니다.")
        else:
            st.sidebar.caption("본인 기록만 보입니다(RLS).")

        items = archive.list_cases(limit=120)
        if not items:
            st.sidebar.info("저장된 기록이 없습니다.")
            return

        labels = []
        id_map = {}
        for it in items:
            created = it.get("created_at", "")
            title = it.get("title") or (it.get("situation", "")[:40] if it.get("situation") else "기록")
            owner = it.get("user_email", "")
            label = f"{created} | {title}"
            if is_admin_email(user_email):
                label += f" | {owner}"
            labels.append(label)
            id_map[label] = it.get("id")

        pick = st.sidebar.selectbox("불러올 기록", labels)
        case_id = id_map.get(pick)

        c1, c2 = st.sidebar.columns(2)
        with c1:
            if st.button("불러오기", use_container_width=True):
                row = archive.get_case(case_id) if case_id else None
                if row and row.get("payload"):
                    st.session_state["workflow_result"] = row["payload"]
                    st.session_state["archive_case_id"] = row.get("id")
                    st.sidebar.success("✅ 복원 완료(메인 화면에 표시)")
                    st.rerun()
                else:
                    st.sidebar.error("불러오기 실패")

        with c2:
            if st.button("삭제", use_container_width=True):
                if case_id:
                    ok, msg = archive.delete_case(case_id)
                    if ok:
                        st.sidebar.success(msg)
                        # 현재 복원된 케이스가 삭제됐으면 리셋
                        if st.session_state.get("archive_case_id") == case_id:
                            st.session_state.pop("workflow_result", None)
                            st.session_state["archive_case_id"] = None
                        st.rerun()
                    else:
                        st.sidebar.error(msg)

        # 관리자 편집(간단)
        if is_admin_email(user_email):
            st.sidebar.markdown("---")
            st.sidebar.caption("관리자 수정(제목/상황만)")
            row = archive.get_case(case_id) if case_id else None
            if row and row.get("payload"):
                payload = row["payload"]
                new_title = st.sidebar.text_input("제목", value=payload.get("title", ""))
                new_sit = st.sidebar.text_area("상황", value=payload.get("situation", ""), height=90)
                if st.sidebar.button("수정 저장", use_container_width=True):
                    payload["title"] = new_title
                    payload["situation"] = new_sit
                    ok, msg = archive.update_case(case_id, payload)
                    if ok:
                        st.sidebar.success(msg)
                        st.rerun()
                    else:
                        st.sidebar.error(msg)


sidebar_auth_history()


# ==========================================
# 9) Main UI
# ==========================================
def render_law_box(raw_law: str):
    cleaned = (raw_law or "").replace("&lt;", "<").replace("&gt;", ">")
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", cleaned)
    cleaned = re.sub(
        r'\[([^\]]+)\]\(([^)]+)\)',
        r'<a href="\2" target="_blank" style="color:#2563eb; text-decoration:none; font-weight:700;">\1</a>',
        cleaned,
    )
    cleaned = cleaned.replace("---", "<br><br>").replace("\n", "<br>")
    st.markdown(
        f"""
        <div style="
            height: 300px;
            overflow-y: auto;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
            background: #f8fafc;
            font-family: 'Pretendard', sans-serif;
            font-size: 0.9rem;
            line-height: 1.6;
            color: #334155;
        ">
        {cleaned}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_news_box(raw_news: str):
    news_body = (raw_news or "").replace("# ", "").replace("## ", "")
    news_body = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", news_body)
    news_html = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        r'<a href="\2" target="_blank" style="color:#2563eb; text-decoration:none; font-weight:600;">\1</a>',
        news_body
    )
    news_html = news_html.replace("\n", "<br>")
    st.markdown(
        f"""
        <div style="
            height: 300px;
            overflow-y: auto;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #dbeafe;
            background: #eff6ff;
            font-family: 'Pretendard', sans-serif;
            font-size: 0.9rem;
            line-height: 1.6;
            color: #1e3a8a;
        ">
        {news_html}
        </div>
        """,
        unsafe_allow_html=True
    )


def main():
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.title("🏢 AI 행정관 Pro 충주시청")
        st.caption("문의 kim0395kk@korea.kr \n 세계최초 행정 Govable AI 에이젼트")
        st.markdown("---")

        # 상태 표시
        ai_ok = "✅AI" if llm_service.is_available() else "❌AI"
        law_ok = "✅LAW" if bool(st.secrets.get("general", {}).get("LAW_API_ID")) else "❌LAW"
        nv_ok = "✅NEWS" if bool(st.secrets.get("general", {}).get("NAVER_CLIENT_ID")) else "❌NEWS"
        sb_ok = "✅SUPABASE" if supabase else "❌SUPABASE"
        login_ok = "✅LOGIN" if st.session_state.get("logged_in") else "❌LOGIN"
        st.caption(f"상태: {ai_ok} | {law_ok} | {nv_ok} | {sb_ok} | {login_ok}")

        st.markdown("### 🗣️ 업무 지시")
        user_input = st.text_area(
            "업무 내용",
            height=160,
            placeholder="예시\n- 상황: (무슨 일 / 어디 / 언제 / 증거 유무...)\n- 의도: (확인하고 싶은 쟁점: 요건/절차/근거...)\n- 요청: (원하는 결과물: 공문 종류/회신/사전통지 등)",
            label_visibility="collapsed",
        )

        st.warning("⚠️ 비공개 문서 부분복사/내부검토 민감정보(성명·연락처·주소·차량번호 등) 입력 금지")

        if st.button("⚡ 스마트 분석 시작", type="primary", use_container_width=True):
            if not user_input:
                st.warning("내용을 입력해주세요.")
            else:
                try:
                    with st.spinner("AI 에이전트 팀이 협업 중입니다..."):
                        res = run_workflow(user_input)

                        # ✅ 로그인 상태면 자동 저장
                        if st.session_state.get("logged_in") and supabase:
                            ok, msg, inserted_id = archive.insert_case(res)
                            res["save_msg"] = msg
                            st.session_state["archive_case_id"] = inserted_id
                        else:
                            res["save_msg"] = "로그인하지 않아 DB 저장을 건너뜀"

                        st.session_state["workflow_result"] = res
                        st.rerun()
                except Exception as e:
                    st.error(f"시스템 오류 발생: {e}")

        if "workflow_result" in st.session_state:
            res = st.session_state["workflow_result"]
            st.markdown("---")

            save_msg = res.get("save_msg", "")
            if "성공" in save_msg:
                st.success(f"✅ {save_msg}")
            else:
                st.info(f"ℹ️ {save_msg}")

            # ✅ Lawbot 버튼명 변경
            pack = res.get("lawbot_pack", {}) or {}
            qb = (pack.get("query_text") or "").strip()
            if qb:
                st.link_button("🔎 법령 AI · Lawbot 실행 (법령/규칙/서식 찾기)", make_lawbot_url(qb), use_container_width=True)

            with st.expander("✅ [검토] 법령 및 유사 사례 확인", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**📜 적용 법령 (법령명 클릭 시 현행 원문 새창)**")
                    render_law_box(res.get("law", ""))

                with col2:
                    st.markdown("**🟩 관련 뉴스/사례**")
                    render_news_box(res.get("search", ""))

            with st.expander("🧭 [방향] 업무 처리 가이드라인", expanded=True):
                st.markdown(res.get("strategy", ""))

    with col_right:
        if "workflow_result" in st.session_state:
            res = st.session_state["workflow_result"]
            doc = res.get("doc")
            meta = res.get("meta", {})

            if doc:
                html_content = f"""
<div class="paper-sheet">
  <div class="stamp">직인생략</div>
  <div class="doc-header">{_escape(_coerce_str(doc.get('title', '공 문 서')))}</div>
  <div class="doc-info">
    <span>문서번호: {_escape(_coerce_str(meta.get('doc_num','')))}</span>
    <span>시행일자: {_escape(_coerce_str(meta.get('today_str','')))}</span>
    <span>수신: {_escape(_coerce_str(doc.get('receiver', '수신자 참조')))}</span>
  </div>
  <hr style="border: 1px solid black; margin-bottom: 30px;">
  <div class="doc-body">
"""
                paragraphs = doc.get("body_paragraphs", [])
                if isinstance(paragraphs, str):
                    paragraphs = [paragraphs]

                for p in paragraphs:
                    p = _coerce_str(p).strip()
                    if p:
                        html_content += f"<p style='margin-bottom: 15px;'>{_escape(p)}</p>"

                html_content += f"""
  </div>
  <div class="doc-footer">{_escape(_coerce_str(doc.get('department_head', '행정기관장')))}</div>
</div>
"""
                st.markdown(html_content, unsafe_allow_html=True)

                st.markdown("---")
                # ✅ expander는 여기 하나만! (내부에서 expander 만들지 않음)
                with st.expander("💬 [후속 질문] 케이스 고정 챗봇 (최대 5회)", expanded=True):
                    render_followup_chat(res)

            else:
                st.warning("공문 생성 결과(doc)가 비어 있습니다. (모델 출력 실패 가능)")

        else:
            st.markdown(
                """<div style='text-align: center; padding: 100px; color: #aaa; background: white; border-radius: 10px; border: 2px dashed #ddd;'>
<h3>📄 Document Preview</h3><p>업무를 지시하면<br>완성된 공문서가 여기에 나타납니다.</p></div>""",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
