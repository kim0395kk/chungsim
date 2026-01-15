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
# Optional deps
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
APP_VERSION = "2026-01-15-agentboost"
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
    s = text or ""
    out = []
    pos = 0
    for m in re.finditer(r"\*\*(.+?)\*\*", s):
        out.append(_escape(s[pos:m.start()]))
        out.append(f"<b>{_escape(m.group(1))}</b>")
        pos = m.end()
    out.append(_escape(s[pos:]))
    html = "".join(out).replace("\n", "<br>")
    return html

def mask_sensitive(text: str) -> str:
    """개인정보/식별정보 간단 마스킹(입력 보호용)"""
    if not text:
        return ""
    t = text
    # phone
    t = re.sub(r"\b0\d{1,2}-\d{3,4}-\d{4}\b", "0**-****-****", t)
    # rrn
    t = re.sub(r"\b\d{6}-\d{7}\b", "******-*******", t)
    # car plate (rough)
    t = re.sub(r"\b\d{2,3}[가-힣]\d{4}\b", "***(차량번호)", t)
    return t

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
    .lawbot-btn:hover { filter: brightness(1.03); transform: translateY(-1px); }
    .lawbot-sub { font-size: 0.82rem; opacity: 0.92; margin-top: 6px; display: block; color: rgba(255,255,255,0.92) !important; font-weight: 700; }

    /* Sidebar history */
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
    div[data-testid="stSidebar"] button[kind="secondary"]:hover{ background: #f3f4f6 !important; }

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
    try:
        if path2:
            return st.secrets.get(path1, {}).get(path2)
        return st.secrets.get(path1)
    except Exception:
        return None

def get_general_secret(key: str) -> Optional[str]:
    # [general] 또는 최상단 모두 지원
    return (st.secrets.get("general", {}) or {}).get(key) or st.secrets.get(key)

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
        if isinstance(u, dict):
            return u.get("user") or u
        if hasattr(u, "user"):
            return u.user
        return u
    except Exception:
        return None

def touch_session(sb):
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
    def __init__(self):
        self.gemini_key = get_general_secret("GEMINI_API_KEY")
        self.groq_key = get_general_secret("GROQ_API_KEY")
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
        strict = prompt + "\n\n반드시 JSON만 출력. 다른 텍스트 금지."
        text = self.generate_text(strict)
        return _safe_json_loads(text)

llm_service = LLMService()

class SearchService:
    def __init__(self):
        self.client_id = get_general_secret("NAVER_CLIENT_ID")
        self.client_secret = get_general_secret("NAVER_CLIENT_SECRET")
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

search_service = SearchService()

class LawOfficialService:
    def __init__(self):
        self.api_id = get_general_secret("LAW_API_ID")  # OC
        self.base_url = "https://www.law.go.kr/DRF/lawSearch.do"
        self.service_url = "https://www.law.go.kr/DRF/lawService.do"

    def _make_current_link(self, mst_id: str) -> Optional[str]:
        if not self.api_id or not mst_id:
            return None
        return f"https://www.law.go.kr/DRF/lawService.do?OC={self.api_id}&target=law&MST={mst_id}&type=HTML"

    def ai_search(self, query: str, top_k: int = 6) -> List[dict]:
        """law.go.kr aiSearch로 후보 법령명을 더 넓게 가져옴"""
        if not requests or not self.api_id or not query:
            return []
        try:
            params = {"OC": self.api_id, "target": "aiSearch", "type": "XML", "query": query, "display": top_k}
            r = requests.get(self.base_url, params=params, timeout=8)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            out = []
            # aiSearch 응답은 케이스별로 달라서, lawName 계열을 최대한 줍기
            for node in root.findall(".//law"):
                name = (node.findtext("법령명") or node.findtext("lawName") or "").strip()
                if name:
                    out.append({"law_name": name})
            # fallback: 텍스트에서 법령명 태그 탐색
            if not out:
                for tag in ["lawName", "법령명", "lawNm"]:
                    for node in root.findall(f".//{tag}"):
                        nm = (node.text or "").strip()
                        if nm:
                            out.append({"law_name": nm})
            # uniq
            seen = set()
            uniq = []
            for x in out:
                nm = x["law_name"]
                if nm not in seen:
                    seen.add(nm)
                    uniq.append(x)
            return uniq[:top_k]
        except Exception:
            return []

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
                want = re.sub(r"\D", "", str(article_num))
                for article in root_detail.findall(".//조문단위"):
                    jo_num_tag = article.find("조문번호")
                    jo_content_tag = article.find("조문내용")
                    if jo_num_tag is None or jo_content_tag is None:
                        continue
                    got = re.sub(r"\D", "", (jo_num_tag.text or "").strip())
                    if want and got == want:
                        target_text = f"[{law_name} 제{got}조 전문]\n" + (jo_content_tag.text or "").strip()
                        for hang in article.findall(".//항"):
                            hang_content = hang.find("항내용")
                            if hang_content is not None:
                                target_text += f"\n  - {(hang_content.text or '').strip()}"
                        return (target_text, current_link) if return_link else target_text

            msg = f"✅ '{law_name}' 확인\n(조문 자동 추출 실패/미지정)\n🔗 현행 원문: {current_link or '-'}"
            return (msg, current_link) if return_link else msg

        except Exception as e:
            msg = f"상세 법령 파싱 실패: {e}"
            return (msg, current_link) if return_link else msg

law_api_service = LawOfficialService()

# =========================================================
# 4) AGENTS (BOOSTED)
# =========================================================
class CaseAnalyzer:
    @staticmethod
    def analyze(situation: str) -> dict:
        s = mask_sensitive(situation)
        prompt = f"""
너는 '민원/업무 케이스 분석관'이다.

[입력]
{s}

[출력 JSON]
{{
  "case_type": "예: 무단방치/번호판훼손/불법주정차/건설기계/기타",
  "core_issue": ["핵심 쟁점 3~6개"],
  "required_facts": ["추가로 필요한 사실확인 질문 5개"],
  "required_evidence": ["필요 증빙 5개"],
  "risk_flags": ["절차상 리스크 3개(예: 통지 누락, 증거 부족...)"],
  "recommended_next_action": ["즉시 다음 행동 3개"]
}}
JSON만.
"""
        data = llm_service.generate_json(prompt)
        if isinstance(data, dict) and data.get("case_type"):
            return data
        # fallback (heuristic)
        t = "기타"
        if "무단방치" in situation: t = "무단방치"
        if "번호판" in situation: t = "번호판훼손"
        return {
            "case_type": t,
            "core_issue": ["사실관계 확정", "증빙 확보", "절차적 정당성 확보"],
            "required_facts": ["장소/시간?", "증빙(사진/영상)?", "소유자 특정 가능?", "반복/상습 여부?", "요청사항(처분/계도/회신)?"],
            "required_evidence": ["현장 사진", "위치/시간 기록", "신고내용 원문", "소유자 확인 자료", "조치/통지 기록"],
            "risk_flags": ["통지/의견제출 기회 누락", "증거 부족", "법적 근거 불명확"],
            "recommended_next_action": ["증빙 정리", "소유자/점유자 확인", "절차 플로우 확정"],
        }

class ProcedureAgent:
    @staticmethod
    def plan(situation: str, legal_basis_summary: str, analysis: dict) -> dict:
        prompt = f"""
너는 '행정 절차 플래너'이다.

[상황]
{situation}

[분석]
{json.dumps(analysis, ensure_ascii=False)}

[법적 근거(요약)]
{legal_basis_summary}

[출력 JSON]
{{
  "timeline": [
    {{"step": 1, "name": "단계명", "goal": "목표", "actions": ["행동1","행동2"], "records": ["기록/증빙"], "legal_note": "근거/유의"}},
    ...
  ],
  "checklist": ["담당자가 체크할 항목 10개"],
  "templates": ["필요 서식/문서 이름 5개"]
}}
JSON만.
"""
        data = llm_service.generate_json(prompt)
        if isinstance(data, dict) and data.get("timeline"):
            return data
        return {
            "timeline": [
                {"step": 1, "name": "사실확인", "goal": "사실관계 확정", "actions": ["현장 확인", "증빙 확보"], "records": ["사진/위치/시간"], "legal_note": "기록이 절차 정당성 핵심"},
                {"step": 2, "name": "대상 특정", "goal": "소유자/점유자 특정", "actions": ["등록정보 조회", "연락/안내"], "records": ["조회 로그", "통화/안내 기록"], "legal_note": "통지/연락 시도 기록"},
                {"step": 3, "name": "통지/계고", "goal": "자진 조치 유도", "actions": ["계고/안내", "기한 부여"], "records": ["통지문", "발송/수령 증빙"], "legal_note": "행정절차상 통지 누락 주의"},
                {"step": 4, "name": "불이행 시 조치", "goal": "강제/처분 검토", "actions": ["불이행 확인", "처분/강제 조치"], "records": ["확인서", "처분문"], "legal_note": "처분 사유/근거 명확화"},
            ],
            "checklist": ["증빙 확보", "법령 근거 확인", "통지/의견제출 기회", "문서번호/기한", "기록 남김"],
            "templates": ["회신 공문", "계고/통지", "의견제출 안내", "공시송달 공고", "처분서"],
        }

class ObjectionAgent:
    @staticmethod
    def build(situation: str, strategy: str) -> List[dict]:
        prompt = f"""
너는 '민원 반발 대응 코치'이다.

[상황]
{situation}

[처리방향]
{strategy}

[출력 JSON 배열]
[
  {{"objection":"반발/항의 문장(실제 톤)", "response":"담당자 답변(짧고 단호)", "record_point":"기록/증빙 포인트"}},
  ...
]
최대 7개. JSON만.
"""
        data = llm_service.generate_json(prompt)
        if isinstance(data, list) and data:
            out = []
            for x in data[:7]:
                if isinstance(x, dict) and x.get("objection") and x.get("response"):
                    out.append(x)
            if out:
                return out
        return [
            {"objection": "왜 나만 단속하냐", "response": "동일 기준으로 확인되면 모두 조치합니다. 현재 신고/확인된 건부터 절차대로 처리합니다.", "record_point": "단속 기준/확인 기록"},
            {"objection": "법적 근거 있냐", "response": "관련 법령 및 조문 근거로 안내드리며, 문서에 근거를 명시해 드립니다.", "record_point": "법령 링크/조문 캡처"},
            {"objection": "그건 내 잘못 아니다", "response": "사실관계 확인 후 책임 범위를 판단합니다. 이의가 있으면 의견제출로 제출해 주세요.", "record_point": "의견제출 안내/접수 기록"},
        ]

class LegalAgents:
    @staticmethod
    def researcher(situation: str, analysis: dict) -> dict:
        """
        강화 포인트:
        - LLM 추출 + aiSearch 후보 보강
        - 각 법령은 링크/조문텍스트/신뢰도 함께
        """
        s = mask_sensitive(situation)
        prompt_extract = f"""
상황: "{s}"
분석: {json.dumps(analysis, ensure_ascii=False)}

이 민원 처리를 위해 법적 근거로 삼을 핵심 법령/조문을
중요도 순으로 최대 5개 JSON 배열로 추출하시오.

형식:
[
 {{"law_name":"정식 법령명","article_num": 26, "why":"왜 필요한지 한 줄"}},
 ...
]
조문 번호 불명확하면 null. JSON만.
"""
        extracted = llm_service.generate_json(prompt_extract)
        targets: List[Dict[str, Any]] = []
        if isinstance(extracted, list):
            targets = extracted
        elif isinstance(extracted, dict):
            targets = [extracted]
        targets = [t for t in targets if isinstance(t, dict) and t.get("law_name")]

        # aiSearch로 법령 후보 보강(키워드 기반)
        kw = " ".join((analysis.get("core_issue") or [])[:4]) or situation[:30]
        extra = law_api_service.ai_search(kw, top_k=6)
        for e in extra:
            nm = e.get("law_name")
            if nm and all((nm != t.get("law_name")) for t in targets):
                targets.append({"law_name": nm, "article_num": None, "why": "aiSearch 후보"})

        # 상한
        targets = targets[:6] if targets else [{"law_name": "행정절차법", "article_num": None, "why": "절차 정당성 기본"}]

        items = []
        for item in targets:
            law_name = (item.get("law_name") or "").strip()
            why = (item.get("why") or "").strip()
            art = item.get("article_num", None)
            if isinstance(art, str):
                m = re.search(r"\d+", art)
                art = int(m.group(0)) if m else None
            if isinstance(art, (int, float)):
                art = int(art)
            else:
                art = None

            law_text, link = law_api_service.get_law_text(law_name, art, return_link=True)
            ok = (link is not None) and ("오류" not in (law_text or "")) and ("없습니다" not in (law_text or ""))
            items.append({
                "law_name": law_name,
                "article_num": art,
                "why": why,
                "ok": bool(ok),
                "link": link,
                "excerpt": (law_text or "")[:1400]
            })

        # 출력용 markdown도 함께
        out_md = [f"🔍 **핵심 법령 근거 ({len(items)}건)**", "---"]
        for i, it in enumerate(items, 1):
            name = it["law_name"]
            art_txt = f" 제{it['article_num']}조" if it["article_num"] else ""
            if it["link"]:
                out_md.append(f"✅ **{i}. [{name}]({it['link']}){art_txt}**  \n- 사유: {it['why']}\n- 발췌:\n{it['excerpt']}\n")
            else:
                out_md.append(f"⚠️ **{i}. {name}{art_txt}**  \n- 사유: {it['why']}\n- 발췌:\n{it['excerpt']}\n")

        return {"items": items, "markdown": "\n".join(out_md)}

    @staticmethod
    def strategist(situation: str, legal_basis_md: str, search_results: str, analysis: dict) -> str:
        prompt = f"""
당신은 행정 업무 베테랑 '주무관'이다.

[민원 상황]
{situation}

[케이스 분석]
{json.dumps(analysis, ensure_ascii=False)}

[법적 근거]
{legal_basis_md}

[유사 사례/기사]
{search_results}

서론(인사말/공감) 금지.
1) 처리 방향(한 줄 결론 + 근거)
2) 핵심 주의사항(절차/증거/기한)
3) 예상 반발 3개 + 대응 원칙
4) 담당자 체크리스트(8개)
"""
        return llm_service.generate_text(prompt)

    @staticmethod
    def clerk() -> dict:
        today = datetime.now()
        return {
            "today_str": today.strftime("%Y. %m. %d."),
            "doc_num": f"행정-{today.strftime('%Y')}-{int(time.time())%1000:03d}호",
        }

    @staticmethod
    def drafter(situation: str, legal_basis_md: str, meta: dict, strategy: str, procedure: dict, objections: List[dict]) -> dict:
        schema = """
{
  "title": "제목",
  "receiver": "수신",
  "body_paragraphs": ["문단1", "문단2", "..."],
  "department_head": "OOO과장"
}
""".strip()

        prompt = f"""
당신은 행정기관의 베테랑 서기이다. 아래 정보를 바탕으로 완결된 공문서를 JSON으로 작성하라.

[입력]
- 민원: {situation}
- 시행일자: {meta.get('today_str')}
- 문서번호: {meta.get('doc_num')}

[법령 근거(필수 인용)]
{legal_basis_md}

[처리방향]
{strategy}

[절차 플랜(반영)]
{json.dumps(procedure, ensure_ascii=False)}

[예상 반발(반영)]
{json.dumps(objections, ensure_ascii=False)}

[원칙]
- 본문에 법 조항/근거를 문장으로 인용할 것
- 구조: 경위 -> 법적 근거 -> 조치/안내 -> 이의제기/문의
- 개인정보는 OOO로 마스킹
- 문단 내에 **1** 같은 번호는 **볼드**로 표시해도 됨(마크다운 허용)

[출력 JSON 스키마]
{schema}

JSON만 출력.
"""
        # 1차
        data = llm_service.generate_json(prompt)
        if isinstance(data, dict) and data.get("title") and data.get("body_paragraphs"):
            return data

        # 2차 재시도(더 강제)
        retry = f"""
방금 출력이 스키마를 만족하지 않았다.
아래 스키마를 정확히 만족하는 JSON만 다시 출력하라.

스키마:
{schema}

(다른 텍스트 금지)
"""
        data2 = llm_service.generate_json(prompt + "\n\n" + retry)
        if isinstance(data2, dict) and data2.get("title") and data2.get("body_paragraphs"):
            return data2

        # fallback
        return {
            "title": "민원 처리 결과 회신(안)",
            "receiver": "수신자 참조",
            "body_paragraphs": [
                "**1**. 경위",
                f"- 민원 요지: {mask_sensitive(situation)}",
                "",
                "**2**. 법적 근거",
                "- 관련 법령 및 조문 근거에 따라 절차를 진행합니다.",
                "",
                "**3**. 조치 내용",
                "- 사실 확인 및 필요 절차를 단계적으로 이행 예정입니다.",
                "",
                "**4**. 이의제기/문의",
                "- 추가 의견이 있는 경우 의견제출 절차로 제출 바랍니다."
            ],
            "department_head": "OOO과장"
        }

def build_lawbot_pack(situation: str, analysis: dict) -> dict:
    prompt = f"""
상황: "{mask_sensitive(situation)}"
분석: {json.dumps(analysis, ensure_ascii=False)}
국가법령정보센터 Lawbot 검색창에 넣을 핵심 키워드 3~7개를 JSON 배열로만 출력.
예: ["무단방치","자동차관리법","공시송달","직권말소"]
"""
    kws = llm_service.generate_json(prompt) or []
    if not isinstance(kws, list):
        kws = []
    kws = [str(x).strip() for x in kws if str(x).strip()]
    query_text = (situation[:60] + " " + " ".join(kws[:7])).strip()
    query_text = re.sub(r"\s+", " ", query_text)
    return {"core_keywords": kws[:10], "query_text": query_text[:180], "url": make_lawbot_url(query_text[:180])}

def run_workflow(user_input: str) -> dict:
    log = st.empty()
    logs: List[str] = []

    def add(msg: str):
        logs.append(f"- {msg}")
        log.markdown("\n".join(logs))

    # 0) analyze
    add("Phase 0) 케이스 분석(유형/쟁점/누락정보/증빙)")
    analysis = CaseAnalyzer.analyze(user_input)

    # 1) law
    add("Phase 1) 법령 근거 강화(LLM + aiSearch + 링크/발췌)")
    law_pack = LegalAgents.researcher(user_input, analysis)
    law_md = law_pack.get("markdown", "")

    # 2) precedents
    add("Phase 2) 뉴스/사례 조회")
    news = search_service.search_precedents(user_input)

    # 3) strategy
    add("Phase 3) 처리방향/주의사항/체크리스트 생성")
    strategy = LegalAgents.strategist(user_input, law_md, news, analysis)

    # 4) objections
    add("Phase 4) 반발/리스크 대응 시나리오")
    objections = ObjectionAgent.build(user_input, strategy)

    # 5) procedure plan
    add("Phase 5) 절차 플랜(타임라인/서식/기록 포인트)")
    procedure = ProcedureAgent.plan(user_input, law_md[:1500], analysis)

    # 6) doc
    add("Phase 6) 공문 조판(JSON 실패 시 자동 복구)")
    meta = LegalAgents.clerk()
    doc = LegalAgents.drafter(user_input, law_md, meta, strategy, procedure, objections)

    # 7) lawbot pack
    add("Phase 7) Lawbot 검색팩 생성")
    lb = build_lawbot_pack(user_input, analysis)

    log.empty()

    return {
        "situation": user_input,
        "analysis": analysis,
        "law_pack": law_pack,
        "law": law_md,
        "search": news,
        "strategy": strategy,
        "objections": objections,
        "procedure": procedure,
        "meta": meta,
        "doc": doc,
        "lawbot_pack": lb,
        "followups": [],
    }

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
        "user_id": user_id if st.session_state.get("logged_in") else None,
        "user_email": user_email if st.session_state.get("logged_in") else None,
        "client_meta": {"app_ver": APP_VERSION},
    }
    try:
        resp = sb.table("work_archive").insert(row).execute()
        if hasattr(resp, "data") and resp.data and isinstance(resp.data, list):
            return resp.data[0].get("id")
    except Exception as e:
        st.warning(f"ℹ️ DB 저장 실패: {e}")
    return None

def db_fetch_history(sb, limit: int = 80) -> List[dict]:
    try:
        q = sb.table("work_archive").select("id,prompt,created_at,user_email,anon_session_id").order("created_at", desc=True).limit(limit)
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
        "user_id": user_id if st.session_state.get("logged_in") else None,
        "user_email": user_email if st.session_state.get("logged_in") else None,
        "anon_session_id": anon_id,
    }
    try:
        sb.table("work_followups").insert(row).execute()
    except Exception:
        pass

# =========================================================
# 6) SIDEBAR AUTH UI (네 기존 유지)
# =========================================================
def sidebar_auth(sb):
    st.sidebar.markdown("## 🔐 로그인")

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_email" not in st.session_state:
        st.session_state.user_email = ""
    if "admin_mode" not in st.session_state:
        st.session_state.admin_mode = False

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

    menu = st.sidebar.radio("메뉴", ["로그인", "회원가입", "비밀번호 찾기"], horizontal=True)

    if "signup_stage" not in st.session_state:
        st.session_state.signup_stage = 1
    if "reset_stage" not in st.session_state:
        st.session_state.reset_stage = 1

    if menu == "로그인":
        email = st.sidebar.text_input("메일", placeholder="name@korea.kr", key="login_email")
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

    elif menu == "회원가입":
        if st.session_state.signup_stage == 1:
            email = st.sidebar.text_input("메일(@korea.kr)", placeholder="name@korea.kr", key="su_email")
            if st.sidebar.button("코리아 메일로 인증번호 발송", use_container_width=True):
                if not (email or "").endswith("@korea.kr"):
                    st.sidebar.error("❌ @korea.kr 메일만 가입 가능")
                else:
                    try:
                        sb.auth.sign_in_with_otp({"email": email, "options": {"should_create_user": True}})
                        st.session_state.pending_email = email.strip()
                        st.session_state.signup_stage = 2
                        log_event(sb, "signup_otp_sent", meta={"email": email.strip()})
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"발송 실패: {e}")
        else:
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

    else:  # reset
        if st.session_state.reset_stage == 1:
            email = st.sidebar.text_input("메일", placeholder="name@korea.kr", key="rp_email")
            if st.sidebar.button("메일로 인증번호 발송", use_container_width=True):
                try:
                    sb.auth.sign_in_with_otp({"email": email, "options": {"should_create_user": False}})
                    st.session_state.reset_email = email.strip()
                    st.session_state.reset_stage = 2
                    log_event(sb, "reset_otp_sent", meta={"email": email.strip()})
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"발송 실패: {e}")
        else:
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
                st.session_state.logged_in = True
                st.session_state.user_email = email.strip()
                st.session_state.reset_stage = 1
                log_event(sb, "reset_done")
                st.rerun()

# =========================================================
# 7) HISTORY (프롬프트만, 클릭 즉시 복원)
# =========================================================
def restore_archive(sb, row_id: str):
    row = db_fetch_payload(sb, row_id)
    if not row:
        st.sidebar.error("복원 실패(권한/RLS 또는 데이터 없음)")
        return
    payload = row.get("payload") or {}
    followups = db_fetch_followups(sb, row_id)
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

    if not st.session_state.get("logged_in") and not admin_all:
        st.sidebar.caption("비로그인: 기록은 저장되지만 조회/복원은 불가")
        return

    hist = db_fetch_history(sb, limit=120)
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
# 8) UI
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

        sidebar_auth(sb)
        render_history_list(sb)
    else:
        st.sidebar.error("Supabase 연결 정보(secrets)가 없습니다.")
        st.sidebar.caption("SUPABASE_URL / SUPABASE_ANON_KEY 필요")

    st.title("🏢 AI 행정관 Pro 충주시청")
    st.caption("문의 kim0395kk@korea.kr  |  Govable AI 에이전트")

    ai_ok = "✅AI" if llm_service.is_available() else "❌AI"
    law_ok = "✅LAW" if bool(get_general_secret("LAW_API_ID")) else "❌LAW"
    nv_ok = "✅NEWS" if bool(get_general_secret("NAVER_CLIENT_ID")) else "❌NEWS"
    db_ok = "✅DB" if sb else "❌DB"
    st.caption(f"상태: {ai_ok} | {law_ok} | {nv_ok} | {db_ok} | ver {APP_VERSION}")

    col_left, col_right = st.columns([1, 1.15], gap="large")

    with col_left:
        st.markdown("### 🗣️ 업무 지시")
        user_input = st.text_area(
            "업무 내용",
            height=170,
            placeholder="예시\n- 상황: (무슨 일 / 어디 / 언제 / 증거 유무...)\n- 쟁점: (요건/절차/근거...)\n- 요청: (원하는 결과물: 회신/사전통지/처분 등)",
            label_visibility="collapsed",
        )
        st.warning("⚠️ 민감정보(성명·연락처·주소·차량번호 등) 입력 금지")

        if st.button("⚡ 스마트 분석 시작", type="primary", use_container_width=True):
            if not user_input:
                st.warning("내용을 입력해주세요.")
            else:
                with st.spinner("AI 에이전트 팀이 협업 중입니다..."):
                    res = run_workflow(user_input)

                    archive_id = None
                    if sb:
                        archive_id = db_insert_archive(sb, user_input, res)
                        if archive_id:
                            st.session_state.current_archive_id = archive_id
                            log_event(sb, "workflow_run", archive_id=archive_id, meta={"prompt_len": len(user_input)})

                    res["archive_id"] = archive_id
                    st.session_state.workflow_result = res
                    st.session_state.followup_messages = []

        if "workflow_result" in st.session_state:
            res = st.session_state.workflow_result
            pack = res.get("lawbot_pack") or {}
            if pack.get("url"):
                render_lawbot_button(pack["url"])

            st.subheader("🧠 케이스 분석(에이전트)")
            a = res.get("analysis", {})
            st.write(f"- 유형: **{a.get('case_type','')}**")
            if a.get("core_issue"): st.write("- 쟁점:", ", ".join(a["core_issue"]))
            with st.expander("누락정보/증빙/리스크/다음행동 보기", expanded=False):
                st.markdown("**추가 확인 질문**")
                for x in a.get("required_facts", []): st.write("- ", x)
                st.markdown("**필요 증빙**")
                for x in a.get("required_evidence", []): st.write("- ", x)
                st.markdown("**절차 리스크**")
                for x in a.get("risk_flags", []): st.write("- ", x)
                st.markdown("**권장 다음 행동**")
                for x in a.get("recommended_next_action", []): st.write("- ", x)

            st.subheader("📜 법령 근거(강화)")
            st.markdown(res.get("law", ""))

            st.subheader("📰 뉴스/사례")
            st.markdown(res.get("search", ""))

            st.subheader("🧭 처리 가이드(주무관)")
            st.markdown(res.get("strategy", ""))

            st.subheader("🧨 예상 반발/대응(에이전트)")
            for ob in res.get("objections", [])[:7]:
                st.markdown(f"- **반발**: {ob.get('objection')}\n  - **대응**: {ob.get('response')}\n  - **기록 포인트**: {ob.get('record_point')}")

            st.subheader("🗺️ 절차 플랜(타임라인/서식/체크리스트)")
            proc = res.get("procedure", {})
            with st.expander("타임라인", expanded=True):
                for step in proc.get("timeline", []):
                    st.markdown(f"**{step.get('step')}. {step.get('name')}** — {step.get('goal')}")
                    for x in step.get("actions", []): st.write("- 행동:", x)
                    for x in step.get("records", []): st.write("- 기록:", x)
                    if step.get("legal_note"): st.caption(f"법/유의: {step['legal_note']}")
                    st.write("")
            with st.expander("체크리스트/서식", expanded=False):
                st.markdown("**체크리스트**")
                for x in proc.get("checklist", []): st.write("- ", x)
                st.markdown("**필요 서식/문서**")
                for x in proc.get("templates", []): st.write("- ", x)

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

        st.subheader("📄 공문서")
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

        st.subheader("💬 후속 질문(복원됨)")
        if not archive_id:
            st.info("저장된 archive_id가 없습니다. (DB 저장 실패 가능)")
            return

        if "followup_messages" not in st.session_state:
            st.session_state.followup_messages = res.get("followups", []) or []

        used = len([m for m in st.session_state.followup_messages if m.get("role") == "user"])
        remain = max(0, MAX_FOLLOWUP_Q - used)
        st.info(f"후속 질문 가능 횟수: **{remain}/{MAX_FOLLOWUP_Q}**")

        pack = res.get("lawbot_pack") or {}
        if pack.get("url"):
            render_lawbot_button(pack["url"])

        for m in st.session_state.followup_messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        if remain == 0:
            st.warning("후속 질문 한도(5회)를 모두 사용했습니다.")
            return

        q = st.chat_input("후속 질문 (최대 5회)")
        if not q:
            return

        st.session_state.followup_messages.append({"role": "user", "content": q})
        turn = len([m for m in st.session_state.followup_messages if m["role"] == "user"])
        db_insert_followup(sb, archive_id, turn=turn * 2 - 1, role="user", content=q)
        log_event(sb, "followup_user", archive_id=archive_id, meta={"turn": turn})

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

반발/대응:
{json.dumps(res.get("objections", []), ensure_ascii=False)[:1500]}

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
                ans = llm_service.generate_text(prompt)
                st.markdown(ans)

        st.session_state.followup_messages.append({"role": "assistant", "content": ans})
        db_insert_followup(sb, archive_id, turn=turn * 2, role="assistant", content=ans)
        log_event(sb, "followup_assistant", archive_id=archive_id, meta={"turn": turn})

        st.rerun()

if __name__ == "__main__":
    main()
