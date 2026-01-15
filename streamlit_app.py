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

# ==========================
# Global Settings
# ==========================
MAX_FOLLOWUP_Q = 5
LAW_BOT_SEARCH_URL = "https://www.law.go.kr/LSW/ais/searchList.do?query="
ADMIN_EMAIL = "kim0395kk@korea.kr"

# ==========================
# Page / Styles
# ==========================
st.set_page_config(
    layout="wide",
    page_title="AI Bureau: The Legal Glass",
    page_icon="⚖️",
    initial_sidebar_state="collapsed",  # ✅ 사이드바 기본 접힘(사용자도 토글로 펼침)
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
        overflow: hidden;
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

# ==========================
# Utils
# ==========================
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
    return text.strip()

def _get_secret(path1: str, path2: str = "") -> Optional[str]:
    """secrets 키가 old/new 섞여 있어도 안전하게 읽기"""
    try:
        if path2:
            return st.secrets.get(path1, {}).get(path2)
        return st.secrets.get(path1)
    except Exception:
        return None

def _is_korea_email(email: str) -> bool:
    return bool(email) and email.lower().endswith("@korea.kr")

# ==========================
# Supabase Auth + Client
# ==========================
def get_supabase_base():
    """anon 기반 supabase client (로그인 전/후 공통)"""
    if not create_client:
        return None

    sb = st.secrets.get("supabase", {})
    # new style
    url = sb.get("SUPABASE_URL") or st.secrets.get("SUPABASE_URL")
    anon = sb.get("SUPABASE_ANON_KEY") or sb.get("SUPABASE_KEY") or st.secrets.get("SUPABASE_ANON_KEY") or st.secrets.get("SUPABASE_KEY")

    if not (url and anon):
        return None
    return create_client(url, anon)

def get_supabase_authed():
    """
    로그인한 유저 토큰으로 RLS 적용되는 client 반환
    (없으면 base client 반환)
    """
    sb = get_supabase_base()
    if not sb:
        return None

    if st.session_state.get("logged_in") and st.session_state.get("sb_access_token"):
        try:
            # supabase-py v2: set_session(access, refresh)
            sb.auth.set_session(st.session_state["sb_access_token"], st.session_state.get("sb_refresh_token", ""))
        except Exception:
            # fallback: postgrest auth
            try:
                sb.postgrest.auth(st.session_state["sb_access_token"])
            except Exception:
                pass
    return sb

def is_admin_user() -> bool:
    return (st.session_state.get("user_email") or "").lower() == ADMIN_EMAIL.lower()

# ==========================
# LLM / Search / Law API (기존 유지 + 안정화)
# ==========================
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
        text = self.generate_text(prompt + "\n\n반드시 JSON만 출력. 다른 텍스트 금지.")
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

    def _extract_keywords_llm(self, situation: str, llm: LLMService) -> str:
        prompt = f"상황: '{situation}'\n뉴스 검색을 위한 핵심 키워드 2~3개만 콤마로 구분해 출력."
        try:
            res = (llm.generate_text(prompt) or "").strip()
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

    def search_precedents(self, situation: str, llm: LLMService, top_k: int = 3) -> str:
        keywords = self._extract_keywords_llm(situation, llm)
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

# ==========================
# Database Service (RLS + Restore)
# ==========================
class DatabaseService:
    def __init__(self):
        self.mode = "off"
        self.is_active = False

        sb = get_supabase_base()
        if sb:
            self.is_active = True
            self.mode = "anon"

    def _pack_summary(self, res: dict, followup: dict) -> dict:
        return {
            "meta": res.get("meta"),
            "strategy": res.get("strategy"),
            "search_initial": res.get("search"),
            "law_initial": res.get("law"),
            "document_content": res.get("doc"),
            "lawbot_pack": res.get("lawbot_pack", {}),
            "followup": followup,
        }

    def insert_initial_report(self, res: dict) -> dict:
        sb = get_supabase_authed()
        if not sb:
            return {"ok": False, "msg": "DB 미연결(슈파베이스 설정 없음)", "id": None}

        if not st.session_state.get("logged_in"):
            return {"ok": False, "msg": "로그인 필요(저장 건너뜀)", "id": None}

        try:
            followup = {"count": 0, "messages": [], "extra_context": ""}
            data = {
                "situation": res.get("situation", ""),
                "law_name": res.get("law", ""),
                "summary": self._pack_summary(res, followup),
            }
            resp = sb.table("law_reports").insert(data).execute()
            inserted_id = None
            if hasattr(resp, "data") and resp.data and isinstance(resp.data, list):
                inserted_id = resp.data[0].get("id")
            return {"ok": True, "msg": "DB 저장 성공", "id": inserted_id}
        except Exception as e:
            return {"ok": False, "msg": f"DB 저장 실패: {e}", "id": None}

    def update_followup(self, report_id, res: dict, followup: dict) -> dict:
        sb = get_supabase_authed()
        if not sb:
            return {"ok": False, "msg": "DB 미연결(업데이트 건너뜀)"}
        if not st.session_state.get("logged_in"):
            return {"ok": False, "msg": "로그인 필요(업데이트 건너뜀)"}

        summary = self._pack_summary(res, followup)

        if report_id:
            try:
                sb.table("law_reports").update({"summary": summary}).eq("id", report_id).execute()
                return {"ok": True, "msg": "DB 업데이트 성공"}
            except Exception:
                pass

        try:
            data = {
                "situation": res.get("situation", ""),
                "law_name": res.get("law", ""),
                "summary": summary,
            }
            sb.table("law_reports").insert(data).execute()
            return {"ok": True, "msg": "DB 업데이트 실패 → 신규 저장(fallback) 완료"}
        except Exception as e:
            return {"ok": False, "msg": f"DB 업데이트/저장 실패: {e}"}

    def list_reports(self, limit: int = 50) -> List[dict]:
        sb = get_supabase_authed()
        if not sb or not st.session_state.get("logged_in"):
            return []
        try:
            resp = (
                sb.table("law_reports")
                .select("id, created_at, situation, law_name, summary, user_email")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return resp.data or []
        except Exception:
            return []

    def delete_report(self, report_id: str) -> bool:
        sb = get_supabase_authed()
        if not sb or not st.session_state.get("logged_in"):
            return False
        try:
            sb.table("law_reports").delete().eq("id", report_id).execute()
            return True
        except Exception:
            return False

    def update_report_basic(self, report_id: str, situation: str, law_name: str, summary: Optional[dict] = None) -> bool:
        sb = get_supabase_authed()
        if not sb or not st.session_state.get("logged_in"):
            return False
        try:
            payload = {"situation": situation, "law_name": law_name}
            if isinstance(summary, dict):
                payload["summary"] = summary
            sb.table("law_reports").update(payload).eq("id", report_id).execute()
            return True
        except Exception:
            return False

    @staticmethod
    def restore_res_from_row(row: dict) -> dict:
        """
        DB row -> workflow_result 형태로 복원
        """
        summary = row.get("summary") or {}
        meta = summary.get("meta") or {}
        res = {
            "situation": row.get("situation") or "",
            "meta": meta,
            "strategy": summary.get("strategy") or "",
            "search": summary.get("search_initial") or "",
            "law": summary.get("law_initial") or row.get("law_name") or "",
            "doc": summary.get("document_content") or None,
            "lawbot_pack": summary.get("lawbot_pack") or {},
        }
        # save msg
        res["save_msg"] = "DB에서 복원됨"
        return res

    @staticmethod
    def restore_followup_from_row(row: dict):
        summary = row.get("summary") or {}
        followup = summary.get("followup") or {}
        st.session_state["report_id"] = row.get("id")
        st.session_state["followup_count"] = int(followup.get("count") or 0)
        st.session_state["followup_messages"] = followup.get("messages") or []
        st.session_state["followup_extra_context"] = followup.get("extra_context") or ""

# ==========================
# Agent Logic
# ==========================
llm_service = LLMService()
search_service = SearchService()
law_api_service = LawOfficialService()
db_service = DatabaseService()

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
    def _normalize_doc(doc_data: Any) -> Optional[dict]:
        if isinstance(doc_data, dict):
            # body_paragraphs가 문자열이면 리스트화
            bp = doc_data.get("body_paragraphs", [])
            if isinstance(bp, str):
                doc_data["body_paragraphs"] = [bp]
            if "title" not in doc_data:
                doc_data["title"] = "공 문 서"
            if "receiver" not in doc_data:
                doc_data["receiver"] = "수신자 참조"
            if "department_head" not in doc_data:
                doc_data["department_head"] = "행정기관장"
            if not isinstance(doc_data.get("body_paragraphs"), list):
                doc_data["body_paragraphs"] = [str(doc_data.get("body_paragraphs", ""))]
            return doc_data
        return None

    @staticmethod
    def drafter(situation: str, legal_basis: str, meta_info: dict, strategy: str) -> Optional[dict]:
        # ✅ 공문서 JSON이 깨질 때가 많아서 "2단계 방어"로 안정화
        prompt = f"""
당신은 행정기관의 베테랑 서기입니다. 아래 정보를 바탕으로 완결된 공문서를 작성하세요.

[입력]
- 민원: {situation}
- 법적 근거: {legal_basis}
- 시행일자: {meta_info.get('today_str','')}
- 기한: {meta_info.get('deadline_str','')} ({meta_info.get('days_added','')}일)

[전략]
{strategy}

[출력 형식(JSON만)]
{{
  "title": "문서 제목",
  "receiver": "수신자",
  "body_paragraphs": ["문단1", "문단2", "..."],
  "department_head": "부서장/기관장"
}}

[원칙]
1) 본문에 법 조항 인용 필수
2) 구조: 경위 -> 법적 근거 -> 처분 내용 -> 이의제기 절차
3) 개인정보 마스킹('OOO')
"""
        doc_data = llm_service.generate_json(prompt)
        doc_norm = LegalAgents._normalize_doc(doc_data)
        if doc_norm:
            return doc_norm

        # 2차: 텍스트로라도 받기 → JSON 블록만 추출
        text = llm_service.generate_text(prompt + "\n\nJSON만 출력!!!")
        doc_data2 = _safe_json_loads(text)
        return LegalAgents._normalize_doc(doc_data2)

# ==========================
# Workflow
# ==========================
def build_lawbot_pack(res: dict) -> dict:
    situation = (res.get("situation") or "").strip()
    prompt = f"""
상황: "{situation}"
국가법령정보센터 법령 AI(Lawbot) 검색에 넣을 핵심 키워드 3~6개를 JSON 배열로만 출력.
예: ["무단방치", "자동차관리법", "공시송달", "직권말소"]
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

    add_log("🔍 Phase 1: 법령 리서치 중...", "legal")
    legal_basis = LegalAgents.researcher(user_input)
    add_log("📜 법적 근거 확보", "legal")

    add_log("🟩 Phase 1-2: 뉴스/사례 조회...", "search")
    try:
        search_results = search_service.search_precedents(user_input, llm_service)
    except Exception:
        search_results = "검색 모듈 미연결 (건너뜀)"

    add_log("🧠 Phase 2: 업무 처리 방향 수립...", "strat")
    strategy = LegalAgents.strategist(user_input, legal_basis, search_results)

    add_log("📅 Phase 3: 기한 산정...", "calc")
    meta_info = LegalAgents.clerk(user_input, legal_basis)

    add_log("✍️ Phase 4: 공문서 조판 중...", "draft")
    doc_data = LegalAgents.drafter(user_input, legal_basis, meta_info, strategy)

    time.sleep(0.2)
    log_placeholder.empty()

    res = {
        "situation": user_input,
        "doc": doc_data,
        "meta": meta_info,
        "law": legal_basis,
        "search": search_results,
        "strategy": strategy,
    }
    res["lawbot_pack"] = build_lawbot_pack(res)
    return res

# ==========================
# Follow-up Chat (expander 중첩 금지)
# ==========================
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
    law_triggers = ["근거", "조문", "법령", "몇 조", "원문", "현행", "추가 조항", "다른 조문", "전문", "절차법", "행정절차", "규칙", "서식"]
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

[출력 JSON 스키마]
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
    if "case_id" not in st.session_state:
        st.session_state["case_id"] = None
    if "followup_count" not in st.session_state:
        st.session_state["followup_count"] = 0
    if "followup_messages" not in st.session_state:
        st.session_state["followup_messages"] = []
    if "followup_extra_context" not in st.session_state:
        st.session_state["followup_extra_context"] = ""
    if "report_id" not in st.session_state:
        st.session_state["report_id"] = None

    current_case_id = (res.get("meta") or {}).get("doc_num", "") or "case"
    if st.session_state["case_id"] != current_case_id:
        st.session_state["case_id"] = current_case_id
        # (복원된 케이스는 별도 로직에서 세팅되므로 여기서 강제 초기화하지 않음)
        if not st.session_state.get("restored_case_lock"):
            st.session_state["followup_count"] = 0
            st.session_state["followup_messages"] = []
            st.session_state["followup_extra_context"] = ""
        st.session_state["restored_case_lock"] = False

    remain = max(0, MAX_FOLLOWUP_Q - st.session_state["followup_count"])
    st.info(f"후속 질문 가능 횟수: **{remain}/{MAX_FOLLOWUP_Q}**")

    # ✅ Lawbot 실행 버튼
    pack = res.get("lawbot_pack", {}) or {}
    qb = (pack.get("query_text") or "").strip()
    if qb:
        st.caption("추가 근거/규칙/서식을 더 파고들 때:")
        st.link_button("⚖️ 법령 AI Lawbot 실행 (법령·규칙·서식 찾기)", make_lawbot_url(qb), use_container_width=True)

    if remain == 0:
        st.warning("후속 질문 한도(5회)를 모두 사용했습니다.")
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

        # Lawbot 빠른 링크
        if plan.get("need_law"):
            q2 = (f"{plan.get('law_name','')} 제{plan.get('article_num',0)}조 {user_q}".strip()).replace("제0조", "")
            q2 = re.sub(r"\s+", " ", q2).strip()[:180]
            extra_ctx += f"\n\n[법령 AI(Lawbot) 빠른검색]\n- 키워드: {q2}\n- 링크: {make_lawbot_url(q2)}"

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

    # ✅ DB에 후속까지 저장
    followup_payload = {
        "count": st.session_state["followup_count"],
        "messages": st.session_state["followup_messages"],
        "extra_context": st.session_state.get("followup_extra_context", ""),
    }
    upd = db_service.update_followup(
        report_id=st.session_state.get("report_id"),
        res=res,
        followup=followup_payload,
    )
    if not upd.get("ok"):
        st.caption(f"DB 후속 저장 실패: {upd.get('msg')}")

# ==========================
# Sidebar: Login / Signup / History / Admin
# ==========================
def sidebar_auth_and_history():
    st.sidebar.title("🔐 로그인 / 히스토리")

    # session init
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "user_email" not in st.session_state:
        st.session_state["user_email"] = ""
    if "sb_access_token" not in st.session_state:
        st.session_state["sb_access_token"] = ""
    if "sb_refresh_token" not in st.session_state:
        st.session_state["sb_refresh_token"] = ""

    sb = get_supabase_base()
    if not sb:
        st.sidebar.error("Supabase 설정이 없습니다. secrets.toml 확인")
        return

    if not st.session_state["logged_in"]:
        choice = st.sidebar.radio("메뉴", ["로그인", "회원가입"], horizontal=False)

        email = st.sidebar.text_input("아이디(이메일)", key="auth_email")
        password = st.sidebar.text_input("비밀번호", type="password", key="auth_pw")

        if choice == "회원가입":
            st.sidebar.caption("✅ @korea.kr 이메일만 가입 허용")
            if st.sidebar.button("가입하기", use_container_width=True):
                if not _is_korea_email(email):
                    st.sidebar.error("❌ @korea.kr 메일로만 가입 가능합니다.")
                elif not password or len(password) < 6:
                    st.sidebar.error("❌ 비밀번호는 6자 이상")
                else:
                    try:
                        sb.auth.sign_up({"email": email, "password": password})
                        st.sidebar.success("✉️ 메일함에서 인증 후 로그인하세요.")
                    except Exception as e:
                        st.sidebar.error(f"가입 실패: {e}")

        else:
            if st.sidebar.button("로그인", use_container_width=True):
                if not _is_korea_email(email):
                    st.sidebar.error("❌ @korea.kr 메일만 로그인 허용")
                else:
                    try:
                        res = sb.auth.sign_in_with_password({"email": email, "password": password})
                        # session tokens
                        access_token = ""
                        refresh_token = ""
                        try:
                            access_token = res.session.access_token
                            refresh_token = res.session.refresh_token
                        except Exception:
                            # dict-like fallback
                            access_token = (res.get("session") or {}).get("access_token", "")
                            refresh_token = (res.get("session") or {}).get("refresh_token", "")

                        st.session_state["logged_in"] = True
                        st.session_state["user_email"] = email
                        st.session_state["sb_access_token"] = access_token
                        st.session_state["sb_refresh_token"] = refresh_token
                        st.rerun()
                    except Exception:
                        st.sidebar.error("아이디/비밀번호를 확인해주세요.")
        return

    # logged in
    user_email = st.session_state["user_email"]
    admin_badge = "👑 관리자" if is_admin_user() else "👤 사용자"
    st.sidebar.success(f"✅ 접속중: {user_email}\n\n{admin_badge}")

    col_a, col_b = st.sidebar.columns(2)
    if col_a.button("로그아웃", use_container_width=True):
        st.session_state["logged_in"] = False
        st.session_state["user_email"] = ""
        st.session_state["sb_access_token"] = ""
        st.session_state["sb_refresh_token"] = ""
        st.session_state.pop("workflow_result", None)
        st.session_state.pop("report_id", None)
        st.rerun()

    if col_b.button("새로고침", use_container_width=True):
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("🗂️ 히스토리 (짠 복원)")

    rows = db_service.list_reports(limit=60)
    if not rows:
        st.sidebar.caption("저장된 기록이 없습니다.")
        return

    # 관리자 편집 패널
    st.sidebar.caption("기록 클릭 → 오른쪽 화면이 그대로 복원됩니다.")

    for row in rows[:60]:
        rid = row.get("id")
        created = (row.get("created_at") or "")[:19].replace("T", " ")
        title = (row.get("situation") or "")[:16].replace("\n", " ")
        who = (row.get("user_email") or "").split("@")[0] if row.get("user_email") else ""

        label = f"{created} | {title}"
        if is_admin_user():
            label += f" | {who}"

        with st.sidebar.container():
            c1, c2 = st.sidebar.columns([3, 1])
            if c1.button(f"📌 {label}", key=f"restore_{rid}", use_container_width=True):
                res = DatabaseService.restore_res_from_row(row)
                st.session_state["workflow_result"] = res
                DatabaseService.restore_followup_from_row(row)
                st.session_state["restored_case_lock"] = True
                st.rerun()

            # 관리자 삭제
            if is_admin_user():
                if c2.button("🗑️", key=f"del_{rid}", use_container_width=True):
                    ok = db_service.delete_report(rid)
                    if ok:
                        st.sidebar.success("삭제 완료")
                        st.rerun()
                    else:
                        st.sidebar.error("삭제 실패")

    # 관리자: 수정 패널(선택적으로)
    if is_admin_user():
        st.sidebar.markdown("---")
        st.sidebar.subheader("🛠️ 관리자 수정(선택)")
        st.sidebar.caption("복원 후 수정하세요. (situation/law_name/summary 일부 수정 가능)")
        if st.session_state.get("report_id") and st.session_state.get("workflow_result"):
            rid = st.session_state["report_id"]
            res = st.session_state["workflow_result"]
            sit = st.sidebar.text_area("situation", value=res.get("situation",""), height=90, key="adm_edit_sit")
            lw = st.sidebar.text_input("law_name(표시용)", value=res.get("law",""), key="adm_edit_law")

            edit_json = st.sidebar.checkbox("summary JSON까지 수정", value=False)
            summary_obj = None
            if edit_json:
                # 현재 workflow_result로부터 summary를 재구성
                current_summary = {
                    "meta": res.get("meta"),
                    "strategy": res.get("strategy"),
                    "search_initial": res.get("search"),
                    "law_initial": res.get("law"),
                    "document_content": res.get("doc"),
                    "lawbot_pack": res.get("lawbot_pack", {}),
                    "followup": {
                        "count": st.session_state.get("followup_count",0),
                        "messages": st.session_state.get("followup_messages",[]),
                        "extra_context": st.session_state.get("followup_extra_context",""),
                    }
                }
                txt = st.sidebar.text_area("summary(JSON)", value=json.dumps(current_summary, ensure_ascii=False), height=180, key="adm_edit_summary")
                try:
                    summary_obj = json.loads(txt)
                except Exception:
                    st.sidebar.error("summary JSON 파싱 실패")
                    summary_obj = None

            if st.sidebar.button("✅ 수정 저장", use_container_width=True):
                ok = db_service.update_report_basic(rid, sit, lw, summary_obj if edit_json else None)
                if ok:
                    st.sidebar.success("수정 저장 완료")
                    # 화면에도 반영
                    st.session_state["workflow_result"]["situation"] = sit
                    st.session_state["workflow_result"]["law"] = lw
                    st.rerun()
                else:
                    st.sidebar.error("수정 실패")

# ==========================
# Main UI
# ==========================
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
    ).replace("\n", "<br>")

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

def render_document(res: dict):
    doc = res.get("doc") or {}
    meta = res.get("meta") or {}

    # 공문 JSON이 비어있으면 안내 + raw로라도 보여주기
    if not isinstance(doc, dict) or not doc:
        st.warning("공문 생성 결과(doc)가 비어 있습니다. (모델 JSON 출력 실패 가능)")
        raw = doc if isinstance(doc, str) else json.dumps(doc, ensure_ascii=False)
        st.code(raw or "(없음)", language="json")
        return

    title = _escape(str(doc.get("title", "공 문 서")))
    receiver = _escape(str(doc.get("receiver", "수신자 참조")))
    dept = _escape(str(doc.get("department_head", "행정기관장")))
    doc_num = _escape(str(meta.get("doc_num", "")))
    today_str = _escape(str(meta.get("today_str", "")))

    paragraphs = doc.get("body_paragraphs", [])
    if isinstance(paragraphs, str):
        paragraphs = [paragraphs]
    if not isinstance(paragraphs, list):
        paragraphs = [str(paragraphs)]

    html_content = f"""
<div class="paper-sheet">
  <div class="stamp">직인생략</div>
  <div class="doc-header">{title}</div>
  <div class="doc-info">
    <span>문서번호: {doc_num}</span>
    <span>시행일자: {today_str}</span>
    <span>수신: {receiver}</span>
  </div>
  <hr style="border: 1px solid black; margin-bottom: 30px;">
  <div class="doc-body">
"""
    for p in paragraphs:
        html_content += f"<p style='margin-bottom: 15px;'>{_escape(str(p))}</p>"

    html_content += f"""
  </div>
  <div class="doc-footer">{dept}</div>
</div>
"""
    st.markdown(html_content, unsafe_allow_html=True)

def main():
    # ✅ sidebar login/history
    sidebar_auth_and_history()

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.title("🏢 AI 행정관 Pro 충주시청")
        st.caption("문의 kim0395kk@korea.kr \n세계최초 행정 Govable AI 에이젼트")
        st.markdown("---")

        ai_ok = "✅AI" if llm_service.is_available() else "❌AI"
        law_ok = "✅LAW" if bool(st.secrets.get("general", {}).get("LAW_API_ID")) else "❌LAW"
        nv_ok = "✅NEWS" if bool(st.secrets.get("general", {}).get("NAVER_CLIENT_ID")) else "❌NEWS"
        db_ok = "✅DB" if db_service.is_active else "❌DB"
        login_ok = "✅LOGIN" if st.session_state.get("logged_in") else "❌LOGIN"
        st.caption(f"상태: {ai_ok} | {law_ok} | {nv_ok} | {db_ok} | {login_ok}")

        st.markdown("### 🗣️ 업무 지시")
        user_input = st.text_area(
            "업무 내용",
            height=150,
            placeholder="예시 \n- 상황: (무슨 일 / 어디 / 언제 / 증거 유무...).... \n- 의도: (확인하고 싶은 쟁점: 요건/절차/근거... )\n- 요청: (원하는 결과물: 공문 종류/회신/사전통지 등)",
            label_visibility="collapsed",
        )
        st.warning("⚠️ 비공개 문서 부분복사/내부검토 민감정보(성명·연락처·주소·차량번호 등) 입력 금지")

        if not st.session_state.get("logged_in"):
            st.info("저장/히스토리/복원 기능 사용하려면 사이드바에서 @korea.kr 로그인 필요")

        if st.button("⚡ 스마트 분석 시작", type="primary", use_container_width=True):
            if not user_input:
                st.warning("내용을 입력해주세요.")
            else:
                try:
                    with st.spinner("AI 에이전트 팀이 협업 중입니다..."):
                        res = run_workflow(user_input)

                        # ✅ 저장(로그인 시에만)
                        ins = db_service.insert_initial_report(res)
                        res["save_msg"] = ins.get("msg")
                        st.session_state["report_id"] = ins.get("id")

                        st.session_state["workflow_result"] = res

                        # 새 케이스 생성이므로 followup 초기화
                        st.session_state["restored_case_lock"] = False
                        st.session_state["followup_count"] = 0
                        st.session_state["followup_messages"] = []
                        st.session_state["followup_extra_context"] = ""

                except Exception as e:
                    st.error(f"시스템 오류 발생: {e}")

        if "workflow_result" in st.session_state:
            res = st.session_state["workflow_result"]
            st.markdown("---")
            if "성공" in (res.get("save_msg") or ""):
                st.success(f"✅ {res['save_msg']}")
            else:
                st.info(f"ℹ️ {res.get('save_msg','')}")

            # ✅ 법령 AI(Lawbot) 실행 버튼(문구 변경)
            pack = res.get("lawbot_pack", {}) or {}
            qb = (pack.get("query_text") or "").strip()
            if qb:
                st.link_button("⚖️ 법령 AI Lawbot 실행 (법령·규칙·서식 찾기)", make_lawbot_url(qb), use_container_width=True)

            st.markdown("### ✅ [검토] 법령 및 유사 사례 확인")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📜 적용 법령 (법령명 클릭 시 현행 원문 새창)**")
                render_law_box(res.get("law", ""))
            with col2:
                st.markdown("**🟩 관련 뉴스/사례**")
                render_news_box(res.get("search", ""))

            st.markdown("### 🧭 [방향] 업무 처리 가이드라인")
            st.markdown(res.get("strategy", ""))

    with col_right:
        if "workflow_result" in st.session_state:
            res = st.session_state["workflow_result"]
            render_document(res)

            st.markdown("---")
            st.subheader("💬 후속 질문 (케이스 고정 / 최대 5회)")
            render_followup_chat(res)
        else:
            st.markdown(
                """<div style='text-align: center; padding: 100px; color: #aaa; background: white; border-radius: 10px; border: 2px dashed #ddd;'>
<h3>📄 Document Preview</h3><p>왼쪽에서 업무를 지시하면<br>완성된 공문서가 여기에 나타납니다.</p></div>""",
                unsafe_allow_html=True,
            )

if __name__ == "__main__":
    main()
