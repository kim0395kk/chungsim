# streamlit_app.py
# ✅ 완세트 (secrets 진단 + 414 방어 + _naver_search 누락 해결 + law_api_service/global 인스턴스 포함)
#
# requirements.txt 예시
# streamlit
# google-generativeai
# groq
# supabase
# requests

import os
import time
import json
import re
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from html import escape as _escape

import streamlit as st

# ---- optional imports (안죽게) ----
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
# 0) Secrets Helpers
# ==========================================
def _get_secret(section: str, key: str, default=None):
    """
    st.secrets 방어 + ENV fallback
    - Streamlit Cloud: st.secrets
    - 로컬/기타: 환경변수 fallback
    """
    # 1) st.secrets
    try:
        sec = st.secrets.get(section, None)
        if isinstance(sec, dict) and key in sec:
            v = sec.get(key)
            if v is not None and str(v).strip() != "":
                return v
    except Exception:
        pass

    # 2) env fallback (예: GENERAL__GEMINI_API_KEY 같은 식으로도 넣을 수 있게)
    #    우선 key 그대로도 확인
    v = os.getenv(key)
    if v and v.strip():
        return v.strip()

    #    섹션+키 조합도 확인
    v2 = os.getenv(f"{section.upper()}__{key}")
    if v2 and v2.strip():
        return v2.strip()

    return default


def _mask(s: str, show=4):
    if not s:
        return "(none)"
    s = str(s)
    if len(s) <= show:
        return "*" * len(s)
    return s[:show] + "*" * (len(s) - show)


def _extract_json(text: str):
    """모델이 JSON을 조금 깨도 최대한 복구."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)

    m = re.search(r"\[[\s\S]*\]", t)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return None


# ==========================================
# 1) Page & Style
# ==========================================
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
.doc-info { display: flex; justify-content: space-between; gap: 12px; font-size: 11pt; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; flex-wrap: wrap; }
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
</style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 2) Services
# ==========================================
class LLMService:
    """
    Gemini (google.generativeai) 우선, 실패 시 Groq fallback
    """
    def __init__(self):
        self.gemini_key = _get_secret("general", "GEMINI_API_KEY")
        self.groq_key = _get_secret("general", "GROQ_API_KEY")

        # ⚠️ 모델명은 계정/지역/정책에 따라 가용성이 달라질 수 있어 "순차 시도" 방식 유지
        self.gemini_models = [
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ]

        self.gemini_ready = False
        if genai and self.gemini_key:
            try:
                genai.configure(api_key=self.gemini_key)
                self.gemini_ready = True
            except Exception:
                self.gemini_ready = False

        self.groq_client = None
        if Groq and self.groq_key:
            try:
                self.groq_client = Groq(api_key=self.groq_key)
            except Exception:
                self.groq_client = None

    def _try_gemini(self, prompt: str):
        if not self.gemini_ready:
            raise Exception("Gemini not configured")

        last_err = None
        for model_name in self.gemini_models:
            try:
                model = genai.GenerativeModel(model_name)
                res = model.generate_content(prompt)
                text = (getattr(res, "text", None) or "").strip()
                if text:
                    return text, model_name
            except Exception as e:
                last_err = e
                continue
        raise Exception(f"All Gemini models failed: {last_err}")

    def _generate_groq(self, prompt: str) -> str:
        if not self.groq_client:
            return "시스템 오류: Groq 미연결(키/라이브러리 확인)"
        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return completion.choices[0].message.content
        except Exception:
            return "System Error"

    def generate_text(self, prompt: str) -> str:
        # 1) Gemini
        try:
            text, _ = self._try_gemini(prompt)
            return text
        except Exception:
            # 2) Groq
            return self._generate_groq(prompt)

    def generate_json(self, prompt: str):
        strict = (
            prompt
            + "\n\n반드시 JSON만 출력. 설명/문장/마크다운 금지. 코드펜스 금지."
        )
        text = self.generate_text(strict)
        return _extract_json(text)


class SearchService:
    """
    ✅ Naver Search (414 방어 포함)
    """
    def __init__(self):
        g = st.secrets.get("general", {}) if hasattr(st, "secrets") else {}
        # st.secrets.get이 실패할 수도 있어서 _get_secret도 같이 지원
        self.client_id = (g.get("NAVER_CLIENT_ID") if isinstance(g, dict) else None) or _get_secret("general", "NAVER_CLIENT_ID")
        self.client_secret = (g.get("NAVER_CLIENT_SECRET") if isinstance(g, dict) else None) or _get_secret("general", "NAVER_CLIENT_SECRET")

        self.web_url = "https://openapi.naver.com/v1/search/webkr.json"
        self.news_url = "https://openapi.naver.com/v1/search/news.json"

        self.whitelist_domains = ["law.go.kr", "scourt.go.kr", "acrc.go.kr", "korea.kr", "go.kr", "moj.go.kr", "easylaw.go.kr", "moleg.go.kr"]
        self.blacklist_domains = ["blog.naver.com", "cafe.naver.com", "tistory.com", "brunch.co.kr", "youtube.com", "instagram.com", "facebook.com", "namu.wiki", "kin.naver.com"]
        self.signal_keywords = ["행정심판", "재결", "처분", "과태료", "이행명령", "사전통지", "의견제출", "청문", "행정절차법", "판결", "판례", "대법원", "조례", "시행규칙", "고시", "훈령", "예규", "지침", "공고"]

    def _headers(self):
        return {
            "X-Naver-Client-Id": self.client_id or "",
            "X-Naver-Client-Secret": self.client_secret or "",
        }

    def _naver_search(self, url: str, query: str, display: int = 10):
        # ✅ 여기 누락돼 있던 함수
        params = {"query": query, "display": display, "start": 1, "sort": "sim"}
        res = requests.get(url, headers=self._headers(), params=params, timeout=8)
        res.raise_for_status()
        return res.json()

    def _clean_html(self, s: str) -> str:
        if not s:
            return ""
        s = re.sub(r"</?b>", "", s)
        s = re.sub(r"<[^>]+>", "", s)
        return s.strip()

    def _get_domain(self, link: str) -> str:
        if not link:
            return ""
        m = re.search(r"https?://([^/]+)", link)
        return (m.group(1).lower() if m else "").strip()

    def _is_blacklisted(self, domain: str) -> bool:
        d = (domain or "").lower()
        return any(bad in d for bad in self.blacklist_domains)

    def _whitelist_score(self, domain: str) -> int:
        d = (domain or "").lower()
        score = 0
        for good in self.whitelist_domains:
            if good == "go.kr":
                if d.endswith(".go.kr") or d == "go.kr" or ".go.kr" in d:
                    score += 8
            else:
                if good in d:
                    score += 10
        return score

    def _keyword_score(self, text: str) -> int:
        t = (text or "").lower()
        score = 0
        for kw in self.signal_keywords:
            if kw.lower() in t:
                score += 2
        return score

    def _score_item(self, title: str, desc: str, link: str) -> int:
        domain = self._get_domain(link)
        if self._is_blacklisted(domain):
            return -999
        score = 0
        score += self._whitelist_score(domain)
        score += self._keyword_score(title) * 2
        score += self._keyword_score(desc)
        if len((desc or "").strip()) < 25:
            score -= 3
        if not (link or "").startswith("http"):
            score -= 5
        return score

    def _shrink_query(self, q: str, max_tokens: int = 10, max_chars: int = 80) -> str:
        q = re.sub(r"\s+", " ", (q or "")).strip()
        q = " ".join(q.split()[:max_tokens])
        if len(q) > max_chars:
            q = q[:max_chars].rstrip()
        return q

    def _optimize_query_llm(self, situation: str) -> str:
        prompt = f"""
너는 행정 데이터 검색 전문가.
아래 민원 상황을 네이버 검색용 '키워드'로 바꿔라.

상황: "{situation}"

규칙:
- 문장 금지, 키워드만
- 10토큰 이하
- 60자 이하
- 실무 용어 포함(처분/불복/재결/과태료/사전통지/의견제출 등)
"""
        q = llm_service.generate_text(prompt).strip()
        q = re.sub(r'["\']', "", q)
        return self._shrink_query(q, max_tokens=10, max_chars=80)

    def _rerank_results_llm(self, situation: str, candidate_items: list) -> list:
        if not candidate_items:
            return []
        ctx = ""
        for idx, item in enumerate(candidate_items[:7]):
            ctx += f"[{idx}] 제목:{item['title']} / 내용:{item['desc']} / 출처:{item['domain']}\n"

        prompt = f"""
상황: "{situation}"
아래 후보 중 실무에 가장 도움되는 순으로 인덱스만 JSON 배열로 출력.
{ctx}
예: [2,0,5]
"""
        ranking = llm_service.generate_json(prompt)
        if isinstance(ranking, list) and ranking:
            out = []
            for i in ranking:
                if isinstance(i, int) and 0 <= i < len(candidate_items):
                    out.append(candidate_items[i])
            return out or candidate_items
        return candidate_items

    def search_precedents(self, situation: str, top_k: int = 3) -> str:
        if not self.client_id or not self.client_secret:
            return "⚠️ 네이버 API 키가 설정되지 않았습니다. (NAVER_CLIENT_ID/SECRET)"

        optimized_query = self._optimize_query_llm(situation)

        # ✅ 414 방지: 괄호/OR 제거 + 짧은 site 필터
        final_query = f"{optimized_query} site:go.kr"

        try:
            web_res = self._naver_search(self.web_url, final_query, display=10)
            news_res = self._naver_search(self.news_url, final_query, display=10)
        except requests.HTTPError as e:
            # ✅ 414면 더 줄여 재시도
            if "414" in str(e):
                shorter = self._shrink_query(optimized_query, max_tokens=6, max_chars=50)
                final_query = f"{shorter} site:go.kr"
                web_res = self._naver_search(self.web_url, final_query, display=10)
                news_res = self._naver_search(self.news_url, final_query, display=10)
            else:
                return f"네이버 API 호출 오류: {e}"
        except Exception as e:
            return f"검색 프로세스 중 오류: {e}"

        merged = []
        seen = set()
        for src_name, payload in [("웹", web_res), ("뉴스", news_res)]:
            for it in (payload.get("items", []) or []):
                link = it.get("link", "#")
                if link in seen:
                    continue
                seen.add(link)
                title = self._clean_html(it.get("title", ""))
                desc = self._clean_html(it.get("description", ""))
                score = self._score_item(title, desc, link)
                if score > -100:
                    merged.append({
                        "src": src_name,
                        "title": title,
                        "desc": desc,
                        "link": link,
                        "domain": self._get_domain(link),
                        "score": score
                    })

        if not merged:
            return f"검색어 '{optimized_query}'에 대한 유의미한 결과가 없습니다."

        merged.sort(key=lambda x: x["score"], reverse=True)
        candidates = merged[:7]
        final_items = self._rerank_results_llm(situation, candidates)[:top_k]

        lines = []
        lines.append(f"🔍 **AI 최적화 검색어:** `{optimized_query}`")
        lines.append(f"🧠 **AI 선별 결과 (Top {len(final_items)})**")
        lines.append("---")
        for it in final_items:
            lines.append(f"- ({it['src']}) **[{it['title']}]({it['link']})** `[{it['domain']}]`\n  : {it['desc']}")
        return "\n".join(lines)


class DatabaseService:
    """Supabase Persistence Layer"""
    def __init__(self):
        self.is_active = False
        self.client = None

        self.url = _get_secret("supabase", "SUPABASE_URL")
        self.key = _get_secret("supabase", "SUPABASE_KEY")

        if create_client and self.url and self.key:
            try:
                self.client = create_client(self.url, self.key)
                self.is_active = True
            except Exception:
                self.is_active = False

    def save_log(self, user_input, legal_basis, strategy, doc_data):
        if not self.is_active:
            return "DB 미연결 (저장 건너뜀)"

        try:
            final_summary_content = {"strategy": strategy, "document_content": doc_data}
            data = {
                "situation": user_input,
                "law_name": legal_basis,
                "summary": json.dumps(final_summary_content, ensure_ascii=False),
            }
            self.client.table("law_reports").insert(data).execute()
            return "DB 저장 성공"
        except Exception as e:
            return f"DB 저장 실패: {e}"


class LawOfficialService:
    """국가법령정보센터(law.go.kr) 공식 API 연동"""
    def __init__(self):
        self.api_id = _get_secret("general", "LAW_API_ID")
        self.base_url = "http://www.law.go.kr/DRF/lawSearch.do"
        self.service_url = "http://www.law.go.kr/DRF/lawService.do"

    def get_law_text(self, law_name, article_num=None):
        if not self.api_id:
            return "⚠️ API ID(OC)가 설정되지 않았습니다. (LAW_API_ID)"

        law_name = (law_name or "").strip()
        if not law_name:
            return "⚠️ 법령명이 비어있습니다."

        # 1) 검색 -> MST
        try:
            params = {"OC": self.api_id, "target": "law", "type": "XML", "query": law_name, "display": 1}
            res = requests.get(self.base_url, params=params, timeout=8)
            root = ET.fromstring(res.content)

            law_node = root.find(".//law")
            if law_node is None:
                return f"🔍 '{law_name}'에 대한 검색 결과가 없습니다."

            mst_id_node = law_node.find("법령일련번호")
            link_node = law_node.find("법령상세링크")

            mst_id = mst_id_node.text.strip() if mst_id_node is not None and mst_id_node.text else None
            full_link = link_node.text.strip() if link_node is not None and link_node.text else ""

            if not mst_id:
                return f"🔍 '{law_name}' 검색은 됐지만 MST 추출 실패"
        except Exception as e:
            return f"API 검색 중 오류: {e}"

        # 2) 상세 -> 조문
        try:
            detail_params = {"OC": self.api_id, "target": "law", "type": "XML", "MST": mst_id}
            res_detail = requests.get(self.service_url, params=detail_params, timeout=12)
            root_detail = ET.fromstring(res_detail.content)

            if not article_num:
                return f"✅ '{law_name}' 확인됨\n🔗 원문 보기: {full_link}"

            for article in root_detail.findall(".//조문단위"):
                jo_num = article.find("조문번호")
                jo_cont = article.find("조문내용")
                if jo_num is None or jo_cont is None:
                    continue
                current_num = (jo_num.text or "").strip()
                if str(article_num) == current_num:
                    body = (jo_cont.text or "").strip()
                    out = f"[{law_name} 제{current_num}조 전문]\n{_escape(body)}"
                    for hang in article.findall(".//항"):
                        hang_cont = hang.find("항내용")
                        if hang_cont is not None and hang_cont.text:
                            out += f"\n  - {hang_cont.text.strip()}"
                    return out

            return f"✅ '{law_name}' 확인됨\n(요청 조문 추출 실패: 제{article_num}조)\n🔗 원문 보기: {full_link}"
        except Exception as e:
            return f"상세 법령 파싱 실패: {e}"


class LegalAgents:
    @staticmethod
    def researcher(situation: str):
        prompt_extract = f"""
상황: "{situation}"

필요한 법령/조문을 중요도 순으로 최대 3개 JSON 리스트로:
형식: [{{"law_name":"도로교통법","article_num":32}},{{"law_name":"행정절차법","article_num":21}}]
규칙: 법령명 정식, 조문 불명확하면 null, JSON만 출력.
"""
        search_targets = []
        try:
            extracted = llm_service.generate_json(prompt_extract)
            if isinstance(extracted, list):
                search_targets = extracted
            elif isinstance(extracted, dict):
                search_targets = [extracted]
        except Exception:
            search_targets = [{"law_name": "도로교통법", "article_num": None}]

        if not search_targets:
            search_targets = [{"law_name": "도로교통법", "article_num": None}]

        report_lines = [f"🔍 **AI가 식별한 핵심 법령 ({len(search_targets)}건)**", "---"]
        api_success = 0

        for idx, item in enumerate(search_targets):
            law_name = item.get("law_name", "관련법령")
            article_num = item.get("article_num")

            real = law_api_service.get_law_text(law_name, article_num)
            error_keywords = ["검색 결과가 없습니다", "오류", "API ID", "실패"]
            ok = not any(k in real for k in error_keywords)

            if ok:
                api_success += 1
                report_lines.append(f"✅ **{idx+1}. {law_name} 제{article_num}조 (확인됨)**\n{real}\n")
            else:
                report_lines.append(f"⚠️ **{idx+1}. {law_name} 제{article_num}조 (API 조회 실패)**\n(법령명/조문번호 확인 필요)\n")

        if api_success == 0:
            fallback = llm_service.generate_text(
                f"""[AI 추론 결과]
상황: "{situation}"
API 실패 상태. 가능한 법령을 설명하되 '법제처 확인 필수' 경고 포함."""
            ).strip()
            return f"⚠️ **[시스템 경고: API 조회 실패]**\n(법제처 확인 필수)\n---\n{fallback}"

        return "\n".join(report_lines)

    @staticmethod
    def strategist(situation: str, legal_basis: str, search_results: str):
        prompt = f"""
너는 행정 업무 베테랑 주무관.
[상황] {situation}
[법령] {legal_basis}
[유사사례] {search_results}

아래 3항목 마크다운으로:
1) 처리 방향
2) 핵심 주의사항
3) 예상 반발 및 대응
"""
        return llm_service.generate_text(prompt)

    @staticmethod
    def clerk(situation: str, legal_basis: str):
        today = datetime.now()
        prompt = f"""
오늘: {today.strftime('%Y-%m-%d')}
상황: {situation}
법령: {legal_basis}
통상 의견제출/이행 기간(일수) 숫자만. 모르면 15.
"""
        try:
            res = llm_service.generate_text(prompt)
            days = int(re.sub(r"[^0-9]", "", res))
            if days <= 0:
                days = 15
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
    def drafter(situation: str, legal_basis: str, meta_info: dict, strategy: str):
        prompt = f"""
반드시 JSON만 출력.
{{
  "title": "...",
  "receiver": "...",
  "body_paragraphs": ["...", "..."],
  "department_head": "..."
}}

상황: {situation}
법령: {legal_basis}
시행: {meta_info['today_str']}
기한: {meta_info['deadline_str']}
전략: {strategy}

원칙:
- 법 조항 인용(가능하면 조문번호 포함)
- 개인정보 OOO 마스킹
"""
        data = llm_service.generate_json(prompt)
        if not isinstance(data, dict):
            return {
                "title": "공 문 서",
                "receiver": "수신자 참조",
                "body_paragraphs": ["(문서 생성 실패: LLM/키 설정을 확인하세요)"],
                "department_head": "행정기관장",
            }

        data.setdefault("title", "공 문 서")
        data.setdefault("receiver", "수신자 참조")
        bp = data.get("body_paragraphs", [])
        if isinstance(bp, str):
            bp = [bp]
        if not isinstance(bp, list) or not bp:
            bp = ["(본문 생성 실패)"]
        data["body_paragraphs"] = [str(x) for x in bp]
        data.setdefault("department_head", "행정기관장")
        return data


# ==========================================
# 2.5) Global Instances (이게 없으면 name error 뜸)
# ==========================================
llm_service = LLMService()
search_service = SearchService()
db_service = DatabaseService()
law_api_service = LawOfficialService()


# ==========================================
# Sidebar: Secrets/Dependency Diagnostic (마스킹)
# ==========================================
with st.sidebar:
    st.markdown("## 🔧 상태 진단")
    st.markdown("### 🔐 Secrets (masked)")
    st.write("GEMINI_API_KEY:", _mask(llm_service.gemini_key))
    st.write("GROQ_API_KEY:", _mask(llm_service.groq_key))
    st.write("NAVER_CLIENT_ID:", _mask(search_service.client_id))
    st.write("NAVER_CLIENT_SECRET:", _mask(search_service.client_secret))
    st.write("LAW_API_ID:", _mask(law_api_service.api_id))
    st.write("SUPABASE_URL:", _mask(db_service.url, show=8))
    st.write("SUPABASE_KEY:", _mask(db_service.key))

    st.markdown("### 📦 Library")
    st.write("google.generativeai:", "OK" if genai else "MISSING")
    st.write("groq:", "OK" if Groq else "MISSING")
    st.write("supabase:", "OK" if create_client else "MISSING")


# ==========================================
# 4) Workflow
# ==========================================
def run_workflow(user_input: str):
    log_placeholder = st.empty()
    logs = []

    def add_log(msg, style="sys"):
        logs.append(f"<div class='agent-log log-{style}'>{_escape(msg)}</div>")
        log_placeholder.markdown("".join(logs), unsafe_allow_html=True)
        time.sleep(0.25)

    add_log("🔍 Phase 1: 법령 리서치 중...", "legal")
    legal_basis = LegalAgents.researcher(user_input)
    add_log("📜 법적 근거 수집 완료", "legal")

    add_log("🟩 Phase 1.5: 네이버 유사 사례 검색...", "search")
    try:
        search_results = search_service.search_precedents(user_input)
    except Exception as e:
        search_results = f"검색 모듈 오류 (건너뜀): {e}"

    add_log("🧠 Phase 2: 처리 전략 수립...", "strat")
    strategy = LegalAgents.strategist(user_input, legal_basis, search_results)

    add_log("📅 Phase 3: 기한 산정...", "calc")
    meta_info = LegalAgents.clerk(user_input, legal_basis)

    add_log("✍️ Phase 4: 공문서 생성...", "draft")
    doc_data = LegalAgents.drafter(user_input, legal_basis, meta_info, strategy)

    add_log("💾 Phase 5: DB 저장...", "sys")
    save_result = db_service.save_log(user_input, legal_basis, strategy, doc_data)

    add_log(f"✅ 완료 ({save_result})", "sys")
    time.sleep(0.6)
    log_placeholder.empty()

    return {
        "doc": doc_data,
        "meta": meta_info,
        "law": legal_basis,
        "search": search_results,
        "strategy": strategy,
        "save_msg": save_result,
    }


# ==========================================
# 5) UI
# ==========================================
def main():
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.title("🏢 AI 행정관 Pro")
        st.caption("Gemini(google.generativeai) + 국가법령정보 + Naver Search + Strategy + DB")
        st.markdown("---")

        # 시작부터 세팅 경고
        warn = []
        if not llm_service.gemini_key and not llm_service.groq_key:
            warn.append("- LLM 키 없음: GEMINI_API_KEY 또는 GROQ_API_KEY 필요")
        if not law_api_service.api_id:
            warn.append("- 법령 API 키 없음: LAW_API_ID(OC) 필요")
        if not search_service.client_id or not search_service.client_secret:
            warn.append("- 네이버 키 없음: NAVER_CLIENT_ID/SECRET 필요")
        if warn:
            st.warning("현재 설정 상태:\n" + "\n".join(warn))

        st.markdown("### 🗣️ 업무 지시")
        user_input = st.text_area(
            "업무 내용",
            height=150,
            placeholder="예시:\n- 아파트 단지 내 소방차 전용구역 불법 주차 차량 과태료 부과 예고 통지서 작성해줘.",
            label_visibility="collapsed",
        )

        if st.button("⚡ 스마트 분석 시작", type="primary", use_container_width=True):
            if not user_input:
                st.warning("내용을 입력해주세요.")
            else:
                try:
                    with st.spinner("AI 에이전트 팀이 협업 중입니다..."):
                        st.session_state["workflow_result"] = run_workflow(user_input)
                except Exception as e:
                    st.error(f"시스템 오류 발생: {e}")

        if "workflow_result" in st.session_state:
            res = st.session_state["workflow_result"]
            st.markdown("---")
            st.info(res.get("save_msg", ""))

            with st.expander("✅ [검토] 법령 및 유사 사례 확인", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**📜 적용 법령**")
                    st.code(res["law"], language="text")
                with c2:
                    st.markdown("**🟩 네이버 유사 사례**")
                    st.info(res["search"])

            with st.expander("🧭 [방향] 업무 처리 가이드라인", expanded=True):
                st.markdown(res["strategy"])

    with col_right:
        if "workflow_result" in st.session_state:
            res = st.session_state["workflow_result"]
            doc = res.get("doc") or {}
            meta = res.get("meta") or {}

            html = f"""
<div class="paper-sheet">
  <div class="stamp">직인생략</div>
  <div class="doc-header">{_escape(str(doc.get('title', '공 문 서')))}</div>
  <div class="doc-info">
    <span>문서번호: {_escape(str(meta.get('doc_num','-')))}</span>
    <span>시행일자: {_escape(str(meta.get('today_str','-')))}</span>
    <span>수신: {_escape(str(doc.get('receiver','수신자 참조')))}</span>
  </div>
  <hr style="border: 1px solid black; margin-bottom: 30px;">
  <div class="doc-body">
"""
            paragraphs = doc.get("body_paragraphs", [])
            if isinstance(paragraphs, str):
                paragraphs = [paragraphs]
            if not isinstance(paragraphs, list):
                paragraphs = ["(본문 데이터 형식 오류)"]

            for p in paragraphs:
                html += f"<p style='margin-bottom: 15px;'>{_escape(str(p))}</p>"

            html += f"""
  </div>
  <div class="doc-footer">{_escape(str(doc.get('department_head', '행정기관장')))}</div>
</div>
"""
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.markdown(
                """<div style='text-align: center; padding: 100px; color: #aaa; background: white; border-radius: 10px; border: 2px dashed #ddd;'>
<h3>📄 Document Preview</h3><p>왼쪽에서 업무를 지시하면<br>완성된 공문서가 여기에 나타납니다.</p></div>""",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
