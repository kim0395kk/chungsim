# streamlit_app.py
# ✅ 완세트 (secrets 인식 fix + 414 방어 + law_api_service 미정의 해결 + google.genai 마이그레이션 + 안전한 fallback)
#
# requirements.txt 예시
# streamlit
# google-genai
# groq
# supabase
# requests
#
# ✅ secrets.toml 예시 (프로젝트/.streamlit/secrets.toml 또는 Streamlit Cloud Secrets에 붙여넣기)
# [general]
# GEMINI_API_KEY = "..."
# LAW_API_ID = "..."
# GROQ_API_KEY = "..."
# NAVER_CLIENT_ID = "..."
# NAVER_CLIENT_SECRET = "..."
#
# [supabase]
# SUPABASE_URL = "..."
# SUPABASE_KEY = "..."

import streamlit as st

# google-genai (신형)
from google import genai
from google.genai import types

from groq import Groq
from supabase import create_client

import json
import re
import time
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from html import escape as _escape


# ==========================================
# 0. Helpers
# ==========================================
def _get_secret(section: str, key: str, default=None):
    """
    ✅ Streamlit Secrets는 dict가 아니라 dict-like 객체일 수 있으므로
    isinstance(dict) 체크 금지. 그냥 dict처럼 접근.
    """
    try:
        if section not in st.secrets:
            return default
        return st.secrets[section].get(key, default)
    except Exception:
        return default


def _extract_json(text: str):
    """모델이 JSON을 조금 깨도 최대한 복구."""
    if not text:
        return None
    t = text.strip()

    # ```json ... ``` 제거
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)

    # 배열 우선
    m = re.search(r"\[[\s\S]*\]", t)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    # 객체
    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return None


# ==========================================
# 1. Configuration & Styles
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
# 2. Infrastructure Layer (Services)
# ==========================================
class LLMService:
    """
    [Model Hierarchy]
    1. Gemini 2.5 Flash
    2. Gemini 2.5 Flash Lite
    3. Gemini 2.0 Flash
    4. Groq (Llama 3 Backup)
    """

    def __init__(self):
        self.gemini_key = _get_secret("general", "GEMINI_API_KEY")
        self.groq_key = _get_secret("general", "GROQ_API_KEY")

        self.gemini_models = [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
        ]

        self.gemini_client = None
        if self.gemini_key:
            try:
                self.gemini_client = genai.Client(api_key=self.gemini_key)
            except Exception:
                self.gemini_client = None

        self.groq_client = Groq(api_key=self.groq_key) if self.groq_key else None

    def _try_gemini(self, prompt: str):
        if not self.gemini_client:
            raise Exception("Gemini client not configured")

        last_err = None
        for model_name in self.gemini_models:
            try:
                res = self.gemini_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=2048,
                    ),
                )
                text = (getattr(res, "text", None) or "").strip()
                if text:
                    return text, model_name
            except Exception as e:
                last_err = e
                continue
        raise Exception(f"All Gemini models failed: {last_err}")

    def _generate_groq(self, prompt: str) -> str:
        if not self.groq_client:
            return "시스템 오류: AI 모델 연결 실패 (GROQ_API_KEY 없음)"
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
        try:
            text, _ = self._try_gemini(prompt)
            return text
        except Exception:
            return self._generate_groq(prompt)

    def generate_json(self, prompt: str):
        strict = (
            prompt
            + "\n\n"
            + "반드시 JSON만 출력하세요. 설명/문장/마크다운/코드펜스 금지.\n"
            + "가능하면 배열/객체 형태로만 출력."
        )
        text = self.generate_text(strict)
        return _extract_json(text)


class SearchService:
    """
    ✅ Hybrid Search Engine
    - 414(URI Too Long) 방어 포함
    """

    def __init__(self):
        self.client_id = _get_secret("general", "NAVER_CLIENT_ID")
        self.client_secret = _get_secret("general", "NAVER_CLIENT_SECRET")

        self.web_url = "https://openapi.naver.com/v1/search/webkr.json"
        self.news_url = "https://openapi.naver.com/v1/search/news.json"

        self.whitelist_domains = [
            "law.go.kr",
            "scourt.go.kr",
            "acrc.go.kr",
            "korea.kr",
            "go.kr",
            "moj.go.kr",
            "police.go.kr",
            "easylaw.go.kr",
            "moleg.go.kr",
        ]
        self.blacklist_domains = [
            "blog.naver.com",
            "m.blog.naver.com",
            "cafe.naver.com",
            "m.cafe.naver.com",
            "post.naver.com",
            "tistory.com",
            "brunch.co.kr",
            "youtube.com",
            "youtu.be",
            "instagram.com",
            "facebook.com",
            "namu.wiki",
            "kin.naver.com",
        ]
        self.signal_keywords = [
            "행정심판",
            "재결",
            "처분",
            "과태료",
            "이행명령",
            "사전통지",
            "의견제출",
            "청문",
            "행정절차법",
            "판결",
            "판례",
            "대법원",
            "조례",
            "시행규칙",
            "고시",
            "훈령",
            "예규",
            "지침",
            "공고",
        ]

    def _headers(self):
        return {
            "X-Naver-Client-Id": self.client_id or "",
            "X-Naver-Client-Secret": self.client_secret or "",
        }

    def _naver_search(self, url: str, query: str, display: int = 10):
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
        # ✅ 414 방어 전제(짧게!)
        prompt = f"""
당신은 행정 데이터 검색 전문가입니다.
아래 상황을 해결하기 위한 '검색 키워드'만 출력하세요.

[상황]: "{situation}"

[규칙]
- 문장 금지, 키워드만
- 공백 기준 10토큰 이하
- 60자 이하
- 행정 실무 용어 포함(처분/불복/재결/과태료/사전통지/의견제출 등)
"""
        try:
            q = (llm_service.generate_text(prompt) or "").strip()
            q = re.sub(r'["\']', "", q)
            return self._shrink_query(q, max_tokens=10, max_chars=80)
        except Exception:
            return self._shrink_query(situation, max_tokens=10, max_chars=80)

    def _rerank_results_llm(self, situation: str, candidate_items: list) -> list:
        if not candidate_items:
            return []
        ctx = ""
        for idx, item in enumerate(candidate_items[:7]):
            ctx += f"[{idx}] 제목: {item['title']} / 내용: {item['desc']} / 출처: {item['domain']}\n"

        prompt = f"""
[역할] 베테랑 행정 공무원
[상황] "{situation}"
[임무] 아래 후보 중 신뢰/실무도움 순서로 인덱스만 JSON 리스트로 출력.

[후보]
{ctx}

출력 예: [2,0,5]
"""
        try:
            ranking = llm_service.generate_json(prompt)
            if isinstance(ranking, list) and ranking:
                out = []
                for i in ranking:
                    if isinstance(i, int) and 0 <= i < len(candidate_items):
                        out.append(candidate_items[i])
                return out or candidate_items
            return candidate_items
        except Exception:
            return candidate_items

    def search_precedents(self, situation: str, top_k: int = 3) -> str:
        if not self.client_id or not self.client_secret:
            return "⚠️ 네이버 API 키가 설정되지 않았습니다."

        optimized_query = self._optimize_query_llm(situation)

        # ✅ 414 방지: 짧게 + site 하나
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
                    merged.append(
                        {
                            "src": src_name,
                            "title": title,
                            "desc": desc,
                            "link": link,
                            "domain": self._get_domain(link),
                            "score": score,
                        }
                    )

        if not merged:
            # site 제한이 너무 세면 한 번 더 완화
            try:
                fallback_query = self._shrink_query(optimized_query, max_tokens=6, max_chars=50)
                web_res = self._naver_search(self.web_url, fallback_query, display=10)
                for it in (web_res.get("items", []) or []):
                    link = it.get("link", "#")
                    title = self._clean_html(it.get("title", ""))
                    desc = self._clean_html(it.get("description", ""))
                    score = self._score_item(title, desc, link)
                    if score > -100:
                        merged.append(
                            {
                                "src": "웹",
                                "title": title,
                                "desc": desc,
                                "link": link,
                                "domain": self._get_domain(link),
                                "score": score,
                            }
                        )
            except Exception:
                pass

        if not merged:
            return f"검색어 '{optimized_query}'에 대한 유의미한 결과가 없습니다."

        merged.sort(key=lambda x: x["score"], reverse=True)
        candidates = merged[:7]
        final_items = self._rerank_results_llm(situation, candidates)[:top_k]

        lines = [
            f"🔍 **AI 최적화 검색어:** `{optimized_query}`",
            f"🧠 **AI 선별 결과 (Top {len(final_items)})**",
            "---",
        ]
        for it in final_items:
            lines.append(f"- ({it['src']}) **[{it['title']}]({it['link']})** `[{it['domain']}]`\n  : {it['desc']}")
        return "\n".join(lines)


class DatabaseService:
    """Supabase Persistence Layer"""

    def __init__(self):
        try:
            self.url = _get_secret("supabase", "SUPABASE_URL")
            self.key = _get_secret("supabase", "SUPABASE_KEY")
            if self.url and self.key:
                self.client = create_client(self.url, self.key)
                self.is_active = True
            else:
                self.client = None
                self.is_active = False
        except Exception:
            self.client = None
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
    """
    국가법령정보센터(law.go.kr) 공식 API 연동
    """

    def __init__(self):
        self.api_id = _get_secret("general", "LAW_API_ID")
        self.base_url = "http://www.law.go.kr/DRF/lawSearch.do"
        self.service_url = "http://www.law.go.kr/DRF/lawService.do"

    def get_law_text(self, law_name, article_num=None):
        if not self.api_id:
            return "⚠️ API ID(OC)가 설정되지 않았습니다."

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

        # 2) 상세 -> 조문 탐색
        try:
            detail_params = {"OC": self.api_id, "target": "law", "type": "XML", "MST": mst_id}
            res_detail = requests.get(self.service_url, params=detail_params, timeout=12)
            root_detail = ET.fromstring(res_detail.content)

            # article_num 없으면 링크만
            if not article_num:
                return f"✅ '{law_name}'이(가) 확인되었습니다.\n🔗 원문 보기: {full_link}"

            for article in root_detail.findall(".//조문단위"):
                jo_num = article.find("조문번호")
                jo_cont = article.find("조문내용")
                if jo_num is None or jo_cont is None:
                    continue
                current_num = (jo_num.text or "").strip()
                if str(article_num) == current_num:
                    body = (jo_cont.text or "").strip()
                    out = f"[{law_name} 제{current_num}조 전문]\n{_escape(body)}"
                    # 항내용
                    for hang in article.findall(".//항"):
                        hang_cont = hang.find("항내용")
                        if hang_cont is not None and hang_cont.text:
                            out += f"\n  - {hang_cont.text.strip()}"
                    return out

            return f"✅ '{law_name}'이(가) 확인되었습니다.\n(요청 조문 자동 추출 실패: 제{article_num}조)\n🔗 원문 보기: {full_link}"

        except Exception as e:
            return f"상세 법령 파싱 실패: {e}"


# ==========================================
# 3. Agents
# ==========================================
class LegalAgents:
    @staticmethod
    def researcher(situation):
        """
        - 실체법 + 정의/반박 + 절차법까지 최대 4개
        - category를 넣어 strategist가 절차법 인용을 강제하기 쉬움
        """
        prompt_extract = f"""
상황: "{situation}"

필요 법령/조문을 아래 카테고리에 맞춰 최대 4개까지 JSON 리스트로 추출.
카테고리:
- violation (실체법 위반조항)
- definition (정의/반박 조항)
- procedure (절차/주의 조항: 행정절차법, 질서위반행위규제법 등)

형식:
[
  {{"category":"violation","law_name":"도로교통법","article_num":32}},
  {{"category":"definition","law_name":"도로교통법","article_num":2}},
  {{"category":"procedure","law_name":"행정절차법","article_num":21}},
  {{"category":"procedure","law_name":"질서위반행위규제법","article_num":16}}
]

규칙:
- 법령명은 정식 명칭
- 조문 번호 불명확하면 null
- 반드시 JSON만 출력
"""
        try:
            extracted = llm_service.generate_json(prompt_extract)
            if isinstance(extracted, dict):
                extracted = [extracted]
            search_targets = extracted if isinstance(extracted, list) and extracted else []
        except Exception:
            search_targets = []

        if not search_targets:
            search_targets = [{"category": "violation", "law_name": "도로교통법", "article_num": None}]

        report_lines = [f"🔍 **AI가 식별한 핵심 법령 (실체+절차) {len(search_targets)}건**", "---"]
        api_success_count = 0

        for idx, item in enumerate(search_targets):
            category = item.get("category", "etc")
            law_name = item.get("law_name", "관련법령")
            article_num = item.get("article_num")

            real_law_text = law_api_service.get_law_text(law_name, article_num)

            error_keywords = ["검색 결과가 없습니다", "오류", "API ID", "실패"]
            is_success = not any(k in real_law_text for k in error_keywords)

            if is_success:
                api_success_count += 1
                header = f"✅ **{idx+1}. [{category}] {law_name} 제{article_num}조 (확인됨)**"
                content = real_law_text
            else:
                header = f"⚠️ **{idx+1}. [{category}] {law_name} 제{article_num}조 (API 조회 실패)**"
                content = "(국가법령정보센터에서 해당 조문을 찾지 못했습니다. 법령명/조문번호를 확인해주세요.)"

            report_lines.append(f"{header}\n{content}\n")

        if api_success_count == 0:
            ai_fallback = llm_service.generate_text(
                f"""[AI 추론 결과]
상황: "{situation}"
법령 API 연결 실패 상태. 필요한 실체법/절차법을 최대한 정확히 설명하되,
반드시 '법제처 확인 필수' 경고를 포함해서 작성."""
            ).strip()
            return f"⚠️ **[시스템 경고: API 조회 실패]**\n(법제처 확인 필수)\n------------------\n{ai_fallback}"

        return "\n".join(report_lines)

    @staticmethod
    def strategist(situation, legal_basis, search_results):
        prompt = f"""
당신은 행정 업무 베테랑 '주무관'입니다.

[민원 상황]
{situation}

[확보된 법령 데이터(실체/정의/절차)]
{legal_basis}

[유사 사례(검색)]
{search_results}

[출력 양식(마크다운)]
1) **처리 방향**
- 적용 위반조항(실체법)으로 무엇을 어떻게 처리할지
- 민원인의 주장 반박(정의/판단 근거)

2) **핵심 주의사항(절차 매핑 필수)**
- 3~4개 항목
- 각 항목 끝에 근거 법령/조문을 반드시 표기 (행정절차법/질서위반행위규제법 등)

3) **예상 반발 및 대응**
- 예상 주장 2~3개와 응대 멘트
"""
        return llm_service.generate_text(prompt)

    @staticmethod
    def clerk(situation, legal_basis):
        today = datetime.now()
        prompt = f"""
오늘: {today.strftime('%Y-%m-%d')}
상황: {situation}
법령: {legal_basis}

사전통지/이행명령/의견제출 등 통상 부여하는 기간(일수)을 숫자만 출력.
(예: 10, 15, 20) 모르면 15.
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
    def drafter(situation, legal_basis, meta_info, strategy):
        prompt = f"""
너는 행정기관 베테랑 서기다. 아래 정보를 바탕으로 공문서 JSON을 생성하라.
반드시 JSON만 출력.

[입력]
- 민원 상황: {situation}
- 법적 근거: {legal_basis}
- 시행 일자: {meta_info['today_str']}
- 기한: {meta_info['deadline_str']} ({meta_info['days_added']}일)
- 전략:
{strategy}

[출력 JSON 스키마]
{{
  "title": "공문서 제목",
  "receiver": "수신인",
  "body_paragraphs": ["문단1", "문단2", "문단3"],
  "department_head": "발신 명의"
}}

[작성 원칙]
- 본문에 실체법 + 절차법 인용(가능하면 조문번호 포함)
- 구조: 경위 -> 법적근거(위반/정의/절차) -> 처분/조치 -> 권리구제/이의제기 안내
- 개인정보는 OOO로 마스킹
"""
        data = llm_service.generate_json(prompt)
        if not isinstance(data, dict):
            return {
                "title": "공 문 서",
                "receiver": "수신자 참조",
                "body_paragraphs": ["(문서 생성에 실패했습니다. 입력/키 설정을 확인하세요.)"],
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
# 2.5 Global service instances (⭐ 미정의 해결)
# ==========================================
llm_service = LLMService()
search_service = SearchService()
db_service = DatabaseService()
law_api_service = LawOfficialService()


# ==========================================
# 4. Workflow (UI 로직)
# ==========================================
def run_workflow(user_input: str):
    log_placeholder = st.empty()
    logs = []

    def add_log(msg, style="sys"):
        logs.append(f"<div class='agent-log log-{style}'>{_escape(msg)}</div>")
        log_placeholder.markdown("".join(logs), unsafe_allow_html=True)
        time.sleep(0.25)

    add_log("🔍 Phase 1: 법령 및 유사 사례 리서치 중...", "legal")
    legal_basis = LegalAgents.researcher(user_input)
    add_log("📜 법적 근거 수집 완료", "legal")

    add_log("🟩 네이버 검색 엔진 가동...", "search")
    try:
        search_results = search_service.search_precedents(user_input)
    except Exception:
        search_results = "검색 모듈 오류 (건너뜀)"

    add_log("🧠 Phase 2: AI 주무관이 업무 처리 방향을 수립합니다...", "strat")
    strategy = LegalAgents.strategist(user_input, legal_basis, search_results)

    add_log("📅 Phase 3: 기한 산정 및 공문서 작성 시작...", "calc")
    meta_info = LegalAgents.clerk(user_input, legal_basis)

    add_log("✍️ 최종 공문서 조판 중...", "draft")
    doc_data = LegalAgents.drafter(user_input, legal_basis, meta_info, strategy)

    add_log("💾 업무 기록을 DB(Supabase)에 저장 중...", "sys")
    save_result = db_service.save_log(user_input, legal_basis, strategy, doc_data)

    add_log(f"✅ 완료되었습니다. ({save_result})", "sys")
    time.sleep(0.8)
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
# 5. Presentation Layer (UI)
# ==========================================
def main():
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.title("🏢 AI 행정관 Pro")
        st.caption("Gemini(google.genai) + 국가법령정보 + Naver Search + Strategy + DB")
        st.markdown("---")

        # ✅ 상태 점검(죽지 않고 경고만)
        warn_lines = []
        if not llm_service.gemini_key and not llm_service.groq_key:
            warn_lines.append("- LLM 키 없음: `GEMINI_API_KEY` 또는 `GROQ_API_KEY`가 필요합니다.")
        if not law_api_service.api_id:
            warn_lines.append("- 법령 API 키 없음: `LAW_API_ID(OC)`가 없으면 법령 전문 추출이 제한됩니다.")
        if not search_service.client_id or not search_service.client_secret:
            warn_lines.append("- 네이버 키 없음: `NAVER_CLIENT_ID/SECRET` 없으면 유사사례 검색이 꺼집니다.")
        if warn_lines:
            st.warning("현재 설정 상태:\n" + "\n".join(warn_lines))

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

            if "성공" in (res.get("save_msg") or ""):
                st.success(f"✅ {res['save_msg']}")
            else:
                st.info(f"ℹ️ {res.get('save_msg')}")

            with st.expander("✅ [검토] 법령 및 유사 사례 확인", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📜 적용 법령**")
                    st.code(res.get("law", ""), language="text")
                with col2:
                    st.markdown("**🟩 네이버 유사 사례**")
                    st.info(res.get("search", ""))

            with st.expander("🧭 [방향] 업무 처리 가이드라인", expanded=True):
                st.markdown(res.get("strategy", ""))

    with col_right:
        if "workflow_result" in st.session_state:
            res = st.session_state["workflow_result"]
            doc = res.get("doc") or {}
            meta = res.get("meta") or {}

            html_content = f"""
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
                html_content += f"<p style='margin-bottom: 15px;'>{_escape(str(p))}</p>"

            html_content += f"""
  </div>
  <div class="doc-footer">{_escape(str(doc.get('department_head', '행정기관장')))}</div>
</div>
"""
            st.markdown(html_content, unsafe_allow_html=True)

        else:
            st.markdown(
                """<div style='text-align: center; padding: 100px; color: #aaa; background: white; border-radius: 10px; border: 2px dashed #ddd;'>
<h3>📄 Document Preview</h3><p>왼쪽에서 업무를 지시하면<br>완성된 공문서가 여기에 나타납니다.</p></div>""",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
