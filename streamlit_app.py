import streamlit as st
import google.generativeai as genai
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
# 1. Configuration & Styles (설정 및 디자인)
# ==========================================
st.set_page_config(layout="wide", page_title="AI Bureau: The Legal Glass", page_icon="⚖️")

st.markdown(
    """
<style>
    /* 배경: 차분한 오피스 톤 */
    .stApp { background-color: #f3f4f6; }

    /* 결과물: A4 용지 스타일 */
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

    /* 공문서 내부 스타일 */
    .doc-header { text-align: center; font-size: 22pt; font-weight: 900; margin-bottom: 30px; letter-spacing: 2px; }
    .doc-info { display: flex; justify-content: space-between; font-size: 11pt; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; }
    .doc-body { font-size: 12pt; text-align: justify; white-space: pre-line; }
    .doc-footer { text-align: center; font-size: 20pt; font-weight: bold; margin-top: 80px; letter-spacing: 5px; }
    .stamp { position: absolute; bottom: 85px; right: 80px; border: 3px solid #cc0000; color: #cc0000; padding: 5px 10px; font-size: 14pt; font-weight: bold; transform: rotate(-15deg); opacity: 0.8; border-radius: 5px; }

    /* 로그 스타일 */
    .agent-log { font-family: 'Consolas', monospace; font-size: 0.85rem; padding: 6px 12px; border-radius: 6px; margin-bottom: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    .log-legal { background-color: #eff6ff; color: #1e40af; border-left: 4px solid #3b82f6; } /* Blue */
    .log-search { background-color: #fff7ed; color: #c2410c; border-left: 4px solid #f97316; } /* Orange */
    .log-strat { background-color: #f5f3ff; color: #6d28d9; border-left: 4px solid #8b5cf6; } /* Purple */
    .log-calc { background-color: #f0fdf4; color: #166534; border-left: 4px solid #22c55e; } /* Green */
    .log-draft { background-color: #fef2f2; color: #991b1b; border-left: 4px solid #ef4444; } /* Red */
    .log-sys { background-color: #f3f4f6; color: #4b5563; border-left: 4px solid #9ca3af; } /* Gray */

    /* 전략 박스 스타일 */
    .strategy-box { background-color: #fffbeb; border: 1px solid #fcd34d; padding: 15px; border-radius: 8px; margin-bottom: 15px; }
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
        self.gemini_key = st.secrets["general"].get("GEMINI_API_KEY")
        self.groq_key = st.secrets["general"].get("GROQ_API_KEY")

        self.gemini_models = [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash"
        ]

        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)

        self.groq_client = Groq(api_key=self.groq_key) if self.groq_key else None

    def _try_gemini(self, prompt, is_json=False, schema=None):
        for model_name in self.gemini_models:
            try:
                model = genai.GenerativeModel(model_name)
                config = genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=schema
                ) if is_json else None

                res = model.generate_content(prompt, generation_config=config)
                return res.text, model_name
            except Exception:
                continue
        raise Exception("All Gemini models failed")

    def generate_text(self, prompt):
        try:
            text, _model_used = self._try_gemini(prompt, is_json=False)
            return text
        except Exception:
            if self.groq_client:
                return self._generate_groq(prompt)
            return "시스템 오류: AI 모델 연결 실패"

    def generate_json(self, prompt, schema=None):
        try:
            text, _model_used = self._try_gemini(prompt, is_json=True, schema=schema)
            return json.loads(text)
        except Exception:
            text = self.generate_text(prompt + "\n\nOutput strictly in JSON.")
            try:
                match = re.search(r"\{.*\}", text, re.DOTALL)
                return json.loads(match.group(0)) if match else None
            except Exception:
                return None

    def _generate_groq(self, prompt):
        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return completion.choices[0].message.content
        except Exception:
            return "System Error"


class SearchService:
    """
    ✅ Hybrid Search Engine
    1. AI Query Optimizer
    2. Heuristic Filter
    3. LLM Re-ranking
    """
    def __init__(self):
        g = st.secrets.get("general", {})
        self.client_id = g.get("NAVER_CLIENT_ID")
        self.client_secret = g.get("NAVER_CLIENT_SECRET")

        self.web_url = "https://openapi.naver.com/v1/search/webkr.json"
        self.news_url = "https://openapi.naver.com/v1/search/news.json"

        self.whitelist_domains = [
            "law.go.kr", "scourt.go.kr", "acrc.go.kr", "korea.kr",
            "go.kr", "moj.go.kr", "police.go.kr", "easylaw.go.kr", "moleg.go.kr"
        ]

        self.blacklist_domains = [
            "blog.naver.com", "m.blog.naver.com", "cafe.naver.com", "m.cafe.naver.com",
            "post.naver.com", "tistory.com", "brunch.co.kr", "youtube.com",
            "youtu.be", "instagram.com", "facebook.com", "namu.wiki", "kin.naver.com"
        ]

        self.signal_keywords = [
            "행정심판", "재결", "처분", "과태료", "이행명령", "사전통지",
            "의견제출", "청문", "행정절차법", "판결", "판례", "대법원",
            "조례", "시행규칙", "고시", "훈령", "예규", "지침", "공고"
        ]

    def _headers(self):
        return {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
        }

    # ✅ 필수: 누락됐던 네이버 검색 함수
    def _naver_search(self, url: str, query: str, display: int = 10):
        params = {"query": query, "display": display, "start": 1, "sort": "sim"}
        res = requests.get(url, headers=self._headers(), params=params, timeout=8)
        res.raise_for_status()
        return res.json()

    def _clean_html(self, s: str) -> str:
        if not s: return ""
        s = re.sub(r"<\/?b>", "", s)
        s = re.sub(r"<[^>]+>", "", s)
        return s.strip()

    def _get_domain(self, link: str) -> str:
        if not link: return ""
        m = re.search(r"https?://([^/]+)", link)
        return (m.group(1).lower() if m else "").strip()

    def _is_blacklisted(self, domain: str) -> bool:
        d = domain.lower()
        for bad in self.blacklist_domains:
            if bad in d: return True
        return False

    def _whitelist_score(self, domain: str) -> int:
        d = domain.lower()
        score = 0
        for good in self.whitelist_domains:
            if good == "go.kr":
                if d.endswith(".go.kr") or d == "go.kr" or ".go.kr" in d: score += 8
            else:
                if good in d: score += 10
        return score

    def _keyword_score(self, text: str) -> int:
        t = (text or "").lower()
        score = 0
        for kw in self.signal_keywords:
            if kw.lower() in t: score += 2
        return score

    def _score_item(self, title: str, desc: str, link: str) -> int:
        domain = self._get_domain(link)
        if self._is_blacklisted(domain): return -999
        score = 0
        score += self._whitelist_score(domain)
        score += self._keyword_score(title) * 2
        score += self._keyword_score(desc)
        if len((desc or "").strip()) < 25: score -= 3
        if not (link or "").startswith("http"): score -= 5
        return score

    def _optimize_query_llm(self, situation: str) -> str:
        prompt = f"""
당신은 행정 데이터 검색 전문가입니다.
아래 민원 상황을 해결하기 위해 네이버에서 검색할 '최적의 키워드'를 생성하세요.

[민원 상황]: "{situation}"

[요청사항]
1. 행정 실무 용어(예: 처분, 불복, 재결례)를 포함하세요.
2. 조사나 서술어를 뺀 '명사형 키워드' 위주로 작성하세요.
출력 예시: 도로교통법 제32조 주정차위반 의견제출 인용 사례
"""
        try:
            query = llm_service.generate_text(prompt).strip()
            return re.sub(r'["\']', "", query)
        except Exception:
            return situation

    def _rerank_results_llm(self, situation: str, candidate_items: list) -> list:
        if not candidate_items:
            return []
        context_text = ""
        for idx, item in enumerate(candidate_items[:7]):
            context_text += f"[{idx}] 제목: {item['title']} / 내용: {item['desc']} / 출처: {item['domain']}\n"

        prompt = f"""
[역할]: 베테랑 행정 공무원
[상황]: "{situation}"
[임무]: 위 상황을 처리할 때, 아래 검색 결과 중 '가장 신뢰할 수 있고 참고가 되는 자료'를 순서대로 선택하시오.

[검색 결과 후보]
{context_text}

[출력 형식 - JSON List]
도움이 되는 순서대로 인덱스 번호(숫자)만 리스트로 출력하세요.
예: [2, 0, 5]
"""
        try:
            ranking_indices = llm_service.generate_json(prompt)
            if isinstance(ranking_indices, list):
                reranked = []
                for idx in ranking_indices:
                    if isinstance(idx, int) and 0 <= idx < len(candidate_items):
                        reranked.append(candidate_items[idx])
                return reranked
            return candidate_items
        except Exception:
            return candidate_items

    def search_precedents(self, situation: str, top_k: int = 3) -> str:
        if not self.client_id or not self.client_secret:
            return "⚠️ 네이버 API 키가 설정되지 않았습니다."

        try:
            optimized_query = self._optimize_query_llm(situation)
            final_query = f"{optimized_query} (site:go.kr OR site:kr OR 판례 OR 재결)"

            web_res = self._naver_search(self.web_url, final_query, display=10)
            news_res = self._naver_search(self.news_url, final_query, display=10)

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

            final_items = self._rerank_results_llm(situation, candidates)
            if not final_items:
                final_items = candidates[:top_k]
            else:
                final_items = final_items[:top_k]

            lines = []
            lines.append(f"🔍 **AI 최적화 검색어:** `{optimized_query}`")
            lines.append(f"🧠 **AI 선별 결과 (Top {len(final_items)})**")
            lines.append("---")
            for it in final_items:
                lines.append(f"- ({it['src']}) **[{it['title']}]({it['link']})** `[{it['domain']}]`\n  : {it['desc']}")
            return "\n".join(lines)

        except requests.HTTPError as e:
            return f"네이버 API 호출 오류: {e}"
        except Exception as e:
            return f"검색 프로세스 중 오류: {e}"


class DatabaseService:
    """Supabase Persistence Layer"""
    def __init__(self):
        try:
            self.url = st.secrets["supabase"]["SUPABASE_URL"]
            self.key = st.secrets["supabase"]["SUPABASE_KEY"]
            self.client = create_client(self.url, self.key)
            self.is_active = True
        except Exception:
            self.is_active = False

    def save_log(self, user_input, legal_basis, strategy, doc_data):
        if not self.is_active:
            return "DB 미연결 (저장 건너뜀)"

        try:
            final_summary_content = {
                "strategy": strategy,
                "document_content": doc_data,
            }

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
    1. 검색: 법령명 -> 법령 ID(MST) 추출
    2. 조회: 법령 ID -> 전체 조문 파싱 -> 특정 조문 검색
    """
    def __init__(self):
        self.api_id = st.secrets["general"].get("LAW_API_ID")
        self.base_url = "http://www.law.go.kr/DRF/lawSearch.do"
        self.service_url = "http://www.law.go.kr/DRF/lawService.do"

    def get_law_text(self, law_name, article_num=None):
        if not self.api_id:
            return "⚠️ API ID(OC)가 설정되지 않았습니다."

        # 1) 법령 ID(MST) 검색
        try:
            params = {
                "OC": self.api_id,
                "target": "law",
                "type": "XML",
                "query": law_name,
                "display": 1
            }
            res = requests.get(self.base_url, params=params, timeout=5)
            root = ET.fromstring(res.content)

            law_node = root.find(".//law")
            if law_node is None:
                return f"🔍 '{law_name}'에 대한 검색 결과가 없습니다."

            mst_id = law_node.find("법령일련번호").text
            full_link = law_node.find("법령상세링크").text
        except Exception as e:
            return f"API 검색 중 오류: {e}"

        # 2) 상세 조문 가져오기
        try:
            detail_params = {
                "OC": self.api_id,
                "target": "law",
                "type": "XML",
                "MST": mst_id
            }
            res_detail = requests.get(self.service_url, params=detail_params, timeout=10)
            root_detail = ET.fromstring(res_detail.content)

            found = False
            target_text = ""

            for article in root_detail.findall(".//조문단위"):
                jo_num_tag = article.find("조문번호")
                jo_content_tag = article.find("조문내용")

                if jo_num_tag is not None and jo_content_tag is not None:
                    current_num = jo_num_tag.text.strip()

                    if article_num and str(article_num) == current_num:
                        target_text = f"[{law_name} 제{current_num}조 전문]\n" + _escape((jo_content_tag.text or "").strip())

                        for hang in article.findall(".//항"):
                            hang_content = hang.find("항내용")
                            if hang_content is not None:
                                target_text += f"\n  - {(hang_content.text or '').strip()}"
                        found = True
                        break

            if found:
                return target_text

            return f"✅ '{law_name}'이(가) 확인되었습니다.\n(상세 조문 자동 추출 실패 또는 전체 법령 참조)\n🔗 원문 보기: {full_link}"

        except Exception as e:
            return f"상세 법령 파싱 실패: {e}"


# ==========================================
# 3. Global Service Instances (핵심!)
# ==========================================
llm_service = LLMService()
search_service = SearchService()
db_service = DatabaseService()
law_api_service = LawOfficialService()


class LegalAgents:
    @staticmethod
    def researcher(situation):
        prompt_extract = f"""
상황: "{situation}"

위 민원 처리를 위해 법적 근거로 삼아야 할 핵심 대한민국 법령과 조문 번호를
**중요도 순으로 최대 3개까지** JSON 리스트로 추출하시오.

형식: [{{"law_name": "도로교통법", "article_num": 32}}, {{"law_name": "도로교통법", "article_num": 2}}, ...]
* 법령명은 정식 명칭 사용. 조문 번호 불명확하면 null.
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

        report_lines = []
        api_success_count = 0

        report_lines.append(f"🔍 **AI가 식별한 핵심 법령 ({len(search_targets)}건)**")
        report_lines.append("---")

        for idx, item in enumerate(search_targets):
            law_name = item.get("law_name", "관련법령")
            article_num = item.get("article_num")

            real_law_text = law_api_service.get_law_text(law_name, article_num)

            error_keywords = ["검색 결과가 없습니다", "오류", "API ID", "실패"]
            is_success = not any(k in real_law_text for k in error_keywords)

            if is_success:
                api_success_count += 1
                header = f"✅ **{idx+1}. {law_name} 제{article_num}조 (확인됨)**"
                content = real_law_text
            else:
                header = f"⚠️ **{idx+1}. {law_name} 제{article_num}조 (API 조회 실패)**"
                content = "(국가법령정보센터에서 해당 조문을 찾지 못했습니다. 법령명이 정확한지 확인이 필요합니다.)"

            report_lines.append(f"{header}\n{content}\n")

        final_report = "\n".join(report_lines)

        if api_success_count == 0:
            prompt_fallback = f"""
Role: 행정 법률 전문가
Task: 아래 상황에 적용될 법령과 조항을 찾아 설명하시오.
상황: "{situation}"

* 경고: 현재 외부 법령 API 연결이 원활하지 않습니다.
당신이 알고 있는 지식을 바탕으로 가장 정확한 법령 정보를 작성하되,
반드시 상단에 [AI 추론 결과]임을 명시하고 환각 가능성을 경고하시오.
"""
            ai_fallback_text = llm_service.generate_text(prompt_fallback).strip()

            return f"""⚠️ **[시스템 경고: API 조회 실패]**
(국가법령정보센터 연결에 실패하여 AI의 지식 기반으로 답변을 생성합니다. **환각(Hallucination)** 가능성이 있으므로 법제처 확인이 필수입니다.)

--------------------------------------------------
{ai_fallback_text}"""

        return final_report

    @staticmethod
    def strategist(situation, legal_basis, search_results):
        prompt = f"""
당신은 행정 업무 베테랑 '주무관'입니다.

[민원 상황]: {situation}
[확보된 법적 근거]:
{legal_basis}

[유사 사례/판례]: {search_results}

위 정보를 종합하여 이 민원을 처리하기 위한 **대략적인 업무 처리 방향(Strategy)**을 수립하세요.

다음 3가지 항목 포함:
1. 처리 방향
2. 핵심 주의사항
3. 예상 반발 및 대응
"""
        return llm_service.generate_text(prompt)

    @staticmethod
    def clerk(situation, legal_basis):
        today = datetime.now()
        prompt = f"""
오늘: {today.strftime('%Y-%m-%d')}
상황: {situation}
법령: {legal_basis}
이행/의견제출 기간은 며칠인가?
숫자만 출력. 모르겠으면 15.
"""
        try:
            res = llm_service.generate_text(prompt)
            days = int(re.sub(r"[^0-9]", "", res))
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
        doc_schema = {
            "type": "OBJECT",
            "properties": {
                "title": {"type": "STRING", "description": "공문서 제목"},
                "receiver": {"type": "STRING", "description": "수신인"},
                "body_paragraphs": {"type": "ARRAY", "items": {"type": "STRING"}},
                "department_head": {"type": "STRING", "description": "발신 명의"},
            },
            "required": ["title", "receiver", "body_paragraphs", "department_head"],
        }

        prompt = f"""
당신은 행정기관의 베테랑 서기입니다. 아래 정보를 바탕으로 완결된 공문서를 작성하세요.

[입력 정보]
- 민원 상황: {situation}
- 법적 근거: {legal_basis}
- 시행 일자: {meta_info['today_str']}
- 기한: {meta_info['deadline_str']} ({meta_info['days_added']}일)

[전략]
{strategy}

[작성 원칙]
1. 본문에 법 조항 인용 필수
2. 본문 구조: 경위 -> 법적 근거 -> 처분 내용 -> 이의제기 절차
3. 개인정보 마스킹('OOO')
"""
        return llm_service.generate_json(prompt, schema=doc_schema)


# ==========================================
# 4. Workflow (UI 로직)
# ==========================================
def run_workflow(user_input):
    log_placeholder = st.empty()
    logs = []

    def add_log(msg, style="sys"):
        logs.append(f"<div class='agent-log log-{style}'>{_escape(msg)}</div>")
        log_placeholder.markdown("".join(logs), unsafe_allow_html=True)
        time.sleep(0.3)

    add_log("🔍 Phase 1: 법령 및 유사 사례 리서치 중...", "legal")
    legal_basis = LegalAgents.researcher(user_input)
    add_log("📜 법적 근거 발견 완료", "legal")

    add_log("🟩 네이버 검색 엔진 가동...", "search")
    try:
        search_results = search_service.search_precedents(user_input)
    except Exception:
        search_results = "검색 모듈 미연결 (건너뜀)"

    add_log("🧠 Phase 2: AI 주무관이 업무 처리 방향을 수립합니다...", "strat")
    strategy = LegalAgents.strategist(user_input, legal_basis, search_results)

    add_log("📅 Phase 3: 기한 산정 및 공문서 작성 시작...", "calc")
    meta_info = LegalAgents.clerk(user_input, legal_basis)

    add_log("✍️ 최종 공문서 조판 중...", "draft")
    doc_data = LegalAgents.drafter(user_input, legal_basis, meta_info, strategy)

    add_log("💾 업무 기록을 데이터베이스(Supabase)에 저장 중...", "sys")
    save_result = db_service.save_log(user_input, legal_basis, strategy, doc_data)

    add_log(f"✅ 모든 행정 절차가 완료되었습니다. ({save_result})", "sys")
    time.sleep(1)
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
        st.caption("Gemini + 국가법령정보 + Naver Search + Strategy + DB")
        st.markdown("---")

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

            if "성공" in res["save_msg"]:
                st.success(f"✅ {res['save_msg']}")
            else:
                st.error(f"❌ {res['save_msg']}")

            with st.expander("✅ [검토] 법령 및 유사 사례 확인", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📜 적용 법령**")
                    st.code(res["law"], language="text")
                with col2:
                    st.markdown("**🟩 네이버 유사 사례**")
                    st.info(res["search"])

            with st.expander("🧭 [방향] 업무 처리 가이드라인", expanded=True):
                st.markdown(res["strategy"])

    with col_right:
        if "workflow_result" in st.session_state:
            res = st.session_state["workflow_result"]
            doc = res["doc"]
            meta = res["meta"]

            if doc:
                html_content = f"""
<div class="paper-sheet">
  <div class="stamp">직인생략</div>
  <div class="doc-header">{_escape(doc.get('title', '공 문 서'))}</div>
  <div class="doc-info">
    <span>문서번호: {_escape(meta['doc_num'])}</span>
    <span>시행일자: {_escape(meta['today_str'])}</span>
    <span>수신: {_escape(doc.get('receiver', '수신자 참조'))}</span>
  </div>
  <hr style="border: 1px solid black; margin-bottom: 30px;">
  <div class="doc-body">
"""
                paragraphs = doc.get("body_paragraphs", [])
                if isinstance(paragraphs, str):
                    paragraphs = [paragraphs]

                for p in paragraphs:
                    html_content += f"<p style='margin-bottom: 15px;'>{_escape(p)}</p>"

                html_content += f"""
  </div>
  <div class="doc-footer">{_escape(doc.get('department_head', '행정기관장'))}</div>
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
