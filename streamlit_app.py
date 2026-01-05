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
    ✅ Naver Search OpenAPI Wrapper (Web + News) + 강력 정제 필터
    - webkr/news 결과를 합쳐서:
      1) 도메인 화이트리스트 가점
      2) 도메인 블랙리스트 강제 제외
      3) 실무 키워드 포함 여부로 점수화
      4) 상위 N개만 반환
    """
    def __init__(self):
        g = st.secrets.get("general", {})
        self.client_id = g.get("NAVER_CLIENT_ID")
        self.client_secret = g.get("NAVER_CLIENT_SECRET")

        self.web_url = "https://openapi.naver.com/v1/search/webkr.json"
        self.news_url = "https://openapi.naver.com/v1/search/news.json"

        # ✅ “정제된 데이터”에 가까운 도메인 위주(가점)
        self.whitelist_domains = [
            "law.go.kr",        # 국가법령정보센터
            "scourt.go.kr",     # 대법원
            "acrc.go.kr",       # 국민권익위(행정심판/민원)
            "korea.kr",         # 대한민국 정책브리핑/정부
            "go.kr",            # 지자체/정부기관
            "moj.go.kr",        # 법무부
            "police.go.kr",     # 경찰청
            "kgsp.go.kr",       # 법제처/유사기관 케이스 대비(있으면)
        ]

        # ❌ 뻘소리 양산 도메인(강제 제외)
        self.blacklist_domains = [
            "blog.naver.com",
            "m.blog.naver.com",
            "cafe.naver.com",
            "m.cafe.naver.com",
            "post.naver.com",
            "m.post.naver.com",
            "tistory.com",
            "brunch.co.kr",
            "youtube.com",
            "youtu.be",
            "instagram.com",
            "facebook.com",
            "namu.wiki",
        ]

        # ✅ 실무형 문서에 자주 등장하는 단서(가점/필터)
        self.signal_keywords = [
            "행정심판", "재결", "처분", "과태료", "이행명령",
            "사전통지", "의견제출", "청문", "행정절차법",
            "판결", "판례", "대법원", "조례", "시행규칙",
            "고시", "훈령", "예규", "지침", "업무처리",
            "공고", "공시송달"
        ]

    def _headers(self):
        return {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
        }

    def _clean_html(self, s: str) -> str:
        if not s:
            return ""
        s = re.sub(r"<\/?b>", "", s)
        s = re.sub(r"<[^>]+>", "", s)
        return s.strip()

    def _naver_search(self, url: str, query: str, display: int = 5):
        params = {
            "query": query,
            "display": display,
            "start": 1,
            "sort": "sim",
        }
        r = requests.get(url, headers=self._headers(), params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def _get_domain(self, link: str) -> str:
        # 링크에서 도메인만 추출 (정규식 간단 추출)
        if not link:
            return ""
        m = re.search(r"https?://([^/]+)", link)
        return (m.group(1).lower() if m else "").strip()

    def _is_blacklisted(self, domain: str) -> bool:
        d = domain.lower()
        for bad in self.blacklist_domains:
            if bad in d:
                return True
        return False

    def _whitelist_score(self, domain: str) -> int:
        d = domain.lower()
        score = 0
        for good in self.whitelist_domains:
            if good == "go.kr":
                # go.kr은 하위 도메인이 많으니 포함 검사
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

        # 블랙리스트면 탈락
        if self._is_blacklisted(domain):
            return -999

        score = 0

        # 화이트리스트 가점
        score += self._whitelist_score(domain)

        # 제목/설명 키워드 가점
        score += self._keyword_score(title) * 2
        score += self._keyword_score(desc)

        # 너무 짧은 설명은 감점 (의미없는 결과가 많음)
        if len((desc or "").strip()) < 25:
            score -= 3

        # 링크가 http(s) 아닌 경우 감점
        if not (link or "").startswith("http"):
            score -= 5

        return score

    def _build_query(self, situation: str) -> str:
        # 입력을 너무 길게 넣으면 검색 품질이 떨어짐
        q_core = re.sub(r"\s+", " ", (situation or "").strip())
        if len(q_core) > 80:
            q_core = q_core[:80] + "..."

        # ✅ 네이버에서도 어느 정도 먹히는 "공식문서" 유도 쿼리
        # (완벽한 site: 필터는 아니지만 효과 있음)
        official_hint = "(site:go.kr OR site:law.go.kr OR site:scourt.go.kr OR site:acrc.go.kr OR site:korea.kr)"
        intent_hint = "(행정심판 OR 재결 OR 판례 OR 처분 OR 과태료 OR 이행명령 OR 사전통지 OR 청문 OR 조례)"

        query = f"{q_core} {intent_hint} {official_hint}"
        return query

    def search_precedents(self, situation: str, top_k: int = 5) -> str:
        if not self.client_id or not self.client_secret:
            return "⚠️ 네이버 검색 API 키(NAVER_CLIENT_ID / NAVER_CLIENT_SECRET)가 없어 유사 사례를 조회할 수 없습니다."

        try:
            query = self._build_query(situation)

            # web + news 넉넉히 가져온 다음 필터링
            web = self._naver_search(self.web_url, query, display=10)
            news = self._naver_search(self.news_url, query, display=10)

            merged = []
            for src_name, payload in [("웹", web), ("뉴스", news)]:
                for it in (payload.get("items", []) or []):
                    title = self._clean_html(it.get("title", "제목 없음"))
                    desc = self._clean_html(it.get("description", "내용 없음"))
                    link = it.get("link", "#")

                    score = self._score_item(title, desc, link)
                    if score <= -100:
                        continue  # 블랙리스트/불량
                    merged.append({
                        "src": src_name,
                        "title": title,
                        "desc": desc,
                        "link": link,
                        "score": score,
                        "domain": self._get_domain(link)
                    })

            if not merged:
                return "관련된 유사 사례 검색 결과가 없습니다. (정제 필터 적용 후 결과가 비었습니다)"

            # 점수 순 정렬 + 중복 링크 제거
            merged.sort(key=lambda x: x["score"], reverse=True)
            seen = set()
            picked = []
            for it in merged:
                if it["link"] in seen:
                    continue
                seen.add(it["link"])
                picked.append(it)
                if len(picked) >= top_k:
                    break

            # 출력: 공식/비공식 분리 느낌으로 표기
            lines = []
            lines.append(f"**[네이버 정제 결과 Top {len(picked)}]**")
            for it in picked:
                title = it["title"]
                link = it["link"]
                desc = it["desc"]
                domain = it["domain"]
                src = it["src"]
                lines.append(f"- ({src}) **[{title}]({link})** `[{domain}]` : {desc}")

            return "\n".join(lines)

        except requests.HTTPError as e:
            return f"검색 중 HTTP 오류 발생: {e}"
        except Exception as e:
            return f"검색 중 오류 발생: {e}"


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
        # secrets.toml의 [general] 섹션에서 ID 로드
        self.api_id = st.secrets["general"].get("LAW_API_ID")
        self.base_url = "http://www.law.go.kr/DRF/lawSearch.do"
        self.service_url = "http://www.law.go.kr/DRF/lawService.do"

    def get_law_text(self, law_name, article_num=None):
        """
        law_name: "도로교통법"
        article_num: 2 (제2조를 찾고 싶을 때, 없으면 전체 요약이나 링크 반환)
        """
        if not self.api_id:
            return "⚠️ API ID(OC)가 설정되지 않았습니다."

        # 1단계: 법령 ID(MST) 검색
        try:
            params = {
                "OC": self.api_id,
                "target": "law",
                "type": "XML",
                "query": law_name,
                "display": 1  # 정확도순 1개만
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

        # 2단계: 상세 조문 가져오기
        try:
            detail_params = {
                "OC": self.api_id,
                "target": "law",
                "type": "XML",
                "MST": mst_id
            }
            res_detail = requests.get(self.service_url, params=detail_params, timeout=10)
            root_detail = ET.fromstring(res_detail.content)
            
            # 특정 조문 찾기 (예: article_num이 2이면 '제2조' 검색)
            target_text = ""
            
            # 조문 단위 순회
            found = False
            for article in root_detail.findall(".//조문단위"):
                # 조문번호 태그 확인
                # API XML 구조: <조문단위><조문번호>2</조문번호><조문내용>...</조문내용></조문단위>
                jo_num_tag = article.find("조문번호")
                jo_content_tag = article.find("조문내용")
                
                if jo_num_tag is not None and jo_content_tag is not None:
                    current_num = jo_num_tag.text.strip() # "2"
                    
                    # 사용자가 요청한 번호와 일치하는지 확인 (숫자만 비교)
                    if article_num and str(article_num) == current_num:
                        target_text = f"[{law_name} 제{current_num}조 전문]\n" + _escape(jo_content_tag.text.strip())
                        
                        # 항/호 내용이 별도 태그로 있는 경우도 긁어오기 (간소화)
                        for hang in article.findall(".//항"):
                            hang_content = hang.find("항내용")
                            if hang_content is not None:
                                target_text += f"\n  - {hang_content.text.strip()}"
                        found = True
                        break
            
            if found:
                return target_text
            else:
                # 조문을 못 찾았거나 번호 지정이 안 된 경우 링크 반환
                return f"✅ '{law_name}'이(가) 확인되었습니다.\n(상세 조문 자동 추출 실패 또는 전체 법령 참조)\n🔗 원문 보기: {full_link}"

        except Exception as e:
            return f"상세 법령 파싱 실패: {e}"


class LegalAgents:
    @staticmethod
    def researcher(situation):
        """
        [다중 법령 하이브리드 검색 시스템]
        1. LLM: 상황을 분석하여 전략적으로 필요한 법령/조문을 '리스트'로 추출 (최대 3개)
           (예: 위반 조항 + 용어 정의 조항 + 과태료 부과 근거)
        2. API: 추출된 리스트를 순회하며 실제 법령 데이터 조회
        3. 통합: API 조회 결과와 실패 시 AI 추론 결과를 종합하여 리포트 생성
        """
        
        # =========================================================
        # 1단계: 다중 검색 키워드 추출 (JSON List)
        # =========================================================
        prompt_extract = f"""
        상황: "{situation}"
        
        위 민원 처리를 위해 법적 근거로 삼아야 할 핵심 대한민국 법령과 조문 번호를 
        **중요도 순으로 최대 3개까지** JSON 리스트로 추출하시오.
        
        [추출 전략 가이드]
        1. 핵심 위반 조항 (예: 주정차 금지)
        2. 반박을 위한 정의 조항 (예: '보도'의 정의, 민원인이 우길 경우 대비)
        3. 처벌/과태료 근거 조항
        
        형식: [{{"law_name": "도로교통법", "article_num": 32}}, {{"law_name": "도로교통법", "article_num": 2}}, ...]
        * 법령명은 정식 명칭 사용. 조문 번호 불명확하면 null.
        """
        
        search_targets = []
        try:
            # 리스트 형태의 JSON 파싱
            extracted = llm_service.generate_json(prompt_extract)
            if isinstance(extracted, list):
                search_targets = extracted
            elif isinstance(extracted, dict): # 혹시 하나만 줄 경우 리스트로 변환
                search_targets = [extracted]
        except Exception:
            # 실패 시 기본값 설정
            search_targets = [{"law_name": "도로교통법", "article_num": None}]

        if not search_targets:
            search_targets = [{"law_name": "도로교통법", "article_num": None}]

        # =========================================================
        # 2단계: API 순회 호출 및 결과 수집
        # =========================================================
        report_lines = []
        api_success_count = 0
        
        report_lines.append(f"🔍 **AI가 식별한 핵심 법령 ({len(search_targets)}건)**")
        report_lines.append("---")

        for idx, item in enumerate(search_targets):
            law_name = item.get("law_name", "관련법령")
            article_num = item.get("article_num")
            
            # API 호출
            real_law_text = law_api_service.get_law_text(law_name, article_num)
            
            # API 성공 여부 판단 (에러 키워드 체크)
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

        # =========================================================
        # 3단계: 결과 종합 (Fallback 로직 포함)
        # =========================================================
        
        final_report = "\n".join(report_lines)

        # 만약 API가 단 하나도 성공하지 못했다면 -> 전면 AI 추론(가상) 모드 가동
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
        
        # 하나라도 성공했다면 API 리포트 반환
        return final_report

    # ... (strategist, clerk, drafter 등 다른 메서드는 기존 그대로 유지) ...
    @staticmethod
    def strategist(situation, legal_basis, search_results):
        prompt = f"""
당신은 행정 업무 베테랑 '주무관'입니다.

[민원 상황]: {situation}
[확보된 법적 근거]: 
{legal_basis}

[유사 사례/판례]: {search_results}

위 정보를 종합하여 이 민원을 처리하기 위한 **대략적인 업무 처리 방향(Strategy)**을 수립하세요.
특히 [확보된 법적 근거]에 여러 조항(위반조항, 정의조항 등)이 있다면 이를 논리적으로 연결하여 방어 논리를 구성하세요.

다음 3가지 항목을 포함하여 마크다운으로 작성하세요:
1. **처리 방향**: (예: '제2조 정의 규정에 의거하여 보도임을 명확히 하고, 제32조 위반으로 단속 유지')
2. **핵심 주의사항**: (절차상 놓치면 안 되는 것, 법적 쟁점)
3. **예상 반발 및 대응**: (민원인이 "여기가 무슨 인도냐"라고 항의할 경우 대응 논리)

간결하고 명확하게 작성하세요.
"""
        return llm_service.generate_text(prompt)

    @staticmethod
    def clerk(situation, legal_basis):
        today = datetime.now()
        prompt = f"""
오늘: {today.strftime('%Y-%m-%d')}
상황: {situation}
법령: {legal_basis}
위 상황에서 행정처분 사전통지나 이행 명령 시, 법적으로(또는 통상적으로) 부여해야 하는 '이행/의견제출 기간'은 며칠인가?
설명 없이 숫자(일수)만 출력하세요. (예: 10, 15, 20)
모르겠으면 15를 출력하세요.
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

[업무 처리 가이드라인 (전략)]
{strategy}

[작성 원칙]
1. 위 '업무 처리 가이드라인'의 논리를 본문에 녹여내세요. (법 조항 인용 필수)
2. 수신인이 불명확하면 상황에 맞춰 추론하세요.
3. 본문 구조: [문서의 목적/경위] -> [법적 근거(정의 및 위반조항)] -> [처분 내용] -> [이의제기 절차]
4. 개인정보(이름, 번호)는 반드시 마스킹('OOO') 처리하세요.
"""
        return llm_service.generate_json(prompt, schema=doc_schema)# ==========================================
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
