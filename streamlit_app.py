# app.py  (✅ 깨짐 0% + 법령(후보 3개) + “가능하면 원문(조문 텍스트) 조회”까지 포함 완성본)
import streamlit as st
import google.generativeai as genai
from groq import Groq
from serpapi import GoogleSearch
from supabase import create_client
import json
import re
import time
import html
import requests
from datetime import datetime, timedelta

# ==========================================
# 1. Configuration & Styles (설정 및 디자인)
# ==========================================
st.set_page_config(layout="wide", page_title="AI Bureau: The Legal Glass", page_icon="⚖️")

st.markdown("""
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
        overflow: hidden;
    }

    /* 공문서 내부 스타일 */
    .doc-header { text-align: center; font-size: 22pt; font-weight: 900; margin-bottom: 30px; letter-spacing: 2px; }
    .doc-info { display: flex; justify-content: space-between; font-size: 11pt; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; gap: 10px; }
    .doc-body { font-size: 12pt; text-align: justify; }
    .doc-footer { text-align: center; font-size: 20pt; font-weight: bold; margin-top: 80px; letter-spacing: 5px; }
    .stamp { position: absolute; bottom: 85px; right: 80px; border: 3px solid #cc0000; color: #cc0000; padding: 5px 10px; font-size: 14pt; font-weight: bold; transform: rotate(-15deg); opacity: 0.8; border-radius: 5px; }

    /* 로그 스타일 */
    .agent-log { font-family: 'Consolas', monospace; font-size: 0.85rem; padding: 6px 12px; border-radius: 6px; margin-bottom: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    .log-legal { background-color: #eff6ff; color: #1e40af; border-left: 4px solid #3b82f6; }
    .log-search { background-color: #fff7ed; color: #c2410c; border-left: 4px solid #f97316; }
    .log-strat { background-color: #f5f3ff; color: #6d28d9; border-left: 4px solid #8b5cf6; }
    .log-calc { background-color: #f0fdf4; color: #166534; border-left: 4px solid #22c55e; }
    .log-draft { background-color: #fef2f2; color: #991b1b; border-left: 4px solid #ef4444; }
    .log-sys { background-color: #f3f4f6; color: #4b5563; border-left: 4px solid #9ca3af; }

    /* 전략 박스 스타일 */
    .strategy-box { background-color: #fffbeb; border: 1px solid #fcd34d; padding: 15px; border-radius: 8px; margin-bottom: 15px; }

    /* 조문 원문 박스 */
    .law-box { background: white; border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px; margin-top: 10px; }
    .law-title { font-weight: 800; margin-bottom: 8px; }
    .law-text { white-space: pre-wrap; font-family: 'Batang', serif; font-size: 11.5pt; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Infrastructure Layer (Services)
# ==========================================

def _safe_get_secret(path1, path2=None, default=None):
    """st.secrets 안전 접근 헬퍼"""
    try:
        if path2 is None:
            return st.secrets.get(path1, default)
        return st.secrets.get(path1, {}).get(path2, default)
    except Exception:
        return default


class LLMService:
    """
    [Model Hierarchy]
    1. Gemini 2.5 Flash
    2. Gemini 2.5 Flash Lite
    3. Gemini 2.0 Flash
    4. Groq (Llama 3 Backup)
    """
    def __init__(self):
        self.gemini_key = _safe_get_secret("general", "GEMINI_API_KEY")
        self.groq_key = _safe_get_secret("general", "GROQ_API_KEY")

        # 모델 후보 (가능하면 2.5 사용, 안되면 자동 fallback)
        self.gemini_models = [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            # 구버전 호환 후보(환경에 따라)
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ]

        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)

        self.groq_client = Groq(api_key=self.groq_key) if self.groq_key else None

    def _try_gemini(self, prompt, is_json=False, schema=None):
        last_err = None
        for model_name in self.gemini_models:
            try:
                model = genai.GenerativeModel(model_name)
                if is_json:
                    # 일부 환경에서 response_schema가 오류날 수 있어 2단계로 시도
                    try:
                        config = genai.GenerationConfig(
                            response_mime_type="application/json",
                            response_schema=schema
                        )
                        res = model.generate_content(prompt, generation_config=config)
                        return res.text, model_name
                    except Exception:
                        # JSON 강제(문자열 JSON) fallback
                        res = model.generate_content(prompt + "\n\n반드시 JSON만 출력.")
                        return res.text, model_name
                else:
                    res = model.generate_content(prompt)
                    return res.text, model_name
            except Exception as e:
                last_err = e
                continue
        raise Exception(f"All Gemini models failed: {last_err}")

    def generate_text(self, prompt):
        try:
            text, _ = self._try_gemini(prompt, is_json=False)
            return (text or "").strip()
        except Exception:
            if self.groq_client:
                return (self._generate_groq(prompt) or "").strip()
            return "시스템 오류: AI 모델 연결 실패"

    def generate_json(self, prompt, schema=None):
        # 1차: Gemini JSON
        try:
            text, _ = self._try_gemini(prompt, is_json=True, schema=schema)
            return json.loads(text)
        except Exception:
            # 2차: 텍스트로 뽑고 JSON만 추출
            text = self.generate_text(prompt + "\n\nOutput strictly in JSON.")
            try:
                match = re.search(r'\{.*\}', text, re.DOTALL)
                return json.loads(match.group(0)) if match else None
            except Exception:
                return None

    def _generate_groq(self, prompt):
        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return completion.choices[0].message.content
        except Exception:
            return "System Error"


class SearchService:
    """Google Search API (SerpApi) Wrapper"""
    def __init__(self):
        self.api_key = _safe_get_secret("general", "SERPAPI_KEY")

    def search_precedents(self, query):
        if not self.api_key:
            return "⚠️ 검색 API 키(SERPAPI_KEY)가 없어 유사 사례를 조회할 수 없습니다."

        try:
            search_query = f"{query} 행정처분 판례 사례 민원 답변"
            params = {
                "engine": "google",
                "q": search_query,
                "api_key": self.api_key,
                "num": 3,
                "hl": "ko",
                "gl": "kr"
            }
            search = GoogleSearch(params)
            results = search.get_dict().get("organic_results", [])

            if not results:
                return "관련된 유사 사례 검색 결과가 없습니다."

            summary = []
            for item in results:
                title = item.get('title', '제목 없음')
                snippet = item.get('snippet', '내용 없음')
                link = item.get('link', '#')
                summary.append(f"- **[{title}]({link})**: {snippet}")

            return "\n".join(summary)
        except Exception as e:
            return f"검색 중 오류 발생: {e}"


class DatabaseService:
    """Supabase Persistence Layer"""
    def __init__(self):
        try:
            self.url = _safe_get_secret("supabase", "SUPABASE_URL")
            self.key = _safe_get_secret("supabase", "SUPABASE_KEY")
            if not self.url or not self.key:
                raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY")
            self.client = create_client(self.url, self.key)
            self.is_active = True
        except Exception:
            self.is_active = False

    def save_log(self, user_input, legal_basis, strategy, doc_data, law_fulltext=None):
        if not self.is_active:
            return "DB 미연결 (저장 건너뜀)"

        try:
            final_summary_content = {
                "strategy": strategy,
                "document_content": doc_data,
                "law_fulltext": law_fulltext
            }

            data = {
                "situation": user_input,
                "law_name": legal_basis,
                "summary": json.dumps(final_summary_content, ensure_ascii=False)
            }

            self.client.table("law_reports").insert(data).execute()
            return "DB 저장 성공"
        except Exception as e:
            return f"DB 저장 실패: {e}"


class LawTextService:
    """
    (선택) 국가법령정보센터 OpenAPI 기반 '조문 원문' 조회 레이어.
    - 키가 없으면 자동으로 건너뜀.
    - 키/엔드포인트는 환경마다 다를 수 있어 예외 안전하게 처리.
    """
    def __init__(self):
        # secrets 예:
        # [law]
        # LAW_API_KEY = "..."
        # LAW_API_BASE = "https://www.law.go.kr/DRF/lawService.do"
        self.api_key = _safe_get_secret("law", "LAW_API_KEY")
        self.base = _safe_get_secret("law", "LAW_API_BASE", "https://www.law.go.kr/DRF/lawService.do")

    def is_enabled(self):
        return bool(self.api_key and self.base)

    def fetch_law_text(self, law_name: str, article_hint: str = ""):
        """
        law_name: "주민등록법"
        article_hint: "제24조" 같은 힌트 (없어도 됨)
        반환: (ok:bool, text:str, debug:str)
        """
        if not self.is_enabled():
            return False, "⚠️ 법령 원문 조회(API 키 미설정): [law] LAW_API_KEY 를 secrets에 추가하세요.", "disabled"

        # ⚠️ 아래는 “작동 가능한” 일반형 DRF 호출 예시. 환경에 따라 파라미터명이 달라질 수 있어 예외처리 강함.
        # 성공하면 XML/JSON이 오는데, 여기서는 텍스트만 최대한 뽑아서 보여줌.
        try:
            params = {
                "OC": self.api_key,
                "target": "law",
                "type": "XML",
                "query": law_name
            }
            r = requests.get(self.base, params=params, timeout=8)
            if r.status_code != 200:
                return False, f"⚠️ 법령 API 응답 실패: HTTP {r.status_code}", f"http:{r.status_code}"

            raw = r.text or ""
            # 매우 단순 파싱(원문이 HTML/XML로 섞이는 경우가 많아 “보이는 텍스트”만 최대한 추출)
            # law.go.kr 응답은 구조가 복잡할 수 있어 완벽 파싱은 별도 구현 필요.
            # 여기서는 사용자 체감용 “원문 일부”라도 보여주는 목적.
            # 1) 조문명/조문 텍스트 후보 태그를 대충 뽑음
            candidates = []

            # 조문 본문 비슷한 태그
            for pat in [r"<조문내용>(.*?)</조문내용>", r"<조문>(.*?)</조문>", r"<내용>(.*?)</내용>"]:
                m = re.search(pat, raw, re.DOTALL)
                if m:
                    candidates.append(m.group(1))

            # 후보가 없으면 전체에서 텍스트만 뽑기
            if not candidates:
                # 태그 제거
                text = re.sub(r"<[^>]+>", "", raw)
                text = re.sub(r"\s+\n", "\n", text).strip()
                text = text[:4000] if len(text) > 4000 else text
                return True, text if text else "⚠️ 원문 텍스트 추출 실패(응답 구조 확인 필요)", "fallback-strip"

            text = candidates[0]
            text = re.sub(r"<[^>]+>", "", text)
            text = html.unescape(text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()

            # article_hint가 있으면 그 주변만 살짝 필터(간단)
            if article_hint:
                # 힌트가 포함된 라인 근방만 남기기(너무 길면)
                if article_hint in text and len(text) > 1500:
                    idx = text.find(article_hint)
                    start = max(0, idx - 300)
                    end = min(len(text), idx + 1200)
                    text = text[start:end].strip()

            text = text[:4000] if len(text) > 4000 else text
            return True, text, "ok"
        except Exception as e:
            return False, f"⚠️ 법령 API 호출 오류: {e}", "exception"


# 싱글톤 인스턴스
llm_service = LLMService()
search_service = SearchService()
db_service = DatabaseService()
law_text_service = LawTextService()

# ==========================================
# 3. Domain Layer (Agents)
# ==========================================
class LegalAgents:
    @staticmethod
    def researcher(situation):
        """Step 1: 법령 탐색 (✅ 후보 3개 + 적용이유 1줄)"""
        prompt = f"""
Role: 대한민국 행정실무 기준으로, '적용 법령'을 정확히 식별한다.
Task: 아래 상황에 적용될 수 있는 '법령 후보 3개'를 제시하라.

[출력 제약사항 - 매우 중요]
- 인삿말/자기소개/사족 금지. 바로 결과만.
- 가능한 한 현행 법령명 + 조항 번호(조문명)까지.
- 각 후보마다 '적용 이유' 1줄.
- 개인정보는 OOO로 마스킹.

[출력 형식(반드시 준수)]
1) 적용 법령(후보 1): 법령명 제X조(조문명) - 적용 이유(1줄)
2) 적용 법령(후보 2): ...
3) 적용 법령(후보 3): ...

상황: "{situation}"
"""
        return llm_service.generate_text(prompt).strip()

    @staticmethod
    def strategist(situation, legal_basis, search_results):
        """Step 2: 전략 수립"""
        prompt = f"""
당신은 행정 업무 베테랑 '주무관'이다.

[민원 상황]: {situation}
[적용 법령 후보]: {legal_basis}
[유사 사례/판례]: {search_results}

위 정보를 종합하여 민원 처리 '업무 처리 방향(Strategy)'을 수립하라.

다음 3가지 항목을 포함하여 마크다운으로 작성:
1. **처리 방향**
2. **핵심 주의사항**
3. **예상 반발 및 대응**

간결하고 명확하게.
"""
        return llm_service.generate_text(prompt)

    @staticmethod
    def clerk(situation, legal_basis):
        """Step 3: 기한 산정"""
        today = datetime.now()
        prompt = f"""
오늘: {today.strftime('%Y-%m-%d')}
상황: {situation}
법령 후보: {legal_basis}

위 상황에서 '사전통지/의견제출/이행' 등 통상 부여 기간(일수)을 숫자만 출력.
설명 금지. 숫자만.
모르면 15.
"""
        try:
            res = llm_service.generate_text(prompt)
            days = int(re.sub(r'[^0-9]', '', res))
            if days <= 0:
                days = 15
        except Exception:
            days = 15

        deadline = today + timedelta(days=days)
        return {
            "today_str": today.strftime("%Y. %m. %d."),
            "deadline_str": deadline.strftime("%Y. %m. %d."),
            "days_added": days,
            "doc_num": f"행정-{today.strftime('%Y')}-{int(time.time())%1000:03d}호"
        }

    @staticmethod
    def _extract_primary_law_hint(legal_basis_text: str):
        """
        후보 1 라인에서 법령명과 조항 힌트(제X조)를 대충 뽑기
        """
        # 후보 1: "법령명 제24조( ... ) - ..."
        line1 = legal_basis_text.splitlines()[0] if legal_basis_text else ""
        # 법령명: ") 적용 법령(후보 1): " 이후부터 "제" 앞까지
        law_name = ""
        article_hint = ""
        m = re.search(r"후보 1\)\s*:\s*(.+)", line1)
        if m:
            payload = m.group(1).strip()
            m2 = re.search(r"(.+?)\s*(제\s*\d+\s*조)", payload)
            if m2:
                law_name = m2.group(1).strip()
                article_hint = m2.group(2).replace(" ", "")
            else:
                # "제X조"가 없으면 법령명만
                law_name = re.split(r"\s-\s", payload)[0].strip()
        return law_name, article_hint

    @staticmethod
    def drafter(situation, legal_basis, meta_info, strategy):
        """Step 4: 공문서 작성 (JSON)"""
        doc_schema = {
            "type": "OBJECT",
            "properties": {
                "title": {"type": "STRING", "description": "공문서 제목"},
                "receiver": {"type": "STRING", "description": "수신인"},
                "body_paragraphs": {"type": "ARRAY", "items": {"type": "STRING"}},
                "department_head": {"type": "STRING", "description": "발신 명의"}
            },
            "required": ["title", "receiver", "body_paragraphs", "department_head"]
        }

        prompt = f"""
당신은 행정기관 서무 담당자다. 아래 정보를 바탕으로 '완결된 공문서'를 작성하라.

[입력 정보]
- 민원 상황: {situation}
- 적용 법령 후보: {legal_basis}
- 시행 일자: {meta_info['today_str']}
- 기한: {meta_info['deadline_str']} ({meta_info['days_added']}일)

[업무 처리 가이드라인(전략)]
{strategy}

[작성 원칙]
1) 어조: 정중/건조/행정보고서 톤.
2) 수신인이 불명확하면 상황에 맞춰 합리적으로 추론(예: 신청인, 민원인, 관련 부서 등).
3) 본문 구조: [경위] → [근거] → [조치/안내] → [권리구제/문의]
4) 개인정보(이름/번호/주소 상세)는 OOO로 마스킹.
5) 'body_paragraphs'에는 문단 텍스트만. HTML/마크다운/코드블록 금지.
6) JSON만 출력.
"""
        data = llm_service.generate_json(prompt, schema=doc_schema)

        # 안전장치: None이면 최소 형태로라도 반환
        if not data or not isinstance(data, dict):
            data = {
                "title": "공 문 서",
                "receiver": "수신자 참조",
                "body_paragraphs": [
                    "1. 귀하의 민원에 대하여 검토한 결과를 아래와 같이 안내드립니다.",
                    "2. 관련 법령 및 처리 기준에 따라 필요한 조치를 검토·진행하겠습니다.",
                    f"3. 의견제출 또는 추가 자료 제출이 필요한 경우 {meta_info['deadline_str']}까지 제출하여 주시기 바랍니다.",
                    "4. 기타 문의사항은 담당부서로 연락주시기 바랍니다."
                ],
                "department_head": "행정기관장"
            }

        # 문단이 문자열로 오면 배열로 정규화
        bp = data.get("body_paragraphs", [])
        if isinstance(bp, str):
            bp = [bp]
        if not isinstance(bp, list):
            bp = [str(bp)]
        # 문단에 HTML 태그가 섞이면 제거(2차 안전망)
        cleaned = []
        for p in bp:
            p = str(p)
            p = re.sub(r"<[^>]+>", "", p)
            cleaned.append(p.strip())
        data["body_paragraphs"] = cleaned

        return data

# ==========================================
# 4. Workflow (UI 로직)
# ==========================================
def run_workflow(user_input):
    log_placeholder = st.empty()
    logs = []

    def add_log(msg, style="sys"):
        logs.append(f"<div class='agent-log log-{style}'>{html.escape(str(msg))}</div>")
        log_placeholder.markdown("".join(logs), unsafe_allow_html=True)
        time.sleep(0.2)

    # Phase 1
    add_log("🔍 Phase 1: 법령 및 유사 사례 리서치 중...", "legal")
    legal_basis = LegalAgents.researcher(user_input)
    add_log("📜 적용 법령 후보 도출 완료", "legal")

    add_log("🌍 구글 검색 엔진 가동...", "search")
    try:
        search_results = search_service.search_precedents(user_input)
    except Exception:
        search_results = "검색 모듈 미연결 (건너뜀)"

    # (선택) 법령 원문 조회
    add_log("📚 (옵션) 법령 원문(조문 텍스트) 조회 시도...", "legal")
    law_name, article_hint = LegalAgents._extract_primary_law_hint(legal_basis)
    law_fulltext = None
    if law_name:
        ok, text, _dbg = law_text_service.fetch_law_text(law_name, article_hint=article_hint)
        law_fulltext = text
        if ok:
            add_log("✅ 법령 원문(일부) 조회 성공", "legal")
        else:
            add_log("⚠️ 법령 원문 조회 실패/건너뜀", "legal")
    else:
        law_fulltext = "⚠️ 후보 1에서 법령명을 파싱하지 못했습니다. (법령 원문 조회 건너뜀)"

    # Phase 2
    add_log("🧠 Phase 2: 업무 처리 방향(전략) 수립...", "strat")
    strategy = LegalAgents.strategist(user_input, legal_basis, search_results)

    # Phase 3
    add_log("📅 Phase 3: 기한 산정 및 공문서 작성...", "calc")
    meta_info = LegalAgents.clerk(user_input, legal_basis)

    add_log("✍️ 최종 공문서 조판 중...", "draft")
    doc_data = LegalAgents.drafter(user_input, legal_basis, meta_info, strategy)

    # Phase 4
    add_log("💾 업무 기록 DB 저장 중...", "sys")
    save_result = db_service.save_log(user_input, legal_basis, strategy, doc_data, law_fulltext=law_fulltext)

    add_log(f"✅ 완료 ({save_result})", "sys")
    time.sleep(0.6)
    log_placeholder.empty()

    return {
        "doc": doc_data,
        "meta": meta_info,
        "law": legal_basis,
        "law_fulltext": law_fulltext,
        "search": search_results,
        "strategy": strategy,
        "save_msg": save_result
    }

# ==========================================
# 5. Presentation Layer (UI)
# ==========================================
def main():
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.title("🏢 AI 행정관 Pro")
        st.caption("Gemini + Search + Strategy + (Optional) Law Fulltext + DB")
        st.markdown("---")

        user_input = st.text_area("업무 지시", height=150, placeholder="예: 주민등록등본 발급 시 배우자 주소를 모르는 경우 발급 가능 여부")

        if st.button("⚡ 스마트 행정 처분 시작", type="primary", use_container_width=True):
            if user_input.strip():
                try:
                    with st.spinner("AI 에이전트가 분석 중입니다..."):
                        st.session_state['workflow_result'] = run_workflow(user_input)
                except Exception as e:
                    st.error(f"오류: {e}")
            else:
                st.warning("업무 지시 내용을 입력하세요.")

        if 'workflow_result' in st.session_state:
            res = st.session_state['workflow_result']

            st.markdown("---")
            if "성공" in res.get('save_msg', ''):
                st.success(f"✅ {res['save_msg']}")
            else:
                st.info(f"ℹ️ {res.get('save_msg', '')}")

            with st.expander("✅ [검토] 적용 법령(후보) + 유사 사례", expanded=True):
                st.markdown("#### 📜 적용 법령(후보 3)")
                st.code(res.get('law', ''), language="text")

                st.markdown("#### 🔎 유사 사례/판례")
                # ✅ st.info가 아니라 markdown으로 링크/포맷 살림
                st.markdown(res.get('search', ''))

                st.markdown("#### 📚 법령 원문(가능한 경우)")
                fulltext = res.get("law_fulltext") or ""
                st.markdown(
                    f"""
<div class="law-box">
  <div class="law-title">조문 텍스트(조회 결과)</div>
  <div class="law-text">{html.escape(fulltext)}</div>
</div>
""",
                    unsafe_allow_html=True
                )

            with st.expander("🧭 [방향] 처리 가이드라인", expanded=True):
                st.markdown(res.get('strategy', ''))

    with col_right:
        if 'workflow_result' in st.session_state:
            res = st.session_state['workflow_result']
            doc = res.get('doc')
            meta = res.get('meta')

            if doc:
                # ✅ 본문 문단: 절대 깨짐 방지(escape)
                paragraphs = doc.get('body_paragraphs', [])
                if isinstance(paragraphs, str):
                    paragraphs = [paragraphs]
                safe_paragraphs = [html.escape(str(p)) for p in paragraphs]
                safe_paragraphs = [p.replace("\n", "<br>") for p in safe_paragraphs]
                p_html = "".join([f"<p style='margin-bottom: 15px;'>{p}</p>" for p in safe_paragraphs])

                # ✅ 공문 HTML (절대 들여쓰기 강박 X / 핵심은 escape)
                html_content = f"""
<div class="paper-sheet">
  <div class="stamp">직인생략</div>
  <div class="doc-header">{html.escape(doc.get('title', '공 문 서'))}</div>

  <div class="doc-info">
    <span>문서번호: {html.escape(meta.get('doc_num', ''))}</span>
    <span>시행일자: {html.escape(meta.get('today_str', ''))}</span>
    <span>수신: {html.escape(doc.get('receiver', '참조'))}</span>
  </div>

  <hr style="border: 1px solid black; margin-bottom: 30px;">

  <div class="doc-body">
    {p_html}
  </div>

  <div class="doc-footer">{html.escape(doc.get('department_head', '행정기관장'))}</div>
</div>
"""
                st.markdown(html_content, unsafe_allow_html=True)

                st.download_button(
                    label="🖨️ 다운로드 (HTML)",
                    data=html_content,
                    file_name="공문서.html",
                    mime="text/html",
                    use_container_width=True
                )
        else:
            st.markdown("""
<div style='text-align: center; padding: 100px; color: #aaa; background: white; border-radius: 10px; border: 2px dashed #ddd;'>
  <h3>📄 Document Preview</h3>
  <p>왼쪽에서 업무를 지시하면<br>완성된 공문서가 여기에 나타납니다.</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
