import streamlit as st
import google.generativeai as genai
from groq import Groq
from serpapi import GoogleSearch
from supabase import create_client
import json
import re
import time
from datetime import datetime, timedelta
from html import escape

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
        font-family: 'Noto Serif KR','Noto Sans KR','Nanum Gothic','Apple SD Gothic Neo','Malgun Gothic',serif;
        color: #111;
        line-height: 1.6;
        position: relative;
    }

    /* 공문서 내부 스타일 */
    .doc-header { text-align: center; font-size: 22pt; font-weight: 900; margin-bottom: 30px; letter-spacing: 2px; }
    .doc-info { display: flex; justify-content: space-between; font-size: 11pt; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; }
    .doc-body { font-size: 12pt; text-align: justify; }
    .doc-footer { text-align: center; font-size: 20pt; font-weight: bold; margin-top: 80px; letter-spacing: 5px; }
    .stamp { position: absolute; bottom: 85px; right: 80px; border: 3px solid #cc0000; color: #cc0000; padding: 5px 10px; font-size: 14pt; font-weight: bold; transform: rotate(-15deg); opacity: 0.8; border-radius: 5px; }

    /* 로그 스타일 */
    .agent-log { font-family: 'Consolas', monospace; font-size: 0.85rem; padding: 6px 12px; border-radius: 6px; margin-bottom: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    .log-legal { background-color: #eff6ff; color: #1e40af; border-left: 4px solid #3b82f6; } /* Blue */
    .log-search { background-color: #fff7ed; color: #c2410c; border-left: 4px solid #f97316; } /* Orange */
    .log-strat { background-color: #f5f3ff; color: #6d28d9; border-left: 4px solid #8b5cf6; } /* Purple */
    .log-calc  { background-color: #f0fdf4; color: #166534; border-left: 4px solid #22c55e; } /* Green */
    .log-draft { background-color: #fef2f2; color: #991b1b; border-left: 4px solid #ef4444; } /* Red */
    .log-sys   { background-color: #f3f4f6; color: #4b5563; border-left: 4px solid #9ca3af; } /* Gray */

    /* 전략 박스 스타일 */
    .strategy-box { background-color: #fffbeb; border: 1px solid #fcd34d; padding: 15px; border-radius: 8px; margin-bottom: 15px; }
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 2. Helpers (Robust JSON + Safe)
# ==========================================

def _safe_html_text(value):
    if value is None:
        return ""
    return escape(str(value), quote=False).replace("\n", "<br>")

def _strip_code_fences(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```", "", text)
    return text.strip()

def _extract_first_json(text: str):
    """
    Tries to extract the first valid JSON object/array from messy LLM output.
    - Removes code fences
    - Scans for first '{' or '[' and then attempts incremental parsing
    """
    if not text:
        return None
    text = _strip_code_fences(text)

    # find first JSON start
    idx_obj = text.find("{")
    idx_arr = text.find("[")
    candidates = [i for i in [idx_obj, idx_arr] if i != -1]
    if not candidates:
        return None
    start = min(candidates)
    tail = text[start:]

    # fast path: whole tail is json
    try:
        return json.loads(tail)
    except Exception:
        pass

    # incremental: try to find an end position that parses
    # (works well for truncated explanations appended after JSON)
    for end in range(len(tail), max(len(tail) - 5000, 0), -1):
        chunk = tail[:end].strip()
        if not chunk:
            continue
        try:
            return json.loads(chunk)
        except Exception:
            continue
    return None

def _ensure_doc_shape(doc):
    """Guarantee the document dict has required keys and correct types."""
    fallback = {
        "title": "공 문 서",
        "receiver": "수신자 참조",
        "body_paragraphs": ["AI 문서 생성에 실패했습니다. (JSON 파싱/스키마/토큰 문제 가능)"],
        "department_head": "행정기관장",
    }
    if not isinstance(doc, dict):
        return fallback

    title = doc.get("title") or fallback["title"]
    receiver = doc.get("receiver") or fallback["receiver"]
    dept = doc.get("department_head") or fallback["department_head"]
    body = doc.get("body_paragraphs")

    if isinstance(body, str):
        body = [body]
    if not isinstance(body, list) or not body:
        body = fallback["body_paragraphs"]

    # sanitize list items
    clean_body = []
    for p in body:
        if p is None:
            continue
        clean_body.append(str(p).strip())
    if not clean_body:
        clean_body = fallback["body_paragraphs"]

    return {
        "title": str(title),
        "receiver": str(receiver),
        "body_paragraphs": clean_body,
        "department_head": str(dept),
    }

# ==========================================
# 3. Infrastructure Layer (Services)
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
        self.gemini_key = st.secrets.get("general", {}).get("GEMINI_API_KEY")
        self.groq_key = st.secrets.get("general", {}).get("GROQ_API_KEY")

        self.gemini_models = [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
        ]

        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)

        self.groq_client = Groq(api_key=self.groq_key) if self.groq_key else None

    def _try_gemini_text(self, prompt: str):
        last_err = None
        for model_name in self.gemini_models:
            try:
                model = genai.GenerativeModel(model_name)
                res = model.generate_content(prompt)
                return (res.text or "").strip(), model_name
            except Exception as e:
                last_err = e
                continue
        raise Exception(f"All Gemini models failed. last_err={last_err}")

    def _try_gemini_json(self, prompt: str, schema=None):
        """
        Uses JSON mime hint if possible, but still parses robustly because SDK/model may return non-pure JSON.
        """
        last_err = None
        for model_name in self.gemini_models:
            try:
                model = genai.GenerativeModel(model_name)
                # JSON 힌트: 모델/SDK 조합에 따라 완전 보장 X -> 파서로 마무리
                config = genai.GenerationConfig(response_mime_type="application/json", response_schema=schema)
                res = model.generate_content(prompt, generation_config=config)
                raw = (res.text or "").strip()
                obj = _extract_first_json(raw)
                if obj is not None:
                    return obj, model_name, raw
                last_err = Exception("Gemini returned non-JSON or unparsable output")
            except Exception as e:
                last_err = e
                continue
        raise Exception(f"All Gemini JSON attempts failed. last_err={last_err}")

    def generate_text(self, prompt: str) -> str:
        try:
            text, _model = self._try_gemini_text(prompt)
            return text
        except Exception:
            if self.groq_client:
                return self._generate_groq(prompt)
            return "시스템 오류: AI 모델 연결 실패"

    def generate_json(self, prompt: str, schema=None, retries: int = 2):
        """
        Robust JSON generation:
        - Gemini JSON hint -> parse
        - If fail: retry with stricter instruction
        - Finally: Groq fallback
        """
        # 1) Gemini JSON attempts
        try:
            obj, model_used, raw = self._try_gemini_json(prompt, schema=schema)
            return obj
        except Exception as first_err:
            # 2) Retry: add strict JSON constraints (shorter + explicit)
            for i in range(retries):
                tightened = (
                    prompt
                    + "\n\n[IMPORTANT]\n"
                      "- Output ONLY valid JSON. No markdown. No code fences. No commentary.\n"
                      "- Ensure all required keys exist.\n"
                )
                try:
                    obj, model_used, raw = self._try_gemini_json(tightened, schema=schema)
                    return obj
                except Exception:
                    continue

            # 3) Fallback: plain text then parse
            text = self.generate_text(
                prompt
                + "\n\n[IMPORTANT]\nOutput ONLY valid JSON object. No markdown. No commentary."
            )
            obj = _extract_first_json(text)
            if obj is not None:
                return obj

            # 4) Groq fallback (text then parse)
            if self.groq_client:
                text2 = self._generate_groq(
                    prompt
                    + "\n\n[IMPORTANT]\nOutput ONLY valid JSON object. No markdown. No commentary."
                )
                obj2 = _extract_first_json(text2)
                if obj2 is not None:
                    return obj2

            # fail hard
            raise first_err

    def _generate_groq(self, prompt: str) -> str:
        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return completion.choices[0].message.content or ""
        except Exception:
            return "System Error"

class SearchService:
    """Google Search API (SerpApi) Wrapper"""
    def __init__(self):
        self.api_key = st.secrets.get("general", {}).get("SERPAPI_KEY")

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
                "gl": "kr",
            }
            search = GoogleSearch(params)
            results = search.get_dict().get("organic_results", [])

            if not results:
                return "관련된 유사 사례 검색 결과가 없습니다."

            summary = []
            for item in results:
                title = item.get("title", "제목 없음")
                snippet = item.get("snippet", "내용 없음")
                link = item.get("link", "#")
                summary.append(f"- **[{title}]({link})**: {snippet}")

            return "\n".join(summary)
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
            data = {
                "input_text": user_input,
                "legal_basis": legal_basis,
                "strategy": strategy,
                "final_doc": json.dumps(doc_data, ensure_ascii=False),
                "created_at": datetime.now().isoformat(),
            }
            self.client.table("law_logs").insert(data).execute()
            return "DB 저장 성공"
        except Exception as e:
            return f"DB 저장 실패: {e}"

# 싱글톤 인스턴스 생성
llm_service = LLMService()
search_service = SearchService()
db_service = DatabaseService()

# ==========================================
# 4. Domain Layer (Agents)
# ==========================================
class LegalAgents:
    @staticmethod
    def researcher(situation):
        """Step 1: 법령 탐색"""
        prompt = f"""
<role>당신은 30년 경력의 법제관입니다.</role>
<instruction>
상황: "{situation}"
위 상황에 적용할 가장 정확한 '법령명'과 '관련 조항'을 하나만 찾으시오.
반드시 현행 대한민국 법령이어야 하며, 조항 번호까지 명시하세요.
(예: 도로교통법 제32조(정차 및 주차의 금지))

*주의: 입력에 실명 등 개인정보가 있다면 마스킹하여 처리하세요.
</instruction>
"""
        return llm_service.generate_text(prompt).strip()

    @staticmethod
    def strategist(situation, legal_basis, search_results):
        """Step 2: 전략 수립"""
        prompt = f"""
당신은 행정 업무 베테랑 '주무관'입니다.

[민원 상황]: {situation}
[법적 근거]: {legal_basis}
[유사 사례/판례]: {search_results}

위 정보를 종합하여 이 민원을 처리하기 위한 **대략적인 업무 처리 방향(Strategy)**을 수립하세요.

다음 3가지 항목을 포함하여 마크다운으로 작성하세요:
1. **처리 방향**: (예: 강경 대응, 계도 위주, 반려 등)
2. **핵심 주의사항**: (절차상 놓치면 안 되는 것, 법적 쟁점)
3. **예상 반발 및 대응**: (민원인이 항의할 경우 대응 논리)

간결하고 명확하게 작성하세요.
"""
        return llm_service.generate_text(prompt)

    @staticmethod
    def strategist_short(strategy_markdown: str):
        """
        LLM 출력 토큰 폭발 방지: drafter에 넣기 전 5~7줄로 압축
        """
        prompt = f"""
아래 '업무 처리 가이드라인'을 공문서 작성에 필요한 핵심만 남겨 5~7줄로 요약해줘.
- 불릿 형태로
- 법적 리스크/절차/반발 대응 포함
- 군더더기 제거

[원문]
{strategy_markdown}
"""
        return llm_service.generate_text(prompt).strip()

    @staticmethod
    def clerk(situation, legal_basis):
        """Step 3: 기한 산정"""
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
            days = int(re.sub(r"[^0-9]", "", res)) if res else 15
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
    def drafter(situation, legal_basis, meta_info, strategy_short):
        """Step 4: 공문서 작성"""
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
당신은 행정기관의 베테랑 서기입니다. 아래 정보를 바탕으로 '완결 공문서'를 JSON으로 작성하세요.

[입력 정보]
- 민원 상황: {situation}
- 법적 근거: {legal_basis}
- 시행 일자: {meta_info['today_str']}
- 기한: {meta_info['deadline_str']} ({meta_info['days_added']}일)

[업무 처리 가이드라인 (요약)]
{strategy_short}

[작성 원칙]
1. 가이드라인의 기조(톤/처리방향)를 반영하세요.
2. 수신인이 불명확하면 상황에 맞춰 합리적으로 추론하세요.
3. 본문 구조: [경위] -> [근거] -> [처분 내용] -> [권리구제 절차]
4. 개인정보(이름, 번호)는 반드시 마스킹('OOO') 처리하세요.
5. body_paragraphs는 문단 배열로 작성하세요.
"""
        obj = llm_service.generate_json(prompt, schema=doc_schema, retries=2)
        return _ensure_doc_shape(obj)

# ==========================================
# 5. Application Layer (Workflow)
# ==========================================
def run_workflow(user_input):
    log_placeholder = st.empty()
    logs = []

    def add_log(msg, style="sys"):
        logs.append(f"<div class='agent-log log-{style}'>{escape(msg)}</div>")
        log_placeholder.markdown("".join(logs), unsafe_allow_html=True)
        time.sleep(0.25)

    def fail_doc(reason: str):
        return _ensure_doc_shape({
            "title": "공 문 서",
            "receiver": "수신자 참조",
            "body_paragraphs": [f"AI 문서 생성에 실패했습니다: {reason}"],
            "department_head": "행정기관장",
        })

    # ----------------------------------------
    # Phase 1: Fact Check & Research
    # ----------------------------------------
    add_log("🔍 Phase 1: 법령 및 유사 사례 리서치 중...", "legal")
    legal_basis = LegalAgents.researcher(user_input)
    add_log(f"📜 법적 근거 발견: {legal_basis}", "legal")

    add_log("🌍 구글 검색 엔진 가동: 유사 사례 판례 수집 중...", "search")
    search_results = search_service.search_precedents(user_input)

    with st.expander("✅ [검토] 법령 및 유사 사례 확인", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**적용 법령**\n\n{legal_basis}")
        with col2:
            st.warning(f"**유사 사례 검색 결과**\n\n{search_results}")

    # ----------------------------------------
    # Phase 2: Strategy Setup
    # ----------------------------------------
    add_log("🧠 Phase 2: AI 주무관이 업무 처리 방향을 수립합니다...", "strat")
    strategy = LegalAgents.strategist(user_input, legal_basis, search_results)
    strategy_short = LegalAgents.strategist_short(strategy)

    with st.expander("🧭 [방향] 업무 처리 가이드라인", expanded=True):
        st.markdown(strategy)

    with st.expander("🧾 [요약] Draft 입력용 Strategy (토큰 절감)", expanded=False):
        st.markdown(strategy_short)

    # ----------------------------------------
    # Phase 3: Execution (Drafting)
    # ----------------------------------------
    add_log("📅 Phase 3: 기한 산정 및 공문서 작성 시작...", "calc")
    meta_info = LegalAgents.clerk(user_input, legal_basis)
    add_log(f"⏳ 기한 설정: {meta_info['days_added']}일 후 ({meta_info['deadline_str']})", "calc")

    add_log("✍️ 최종 공문서 조판 중 (Formatting)...", "draft")
    try:
        doc_data = LegalAgents.drafter(user_input, legal_basis, meta_info, strategy_short)
    except Exception as e:
        doc_data = fail_doc(f"drafter 예외: {e}")

    # ----------------------------------------
    # Phase 4: Persistence (Saving)
    # ----------------------------------------
    add_log("💾 업무 기록을 데이터베이스(Supabase)에 저장 중...", "sys")
    save_result = db_service.save_log(user_input, legal_basis, strategy, doc_data)

    add_log(f"✅ 모든 행정 절차가 완료되었습니다. ({save_result})", "sys")
    time.sleep(0.8)
    log_placeholder.empty()

    return doc_data, meta_info

# ==========================================
# 6. Presentation Layer (UI)
# ==========================================
def main():
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.title("🏢 AI 행정관 Pro")
        st.caption("Gemini 2.5 + Search + Strategy + DB (Robust JSON + Fallback)")
        st.markdown("---")

        st.markdown("### 🗣️ 업무 지시")
        user_input = st.text_area(
            "업무 내용",
            height=150,
            placeholder="예시:\n- 아파트 단지 내 소방차 전용구역 불법 주차 차량 과태료 부과 예고 통지서 작성해줘.\n- 식품위생법 위반 식당 영업정지 사전 통지서 써줘.",
            label_visibility="collapsed",
        )

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            run_btn = st.button("⚡ 스마트 행정 처분 시작", type="primary", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("🧹 결과 초기화", use_container_width=True)

        if clear_btn:
            st.session_state.pop("final_doc", None)
            st.session_state.pop("debug_last_raw_json", None)
            st.rerun()

        if run_btn:
            if not user_input:
                st.warning("내용을 입력해주세요.")
            else:
                try:
                    with st.spinner("AI 에이전트 팀이 협업 중입니다..."):
                        doc, meta = run_workflow(user_input)
                        st.session_state["final_doc"] = (doc, meta)
                except Exception as e:
                    st.error(f"시스템 오류 발생: {e}")

        st.markdown("---")
        st.info("💡 **Tip:** 법령/판례 검색 -> 전략 수립 -> 문서 작성 -> DB 저장까지 일괄 처리합니다.")

    with col_right:
        if "final_doc" in st.session_state:
            doc, meta = st.session_state["final_doc"]
            doc = _ensure_doc_shape(doc)

            html_content = f"""
<div class="paper-sheet">
  <div class="stamp">직인생략</div>
  <div class="doc-header">{_safe_html_text(doc.get('title', '공 문 서'))}</div>
  <div class="doc-info">
    <span>문서번호: {_safe_html_text(meta.get('doc_num'))}</span>
    <span>시행일자: {_safe_html_text(meta.get('today_str'))}</span>
    <span>수신: {_safe_html_text(doc.get('receiver', '수신자 참조'))}</span>
  </div>
  <hr style="border: 1px solid black; margin-bottom: 30px;">
  <div class="doc-body">
"""
            paragraphs = doc.get("body_paragraphs", [])
            if isinstance(paragraphs, str):
                paragraphs = [paragraphs]
            for p in paragraphs:
                html_content += f"<p style='margin-bottom: 15px;'>{_safe_html_text(p)}</p>"

            html_content += f"""
  </div>
  <div class="doc-footer">{_safe_html_text(doc.get('department_head', '행정기관장'))}</div>
</div>
"""

            st.markdown(html_content, unsafe_allow_html=True)
            st.download_button(
                label="🖨️ 다운로드 (HTML)",
                data=html_content,
                file_name="공문서.html",
                mime="text/html",
                use_container_width=True,
            )

        else:
            st.markdown(
                """
<div style='text-align: center; padding: 100px; color: #aaa; background: white; border-radius: 10px; border: 2px dashed #ddd;'>
  <h3>📄 Document Preview</h3>
  <p>왼쪽에서 업무를 지시하면<br>완성된 공문서가 여기에 나타납니다.</p>
</div>
""",
                unsafe_allow_html=True,
            )

if __name__ == "__main__":
    main()
