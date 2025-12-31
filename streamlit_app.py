import streamlit as st
import streamlit.components.v1 as components

import json
import re
import time
from datetime import datetime, timedelta
from html import escape, unescape
from urllib.parse import urlparse

# =========================
# 0) Optional Imports (Safety)
# =========================
try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import requests
except ImportError:
    requests = None

try:
    import xmltodict
except ImportError:
    xmltodict = None

try:
    from supabase import create_client
except ImportError:
    create_client = None

# =========================
# 1) Page & Style
# =========================
st.set_page_config(
    layout="wide",
    page_title="AI 행정관 Pro (Final v4.1)",
    page_icon="⚖️",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.stApp { background-color: #f8f9fa; }
/* 문서 스타일 */
.paper-sheet {
  background: #fff; width: 100%; max-width: 210mm; min-height: 297mm;
  padding: 25mm; margin: auto; box-shadow: 0 4px 15px rgba(0,0,0,0.08);
  font-family: 'Noto Serif KR','Nanum Myeongjo',serif;
  color:#111; line-height:1.6; position:relative;
}
.doc-header { text-align:center; font-size:24pt; font-weight:800; margin-bottom:35px; letter-spacing:2px; }
.doc-info {
  display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap;
  font-size:11pt; border-bottom:2px solid #333; padding-bottom:12px; margin-bottom:25px;
}
.doc-body { font-size:12pt; text-align: justify; }
.doc-footer { text-align:center; font-size:22pt; font-weight:bold; margin-top:80px; letter-spacing:4px; }
.stamp {
  position:absolute; bottom:85px; right:80px; border:3px solid #d32f2f; color: #d32f2f;
  padding:6px 12px; font-size:14pt; font-weight:bold; transform:rotate(-15deg);
  opacity:0.8; border-radius:4px; font-family: 'Nanum Gothic', sans-serif;
}
/* 로그 스타일 */
.agent-log { font-family: 'Pretendard', sans-serif; font-size: 0.9rem; padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; background: white; border: 1px solid #e5e7eb; }
.log-legal { border-left: 4px solid #3b82f6; color: #1e40af; }
.log-search { border-left: 4px solid #f97316; color: #c2410c; }
.log-strat { border-left: 4px solid #8b5cf6; color: #6d28d9; }
.log-calc  { border-left: 4px solid #22c55e; color: #166534; }
.log-draft { border-left: 4px solid #ef4444; color: #991b1b; }
.log-sys   { border-left: 4px solid #9ca3af; color: #4b5563; }
</style>
""", unsafe_allow_html=True)

# 정규식 컴파일
_TAG_RE = re.compile(r"<[^>]+>")

# =========================
# 2) Helper Functions
# =========================
def clean_text(value) -> str:
    """HTML 태그 및 특수문자 제거"""
    if value is None:
        return ""
    s = str(value)
    s = unescape(s)
    s = _TAG_RE.sub("", s)
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", s)
    return s.strip()

def safe_html(value) -> str:
    return escape(clean_text(value), quote=False).replace("\n", "<br>")

def truncate_text(s: str, max_chars: int = 3000) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n...(내용 축소됨)"

def ensure_doc_shape(doc):
    """LLM 응답이 깨졌을 때 기본값 보장"""
    fallback = {
        "title": "문 서 (생성 실패)",
        "receiver": "수신자 참조",
        "body_paragraphs": ["시스템 오류로 인해 문서 생성에 실패했습니다."],
        "department_head": "행정기관장",
    }
    if not isinstance(doc, dict):
        return fallback
    
    return {
        "title": clean_text(doc.get("title") or fallback["title"]),
        "receiver": clean_text(doc.get("receiver") or fallback["receiver"]),
        "body_paragraphs": doc.get("body_paragraphs") if isinstance(doc.get("body_paragraphs"), list) else fallback["body_paragraphs"],
        "department_head": clean_text(doc.get("department_head") or fallback["department_head"]),
    }

def safe_json_dump(obj):
    """Supabase 저장 시 터지지 않게 직렬화"""
    try:
        # set이나 기타 객체가 있어도 str로 변환하여 저장
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return "{}"

# =========================
# 3) Services
# =========================

# --- Metrics ---
def metrics_init():
    if "metrics" not in st.session_state:
        st.session_state["metrics"] = {"calls": {}, "tokens": 0}

def metrics_add(model_name: str):
    metrics_init()
    m = st.session_state["metrics"]
    m["calls"][model_name] = m["calls"].get(model_name, 0) + 1

# --- LLM Service ---
class LLMService:
    def __init__(self):
        self.groq_key = st.secrets.get("general", {}).get("GROQ_API_KEY")
        self.client = None
        self.last_model = "N/A"
        if Groq and self.groq_key:
            try:
                self.client = Groq(api_key=self.groq_key)
            except Exception:
                pass

    def generate(self, prompt: str, json_mode: bool = False, temp: float = 0.1):
        if not self.client:
            return {} if json_mode else "Groq API Key가 없거나 라이브러리 미설치"
        
        try:
            model = "llama-3.3-70b-versatile"
            resp = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Output JSON only." if json_mode else "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temp,
                response_format={"type": "json_object"} if json_mode else None
            )
            self.last_model = model
            metrics_add(model)
            text = resp.choices[0].message.content or ""
            
            if json_mode:
                return self._parse_json(text)
            return text
        except Exception as e:
            self.last_model = "ERROR"
            return {} if json_mode else f"LLM Error: {str(e)}"

    def _parse_json(self, text):
        try:
            return json.loads(text)
        except:
            # Markdown code block 제거 후 재시도
            cleaned = re.sub(r"```json|```", "", text).strip()
            try:
                return json.loads(cleaned)
            except:
                return {}

llm_service = LLMService()

# --- LAW API ---
class LawAPIService:
    def __init__(self):
        self.oc = st.secrets.get("law", {}).get("LAW_API_ID")
        self.base_url = "https://www.law.go.kr/DRF/lawService.do"
        self.enabled = bool(requests and xmltodict and self.oc)

    def search_law(self, query):
        if not self.enabled or not query: return []
        try:
            params = {"OC": self.oc, "target": "law", "type": "XML", "query": query, "display": 10}
            r = requests.get(self.base_url, params=params, timeout=5) # 타임아웃 5초
            data = xmltodict.parse(r.text)
            law_list = data.get("LawSearch", {}).get("law", [])
            if isinstance(law_list, dict): law_list = [law_list]
            return law_list
        except Exception:
            return []

    def get_article(self, law_id, article_no):
        if not self.enabled or not law_id: return ""
        try:
            params = {"OC": self.oc, "target": "law", "type": "XML", "ID": law_id}
            r = requests.get(self.base_url, params=params, timeout=8)
            data = xmltodict.parse(r.text)
            
            # 조문 찾기 로직 (간소화)
            articles = data.get("Law", {}).get("Article", [])
            if isinstance(articles, dict): articles = [articles]
            
            tgt_num = re.sub(r"[^0-9]", "", str(article_no))
            for art in articles:
                # 조문번호 확인
                an = str(art.get("@조문번호", ""))
                at = str(art.get("ArticleTitle", ""))
                if tgt_num in an or (tgt_num and f"제{tgt_num}조" in at):
                    # 본문 + 항 내용 합치기
                    content = art.get("ArticleContent", "")
                    paras = art.get("Paragraph", [])
                    if isinstance(paras, dict): paras = [paras]
                    p_text = "\n".join([str(p.get("ParagraphContent", "")) for p in paras if p])
                    return f"{at}\n{content}\n{p_text}"
            return ""
        except Exception:
            return ""

law_api = LawAPIService()

# --- Naver Search ---
class NaverSearchService:
    def __init__(self):
        self.cid = st.secrets.get("naver", {}).get("CLIENT_ID")
        self.csec = st.secrets.get("naver", {}).get("CLIENT_SECRET")
        self.enabled = bool(requests and self.cid and self.csec)

    def search(self, query, cat="news"):
        if not self.enabled or not query: return []
        try:
            url = f"https://openapi.naver.com/v1/search/{cat}.json"
            headers = {"X-Naver-Client-Id": self.cid, "X-Naver-Client-Secret": self.csec}
            params = {"query": query, "display": 5, "sort": "date"}
            r = requests.get(url, headers=headers, params=params, timeout=5)
            return r.json().get("items", [])
        except Exception:
            return []

naver_search = NaverSearchService()

# --- Database ---
class DatabaseService:
    def __init__(self):
        self.client = None
        url = st.secrets.get("supabase", {}).get("SUPABASE_URL")
        key = st.secrets.get("supabase", {}).get("SUPABASE_KEY")
        if create_client and url and key:
            try:
                self.client = create_client(url, key)
            except:
                pass

    def save_log(self, data: dict):
        if not self.client: return "DB 미연결"
        try:
            # 안전하게 직렬화 후 다시 dict로 변환 (Supabase 라이브러리 특성상)
            # 혹은 그냥 data를 넘기되, data 내부 값들이 safe해야 함.
            safe_data = json.loads(safe_json_dump(data))
            self.client.table("law_logs").insert(safe_data).execute()
            return "저장 성공"
        except Exception as e:
            return f"저장 실패: {str(e)}"

db_service = DatabaseService()

# =========================
# 4) Agents & Workflow
# =========================
def run_workflow(user_input, dept, officer):
    log_area = st.empty()
    logs = []
    
    def add_log(msg, style="sys"):
        logs.append(f"<div class='agent-log log-{style}'>{msg}</div>")
        log_area.markdown("".join(logs), unsafe_allow_html=True)
        time.sleep(0.05)

    # 1. Planner
    add_log("🧭 [Planner] 업무 분석 및 법령/검색어 추출...", "sys")
    prompt_plan = f"""
    입력: "{user_input}"
    JSON 형식으로 다음을 추출하라:
    {{
        "task_type": "업무유형",
        "law_hint": {{"law_name": "법령명(공식)", "article_no": "조번호(숫자만)"}},
        "keywords": ["검색어1", "검색어2"]
    }}
    """
    plan = llm_service.generate(prompt_plan, json_mode=True)
    
    # 2. Law Search
    add_log("📚 [Law] 법령 검색 및 조문 확인...", "legal")
    law_hint = plan.get("law_hint", {})
    law_name = law_hint.get("law_name", "")
    art_no = law_hint.get("article_no", "")
    
    legal_basis = "법령 정보를 찾을 수 없습니다."
    legal_status = "PENDING"
    law_debug = {}
    
    if law_name:
        candidates = law_api.search_law(law_name)
        if candidates:
            top_law = candidates[0]
            law_id = top_law.get("lawId")
            full_text = law_api.get_article(law_id, art_no)
            if full_text:
                legal_basis = f"[{top_law.get('lawNm')} 제{art_no}조]\n{full_text}"
                legal_status = "CONFIRMED"
                law_debug = {"id": law_id, "name": top_law.get("lawNm")}
            else:
                legal_basis = f"법령({top_law.get('lawNm')})은 찾았으나 제{art_no}조 원문 확보 실패."
        else:
            legal_basis = f"'{law_name}' 관련 법령 검색 실패."

    # 3. Naver Evidence
    add_log("🌍 [Search] 사실관계 및 리스크 점검 (Naver)...", "search")
    ev_text = ""
    ev_items = []
    keywords = plan.get("keywords", [])
    if keywords:
        raw_items = naver_search.search(keywords[0])
        for item in raw_items:
            # HTML 태그 제거 및 길이 제한
            clean_t = clean_text(item.get("title"))
            clean_d = clean_text(item.get("description"))
            ev_items.append({"title": clean_t, "link": item.get("link"), "desc": clean_d})
            ev_text += f"- {clean_t}: {clean_d}\n"
    
    # 4. Strategy
    add_log("🧠 [Analyst] 처리 전략 수립...", "strat")
    prompt_strat = f"""
    상황: {user_input}
    법적근거: {legal_basis}
    참고자료: {truncate_text(ev_text, 1000)}
    
    업무 처리 방향과 주의사항을 마크다운으로 요약하라.
    """
    strategy = llm_service.generate(prompt_strat)
    
    # 5. Drafter
    add_log("✍️ [Drafter] 공문서 초안 작성...", "draft")
    today_str = datetime.now().strftime("%Y. %m. %d.")
    # 문서번호 생성 시 time.time() 오류 방지
    doc_num = f"행정-{datetime.now().strftime('%Y')}-{int(time.time()) % 10000:04d}호"
    
    prompt_draft = f"""
    당신은 행정 공무원이다. 아래 정보를 바탕으로 완결된 공문서 JSON을 작성하라.
    수신, 발신, 제목, 본문(body_paragraphs 배열) 필수.
    
    - 부서: {dept}
    - 담당자: {officer}
    - 상황: {user_input}
    - 법적근거: {legal_basis}
    - 시행일: {today_str}
    - 문서번호: {doc_num}
    """
    doc_json = llm_service.generate(prompt_draft, json_mode=True)
    doc_final = ensure_doc_shape(doc_json)
    
    # 6. Save
    add_log("💾 [System] 결과 저장 중...", "sys")
    payload = {
        "created_at": datetime.now().isoformat(),
        "dept": dept,
        "officer": officer,
        "input": user_input,
        "legal_basis": legal_basis,
        "final_doc": safe_json_dump(doc_final), # 여기서 터지는 것 방지
        "strategy": strategy,
        "provenance": safe_json_dump(ev_items)  # 객체 포함 시 안전변환
    }
    db_msg = db_service.save_log(payload)
    add_log(f"✅ 완료 ({db_msg})", "sys")
    
    time.sleep(0.5)
    log_area.empty()
    
    return {
        "doc": doc_final,
        "meta": {"doc_num": doc_num, "today": today_str},
        "legal_basis": legal_basis,
        "strategy": strategy,
        "ev_items": ev_items
    }

# =========================
# 5) Main UI
# =========================
def main():
    st.session_state.setdefault("dept", "OO시청 OO과")
    st.session_state.setdefault("officer", "김주무관")
    
    col_l, col_r = st.columns([1, 1.2])
    
    with col_l:
        st.title("AI 행정관 Pro (Stable)")
        st.caption("v4.1 - Anti-Crash & Auto-Recovery")
        
        with st.expander("📝 사용자 정보 설정", expanded=False):
            st.text_input("부서명", key="dept")
            st.text_input("담당자", key="officer")
            
        user_input = st.text_area("업무 지시 사항", height=200, placeholder="예: 불법주정차 과태료 부과에 대한 이의신청 기각 통지서 작성해줘.")
        
        if st.button("🚀 문서 생성 실행", type="primary", use_container_width=True):
            if not user_input:
                st.warning("내용을 입력하세요.")
            else:
                with st.spinner("AI 에이전트 협업 중..."):
                    try:
                        res = run_workflow(user_input, st.session_state["dept"], st.session_state["officer"])
                        st.session_state["result"] = res
                    except Exception as e:
                        st.error(f"치명적 오류 발생: {e}")
                        
        # 대시보드 (간략)
        st.markdown("---")
        st.subheader("📊 시스템 상태")
        m = st.session_state.get("metrics", {})
        if m.get("calls"):
            for k, v in m["calls"].items():
                st.write(f"- {k}: {v}회 호출")
        else:
            st.info("대기 중...")

    with col_r:
        res = st.session_state.get("result")
        
        if not res:
            st.markdown("""
            <div style='text-align: center; padding: 120px 20px; color: #aaa; border: 2px dashed #ddd; border-radius: 12px;'>
                <h3>📄 Document Preview</h3>
                <p>왼쪽에서 업무를 입력하고 실행 버튼을 누르세요.<br>자동으로 법령을 검토하고 공문을 작성합니다.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            doc = res["doc"]
            meta = res["meta"]
            
            tab1, tab2 = st.tabs(["📄 공문서 결과", "🔍 근거 및 분석"])
            
            with tab1:
                # HTML 공문 렌더링
                html = f"""
                <div class="paper-sheet">
                    <div class="stamp">직인생략</div>
                    <div class="doc-header">{doc['title']}</div>
                    <div class="doc-info">
                        <span>문서번호: {meta['doc_num']}</span>
                        <span>시행일자: {meta['today']}</span>
                        <span>수신: {doc['receiver']}</span>
                    </div>
                    <div class="doc-body">
                        {''.join(f'<p>{p}</p>' for p in doc['body_paragraphs'])}
                    </div>
                    <div class="doc-footer">{doc['department_head']}</div>
                </div>
                """
                components.html(f"<html><head><style>{st.markdown}</style></head><body>{html}</body></html>", height=800, scrolling=True)
            
            with tab2:
                st.info(f"📜 법적 근거:\n{res['legal_basis']}")
                st.markdown("### 💡 처리 전략")
                st.markdown(res['strategy'])
                st.markdown("### 📎 참고 자료 (Naver)")
                for item in res['ev_items']:
                    st.markdown(f"- [{item['title']}]({item['link']})")

if __name__ == "__main__":
    main()
