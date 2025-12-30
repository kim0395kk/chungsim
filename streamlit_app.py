import streamlit as st
import google.generativeai as genai
from groq import Groq
from serpapi import GoogleSearch
from supabase import create_client
import requests
import xml.etree.ElementTree as ET
import json
import re
import time
from datetime import datetime, timedelta

# ==========================================
# 1. Configuration & Styles
# ==========================================
st.set_page_config(layout="wide", page_title="AI Bureau: The Legal Glass", page_icon="⚖️")

st.markdown("""
<style>
    .stApp { background-color: #f3f4f6; }
    
    /* A4 Paper Style */
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
    .doc-info { display: flex; justify-content: space-between; font-size: 11pt; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; }
    .doc-body { font-size: 12pt; text-align: justify; }
    .doc-footer { text-align: center; font-size: 20pt; font-weight: bold; margin-top: 80px; letter-spacing: 5px; }
    .stamp { position: absolute; bottom: 85px; right: 80px; border: 3px solid #cc0000; color: #cc0000; padding: 5px 10px; font-size: 14pt; font-weight: bold; transform: rotate(-15deg); opacity: 0.8; border-radius: 5px; }
    
    /* Logs & Strategy Box */
    .agent-log { font-family: 'Consolas', monospace; font-size: 0.85rem; padding: 6px 12px; border-radius: 6px; margin-bottom: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    .log-legal { background-color: #eff6ff; color: #1e40af; border-left: 4px solid #3b82f6; }
    .log-search { background-color: #fff7ed; color: #c2410c; border-left: 4px solid #f97316; }
    .log-strat { background-color: #f5f3ff; color: #6d28d9; border-left: 4px solid #8b5cf6; }
    .log-calc { background-color: #f0fdf4; color: #166534; border-left: 4px solid #22c55e; }
    .log-draft { background-color: #fef2f2; color: #991b1b; border-left: 4px solid #ef4444; }
    .log-sys { background-color: #f3f4f6; color: #4b5563; border-left: 4px solid #9ca3af; }
    
    .strategy-box { 
        background-color: #fffbeb; 
        border: 2px solid #fcd34d; 
        padding: 20px; 
        border-radius: 10px; 
        margin-bottom: 20px;
        color: #451a03;
        font-size: 1.05rem;
        line-height: 1.6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Infrastructure Layer (Services)
# ==========================================

class LLMService:
    def __init__(self):
        self.gemini_key = st.secrets["general"].get("GEMINI_API_KEY")
        self.groq_key = st.secrets["general"].get("GROQ_API_KEY")
        self.gemini_models = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]
        
        if self.gemini_key: genai.configure(api_key=self.gemini_key)
        self.groq_client = Groq(api_key=self.groq_key) if self.groq_key else None

    def _try_gemini(self, prompt, is_json=False, schema=None):
        for model_name in self.gemini_models:
            try:
                model = genai.GenerativeModel(model_name)
                config = genai.GenerationConfig(response_mime_type="application/json", response_schema=schema) if is_json else None
                res = model.generate_content(prompt, generation_config=config)
                return res.text
            except: continue
        raise Exception("All Gemini models failed")

    def generate_text(self, prompt):
        try: return self._try_gemini(prompt, is_json=False)
        except: return self._generate_groq(prompt) if self.groq_client else "AI 모델 오류"

    def generate_json(self, prompt, schema=None):
        try:
            text = self._try_gemini(prompt, is_json=True, schema=schema)
            return json.loads(text)
        except:
            text = self.generate_text(prompt + "\n\nOutput strictly in JSON.")
            try: return json.loads(re.search(r'\{.*\}', text, re.DOTALL).group(0))
            except: return None

    def _generate_groq(self, prompt):
        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return completion.choices[0].message.content
        except: return "System Error"

class NationalLawService:
    """[NEW] 국가법령정보센터 공식 API 연동"""
    def __init__(self):
        self.api_id = st.secrets["general"].get("LAW_API_ID") # secrets에 ID 필요
        self.base_url = "https://www.law.go.kr/DRF/lawSearch.do"
        self.detail_url = "https://www.law.go.kr/DRF/lawService.do"

    def get_law_text(self, keyword):
        """법령명을 검색하여 본문(조문) 일부를 가져옴"""
        if not self.api_id:
            return "(시스템 알림: LAW_API_ID가 설정되지 않아 LLM 지식으로 대체합니다.)"
            
        try:
            # 1. 법령 검색 (XML)
            params = {"OC": self.api_id, "target": "law", "type": "XML", "query": keyword, "display": 1}
            res = requests.get(self.base_url, params=params, timeout=5)
            root = ET.fromstring(res.content)
            
            law_node = root.find(".//law")
            if law_node is None: return "검색된 법령이 없습니다."
            
            law_id = law_node.find("법령일련번호").text
            law_name = law_node.find("법령명한글").text
            
            # 2. 상세 본문 조회
            d_params = {"OC": self.api_id, "target": "law", "type": "XML", "MST": law_id}
            d_res = requests.get(self.detail_url, params=d_params, timeout=10)
            d_root = ET.fromstring(d_res.content)
            
            # 3. 조문 텍스트 추출 (앞부분 15,000자만 - 토큰 절약)
            articles = []
            for article in d_root.findall(".//조문")[:30]: # 상위 30개 조항만 예시로
                num = article.find("조문번호").text or ""
                content = article.find("조문내용").text or ""
                articles.append(f"[제{num}조] {content}")
                
            return f"공식 법령명: {law_name}\n\n" + "\n".join(articles)
            
        except Exception as e:
            return f"법령 API 호출 오류: {e}"

class SearchService:
    """구글 검색 (유사 사례/판례용)"""
    def __init__(self):
        self.api_key = st.secrets["general"].get("SERPAPI_KEY")

    def search_google(self, query):
        if not self.api_key: return "API 키 없음"
        try:
            params = {"engine": "google", "q": query + " 행정처분 판례 사례", "api_key": self.api_key, "num": 3, "hl": "ko", "gl": "kr"}
            search = GoogleSearch(params)
            results = search.get_dict().get("organic_results", [])
            return "\n".join([f"- {item['title']}: {item['snippet']}" for item in results]) if results else "결과 없음"
        except: return "검색 오류"

class DatabaseService:
    def __init__(self):
        try:
            self.url = st.secrets["supabase"]["SUPABASE_URL"]
            self.key = st.secrets["supabase"]["SUPABASE_KEY"]
            self.client = create_client(self.url, self.key)
            self.is_active = True
        except: self.is_active = False

    def save_report(self, user_input, legal_basis, doc_data):
        if not self.is_active: return "DB 미연결"
        try:
            summary_text = json.dumps(doc_data, ensure_ascii=False, indent=2)
            data = {"situation": user_input, "law_name": legal_basis, "summary": summary_text}
            self.client.table("law_reports").insert(data).execute()
            return "저장 성공"
        except Exception as e: return f"저장 실패({e})"

llm_service = LLMService()
law_api = NationalLawService()
search_service = SearchService()
db_service = DatabaseService()

# ==========================================
# 3. Domain Layer (Agents - 로직 이원화)
# ==========================================
class LegalAgents:
    @staticmethod
    def researcher(situation):
        """
        [이원화 로직]
        1. LLM이 '검색할 법령 키워드' 추출 (예: '도로교통법')
        2. Law API가 실제 법령 텍스트(Official Text)를 가져옴
        3. LLM이 그 텍스트 안에서 상황에 맞는 조항을 Pick
        """
        # Step 1. 키워드 추출
        kw_prompt = f"상황: '{situation}'\n이 상황에 적용될 가장 유력한 대한민국 법령 이름 1개만 단어로 출력. (예: 도로교통법)"
        target_law = llm_service.generate_text(kw_prompt).strip()
        
        # Step 2. 공식 API 호출
        official_text = law_api.get_law_text(target_law)
        
        # Step 3. 조항 추출 (Context 주입)
        final_prompt = f"""
        당신은 법제관입니다.
        
        [민원 상황]: {situation}
        [국가법령정보센터 원문 데이터]: 
        {official_text} (...생략...)
        
        위 원문 데이터를 근거로, 이 상황에 적용할 정확한 '법령명'과 '제O조(조문 제목)'을 찾아내세요.
        만약 원문에 정확한 조항이 없다면 당신의 지식을 보태어 가장 적절한 조항을 제시하세요.
        """
        return llm_service.generate_text(final_prompt).strip()

    @staticmethod
    def strategist(situation, legal_basis, search_results):
        prompt = f"""
        당신은 행정 주무관입니다.
        [상황]: {situation}
        [법적 근거]: {legal_basis}
        [유사 사례]: {search_results}
        
        위 정보를 종합하여 **업무 처리 전략(Strategy)**을 수립하세요.
        다음 3가지를 포함하여 마크다운으로 작성하세요:
        1. **처리 방향**: (강경/계도/반려 등)
        2. **핵심 주의사항**: (절차적 흠결 방지)
        3. **대응 논리**: (민원인 반발 시)
        """
        return llm_service.generate_text(prompt)

    @staticmethod
    def clerk(situation, legal_basis):
        today = datetime.now()
        prompt = f"""
        오늘: {today.strftime('%Y-%m-%d')}, 법령: {legal_basis}
        법적/통상적 의견제출 기한(일수) 숫자만 출력. (기본 15)
        """
        try:
            res = llm_service.generate_text(prompt)
            days = int(re.sub(r'[^0-9]', '', res))
        except: days = 15
        
        deadline = today + timedelta(days=days)
        return {
            "today_str": today.strftime("%Y. %m. %d."),
            "deadline_str": deadline.strftime("%Y. %m. %d."),
            "days_added": days,
            "doc_num": f"행정-{today.strftime('%Y')}-{int(time.time())%1000:03d}호"
        }

    @staticmethod
    def drafter(situation, legal_basis, meta_info, strategy):
        doc_schema = {
            "type": "OBJECT",
            "properties": {
                "title": {"type": "STRING"},
                "receiver": {"type": "STRING"},
                "body_paragraphs": {"type": "ARRAY", "items": {"type": "STRING"}},
                "department_head": {"type": "STRING"}
            },
            "required": ["title", "receiver", "body_paragraphs", "department_head"]
        }
        prompt = f"""
        베테랑 서기입니다. 공문서를 작성하세요.
        상황: {situation}, 법령: {legal_basis}, 기한: {meta_info['deadline_str']}
        전략: {strategy}
        작성원칙: 정중하고 단호하게. 개인정보 마스킹.
        """
        return llm_service.generate_json(prompt, schema=doc_schema)

# ==========================================
# 4. Workflow (Orchestration)
# ==========================================
def run_workflow(user_input):
    log_placeholder = st.empty()
    logs = []
    def add_log(msg, style="sys"):
        logs.append(f"<div class='agent-log log-{style}'>{msg}</div>")
        log_placeholder.markdown("".join(logs), unsafe_allow_html=True)
        time.sleep(0.3)

    # 1. 리서치 (API 이원화)
    add_log("📡 Phase 1: 국가법령정보센터(API) 원문 조회 중...", "legal")
    legal_basis = LegalAgents.researcher(user_input)
    
    add_log("🌍 Phase 1-2: 구글 검색(판례/사례) 조회 중...", "search")
    search_results = search_service.search_google(user_input)
    
    with st.expander("✅ [팩트체크] 법령 원문 및 유사 사례", expanded=True):
        col1, col2 = st.columns(2)
        with col1: st.info(f"**법령(law.go.kr)**\n\n{legal_basis}")
        with col2: st.warning(f"**판례(Google)**\n\n{search_results}")

    # 2. 전략 수립
    add_log("🧠 Phase 2: 업무 처리 전략 수립...", "strat")
    strategy = LegalAgents.strategist(user_input, legal_basis, search_results)
    
    # [전략 박스 UI 적용]
    st.markdown(f"""
    <div class="strategy-box">
        <div class="strategy-title">🧭 AI 주무관의 업무 가이드라인</div>
        {strategy}
    </div>
    """, unsafe_allow_html=True)

    # 3. 문서 작성
    add_log("✍️ Phase 3: 문서 작성 및 기한 산정...", "draft")
    meta_info = LegalAgents.clerk(user_input, legal_basis)
    doc_data = LegalAgents.drafter(user_input, legal_basis, meta_info, strategy)
    
    # 4. 저장
    add_log("💾 law_reports 테이블에 저장 중...", "sys")
    save_msg = db_service.save_report(user_input, legal_basis, doc_data)
    
    add_log(f"✅ 완료 ({save_msg})", "sys")
    time.sleep(1)
    log_placeholder.empty()

    return doc_data, meta_info

# ==========================================
# 5. Main UI
# ==========================================
def main():
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.title("🏢 AI 행정관 Pro")
        st.caption("Law API(Statute) + Google(Case) + Gemini 2.5")
        st.markdown("---")
        
        user_input = st.text_area("업무 내용", height=150, placeholder="예: 아파트 단지 내 개인형 이동장치 수거 안내문 작성해줘")
        
        if st.button("⚡ 실행", type="primary", use_container_width=True):
            if not user_input:
                st.warning("내용을 입력하세요.")
            else:
                try:
                    with st.spinner("처리 중..."):
                        doc, meta = run_workflow(user_input)
                        st.session_state['final_doc'] = (doc, meta)
                except Exception as e:
                    st.error(f"오류: {e}")

    with col_right:
        if 'final_doc' in st.session_state:
            doc, meta = st.session_state['final_doc']
            if doc:
                html_content = f"""
                <div class="paper-sheet">
                    <div class="stamp">직인생략</div>
                    <div class="doc-header">{doc.get('title', '공 문 서')}</div>
                    <div class="doc-info">
                        <span>문서번호: {meta['doc_num']}</span>
                        <span>시행일자: {meta['today_str']}</span>
                        <span>수신: {doc.get('receiver', '수신자 참조')}</span>
                    </div>
                    <hr style="border: 1px solid black; margin-bottom: 30px;">
                    <div class="doc-body">
                """
                paragraphs = doc.get('body_paragraphs', [])
                if isinstance(paragraphs, str): paragraphs = [paragraphs]
                for p in paragraphs:
                    html_content += f"<p style='margin-bottom: 15px;'>{p}</p>"
                html_content += f"""
                    </div>
                    <div class="doc-footer">{doc.get('department_head', '행정기관장')}</div>
                </div>
                """
                st.markdown(html_content, unsafe_allow_html=True)
                st.download_button(label="🖨️ 다운로드", data=html_content, file_name="공문서.html", mime="text/html", use_container_width=True)
        else:
            st.markdown("<div style='text-align:center;padding:100px;color:#aaa;'><h3>📄 Preview</h3></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
