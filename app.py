import streamlit as st
import google.generativeai as genai
from groq import Groq
from serpapi import GoogleSearch
from supabase import create_client
import json
import re
import time
from datetime import datetime, timedelta

# ==========================================
# 1. 환경 설정 & 스타일 (UI 깨짐 방지 수정)
# ==========================================
st.set_page_config(layout="wide", page_title="AI 행정관 Pro", page_icon="⚖️")

st.markdown("""
<style>
    .stApp { background-color: #f3f4f6; }
    /* 문서 디자인: A4 용지 느낌 */
    .paper-sheet {
        background-color: white;
        width: 100%;
        max-width: 210mm;
        min-height: 297mm;
        padding: 20mm;
        margin: auto;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        font-family: 'Batang', serif; /* 바탕체 */
        color: #111;
        line-height: 1.8;
        position: relative;
    }
    .doc-header { text-align: center; font-size: 24pt; font-weight: 900; margin-bottom: 40px; }
    .doc-info { 
        border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 30px;
        font-size: 11pt; display: flex; justify-content: space-between;
    }
    .doc-body { font-size: 12pt; text-align: justify; white-space: pre-line; }
    .doc-footer { text-align: center; font-size: 22pt; font-weight: bold; margin-top: 60px; }
    .stamp { 
        position: absolute; bottom: 65px; right: 50px; 
        border: 3px solid #cc0000; color: #cc0000; 
        padding: 5px 10px; font-size: 14pt; font-weight: bold; 
        transform: rotate(-15deg); opacity: 0.8; border-radius: 5px; 
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 서비스 연결 (Supabase, AI)
# ==========================================
class LLMService:
    def __init__(self):
        self.gemini_key = st.secrets["general"].get("GEMINI_API_KEY")
        self.groq_key = st.secrets["general"].get("GROQ_API_KEY")
        # [수정] 에러 나던 2.5 모델 삭제 -> 1.5 Flash로 변경 (안전)
        self.gemini_models = ["gemini-1.5-flash", "gemini-2.0-flash-exp"]
        
        if self.gemini_key: genai.configure(api_key=self.gemini_key)
        self.groq_client = Groq(api_key=self.groq_key) if self.groq_key else None

    def generate_text(self, prompt):
        # Gemini 시도
        for model_name in self.gemini_models:
            try:
                model = genai.GenerativeModel(model_name)
                return model.generate_content(prompt).text
            except: continue
        
        # Groq 시도 (백업)
        if self.groq_client:
            try:
                return self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}]
                ).choices[0].message.content
            except: pass
            
        return "AI 모델 연결 실패"

    def generate_json(self, prompt):
        # 텍스트 생성 후 JSON 파싱
        text = self.generate_text(prompt + "\n\nOutput strictly in JSON format.")
        try:
            # ```json ... ``` 제거 및 파싱
            clean_text = re.sub(r"```json|```", "", text).strip()
            match = re.search(r'\{.*\}', clean_text, re.DOTALL)
            return json.loads(match.group(0)) if match else json.loads(clean_text)
        except: return None

class DatabaseService:
    def __init__(self):
        try:
            self.url = st.secrets["supabase"]["SUPABASE_URL"]
            self.key = st.secrets["supabase"]["SUPABASE_KEY"]
            self.client = create_client(self.url, self.key)
            self.is_active = True
        except: self.is_active = False

    def save_report(self, user_input, legal_basis, doc_data):
        if not self.is_active: return "❌ 비밀키(Secrets) 설정 확인 필요"
        try:
            summary_json = json.dumps(doc_data, ensure_ascii=False)
            data = {"situation": user_input, "law_name": legal_basis, "summary": summary_json}
            # 테이블 이름: law_reports
            self.client.table("law_reports").insert(data).execute()
            return "✅ DB 저장 성공"
        except Exception as e:
            # [디버그] 에러 원인을 화면에 표시
            return f"❌ 저장 실패: {str(e)}"

llm = LLMService()
db = DatabaseService()

# ==========================================
# 3. AI 에이전트 (업무 로직)
# ==========================================
class Agents:
    @staticmethod
    def researcher(text):
        return llm.generate_text(f"상황: '{text}'\n관련된 대한민국 법령명과 조항 번호만 간단히 알려줘.")

    @staticmethod
    def clerk(text):
        today = datetime.now()
        return {
            "today": today.strftime("%Y. %m. %d."),
            "doc_num": f"행정-{today.strftime('%Y')}-{int(time.time())%1000}호"
        }

    @staticmethod
    def drafter(text, law, meta):
        prompt = f"""
        공문서를 작성해줘. JSON 포맷으로 출력해.
        
        상황: {text}
        근거: {law}
        문서번호: {meta['doc_num']}
        시행일: {meta['today']}
        
        필수 항목:
        1. title (제목)
        2. receiver (수신자)
        3. body_paragraphs (본문 문단 리스트, 배열 형태)
        4. department_head (발신 명의 - 예: OO시장, OO구청장) *'행정'이라고 줄이지 말 것.
        """
        res = llm.generate_json(prompt)
        # 실패 시 기본값
        if not res:
            return {
                "title": "공 문 서", "receiver": "수신자 참조",
                "body_paragraphs": ["내용 생성 실패. 다시 시도해주세요."], "department_head": "행정기관장"
            }
        return res

# ==========================================
# 4. 메인 화면 (UI)
# ==========================================
def main():
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.title("🏢 AI 행정관")
        user_input = st.text_area("업무 지시", height=150, placeholder="예: 무단투기 과태료 부과 예고 통지서 작성해")
        
        if st.button("⚡ 문서 생성", type="primary"):
            if user_input:
                with st.spinner("AI가 작성 중..."):
                    # 1. AI 작업
                    law = Agents.researcher(user_input)
                    meta = Agents.clerk(user_input)
                    doc = Agents.drafter(user_input, law, meta)
                    
                    # 2. DB 저장 시도
                    msg = db.save_report(user_input, law, doc)
                    
                    # 3. 결과 저장
                    st.session_state['result'] = (doc, meta, msg)
            else:
                st.warning("내용을 입력하세요.")

    with col2:
        if 'result' in st.session_state:
            doc, meta, msg = st.session_state['result']
            
            # 저장 결과 표시 (성공/실패)
            if "성공" in msg: st.success(msg)
            else: st.error(msg) # 여기서 에러 메시지를 확인하세요!

            # HTML 조립 (깨짐 방지)
            body_text = "\n\n".join(doc.get('body_paragraphs', []))
            
            # f-string 안에서 HTML 구조를 단순화
            html = f"""
            <div class="paper-sheet">
                <div class="stamp">직인생략</div>
                <div class="doc-header">{doc.get('title', '제목 없음')}</div>
                <div class="doc-info">
                    <span>문서번호: {meta['doc_num']}</span>
                    <span>시행일자: {meta['today']}</span>
                    <span>수신: {doc.get('receiver')}</span>
                </div>
                <div class="doc-body">{body_text}</div>
                <div class="doc-footer">{doc.get('department_head', '기관장')}</div>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
