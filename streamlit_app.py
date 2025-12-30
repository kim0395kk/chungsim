import streamlit as st
import google.generativeai as genai
from groq import Groq
import json
import re
import time
from datetime import datetime, timedelta

# ==========================================
# 1. Configuration & Styles (설정 및 디자인)
# ==========================================
st.set_page_config(layout="wide", page_title="AI Bureau: The Legal Glass", page_icon="⚖️")

st.markdown("""
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
    .doc-info { display: flex; justify-content: space-between; font-size: 11pt; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; }
    .doc-body { font-size: 12pt; text-align: justify; }
    .doc-footer { text-align: center; font-size: 20pt; font-weight: bold; margin-top: 80px; letter-spacing: 5px; }
    .stamp { position: absolute; bottom: 85px; right: 80px; border: 3px solid #cc0000; color: #cc0000; padding: 5px 10px; font-size: 14pt; font-weight: bold; transform: rotate(-15deg); opacity: 0.8; border-radius: 5px; }
    
    .agent-log { font-family: 'Consolas', monospace; font-size: 0.85rem; padding: 6px 12px; border-radius: 6px; margin-bottom: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    .log-legal { background-color: #eff6ff; color: #1e40af; border-left: 4px solid #3b82f6; }
    .log-calc { background-color: #f0fdf4; color: #166534; border-left: 4px solid #22c55e; }
    .log-draft { background-color: #fef2f2; color: #991b1b; border-left: 4px solid #ef4444; }
    .log-sys { background-color: #f3f4f6; color: #4b5563; border-left: 4px solid #9ca3af; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Service Layer (Infrastructure)
# ==========================================
class LLMService:
    """Gemini 2.5 및 2.0 모델만 사용하는 서비스"""
    def __init__(self):
        self.gemini_key = st.secrets["general"].get("GEMINI_API_KEY")
        self.groq_key = st.secrets["general"].get("GROQ_API_KEY")
        
        # [업데이트] 사용자 환경에 존재하는 모델만 등록 (우선순위 순)
        self.gemini_models = [
            "gemini-2.5-flash",       # 1순위: 최신 표준
            "gemini-2.5-flash-lite",  # 2순위: 최신 경량
            "gemini-2.0-flash",       # 3순위: 구버전 표준
            "gemini-2.0-flash-lite"   # 4순위: 구버전 경량
        ]
        
        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)
            
        self.groq_client = Groq(api_key=self.groq_key) if self.groq_key else None

    def _try_gemini(self, prompt, is_json=False, schema=None):
        """리스트에 있는 모델을 순차적으로 시도"""
        for model_name in self.gemini_models:
            try:
                # 2.5 버전 호환성을 위해 모델명 소문자 처리
                model_id = model_name.lower()
                model = genai.GenerativeModel(model_id)
                
                config = genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=schema
                ) if is_json else None
                
                res = model.generate_content(prompt, generation_config=config)
                return res.text, model_name
                
            except Exception as e:
                # 해당 모델 실패 시 다음 모델로 (로그는 생략)
                continue
                
        raise Exception("모든 Gemini 모델(2.5/2.0) 호출 실패")

    def generate_text(self, prompt):
        try:
            text, model_used = self._try_gemini(prompt, is_json=False)
            return text
        except Exception:
            if self.groq_client:
                return self._generate_groq(prompt)
            return "시스템 오류: AI 모델 연결 실패 (API Key 및 모델 권한을 확인하세요)"

    def generate_json(self, prompt, schema=None):
        try:
            text, model_used = self._try_gemini(prompt, is_json=True, schema=schema)
            return json.loads(text)
        except Exception:
            # Fallback
            text = self.generate_text(prompt + "\n\nOutput strictly in JSON.")
            try:
                match = re.search(r'\{.*\}', text, re.DOTALL)
                return json.loads(match.group(0)) if match else None
            except:
                return None

    def _generate_groq(self, prompt):
        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return completion.choices[0].message.content
        except:
            return "System Error"

# 싱글톤 인스턴스
llm_service = LLMService()

# ==========================================
# 3. Agent Layer (Business Logic)
# ==========================================
class LegalAgents:
    @staticmethod
    def researcher(situation):
        # [보안] 프롬프트 레벨 익명화 지침 추가
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
            days = int(re.sub(r'[^0-9]', '', res))
        except:
            days = 15
        deadline = today + timedelta(days=days)
        return {
            "today_str": today.strftime("%Y. %m. %d."),
            "deadline_str": deadline.strftime("%Y. %m. %d."),
            "days_added": days,
            "doc_num": f"행정-{today.strftime('%Y')}-{int(time.time())%1000:03d}호"
        }

    @staticmethod
    def drafter(situation, legal_basis, meta_info):
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
        당신은 행정기관의 베테랑 서기입니다. 아래 정보를 바탕으로 완결된 공문서를 작성하세요.
        
        [입력 정보]
        - 민원 상황: {situation}
        - 법적 근거: {legal_basis}
        - 문서 번호: {meta_info['doc_num']}
        - 시행 일자: {meta_info['today_str']}
        - 제출 기한: {meta_info['deadline_str']} ({meta_info['days_added']}일 부여됨)
        
        [작성 원칙]
        1. 수신인이 불명확하면 상황에 맞춰 'OOO 귀하', '차량소유주 귀하' 등으로 추론.
        2. 본문은 [처분 원인 및 경과] -> [법적 근거] -> [처분 내용 및 기한] -> [불이행 시 조치/구제절차] 순서로 작성.
        3. 어조는 정중하되 단호한 공문서 표준어 사용.
        4. (중요) 개인정보 보호를 위해 실명, 전화번호는 'OOO', '010-****-****' 형태로 마스킹하여 작성하세요.
        """
        return llm_service.generate_json(prompt, schema=doc_schema)

# ==========================================
# 4. Use Case & UI
# ==========================================
def run_workflow(user_input):
    log_placeholder = st.empty()
    logs = []
    def add_log(msg, style="sys"):
        logs.append(f"<div class='agent-log log-{style}'>{msg}</div>")
        log_placeholder.markdown("".join(logs), unsafe_allow_html=True)
        time.sleep(0.5)

    add_log("👨‍⚖️ Legal Agent: 법령 데이터베이스 검색 중...", "legal")
    legal_basis = LegalAgents.researcher(user_input)
    add_log(f"📜 법적 근거 확보: {legal_basis}", "legal")

    add_log("📅 Clerk Agent: 기한 산정 중...", "calc")
    meta_info = LegalAgents.clerk(user_input, legal_basis)
    add_log(f"⏳ 기한 설정: {meta_info['days_added']}일 ({meta_info['deadline_str']} 까지)", "calc")

    add_log("✍️ Drafter Agent: 공문서 작성 중 (Gemini 2.5)...", "draft")
    doc_data = LegalAgents.drafter(user_input, legal_basis, meta_info)
    
    add_log("✅ 절차 완료.", "sys")
    time.sleep(1)
    log_placeholder.empty()
    return doc_data, meta_info

def main():
    col_left, col_right = st.columns([1, 1.2])
    with col_left:
        st.title("🏢 AI 행정관 Pro")
        st.caption("Powered by Gemini 2.5 Flash")
        st.markdown("---")
        st.markdown("### 🗣️ 업무 지시")
        user_input = st.text_area("업무 내용", height=150, placeholder="예시:\n- 식품위생법 위반 업소 영업정지 사전통지서 작성해줘 (업소명: 대박식당)", label_visibility="collapsed")
        
        if st.button("⚡ 행정 처분 시작", type="primary", use_container_width=True):
            if not user_input:
                st.warning("내용을 입력해주세요.")
            else:
                try:
                    with st.spinner("Gemini 2.5 에이전트 구동 중..."):
                        doc, meta = run_workflow(user_input)
                        st.session_state['final_doc'] = (doc, meta)
                except Exception as e:
                    st.error(f"오류: {e}")
        st.markdown("---")
        st.info("💡 **Security Notice:** 본 시스템은 Free Tier를 사용 중이므로, 실제 민원인의 민감정보(주민번호 등)는 입력하지 마시거나 마스킹하여 입력해주세요.")

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
                st.download_button(label="🖨️ 다운로드 (HTML)", data=html_content, file_name="공문서.html", mime="text/html", use_container_width=True)
        else:
            st.markdown("""<div style='text-align: center; padding: 100px; color: #aaa; background: white; border-radius: 10px; border: 2px dashed #ddd;'><h3>📄 Document Preview</h3><p>왼쪽에서 업무를 지시하면<br>완성된 공문서가 여기에 나타납니다.</p></div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
