# app.py
import streamlit as st
import streamlit.components.v1 as components
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
# 1. Page Config & Styles
# ==========================================
st.set_page_config(
    layout="wide",
    page_title="AI Bureau: The Legal Glass",
    page_icon="⚖️"
)

st.markdown("""
<style>
.stApp { background-color: #f3f4f6; }

.paper-sheet {
    background: white;
    max-width: 210mm;
    min-height: 297mm;
    padding: 25mm;
    margin: auto;
    font-family: 'Noto Serif KR','Nanum Gothic','Malgun Gothic',serif;
    color: #111;
    line-height: 1.7;
}

.doc-header {
    text-align: center;
    font-size: 22pt;
    font-weight: 900;
    margin-bottom: 30px;
}

.doc-info {
    display: flex;
    justify-content: space-between;
    font-size: 11pt;
    border-bottom: 2px solid #000;
    padding-bottom: 10px;
    margin-bottom: 25px;
}

.doc-body { font-size: 12pt; }

.doc-footer {
    text-align: center;
    font-size: 18pt;
    font-weight: bold;
    margin-top: 80px;
}

.stamp {
    position: absolute;
    right: 80px;
    bottom: 90px;
    border: 3px solid #c00;
    color: #c00;
    padding: 6px 12px;
    transform: rotate(-15deg);
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Text Sanitizer (핵심)
# ==========================================
def strip_html_and_control(text: str) -> str:
    if not text:
        return ""
    text = str(text)
    # HTML 태그 제거
    text = re.sub(r"<[^>]+>", "", text)
    # 제어문자 제거
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    # zero-width 제거
    text = text.replace("\u200b", "")
    return text.strip()

def safe_html(text: str) -> str:
    return escape(strip_html_and_control(text)).replace("\n", "<br>")

# ==========================================
# 3. LLM Service
# ==========================================
class LLMService:
    def __init__(self):
        self.gemini_key = st.secrets["general"]["GEMINI_API_KEY"]
        self.groq_key = st.secrets["general"].get("GROQ_API_KEY")

        genai.configure(api_key=self.gemini_key)
        self.groq = Groq(api_key=self.groq_key) if self.groq_key else None

        self.models = [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash"
        ]

    def text(self, prompt: str) -> str:
        for m in self.models:
            try:
                model = genai.GenerativeModel(m)
                res = model.generate_content(prompt)
                return res.text
            except:
                continue
        if self.groq:
            return self.groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            ).choices[0].message.content
        return ""

    def json(self, prompt: str) -> dict:
        text = self.text(prompt + "\n\n[중요] JSON만 출력")
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        match = re.search(r"\{.*\}", text, re.S)
        return json.loads(match.group()) if match else {}

llm = LLMService()

# ==========================================
# 4. Domain Agents
# ==========================================
class LegalAgents:

    @staticmethod
    def researcher(situation):
        return llm.text(f"""
당신은 30년 경력의 법제관입니다.
상황: "{situation}"

적용할 수 있는 **현행 대한민국 법령과 조항**을
원문 형식으로 하나만 제시하세요.
""")

    @staticmethod
    def strategist(situation, legal_basis):
        return llm.text(f"""
당신은 행정 실무 주무관입니다.

[민원 상황]
{situation}

[적용 법령]
{legal_basis}

이 민원의 처리 전략을 5줄 이내로 작성하세요.
""")

    @staticmethod
    def clerk():
        today = datetime.now()
        deadline = today + timedelta(days=15)
        return {
            "today": today.strftime("%Y. %m. %d."),
            "deadline": deadline.strftime("%Y. %m. %d."),
            "doc_num": f"행정-{today.strftime('%Y')}-{int(time.time())%1000:03d}호"
        }

    @staticmethod
    def drafter(situation, legal_basis, meta, strategy):
        return llm.json(f"""
너는 행정기관 서기다.

[민원]
{situation}

[법령]
{legal_basis}

[전략]
{strategy}

[작성 규칙]
- HTML, 태그, 마크다운 사용 금지
- 순수 텍스트 문단만 작성
- body는 배열

출력 JSON 형식:
{{
 "title": "...",
 "receiver": "...",
 "body": ["문단1", "문단2"],
 "sender": "OO시장"
}}
""")

# ==========================================
# 5. Workflow
# ==========================================
def run(user_input):
    legal = LegalAgents.researcher(user_input)
    strategy = LegalAgents.strategist(user_input, legal)
    meta = LegalAgents.clerk()
    doc = LegalAgents.drafter(user_input, legal, meta, strategy)

    # 강제 정화
    body = [strip_html_and_control(p) for p in doc.get("body", [])]

    return {
        "title": strip_html_and_control(doc.get("title", "공 문 서")),
        "receiver": strip_html_and_control(doc.get("receiver", "수신자 참조")),
        "body": body,
        "sender": strip_html_and_control(doc.get("sender", "행정기관장")),
        "legal": legal,
        "meta": meta
    }

# ==========================================
# 6. UI
# ==========================================
def main():
    left, right = st.columns([1, 1.2])

    with left:
        st.title("🏛️ AI 행정관 Pro")
        user_input = st.text_area("업무 내용", height=150)

        if st.button("행정 처리 시작"):
            with st.spinner("처리 중..."):
                st.session_state.result = run(user_input)

    with right:
        if "result" in st.session_state:
            r = st.session_state.result

            st.subheader("📜 적용 법령 (원문)")
            st.info(r["legal"])

            html = f"""
<div class="paper-sheet">
<div class="stamp">직인생략</div>
<div class="doc-header">{safe_html(r["title"])}</div>

<div class="doc-info">
<span>문서번호: {r["meta"]["doc_num"]}</span>
<span>시행일자: {r["meta"]["today"]}</span>
<span>수신: {safe_html(r["receiver"])}</span>
</div>

<div class="doc-body">
"""
            for p in r["body"]:
                html += f"<p>{safe_html(p)}</p>"

            html += f"""
</div>
<div class="doc-footer">{safe_html(r["sender"])}</div>
</div>
"""

            components.html(html, height=1100, scrolling=True)

if __name__ == "__main__":
    main()
