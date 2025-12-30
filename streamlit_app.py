import streamlit as st
import time, json, re
from datetime import datetime, timedelta
from html import escape

# ===============================
# OPTIONAL IMPORTS (안 죽게)
# ===============================
try:
    import requests
except ImportError:
    requests = None

try:
    import xmltodict
except ImportError:
    xmltodict = None

try:
    from serpapi import GoogleSearch
except ImportError:
    GoogleSearch = None

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="AI 행정관 (Stable)",
    page_icon="⚖️",
    layout="wide"
)

# ===============================
# LLM SERVICE (안정판)
# ===============================
import google.generativeai as genai
from groq import Groq

class LLMService:
    def __init__(self):
        self.last_model = "N/A"
        self.gemini_key = st.secrets.get("general", {}).get("GEMINI_API_KEY")
        self.groq_key = st.secrets.get("general", {}).get("GROQ_API_KEY")

        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)

        self.groq = Groq(api_key=self.groq_key) if self.groq_key else None

    def text(self, prompt: str) -> str:
        if self.gemini_key:
            try:
                model = genai.GenerativeModel("gemini-2.5-flash")
                res = model.generate_content(prompt)
                self.last_model = "Gemini 2.5 Flash"
                return res.text.strip()
            except:
                pass

        if self.groq:
            try:
                res = self.groq.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                self.last_model = "Groq / llama-3.3-70b"
                return res.choices[0].message.content.strip()
            except:
                pass

        self.last_model = "LLM FAILED"
        return ""

llm = LLMService()

# ===============================
# LAW API (완전 방어형)
# ===============================
class LawService:
    def __init__(self):
        self.enabled = False
        self.oc = None
        self.base = None

        try:
            self.oc = st.secrets["law"]["LAW_API_ID"]
            self.base = "https://www.law.go.kr/DRF/lawService.do"
            self.enabled = requests is not None
        except:
            self.enabled = False

    def get_law_text(self, law_name: str) -> str:
        if not self.enabled:
            return ""

        try:
            r = requests.get(self.base, params={
                "OC": self.oc,
                "target": "law",
                "type": "XML",
                "query": law_name,
                "display": 1
            }, timeout=10)

            if xmltodict is None:
                return ""

            data = xmltodict.parse(r.text)
            law = data["LawSearch"]["law"]
            return law.get("lawNm", "")
        except:
            return ""

law_api = LawService()

# ===============================
# WORKFLOW
# ===============================
def run(user_input: str):
    t0 = time.time()
    timing = {}

    # 1. 법령 힌트 (LLM)
    s = time.time()
    law_hint = llm.text(
        f"상황: {user_input}\n"
        "적용될 법령명 하나만 추론해라. 예: 도로교통법"
    )
    timing["법령 힌트"] = int((time.time()-s)*1000)

    # 2. LAW API 검증
    s = time.time()
    law_confirmed = law_api.get_law_text(law_hint)
    timing["법령 API"] = int((time.time()-s)*1000)

    final_law = law_confirmed or law_hint or "관련 법령 없음"

    # 3. 공문 JSON 생성
    s = time.time()
    doc_raw = llm.text(f"""
상황: {user_input}
법령: {final_law}

아래 JSON만 출력:
{{
 "title": "공문 제목",
 "receiver": "수신인",
 "body_paragraphs": ["경위", "근거", "처분 내용", "권리구제"],
 "department_head": "충주시장"
}}
""")
    timing["공문 생성"] = int((time.time()-s)*1000)

    # JSON 정제
    try:
        doc = json.loads(re.search(r"\{.*\}", doc_raw, re.S).group())
    except:
        doc = {
            "title": "공문",
            "receiver": "수신인",
            "body_paragraphs": ["내용 생성 실패"],
            "department_head": "행정기관장"
        }

    timing["TOTAL"] = int((time.time()-t0)*1000)

    return doc, final_law, timing, llm.last_model

# ===============================
# UI
# ===============================
st.title("⚖️ AI 행정관 (오류 제거 안정판)")

user_input = st.text_area("업무 내용 입력", height=150)

if st.button("실행"):
    doc, law, timing, model = run(user_input)

    st.success("완료")

    st.markdown("### 📜 적용 법령 (원문 유지)")
    st.write(law)

    st.markdown("### 🤖 사용된 LLM 모델")
    st.code(model)

    st.markdown("### ⏱️ 단계별 응답 시간(ms)")
    for k,v in timing.items():
        st.write(f"- {k}: {v}ms")

    st.markdown("### 📄 공문 미리보기")
    st.markdown(f"""
<div style="background:white;padding:30px">
<h2 style="text-align:center">{escape(doc['title'])}</h2>
<p><b>수신:</b> {escape(doc['receiver'])}</p>
<hr>
{''.join(f'<p>{escape(p)}</p>' for p in doc['body_paragraphs'])}
<br><br>
<div style="text-align:center;font-weight:bold">
{escape(doc['department_head'])}
</div>
</div>
""", unsafe_allow_html=True)
