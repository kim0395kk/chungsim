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
# 0. secrets safe access (1번)
# ==========================================
def sget(section, key, default=None):
    try:
        return st.secrets.get(section, {}).get(key, default)
    except Exception:
        return default

def sget_required(section, key):
    v = sget(section, key)
    if not v:
        st.error(f"❌ secrets 누락: [{section}] {key}")
        st.stop()
    return v

# ==========================================
# 1. Configuration & Styles
# ==========================================
st.set_page_config(layout="wide", page_title="AI Bureau: The Legal Glass", page_icon="⚖️")

st.markdown("""
<style>
.stApp { background-color:#f3f4f6; }
.paper-sheet {
  background:white; max-width:210mm; min-height:297mm;
  padding:25mm; margin:auto; box-shadow:0 10px 30px rgba(0,0,0,.1);
  font-family:'Batang', serif;
}
.doc-header { text-align:center; font-size:22pt; font-weight:900; margin-bottom:30px; }
.doc-info { display:flex; justify-content:space-between; font-size:11pt; border-bottom:2px solid #333; }
.doc-footer { text-align:center; font-size:20pt; margin-top:80px; }
.stamp { position:absolute; bottom:85px; right:80px; border:3px solid #c00; color:#c00; transform:rotate(-15deg); }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Services
# ==========================================
class LLMService:
    def __init__(self):
        self.gemini_key = sget("general", "GEMINI_API_KEY")
        self.groq_key = sget("general", "GROQ_API_KEY")
        self.models = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]
        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)
        self.groq = Groq(api_key=self.groq_key) if self.groq_key else None

    def _try_gemini(self, prompt, is_json=False, schema=None):
        for m in self.models:
            try:
                model = genai.GenerativeModel(m)
                cfg = genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=schema
                ) if is_json else None
                return model.generate_content(prompt, generation_config=cfg).text
            except:
                continue
        raise Exception("Gemini failed")

    def generate_text(self, prompt):
        try:
            return self._try_gemini(prompt)
        except:
            if not self.groq:
                return "AI ERROR"
            return self.groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role":"user","content":prompt}],
                temperature=0.1
            ).choices[0].message.content

    def generate_json(self, prompt, schema=None):
        try:
            return json.loads(self._try_gemini(prompt, True, schema))
        except:
            txt = self.generate_text(prompt + "\nJSON만 출력")
            try:
                return json.loads(re.search(r"\{.*\}", txt, re.DOTALL).group())
            except:
                return None

class NationalLawService:
    # 3번: display=5 + 정확도 스코어
    def __init__(self):
        self.api_id = sget_required("general", "LAW_API_ID")
        self.search_url = "https://www.law.go.kr/DRF/lawSearch.do"
        self.detail_url = "https://www.law.go.kr/DRF/lawService.do"

    def get_specific_article(self, law_name, article_num):
        try:
            params = {
                "OC": self.api_id,
                "target": "law",
                "type": "XML",
                "query": law_name,
                "display": 5
            }
            root = ET.fromstring(requests.get(self.search_url, params=params).content)
            laws = root.findall(".//law")
            if not laws:
                return "법령 없음"

            def score(n):
                nm = n.findtext("법령명한글","")
                return 100 if nm == law_name else 50 if law_name in nm else 0

            law = sorted(laws, key=score, reverse=True)[0]
            mst = law.findtext("법령일련번호")
            name = law.findtext("법령명한글", law_name)

            d = ET.fromstring(requests.get(self.detail_url, params={
                "OC": self.api_id, "target": "law", "type": "XML", "MST": mst
            }).content)

            num = re.sub(r"\D","",str(article_num))
            for j in d.findall(".//조문"):
                if j.findtext("조문번호") == num:
                    txt = j.findtext("조문내용","")
                    return f"[{name} 제{num}조] {txt}"
            return "조문 없음"
        except Exception as e:
            return f"법령 오류: {e}"

class SearchService:
    def __init__(self):
        self.key = sget("general","SERPAPI_KEY")

    def search(self, q):
        if not self.key:
            return "검색 비활성"
        try:
            r = GoogleSearch({
                "engine":"google","q":q+" 행정처분 판례",
                "api_key":self.key,"hl":"ko","gl":"kr","num":3
            }).get_dict().get("organic_results",[])
            return "\n".join(f"- {x['title']}" for x in r) if r else "없음"
        except:
            return "검색 오류"

class DatabaseService:
    def __init__(self):
        self.url = sget("supabase","SUPABASE_URL")
        self.key = sget("supabase","SUPABASE_KEY")
        self.active = bool(self.url and self.key)
        self.client = create_client(self.url,self.key) if self.active else None

    def save(self, situation, law, doc):
        if not self.active:
            return "DB OFF"
        self.client.table("law_reports").insert({
            "situation": situation,
            "law_name": law,
            "summary": json.dumps(doc, ensure_ascii=False)
        }).execute()
        return "OK"

# ==========================================
# 3. Workflow
# ==========================================
llm = LLMService()
law_api = NationalLawService()
search_api = SearchService()
db = DatabaseService()

def run(situation):
    law = llm.generate_json(f"상황:{situation} 법령/조항 JSON {{law,article_num}}") or {}
    text = law_api.get_specific_article(law.get("law","도로교통법"), law.get("article_num",32))
    strategy = llm.generate_text(f"{situation}\n{text}")
    doc = llm.generate_json(f"공문 JSON 작성\n{situation}\n{text}\n{strategy}") or {}
    db.save(situation, text, doc)
    return doc

# ==========================================
# 4. UI
# ==========================================
st.title("🏢 AI 행정관 Pro")

q = st.text_area("업무 내용")

if st.button("실행"):
    with st.spinner("처리중"):
        doc = run(q)
        st.json(doc)
