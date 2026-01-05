import json
import re
import time
from datetime import datetime, timedelta
from html import escape as _escape

import requests
import streamlit as st

# optional imports (없어도 안 죽게)
try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from groq import Groq
except Exception:
    Groq = None

try:
    from supabase import create_client
except Exception:
    create_client = None


# =====================================================
# 1. Page & Style
# =====================================================
st.set_page_config(layout="wide", page_title="AI 행정관 Pro", page_icon="⚖️")

st.markdown("""
<style>
.stApp { background-color: #f3f4f6; }

.paper-sheet {
  background: white;
  width: 100%;
  max-width: 210mm;
  min-height: 297mm;
  padding: 25mm;
  margin: auto;
  box-shadow: 0 10px 30px rgba(0,0,0,0.1);
  font-family: 'Batang', serif;
}

.doc-header { text-align:center; font-size:22pt; font-weight:900; margin-bottom:30px; }
.doc-info { display:flex; justify-content:space-between; border-bottom:2px solid #333; padding-bottom:10px; margin-bottom:20px; font-size:11pt; }
.doc-body { font-size:12pt; }
.doc-footer { text-align:center; font-size:20pt; font-weight:bold; margin-top:80px; }
.stamp {
  position:absolute; right:80px; bottom:85px;
  border:3px solid #cc0000; color:#cc0000;
  padding:5px 10px; font-weight:bold;
  transform:rotate(-15deg);
}
.agent-log { font-family:Consolas; font-size:0.85rem; padding:6px 12px; margin-bottom:6px; border-left:4px solid #999; background:#f9fafb; }
.log-legal { border-color:#2563eb; background:#eff6ff; }
.log-search { border-color:#ea580c; background:#fff7ed; }
.log-strat { border-color:#7c3aed; background:#f5f3ff; }
.log-draft { border-color:#dc2626; background:#fef2f2; }
</style>
""", unsafe_allow_html=True)


# =====================================================
# 2. Secrets
# =====================================================
G = st.secrets.get("general", {})
S = st.secrets.get("supabase", {})

GEMINI_KEY = G.get("GEMINI_API_KEY")
GROQ_KEY = G.get("GROQ_API_KEY")
SERPAPI_KEY = G.get("SERPAPI_KEY")
NAVER_ID = G.get("NAVER_CLIENT_ID")
NAVER_SECRET = G.get("NAVER_CLIENT_SECRET")
LAW_OC = G.get("LAW_OC")

SUPABASE_URL = S.get("SUPABASE_URL")
SUPABASE_KEY = S.get("SUPABASE_KEY")


# =====================================================
# 3. LLM Service
# =====================================================
class LLMService:
    def __init__(self):
        self.gemini_ok = bool(GEMINI_KEY and genai)
        self.groq_ok = bool(GROQ_KEY and Groq)

        if self.gemini_ok:
            genai.configure(api_key=GEMINI_KEY)

        self.groq = Groq(api_key=GROQ_KEY) if self.groq_ok else None
        self.models = ["gemini-2.5-flash", "gemini-2.0-flash"]

    def text(self, prompt: str) -> str:
        if self.gemini_ok:
            for m in self.models:
                try:
                    res = genai.GenerativeModel(m).generate_content(prompt)
                    return res.text.strip()
                except Exception:
                    continue

        if self.groq_ok:
            r = self.groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return r.choices[0].message.content.strip()

        return "LLM 오류"

    def json(self, prompt: str) -> dict | None:
        txt = self.text(prompt + "\n\nJSON만 출력")
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        return json.loads(m.group()) if m else None


llm = LLMService()


# =====================================================
# 4. Google / Naver Search
# =====================================================
def google_search(q):
    if not SERPAPI_KEY:
        return "⚠️ Google 검색 키 없음"

    params = {
        "engine": "google",
        "q": f"{q} 행정처분 판례 site:go.kr OR site:law.go.kr",
        "api_key": SERPAPI_KEY,
        "hl": "ko",
        "gl": "kr",
        "num": 5
    }
    r = requests.get("https://serpapi.com/search.json", params=params, timeout=15)
    items = r.json().get("organic_results", [])

    if not items:
        return "Google 결과 없음"

    out = ["**[Google 유사사례]**"]
    for it in items[:5]:
        out.append(f"- **[{it.get('title')}]({it.get('link')})**: {it.get('snippet')}")
    return "\n".join(out)


def naver_search(q):
    if not (NAVER_ID and NAVER_SECRET):
        return "⚠️ Naver 검색 키 없음"

    headers = {
        "X-Naver-Client-Id": NAVER_ID,
        "X-Naver-Client-Secret": NAVER_SECRET
    }

    params = {
        "query": f"{q} 행정처분 판례 law.go.kr",
        "display": 5,
        "sort": "sim"
    }

    r = requests.get("https://openapi.naver.com/v1/search/webkr.json",
                     headers=headers, params=params, timeout=15)
    items = r.json().get("items", [])

    clean = []
    for it in items:
        link = it.get("link", "")
        if any(x in link for x in ["blog.", "cafe.", "tistory", "velog"]):
            continue
        title = re.sub("<[^>]+>", "", it.get("title", ""))
        desc = re.sub("<[^>]+>", "", it.get("description", ""))
        clean.append(f"- **[{title}]({link})**: {desc}")

    return "**[Naver 유사사례(정제)]**\n" + "\n".join(clean) if clean else "Naver 결과 없음"


# =====================================================
# 5. Law Resolver (핵심 수정판)
# =====================================================
def resolve_law(law_text: str):
    """
    - 시행령 / 시행규칙 자동 제거
    - 본법 1개만 선택
    - 절대 URL + OC 마스킹
    """
    if not LAW_OC:
        return {"ok": False, "text": law_text}

    name = law_text.split("제")[0].strip()

    params = {
        "OC": LAW_OC,
        "target": "law",
        "type": "JSON",
        "query": name,
        "display": 5
    }

    r = requests.get("https://www.law.go.kr/DRF/lawSearch.do",
                     params=params, timeout=15)
    data = r.json()

    laws = data.get("law", [])
    if not laws:
        return {"ok": False, "text": law_text}

    # 시행령/규칙 제거
    base = [l for l in laws if "시행" not in l.get("법령명한글", "")]

    chosen = base[0] if base else laws[0]

    link = chosen.get("법령상세링크", "")
    if link.startswith("/"):
        link = "https://www.law.go.kr" + link
    link = re.sub(r"(OC=)[^&]+", r"\1***", link)

    return {
        "ok": True,
        "name": chosen.get("법령명한글"),
        "dept": chosen.get("소관부처명"),
        "effective": chosen.get("시행일자"),
        "link": link,
        "text": law_text
    }


# =====================================================
# 6. Workflow
# =====================================================
def run(user_input):
    log_box = st.empty()
    logs = []

    def log(msg, cls="agent-log"):
        logs.append(f"<div class='{cls}'>{_escape(msg)}</div>")
        log_box.markdown("".join(logs), unsafe_allow_html=True)
        time.sleep(0.25)

    log("법령 추출 중...", "agent-log log-legal")
    law_raw = llm.text(f"상황에 적용될 대한민국 법령 1개만 '법령명 제N조' 형식으로 출력:\n{user_input}")

    law_info = resolve_law(law_raw)

    log("유사 사례 검색 중...", "agent-log log-search")
    search_txt = google_search(user_input) + "\n\n" + naver_search(user_input)

    log("처리 전략 수립 중...", "agent-log log-strat")
    strategy = llm.text(f"""
민원 상황:
{user_input}

법령:
{law_info.get('text')}

처리 전략을 간결히 작성:
- 처리 방향
- 주의사항
- 예상 반발 대응
""")

    log("공문 작성 중...", "agent-log log-draft")
    doc = llm.json(f"""
민원 상황: {user_input}
법령: {law_info.get('text')}

공문 JSON 작성:
title, receiver, body_paragraphs[], department_head
""")

    log_box.empty()

    return law_info, search_txt, strategy, doc


# =====================================================
# 7. UI
# =====================================================
left, right = st.columns([1, 1.2])

with left:
    st.title("🏢 AI 행정관 Pro")

    user_input = st.text_area("업무 내용", height=160,
                              placeholder="예: 자동차관리법 위반 무단방치 차량 행정처분")

    if st.button("분석 시작", use_container_width=True):
        law, search, strategy, doc = run(user_input)
        st.session_state["res"] = (law, search, strategy, doc)

    if "res" in st.session_state:
        law, search, strategy, doc = st.session_state["res"]

        st.markdown("### 📜 적용 법령")
        st.write(f"**{law.get('name')}** ({law.get('dept')})")
        st.write(law.get("link"))

        st.markdown("### 🔍 유사 사례")
        st.markdown(search)

        st.markdown("### 🧭 처리 전략")
        st.markdown(strategy)

with right:
    if "res" in st.session_state:
        _, _, _, doc = st.session_state["res"]
        if doc:
            st.markdown(f"""
<div class="paper-sheet">
<div class="stamp">직인생략</div>
<div class="doc-header">{doc.get("title","공문")}</div>
<div class="doc-body">
{"".join(f"<p>{_escape(p)}</p>" for p in doc.get("body_paragraphs",[]))}
</div>
<div class="doc-footer">{doc.get("department_head","행정기관장")}</div>
</div>
""", unsafe_allow_html=True)
