import streamlit as st
import google.generativeai as genai
from groq import Groq
from supabase import create_client
import json
import re
import time
import requests
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from html import escape as _escape

# ==========================================
# 1. Configuration & Styles
# ==========================================
st.set_page_config(layout="wide", page_title="AI Bureau: The Legal Glass", page_icon="⚖️")

st.markdown(
    """
<style>
    .stApp { background-color: #f3f4f6; }
    .paper-sheet {
        background-color: white; width: 100%; max-width: 210mm; min-height: 297mm;
        padding: 25mm; margin: auto; box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        font-family: 'Batang', serif; color: #111; line-height: 1.6; position: relative;
    }
    .doc-header { text-align: center; font-size: 22pt; font-weight: 900; margin-bottom: 30px; letter-spacing: 2px; }
    .doc-info { display: flex; justify-content: space-between; font-size: 11pt; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; }
    .doc-body { font-size: 12pt; text-align: justify; white-space: pre-line; }
    .doc-footer { text-align: center; font-size: 20pt; font-weight: bold; margin-top: 80px; letter-spacing: 5px; }
    .stamp { position: absolute; bottom: 85px; right: 80px; border: 3px solid #cc0000; color: #cc0000; padding: 5px 10px; font-size: 14pt; font-weight: bold; transform: rotate(-15deg); opacity: 0.8; border-radius: 5px; }

    .agent-log { font-family: 'Consolas', monospace; font-size: 0.85rem; padding: 6px 12px; border-radius: 6px; margin-bottom: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    .log-legal { background-color: #eff6ff; color: #1e40af; border-left: 4px solid #3b82f6; }
    .log-search { background-color: #fff7ed; color: #c2410c; border-left: 4px solid #f97316; }
    .log-strat { background-color: #f5f3ff; color: #6d28d9; border-left: 4px solid #8b5cf6; }
    .log-calc { background-color: #f0fdf4; color: #166534; border-left: 4px solid #22c55e; }
    .log-draft { background-color: #fef2f2; color: #991b1b; border-left: 4px solid #ef4444; }
    .log-sys { background-color: #f3f4f6; color: #4b5563; border-left: 4px solid #9ca3af; }
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 2. Infrastructure Layer
# ==========================================

class LLMService:
    def __init__(self):
        self.gemini_key = st.secrets["general"].get("GEMINI_API_KEY")
        self.groq_key = st.secrets["general"].get("GROQ_API_KEY")
        
        self.gemini_models = ["gemini-2.0-flash", "gemini-1.5-flash"] # 모델명 최신화

        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)
        self.groq_client = Groq(api_key=self.groq_key) if self.groq_key else None

    def generate_text(self, prompt):
        # Gemini 우선 시도
        for model_name in self.gemini_models:
            try:
                model = genai.GenerativeModel(model_name)
                res = model.generate_content(prompt)
                return res.text
            except Exception:
                continue
        # Groq 백업
        if self.groq_client:
            try:
                completion = self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                return completion.choices[0].message.content
            except: pass
        return "시스템 오류: AI 응답 불가"

    def generate_json(self, prompt, schema=None):
        # JSON 모드 시도
        for model_name in self.gemini_models:
            try:
                model = genai.GenerativeModel(model_name)
                config = genai.GenerationConfig(response_mime_type="application/json", response_schema=schema)
                res = model.generate_content(prompt, generation_config=config)
                return json.loads(res.text)
            except Exception:
                continue
        
        # 텍스트로 받아서 파싱 시도
        text = self.generate_text(prompt + "\n\nOutput strictly in JSON.")
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(match.group(0)) if match else {}
        except:
            return {}

class LawService:
    """국가법령정보센터 Open API"""
    def __init__(self):
        self.user_id = st.secrets["general"].get("LAW_API_ID")
        self.base_url = "http://www.law.go.kr/DRF/lawSearch.do"

    def search_laws(self, keywords, top_k=3):
        if not self.user_id:
            return [{"법령명": "API ID 미설정", "정보": "설정 확인 필요", "링크": "#"}]

        query = urllib.parse.quote(keywords)
        # XML 방식 호출 (target=law)
        url = f"{self.base_url}?OC={self.user_id}&target=law&type=XML&query={query}&display={top_k}"
        
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            
            results = []
            for law in root.findall(".//law"):
                name = law.findtext("lawNm")
                info = law.findtext("lawInfo")
                link = law.findtext("link")
                if link and not link.startswith("http"):
                    link = "https://www.law.go.kr" + link
                
                results.append({"법령명": name, "정보": info, "링크": link})
            
            return results[:top_k] if results else []
        except Exception as e:
            return [{"법령명": "검색 오류", "정보": str(e), "링크": "#"}]

class SearchService:
    """네이버 검색 (키워드 기반 정밀 검색)"""
    def __init__(self):
        g = st.secrets.get("general", {})
        self.client_id = g.get("NAVER_CLIENT_ID")
        self.client_secret = g.get("NAVER_CLIENT_SECRET")
        self.url = "https://openapi.naver.com/v1/search/webkr.json"

    def search_naver(self, keywords, top_k=5):
        if not self.client_id:
            return "API Key 미설정"
        
        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret
        }
        # 검색어에 '행정' 관련 필터 추가하여 정확도 향상
        query = f"{keywords} (과태료 OR 처분 OR 판례 OR 행정심판)"
        params = {"query": query, "display": 10, "start": 1, "sort": "sim"}

        try:
            r = requests.get(self.url, headers=headers, params=params, timeout=5)
            data = r.json()
            items = []
            
            for item in data.get("items", []):
                link = item.get("link", "")
                # 신뢰도 낮은 도메인 1차 필터링
                if any(x in link for x in ["youtube", "cafe.naver", "kin.naver"]):
                    continue
                
                title = re.sub(r"<[^>]+>", "", item.get("title", ""))
                desc = re.sub(r"<[^>]+>", "", item.get("description", ""))
                items.append(f"- **[{title}]({link})**\n  : {desc[:120]}...")
                
                if len(items) >= top_k: break
            
            return "\n".join(items) if items else "관련된 신뢰할 수 있는 결과가 없습니다."
        except Exception as e:
            return f"검색 오류: {e}"

class DatabaseService:
    def __init__(self):
        try:
            self.client = create_client(st.secrets["supabase"]["SUPABASE_URL"], st.secrets["supabase"]["SUPABASE_KEY"])
            self.is_active = True
        except: self.is_active = False

    def save_log(self, situation, law_text, strategy, doc_data):
        if not self.is_active: return "DB 미연결"
        try:
            data = {
                "situation": situation,
                "law_name": law_text[:100], # 길이 제한 고려
                "summary": json.dumps({"strategy": strategy, "doc": doc_data}, ensure_ascii=False)
            }
            self.client.table("law_reports").insert(data).execute()
            return "저장 성공"
        except Exception as e: return f"저장 실패: {e}"

# 인스턴스
llm = LLMService()
law_api = LawService()
search_api = SearchService()
db = DatabaseService()

# ==========================================
# 3. Domain Layer (Agents)
# ==========================================
class LegalAgents:
    @staticmethod
    def analyzer(situation):
        """[Step 1] 사용자 입력을 분석해 최적의 검색 키워드 추출 (핵심!)"""
        prompt = f"""
Role: 행정 법률 분석가
Input: "{situation}"
Task:
1. 국가법령정보센터에서 검색할 '정확한 법령명 키워드' 1개 (예: 소방기본법, 도로교통법)
2. 네이버에서 유사 사례를 찾을 '핵심 검색 키워드' (조사 제거, 명사 위주)

Output JSON: {{ "law_keyword": "...", "search_keyword": "..." }}
"""
        return llm.generate_json(prompt)

    @staticmethod
    def researcher(situation, law_keyword):
        """[Step 2] 법령 API 결과 + LLM 조항 추론"""
        # 1. API로 실제 법령 리스트 확보
        laws_found = law_api.search_laws(law_keyword)
        
        law_list_str = "\n".join([f"- {l['법령명']} ({l['정보']})" for l in laws_found])
        
        # 2. 상황에 맞는 구체적 조항 추론
        prompt = f"""
상황: {situation}
검색된 법령 목록:
{law_list_str}

위 목록 중 이 상황에 가장 적합한 법령을 고르고, 적용될 것으로 예상되는 '조항 번호'와 그 이유를 설명하세요.
(실제 조문 내용은 생략하고, 몇 조가 적용될지 추론하여 작성)

출력 예시:
1. **도로교통법 제32조(정차 및 주차의 금지)**: 소방시설 주변 주정차 금지 조항 적용 예상.
...
"""
        analysis_text = llm.generate_text(prompt)
        return analysis_text, laws_found

    @staticmethod
    def strategist(situation, legal_text, search_text):
        prompt = f"""
당신은 베테랑 공무원입니다.
상황: {situation}
법적 근거: {legal_text}
유사 사례: {search_text}

민원 처리 방향(Strategy)을 수립하세요.
1. 처리 방향 (강경/계도/반려 등)
2. 핵심 주의사항 (절차상 쟁점)
3. 예상 민원 대응 논리
"""
        return llm.generate_text(prompt)

    @staticmethod
    def clerk(situation):
        # 날짜 계산
        today = datetime.now()
        prompt = f"상황: '{situation}'. 행정처분 사전통지나 의견제출 기한으로 적절한 일수(숫자만):"
        try:
            txt = llm.generate_text(prompt)
            days = int(re.sub(r"[^0-9]", "", txt))
        except: days = 15
        
        return {
            "today_str": today.strftime("%Y. %m. %d."),
            "deadline_str": (today + timedelta(days=days)).strftime("%Y. %m. %d."),
            "days_added": days,
            "doc_num": f"행정-{today.strftime('%Y')}-{int(time.time())%1000:03d}호"
        }

    @staticmethod
    def drafter(situation, legal_text, meta, strategy):
        schema = {
            "type": "OBJECT",
            "properties": {
                "title": {"type": "STRING"},
                "receiver": {"type": "STRING"},
                "body_paragraphs": {"type": "ARRAY", "items": {"type": "STRING"}},
                "department_head": {"type": "STRING"},
            },
            "required": ["title", "receiver", "body_paragraphs", "department_head"],
        }
        prompt = f"""
작성자: 베테랑 서기
상황: {situation}
근거: {legal_text}
전략: {strategy}
일자: {meta['today_str']}, 기한: {meta['deadline_str']}

완결된 공문서 작성 (JSON).
"""
        return llm.generate_json(prompt, schema=schema)

# ==========================================
# 4. Workflow & UI
# ==========================================
def run_workflow(user_input):
    log_placeholder = st.empty()
    logs = []
    def log(msg, style="sys"):
        logs.append(f"<div class='agent-log log-{style}'>{_escape(msg)}</div>")
        log_placeholder.markdown("".join(logs), unsafe_allow_html=True)
        time.sleep(0.2)

    # 1. 분석 (키워드 추출)
    log("🧠 상황 분석 및 검색 키워드 추출 중...", "sys")
    keys = LegalAgents.analyzer(user_input)
    law_kw = keys.get("law_keyword", "행정절차법")
    search_kw = keys.get("search_keyword", user_input[:10])
    log(f"🔑 키워드 추출: [법령] {law_kw} / [검색] {search_kw}", "sys")

    # 2. 법령 리서치 (API)
    log(f"📚 국가법령정보센터 조회: '{law_kw}'", "legal")
    legal_text, raw_laws = LegalAgents.researcher(user_input, law_kw)
    
    # 3. 판례/사례 검색 (Naver)
    log(f"🌍 유사 행정 사례 검색: '{search_kw}'", "search")
    search_res = search_api.search_naver(search_kw)

    # 4. 전략 및 기안
    log("🤔 업무 처리 방향 수립 중...", "strat")
    strat = LegalAgents.strategist(user_input, legal_text, search_res)
    
    log("✍️ 공문서 기안 및 기한 산정...", "calc")
    meta = LegalAgents.clerk(user_input)
    doc = LegalAgents.drafter(user_input, legal_text, meta, strat)

    # 5. 저장
    log("💾 시스템 기록 저장 중...", "sys")
    save_msg = db.save_log(user_input, legal_text, strat, doc)
    
    log(f"✅ 처리 완료 ({save_msg})", "sys")
    time.sleep(1)
    log_placeholder.empty()

    return {
        "doc": doc, "meta": meta, "law_txt": legal_text, "raw_laws": raw_laws,
        "search": search_res, "strat": strat, "msg": save_msg
    }

def main():
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.title("🏢 AI 행정관 Pro")
        st.caption("Ver 2.0: Law API Integrated")
        
        user_input = st.text_area("업무 지시", height=150, placeholder="예: 소방차 전용구역 불법주차 과태료 부과 통지서 작성해줘")
        
        if st.button("⚡ 분석 및 기안 시작", type="primary", use_container_width=True):
            if user_input:
                with st.spinner("AI 주무관들이 업무를 처리하고 있습니다..."):
                    st.session_state["result"] = run_workflow(user_input)
            else:
                st.warning("내용을 입력하세요.")

        if "result" in st.session_state:
            res = st.session_state["result"]
            st.markdown("---")
            if "성공" in res["msg"]: st.success(f"✅ {res['msg']}")
            else: st.error(f"❌ {res['msg']}")

            with st.expander("🔎 법적 근거 및 유사 사례 (검토)", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("### 📜 법령 분석")
                    st.info(res["law_txt"])
                    st.caption("▼ 국가법령정보센터 원문 링크")
                    for l in res["raw_laws"]:
                        st.markdown(f"- [{l['법령명']}]({l['링크']})")
                with c2:
                    st.markdown("### 🟩 유사 사례")
                    st.markdown(res["search"])
            
            with st.expander("🧭 업무 처리 전략"):
                st.markdown(res["strat"])

    with col_right:
        if "result" in st.session_state:
            res = st.session_state["result"]
            doc = res["doc"]
            meta = res["meta"]
            
            if doc:
                html = f"""
<div class="paper-sheet">
  <div class="stamp">직인생략</div>
  <div class="doc-header">{_escape(doc.get('title','공문서'))}</div>
  <div class="doc-info">
    <span>문서번호: {_escape(meta['doc_num'])}</span>
    <span>시행일자: {_escape(meta['today_str'])}</span>
    <span>수신: {_escape(doc.get('receiver',''))}</span>
  </div>
  <hr style="border:1px solid black; margin-bottom:30px;">
  <div class="doc-body">
"""
                paragraphs = doc.get("body_paragraphs", [])
                if isinstance(paragraphs, str): paragraphs = [paragraphs]
                for p in paragraphs:
                    html += f"<p style='margin-bottom:15px;'>{_escape(p)}</p>"
                
                html += f"""
  </div>
  <div class="doc-footer">{_escape(doc.get('department_head',''))}</div>
</div>
"""
                st.markdown(html, unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align:center; padding:100px; color:#aaa; border:2px dashed #ddd;'>📄 문서 미리보기</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
