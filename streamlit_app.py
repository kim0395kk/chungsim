import streamlit as st
import streamlit.components.v1 as components

import json
import re
import time
from datetime import datetime, timedelta
from html import escape, unescape

import google.generativeai as genai
from groq import Groq

# =========================
# Optional imports (안죽게)
# =========================
try:
    import requests
except Exception:
    requests = None

try:
    import xmltodict
except Exception:
    xmltodict = None

try:
    from supabase import create_client
except Exception:
    create_client = None


# =========================
# 1) Page & Style
# =========================
st.set_page_config(layout="wide", page_title="AI Bureau: Legal Glass (Ops-Final)", page_icon="⚖️")

st.markdown(
    """
<style>
.stApp { background-color: #f3f4f6; }
.paper-sheet {
  background: #fff; width: 100%; max-width: 210mm; min-height: 297mm;
  padding: 25mm; margin: auto; box-shadow: 0 10px 30px rgba(0,0,0,0.1);
  font-family: 'Noto Serif KR','Noto Sans KR','Nanum Gothic','Apple SD Gothic Neo','Malgun Gothic',serif;
  color:#111; line-height:1.7; position:relative;
}
.doc-header { text-align:center; font-size:22pt; font-weight:900; margin-bottom:30px; letter-spacing:2px; }
.doc-info {
  display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap;
  font-size:11pt; border-bottom:2px solid #111; padding-bottom:10px; margin-bottom:20px;
}
.doc-body { font-size:12pt; }
.doc-footer { text-align:center; font-size:20pt; font-weight:bold; margin-top:80px; letter-spacing:5px; }
.stamp {
  position:absolute; bottom:85px; right:80px; border:3px solid #c00; color:#c00;
  padding:5px 10px; font-size:14pt; font-weight:bold; transform:rotate(-15deg);
  opacity:0.85; border-radius:5px;
}
.agent-log { font-family: Consolas, monospace; font-size: 0.85rem; padding: 6px 12px; border-radius: 6px; margin-bottom: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
.log-legal { background-color: #eff6ff; color: #1e40af; border-left: 4px solid #3b82f6; }
.log-search { background-color: #fff7ed; color: #c2410c; border-left: 4px solid #f97316; }
.log-strat { background-color: #f5f3ff; color: #6d28d9; border-left: 4px solid #8b5cf6; }
.log-calc  { background-color: #f0fdf4; color: #166534; border-left: 4px solid #22c55e; }
.log-draft { background-color: #fef2f2; color: #991b1b; border-left: 4px solid #ef4444; }
.log-sys   { background-color: #f3f4f6; color: #4b5563; border-left: 4px solid #9ca3af; }
</style>
""",
    unsafe_allow_html=True,
)

_TAG_RE = re.compile(r"<[^>]+>")

def clean_text(value) -> str:
    if value is None:
        return ""
    s = str(value)
    s = unescape(s)
    s = _TAG_RE.sub("", s)
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", s)
    s = s.replace("\u200b", "")
    s = s.replace("</", "").replace("/>", "").replace("<", "").replace(">", "")
    return s.strip()

def safe_html(value) -> str:
    return escape(clean_text(value), quote=False).replace("\n", "<br>")

def ensure_doc_shape(doc):
    fallback = {
        "title": "공 문 서",
        "receiver": "수신자 참조",
        "body_paragraphs": ["AI 문서 생성에 실패했습니다. (JSON 파싱/모델 응답 오류 가능)"],
        "department_head": "행정기관장",
    }
    if not isinstance(doc, dict):
        return fallback

    title = clean_text(doc.get("title") or fallback["title"])
    receiver = clean_text(doc.get("receiver") or fallback["receiver"])
    head = clean_text(doc.get("department_head") or fallback["department_head"])

    body = doc.get("body_paragraphs", fallback["body_paragraphs"])
    if isinstance(body, str):
        body = [body]
    if not isinstance(body, list) or not body:
        body = fallback["body_paragraphs"]

    cleaned = []
    for p in body:
        p2 = clean_text(p)
        if p2:
            cleaned.append(p2)
    if not cleaned:
        cleaned = fallback["body_paragraphs"]

    # 태그 잔재 필터
    cleaned2 = []
    for p in cleaned:
        low = p.lower()
        if "</" in low or "<div" in low or "class=" in low:
            continue
        cleaned2.append(p)
    if cleaned2:
        cleaned = cleaned2

    return {"title": title, "receiver": receiver, "body_paragraphs": cleaned, "department_head": head}


# =========================
# 2) Metrics (모델/비용/시간)
# =========================
MODEL_PRICES_PER_1M = {
    # 실제 단가(USD/1M tokens)로 바꾸면 됨
    "Gemini / gemini-2.5-flash": 0.0,
    "Gemini / gemini-2.5-flash-lite": 0.0,
    "Gemini / gemini-2.0-flash": 0.0,
    "Groq / llama-3.3-70b-versatile": 0.0,
    "LLM FAILED": 0.0,
}

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / 3.5))

def metrics_init():
    if "metrics" not in st.session_state:
        st.session_state["metrics"] = {
            "runs": 0,
            "calls": {},       # model -> count
            "tokens_est": {},  # model -> tokens
            "cost_est": {},    # model -> cost
            "timing": [],      # list of dict
        }

def metrics_add(model_name: str, prompt: str, output: str):
    m = st.session_state["metrics"]
    m["calls"][model_name] = m["calls"].get(model_name, 0) + 1

    t = estimate_tokens(prompt) + estimate_tokens(output)
    m["tokens_est"][model_name] = m["tokens_est"].get(model_name, 0) + t

    price = MODEL_PRICES_PER_1M.get(model_name, 0.0)
    m["cost_est"][model_name] = m["cost_est"].get(model_name, 0.0) + (t / 1_000_000) * price


# =========================
# 3) LLM Service (모델 기록)
# =========================
class LLMService:
    def __init__(self):
        self.gemini_key = st.secrets.get("general", {}).get("GEMINI_API_KEY")
        self.groq_key = st.secrets.get("general", {}).get("GROQ_API_KEY")

        self.gemini_models = [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
        ]

        self.last_model_used = "N/A"
        self.groq_client = Groq(api_key=self.groq_key) if self.groq_key else None

        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)

    def generate_text(self, prompt: str) -> str:
        last_err = None

        if self.gemini_key:
            for model_name in self.gemini_models:
                try:
                    model = genai.GenerativeModel(model_name)
                    res = model.generate_content(prompt)
                    self.last_model_used = f"Gemini / {model_name}"
                    out = (res.text or "").strip()
                    metrics_add(self.last_model_used, prompt, out)
                    return out
                except Exception as e:
                    last_err = e

        if self.groq_client:
            try:
                completion = self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                self.last_model_used = "Groq / llama-3.3-70b-versatile"
                out = (completion.choices[0].message.content or "").strip()
                metrics_add(self.last_model_used, prompt, out)
                return out
            except Exception as e:
                last_err = e

        self.last_model_used = "LLM FAILED"
        metrics_add(self.last_model_used, prompt, "")
        return f"시스템 오류: AI 모델 연결 실패 ({last_err})"

    def generate_json(self, prompt: str) -> dict:
        raw = self.generate_text(prompt + "\n\n[IMPORTANT] Output ONLY valid JSON. No markdown. No code fences.")
        raw2 = re.sub(r"```(?:json)?\s*|\s*```", "", raw, flags=re.IGNORECASE)
        m = re.search(r"\{.*\}", raw2, re.DOTALL)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}

llm_service = LLMService()


# =========================
# helpers (정규화 + 항 번호 표기)
# =========================
def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", clean_text(s or "")).strip()

def only_digits(s: str) -> str:
    return re.sub(r"[^0-9]", "", clean_text(s or ""))

_CIRCLED = {
    1:"①", 2:"②", 3:"③", 4:"④", 5:"⑤", 6:"⑥", 7:"⑦", 8:"⑧", 9:"⑨", 10:"⑩",
    11:"⑪", 12:"⑫", 13:"⑬", 14:"⑭", 15:"⑮", 16:"⑯", 17:"⑰", 18:"⑱", 19:"⑲", 20:"⑳"
}

def to_circled(n: str) -> str:
    try:
        i = int(re.sub(r"[^0-9]", "", n or ""))
        return _CIRCLED.get(i, f"({i})")
    except Exception:
        return ""

def make_law_query_candidates(hint: dict) -> list:
    # law_name + keywords 기반으로 재탐색 후보 만들기
    law_name = norm_space(hint.get("law_name", ""))
    keywords = hint.get("keywords", []) if isinstance(hint.get("keywords", []), list) else []

    cands = []
    if law_name:
        cands += [law_name]
        cands += [law_name.replace(" ", "")]
        if not law_name.endswith("법"):
            cands += [law_name + "법"]

    for kw in keywords[:5]:
        kw = norm_space(kw)
        if not kw:
            continue
        cands += [kw, kw.replace(" ", "")]
        if not kw.endswith("법") and len(kw) <= 10:
            cands += [kw + "법"]

    # 중복 제거(순서 유지)
    seen = set()
    out = []
    for x in cands:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out[:10]


# =========================
# 4) LAW API Service (운영형 강화)
# =========================
class LawAPIService:
    """
    secrets.toml 예시:
    [law]
    LAW_API_ID="kim03"
    BASE_URL="https://www.law.go.kr/DRF/lawService.do"
    """
    def __init__(self):
        self.enabled = False
        try:
            self.oc = st.secrets["law"]["LAW_API_ID"]
            self.base_url = st.secrets["law"].get("BASE_URL", "https://www.law.go.kr/DRF/lawService.do")
            self.enabled = (requests is not None) and (xmltodict is not None)
        except Exception:
            self.enabled = False

        # 간단 캐시(같은 법령 반복 호출 방지)
        self._law_xml_cache = {}

    def _call_xml(self, params: dict) -> dict:
        if not self.enabled:
            return {}
        r = requests.get(self.base_url, params=params, timeout=15)
        r.raise_for_status()
        return xmltodict.parse(r.text)

    def search_law_candidates(self, query: str, display: int = 20) -> list:
        if not self.enabled or not query:
            return []
        params = {
            "OC": self.oc,
            "target": "law",
            "type": "XML",
            "query": query,
            "display": max(1, min(display, 50)),
        }
        data = self._call_xml(params)
        try:
            law = data.get("LawSearch", {}).get("law")
            if not law:
                return []
            if isinstance(law, dict):
                law = [law]
            out = []
            for item in law:
                if not isinstance(item, dict):
                    continue
                out.append({
                    "law_id": item.get("lawId", ""),
                    "law_name": item.get("lawNm", ""),
                    "law_type": item.get("lawType", ""),
                })
            return out
        except Exception:
            return []

    def choose_best_law(self, candidates: list, query: str) -> dict:
        q = norm_space(query).replace(" ", "")
        if not candidates:
            return {}

        def score(item):
            name = norm_space(item.get("law_name", ""))
            n2 = name.replace(" ", "")
            s = 0
            if not name:
                return -999
            if q and q in n2:
                s += 50
            # 시행령/시행규칙은 본법 대비 살짝 감점(상황 따라 다르니 -2 정도)
            if "시행령" in name:
                s -= 2
            if "시행규칙" in name:
                s -= 2
            s -= max(0, len(name) - 12) * 0.2
            return s

        best = sorted(candidates, key=score, reverse=True)[0]
        return best if best.get("law_id") else {}

    def get_law_xml(self, law_id: str) -> dict:
        if not self.enabled or not law_id:
            return {}
        if law_id in self._law_xml_cache:
            return self._law_xml_cache[law_id]
        params = {"OC": self.oc, "target": "law", "type": "XML", "ID": law_id}
        data = self._call_xml(params)
        self._law_xml_cache[law_id] = data
        return data

    def _as_list(self, x):
        if x is None:
            return []
        if isinstance(x, list):
            return x
        return [x]

    def extract_article_text(self, law_xml: dict, article_no: str) -> str:
        """
        - '제32조', '32조', '제 32 조' 등 입력을 숫자로 정규화
        - ArticleTitle/조문번호(@조문번호)/content/Paragraph(항번호 포함)까지 합쳐서 반환
        """
        if not law_xml or not article_no:
            return ""
        try:
            target_num = only_digits(article_no)  # '제32조' -> '32'
            if not target_num:
                return ""

            articles = law_xml.get("Law", {}).get("Article", [])
            articles = self._as_list(articles)

            for art in articles:
                if not isinstance(art, dict):
                    continue

                curr_art_no = clean_text(art.get("@조문번호", ""))  # 예: 003200 등
                title = clean_text(art.get("ArticleTitle") or art.get("title") or "")
                content = clean_text(art.get("ArticleContent") or art.get("content") or "")

                title_hit = (target_num in only_digits(title)) or (target_num in title)
                no_hit = (target_num in curr_art_no)

                if title_hit or no_hit:
                    paragraphs = self._as_list(art.get("Paragraph"))
                    p_texts = []
                    for p in paragraphs:
                        if not isinstance(p, dict):
                            continue

                        # 항 번호 필드명 변형 방어
                        pno = (
                            clean_text(p.get("ParagraphNumber", "")) or
                            clean_text(p.get("@항번호", "")) or
                            clean_text(p.get("ParagraphNo", "")) or
                            clean_text(p.get("@번호", ""))
                        )

                        ptxt = clean_text(p.get("ParagraphContent", "")) or clean_text(p.get("content", ""))
                        if not ptxt:
                            continue

                        prefix = (to_circled(pno) + " ") if pno else ""
                        p_texts.append(f"{prefix}{ptxt}")

                    joined = "\n".join([x for x in [title, content, "\n".join(p_texts)] if x])
                    joined = clean_text(joined)
                    if joined:
                        return joined
        except Exception:
            pass
        return ""

law_api = LawAPIService()


# =========================
# 5) Search Service (SerpApi requests direct)
# =========================
class SearchService:
    """
    SerpApi를 python 패키지로 import하지 않고,
    requests로 https://serpapi.com/search.json 직접 호출
    """
    def __init__(self):
        self.api_key = st.secrets.get("general", {}).get("SERPAPI_KEY")

    def search_precedents(self, query: str) -> str:
        if not self.api_key:
            return "⚠️ SERPAPI_KEY가 없어 유사 사례 검색을 생략했습니다."
        if requests is None:
            return "⚠️ requests 미설치: 유사 사례 검색을 생략했습니다."

        try:
            url = "https://serpapi.com/search.json"
            search_query = f"{query} 행정처분 판례 사례 민원 답변"
            params = {
                "engine": "google",
                "q": search_query,
                "api_key": self.api_key,
                "num": 3,
                "hl": "ko",
                "gl": "kr",
            }
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()

            results = data.get("organic_results", []) or []
            if not results:
                return "관련된 유사 사례 검색 결과가 없습니다."

            out = []
            for item in results[:3]:
                title = item.get("title", "제목 없음")
                snippet = item.get("snippet", "내용 없음")
                link = item.get("link", "")
                out.append(f"- **{title}**: {snippet}\n  - {link}")

            return "\n".join(out)
        except Exception as e:
            return f"검색 중 오류 발생: {e}"

search_service = SearchService()


# =========================
# 6) DB Service (선택)
# =========================
class DatabaseService:
    def __init__(self):
        self.is_active = False
        self.client = None
        if create_client is None:
            self.is_active = False
            return
        try:
            self.url = st.secrets["supabase"]["SUPABASE_URL"]
            self.key = st.secrets["supabase"]["SUPABASE_KEY"]
            self.client = create_client(self.url, self.key)
            self.is_active = True
        except Exception:
            self.is_active = False

    def save_log(self, user_input, legal_basis, strategy, doc_data, model_usage=None, timing=None):
        if not self.is_active:
            return "DB 미연결 (저장 건너뜀)"
        try:
            data = {
                "input_text": user_input,
                "legal_basis": legal_basis,
                "strategy": strategy,
                "final_doc": json.dumps(doc_data, ensure_ascii=False),
                "model_usage": json.dumps(model_usage or {}, ensure_ascii=False),
                "timing": json.dumps(timing or {}, ensure_ascii=False),
                "created_at": datetime.now().isoformat(),
            }
            self.client.table("law_logs").insert(data).execute()
            return "DB 저장 성공"
        except Exception as e:
            return f"DB 저장 실패: {e}"

db_service = DatabaseService()


# =========================
# 7) Domain Agents
# =========================
class LegalAgents:
    @staticmethod
    def law_hint(situation: str) -> dict:
        prompt = f"""
상황: "{situation}"

너는 행정 실무자가 쓰는 '법령 검색 힌트'만 만든다.
절대 확정/단정하지 말고, 아래 JSON만 출력하라.

규칙:
- law_name: 법령명 1개(추정, 모르면 빈 문자열)
- article_no: 조문번호(예: 제32조) 모르면 빈 문자열
- keywords: 검색 키워드 3~7개(법령명 후보 포함 가능)

{{
  "law_name": "",
  "article_no": "",
  "keywords": []
}}
"""
        obj = llm_service.generate_json(prompt)
        if not isinstance(obj, dict):
            return {"law_name": "", "article_no": "", "keywords": []}
        kws = obj.get("keywords", [])
        if not isinstance(kws, list):
            kws = []
        kws = [norm_space(x) for x in kws if norm_space(x)]
        return {
            "law_name": norm_space(obj.get("law_name", "")),
            "article_no": norm_space(obj.get("article_no", "")),
            "keywords": kws[:7],
        }

    @staticmethod
    def fallback_candidates_only(situation: str, hint: dict) -> dict:
        """
        ✅ API가 실패했을 때 '원문을 쓰지 말고'
        - 후보 법령/조문/검색어만 JSON으로 내게 함
        """
        prompt = f"""
<role>당신은 법령검색 보조자입니다.</role>
<instruction>
- 절대 법령 원문을 지어내지 마시오.
- 확신이 없으면 "알 수 없음" 처리하시오.
- 목적은 API/검색을 위한 '후보'를 만드는 것입니다.

상황: "{situation}"
현재 힌트:
- law_name(추정): "{hint.get('law_name','')}"
- article_no(추정): "{hint.get('article_no','')}"
- keywords: {hint.get('keywords', [])}

아래 JSON만 출력:
{{
  "law_candidates": ["..."],         // 1~5개 (정식명칭 우선)
  "article_candidates": ["..."],     // 0~5개 (예: '제32조', '제33조')
  "search_queries": ["..."]          // 3~7개 (API/웹 검색용 문장)
}}
</instruction>
"""
        obj = llm_service.generate_json(prompt)
        if not isinstance(obj, dict):
            return {"law_candidates": [], "article_candidates": [], "search_queries": []}

        def as_list(x):
            return x if isinstance(x, list) else []

        return {
            "law_candidates": [norm_space(x) for x in as_list(obj.get("law_candidates")) if norm_space(x)][:5],
            "article_candidates": [norm_space(x) for x in as_list(obj.get("article_candidates")) if norm_space(x)][:5],
            "search_queries": [norm_space(x) for x in as_list(obj.get("search_queries")) if norm_space(x)][:7],
        }

    @staticmethod
    def strategist(situation, legal_basis, search_results):
        prompt = f"""
당신은 행정 업무 베테랑 '주무관'입니다.

[민원 상황]: {situation}
[법적 근거]: {legal_basis}
[유사 사례/판례]: {search_results}

위 정보를 종합하여 이 민원을 처리하기 위한 **대략적인 업무 처리 방향(Strategy)**을 수립하세요.
다음 3가지 항목 포함(마크다운):
1. 처리 방향
2. 핵심 주의사항
3. 예상 반발 및 대응
"""
        return llm_service.generate_text(prompt).strip()

    @staticmethod
    def clerk(situation, legal_basis):
        """
        운영 안전:
        - legal_basis가 비어있거나 PENDING일 때는 모델이 흔들릴 수 있어 기본값 15로 수렴되게 설계.
        """
        today = datetime.now()
        prompt = f"""
오늘: {today.strftime('%Y-%m-%d')}
상황: {situation}
법령: {legal_basis}
의견제출/이행 기간(일수)을 숫자만 출력. 모르면 15.
"""
        try:
            res = llm_service.generate_text(prompt)
            days = int(re.sub(r"[^0-9]", "", res)) if res else 15
            if days <= 0:
                days = 15
        except:
            days = 15

        deadline = today + timedelta(days=days)
        return {
            "today_str": today.strftime("%Y. %m. %d."),
            "deadline_str": deadline.strftime("%Y. %m. %d."),
            "days_added": days,
            "doc_num": f"행정-{today.strftime('%Y')}-{int(time.time())%1000:03d}호",
        }

    @staticmethod
    def drafter(situation, legal_basis, meta_info, strategy, legal_status="PENDING"):
        prompt = f"""
당신은 행정기관의 베테랑 서기입니다. 아래 정보를 바탕으로 완결 공문서를 작성하세요.

[입력 정보]
- 민원 상황: {situation}
- 법적 근거(원문 유지): {legal_basis}
- 법적 근거 상태: {legal_status}   # CONFIRMED 또는 PENDING
- 시행 일자: {meta_info['today_str']}
- 기한: {meta_info['deadline_str']} ({meta_info['days_added']}일)

[업무 처리 가이드라인 (전략)]
{strategy}

[필독 지침]
- 법적 근거 상태가 PENDING 이거나, 법적 근거 문자열에 '⚠️' 또는 '원문을 확정하지 못' 문구가 포함되어 있으면:
  1) 공문 본문 [근거] 섹션에는 경고문/후보문을 그대로 복사하지 말 것
  2) 대신 "관련 법령 검토 중" 또는 "OO법 관련 조항 확인 필요"처럼 실무자가 인지할 수 있는 표현으로 작성할 것
  3) 법령 조문 번호/원문을 단정하여 기재하지 말 것

[중요 금지 규칙]
- HTML/태그/마크다운/코드블록 절대 사용 금지
- 본문은 "순수 텍스트 문단"만 작성

[본문 구조]
[경위] -> [근거] -> [처분 내용] -> [권리구제 절차]

[출력 형식: JSON ONLY]
{{
  "title": "공문 제목",
  "receiver": "수신인",
  "body_paragraphs": ["문단1", "문단2", "문단3", "문단4"],
  "department_head": "발신 명의"
}}
"""
        obj = llm_service.generate_json(prompt)
        return ensure_doc_shape(obj)


# =========================
# 8) Workflow (운영용 최종 보정)
# =========================
def run_workflow(user_input: str):
    log_placeholder = st.empty()
    logs = []
    model_usage = {}
    timing = {}

    def add_log(msg, style="sys"):
        style = style if style in ["legal", "search", "strat", "calc", "draft", "sys"] else "sys"
        logs.append(f"<div class='agent-log log-{style}'>{escape(msg)}</div>")
        log_placeholder.markdown("".join(logs), unsafe_allow_html=True)
        time.sleep(0.10)

    def tick():
        return time.perf_counter()

    # Phase 1
    add_log("🔍 Phase 1: 법령(힌트→API 확정) 및 유사 사례 리서치 중...", "legal")

    t0 = tick()
    hint = LegalAgents.law_hint(user_input)
    timing["법령 힌트(ms)"] = int((tick() - t0) * 1000)
    model_usage["법령 힌트"] = llm_service.last_model_used
    add_log(f"🤖 법령 힌트 모델: {llm_service.last_model_used}", "sys")

    legal_basis = ""
    law_debug = {"source": "NONE"}

    # ✅ LAW API로 원문 확정(강화된 재탐색)
    if law_api.enabled:
        add_log("📚 LAW API로 법령/조문 원문 확보 시도...", "legal")
        try:
            t0 = tick()
            candidates = make_law_query_candidates(hint)
            article_no = hint.get("article_no", "")

            best_law = {}
            best_from_query = ""

            for q in candidates:
                law_cands = law_api.search_law_candidates(q, display=20)
                chosen = law_api.choose_best_law(law_cands, q)
                if chosen.get("law_id"):
                    best_law = chosen
                    best_from_query = q
                    break  # 운영: 속도 우선(첫 성공)

            if best_law.get("law_id"):
                law_xml = law_api.get_law_xml(best_law["law_id"])
                article_text = law_api.extract_article_text(law_xml, article_no) if article_no else ""

                timing["LAW API(ms)"] = int((tick() - t0) * 1000)

                if article_no and article_text:
                    legal_basis = f"[{best_law['law_name']} {article_no}]\n\n{article_text}"
                    law_debug = {
                        "source": "LAW_API_SUCCESS",
                        "law_id": best_law.get("law_id"),
                        "law_name": best_law.get("law_name"),
                        "article_no": article_no,
                        "query_used": best_from_query,
                    }
                    add_log("✅ LAW API로 법령 원문 확정 완료", "legal")
                else:
                    legal_basis = (
                        f"⚠️ LAW API로 '법령'은 확인했으나, 조문 원문을 확정하지 못했습니다.\n"
                        f"- 법령명: {best_law.get('law_name','')}\n"
                        f"- 조문: {article_no or '(미지정)'}\n"
                        f"- 조치: 조문번호/검색어를 보정하거나, 조문을 지정해 다시 시도 필요\n"
                    )
                    law_debug = {
                        "source": "LAW_API_PARTIAL",
                        "law_id": best_law.get("law_id"),
                        "law_name": best_law.get("law_name"),
                        "article_no": article_no,
                        "query_used": best_from_query,
                    }
                    add_log("⚠️ 법령명은 확인, 조문 원문 추출 실패(부분 성공)", "legal")
            else:
                timing["LAW API(ms)"] = int((tick() - t0) * 1000)
                add_log("❌ LAW API 검색 실패: 법령 후보를 찾지 못함", "legal")
        except Exception as e:
            add_log(f"❌ LAW API 오류: {e}", "legal")
            law_debug = {"source": "LAW_API_ERROR", "error": str(e)}
    else:
        add_log("⚠️ LAW API OFF (requests/xmltodict/secrets 확인)", "legal")

    # ✅ 완전 실패 시: LLM이 '원문'을 만들지 못하게 차단하고, 후보만 생성
    if (not legal_basis.strip()) or (law_debug.get("source") in ["NONE", "LAW_API_ERROR"]):
        add_log("🧯 법령 원문 미확보: LLM은 '후보/검색어'만 제시하도록 전환", "sys")
        t0 = tick()
        fb = LegalAgents.fallback_candidates_only(user_input, hint)
        timing["법령 후보(LLM)(ms)"] = int((tick() - t0) * 1000)
        model_usage["법령 후보(LLM)"] = llm_service.last_model_used

        legal_basis = (
            "⚠️ 법령 원문을 API로 확정하지 못했습니다. (환각 방지: 원문 생성 금지)\n\n"
            f"- 1차 힌트(law_name/article_no): {hint.get('law_name','') or '(없음)'} / {hint.get('article_no','') or '(없음)'}\n"
            f"- LLM 법령 후보: {', '.join(fb.get('law_candidates', [])) or '(없음)'}\n"
            f"- LLM 조문 후보: {', '.join(fb.get('article_candidates', [])) or '(없음)'}\n"
            "----------------------------------------\n"
            "재검색 쿼리(복붙용):\n"
            + "\n".join([f"- {q}" for q in fb.get("search_queries", [])]) +
            "\n----------------------------------------\n"
            "※ 위 내용은 '검색 후보'이며, 원문 근거는 반드시 API/공식출처로 확인 필요"
        )
        law_debug = {"source": "LLM_CANDIDATES_ONLY"}

    # ✅ 법적근거 상태 플래그(공문에 경고문 그대로 박히는 사고 방지용)
    legal_basis_is_confirmed = (law_debug.get("source") == "LAW_API_SUCCESS") and ("⚠️" not in (legal_basis or ""))
    legal_status_msg = "CONFIRMED" if legal_basis_is_confirmed else "PENDING"

    # Search
    add_log("🌍 유사 사례(SerpApi) 검색 중...", "search")
    t0 = tick()
    search_results = search_service.search_precedents(user_input)
    timing["유사사례 검색(ms)"] = int((tick() - t0) * 1000)

    with st.expander("✅ [검토] 법령 및 유사 사례 확인", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"**적용 법령(원문 유지/후보는 경고 표시)**\n\n{legal_basis}")
            st.caption(f"법령 소스: {law_debug.get('source')} / 상태: {legal_status_msg}")
        with c2:
            st.warning(f"**유사 사례 검색 결과**\n\n{search_results}")

    # Strategy
    add_log("🧠 Phase 2: 업무 처리 방향(전략) 수립 중...", "strat")
    t0 = tick()
    strategy = LegalAgents.strategist(user_input, legal_basis, search_results)
    timing["전략 수립(ms)"] = int((tick() - t0) * 1000)
    model_usage["전략 수립"] = llm_service.last_model_used
    add_log(f"🤖 전략 모델: {llm_service.last_model_used}", "sys")

    with st.expander("🧭 [방향] 업무 처리 가이드라인", expanded=True):
        st.markdown(strategy)

    # Deadline + Draft
    add_log("📅 Phase 3: 기한 산정 및 공문(JSON) 작성 중...", "calc")

    # ✅ 운영 안전: 법적근거가 미확정이면 clerk가 법령에 끌려가지 않게 비워서 기본값(15일)에 수렴
    meta_info = LegalAgents.clerk(user_input, legal_basis if legal_basis_is_confirmed else "")
    add_log(f"⏳ 기한 설정: {meta_info['days_added']}일 후 ({meta_info['deadline_str']})", "calc")

    add_log("✍️ 공문(JSON) 생성 중...", "draft")
    t0 = tick()
    doc_data = LegalAgents.drafter(user_input, legal_basis, meta_info, strategy, legal_status_msg)
    timing["공문 작성(ms)"] = int((tick() - t0) * 1000)
    model_usage["공문 작성"] = llm_service.last_model_used
    add_log(f"🤖 공문 모델: {llm_service.last_model_used}", "sys")

    doc_data = ensure_doc_shape(doc_data)

    # DB Save
    add_log("💾 DB 저장 중(Supabase)...", "sys")
    t0 = tick()
    save_result = db_service.save_log(user_input, legal_basis, strategy, doc_data, model_usage=model_usage, timing=timing)
    timing["DB 저장(ms)"] = int((tick() - t0) * 1000)

    add_log(f"✅ 완료 ({save_result})", "sys")
    time.sleep(0.15)
    log_placeholder.empty()

    # Metrics 누적 (run)
    m = st.session_state["metrics"]
    m["runs"] += 1
    m["timing"].append(timing)

    return doc_data, meta_info, legal_basis, model_usage, timing


# =========================
# 9) UI
# =========================
def render_dashboard():
    st.markdown("## 📊 운영 계기판")
    m = st.session_state["metrics"]
    calls = m.get("calls", {})
    tokens = m.get("tokens_est", {})
    cost = m.get("cost_est", {})
    runs = m.get("runs", 0)

    if not calls:
        st.info("아직 실행 기록이 없습니다.")
        return

    total_calls = sum(calls.values()) or 1

    st.markdown("### ✅ 모델별 호출 횟수/비율")
    for k, v in sorted(calls.items(), key=lambda x: -x[1]):
        st.write(f"- **{k}**: {v}회 ({(v/total_calls)*100:.1f}%)")

    st.markdown("### 💸 모델별 예상 비용(근사)")
    for k in sorted(tokens.keys(), key=lambda x: -tokens[x]):
        st.write(f"- **{k}**: 토큰≈{tokens.get(k,0):,} / 비용≈${cost.get(k,0):.6f}")

    st.markdown("### ⏱️ 최근 1회 단계별 시간(ms)")
    if m.get("timing"):
        last = m["timing"][-1]
        for step, ms in last.items():
            st.write(f"- {step}: {ms} ms")

    st.caption(f"Runs: {runs}, Total model calls: {total_calls}")


def main():
    metrics_init()

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.title("⚖️ AI 행정관 Pro (Ops-Final)")
        st.caption("LAW API + LLM + SerpApi(requests) + DB + Metrics (Hallucination-Guard + Pending-Aware)")
        st.markdown("---")

        user_input = st.text_area(
            "업무 내용",
            height=160,
            placeholder="예시:\n- 아파트 단지 내 소방차 전용구역 불법 주차 차량 과태료 부과 예고 통지서 작성\n- 식품위생법 위반 업소 영업정지 사전통지서 작성\n- 건설기계 주기위반 관련 행정처분 사전통지",
        )

        c1, c2 = st.columns([1, 1])
        run_btn = c1.button("⚡ 실행", type="primary", use_container_width=True)
        clear_btn = c2.button("🧹 초기화", use_container_width=True)

        if clear_btn:
            for k in ["final_doc", "final_meta", "final_legal", "final_models", "final_timing", "metrics"]:
                st.session_state.pop(k, None)
            st.rerun()

        st.markdown("### ⚙️ 상태")
        st.write(f"- LAW API: {'ON' if law_api.enabled else 'OFF'} (requests+xmltodict+secrets 필요)")
        st.write(f"- SerpApi 검색: {'ON' if (requests is not None and st.secrets.get('general', {}).get('SERPAPI_KEY')) else 'OFF'} (requests+SERPAPI_KEY 필요)")
        st.write(f"- DB(Supabase): {'ON' if db_service.is_active else 'OFF'}")

        if run_btn:
            if not user_input.strip():
                st.warning("내용을 입력해주세요.")
            else:
                try:
                    with st.spinner("에이전트 처리 중..."):
                        doc, meta, legal, models, timing = run_workflow(user_input)
                        st.session_state["final_doc"] = doc
                        st.session_state["final_meta"] = meta
                        st.session_state["final_legal"] = legal
                        st.session_state["final_models"] = models
                        st.session_state["final_timing"] = timing
                except Exception as e:
                    st.error(f"시스템 오류 발생: {e}")

        st.markdown("---")
        render_dashboard()

        if "final_models" in st.session_state:
            st.markdown("### 🤖 이번 실행에 사용된 LLM 모델(단계별)")
            for step, model in st.session_state["final_models"].items():
                st.write(f"- **{step}**: `{model}`")

        if "final_timing" in st.session_state:
            st.markdown("### ⏱️ 이번 실행 단계별 응답시간(ms)")
            for step, ms in st.session_state["final_timing"].items():
                st.write(f"- **{step}**: `{ms} ms`")

    with col_right:
        if ("final_doc" in st.session_state) and ("final_meta" in st.session_state) and ("final_legal" in st.session_state):
            doc = ensure_doc_shape(st.session_state["final_doc"])
            meta = st.session_state["final_meta"]
            legal_basis = st.session_state["final_legal"]

            st.subheader("📜 적용 법령(원문 유지/후보는 경고)")
            st.info(legal_basis)

            html_content = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ margin:0; padding:0; background:#f3f4f6; }}
  .paper-sheet {{
    background:#fff; max-width:210mm; min-height:297mm; padding:25mm; margin:0 auto;
    font-family: 'Noto Serif KR','Noto Sans KR','Nanum Gothic','Apple SD Gothic Neo','Malgun Gothic',serif;
    color:#111; line-height:1.7; position:relative;
  }}
  .doc-header {{ text-align:center; font-size:22pt; font-weight:900; margin-bottom:30px; letter-spacing:2px; }}
  .doc-info {{
    display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap;
    font-size:11pt; border-bottom:2px solid #111; padding-bottom:10px; margin-bottom:20px;
  }}
  .doc-body {{ font-size:12pt; }}
  .doc-footer {{ text-align:center; font-size:20pt; font-weight:bold; margin-top:80px; letter-spacing:5px; }}
  .stamp {{
    position:absolute; bottom:85px; right:80px; border:3px solid #c00; color:#c00;
    padding:5px 10px; font-size:14pt; font-weight:bold; transform:rotate(-15deg); opacity:0.85; border-radius:5px;
  }}
  p {{ margin: 0 0 15px 0; }}
</style>
</head>
<body>
  <div class="paper-sheet">
    <div class="stamp">직인생략</div>
    <div class="doc-header">{safe_html(doc.get("title"))}</div>
    <div class="doc-info">
      <span>문서번호: {safe_html(meta.get("doc_num"))}</span>
      <span>시행일자: {safe_html(meta.get("today_str"))}</span>
      <span>수신: {safe_html(doc.get("receiver"))}</span>
    </div>
    <div class="doc-body">
"""
            for p in doc.get("body_paragraphs", []):
                html_content += f"<p>{safe_html(p)}</p>\n"

            html_content += f"""
    </div>
    <div class="doc-footer">{safe_html(doc.get("department_head"))}</div>
  </div>
</body>
</html>
"""
            components.html(html_content, height=1100, scrolling=True)

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
