# streamlit_app.py — AI 행정관 Pro (v7.1)
# Dual-Model Router (FAST: qwen/qwen3-32b, STRICT: llama-3.3-70b-versatile)
# law.go.kr DRF + Naver Search + (Optional) Supabase
#
# 핵심 UX:
# - 법령 후보를 "클릭(선택)" -> 조문 원문(정리본) + law.go.kr 링크 + 사례(네이버) 카드
# - 공문 결과 A4 HTML 미리보기 + HTML 다운로드
# - U+EA01 등 비표시문자(Private Use) 제거로 SyntaxError/렌더링 크래시 방지

import streamlit as st
import streamlit.components.v1 as components
import google.generativeai as genai
import json
import re
import time
from datetime import datetime
from html import escape, unescape
from typing import Any, Dict, List, Optional, Tuple

# -------------------------
# Optional imports
# -------------------------
try:
    from groq import Groq
except Exception:
    Groq = None

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
st.set_page_config(
    layout="wide",
    page_title="AI 행정관 Pro (v7.1)",
    page_icon="⚖️",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.stApp { background-color: #f8f9fa; }

/* A4 문서 스타일 */
.paper-sheet {
  background: #fff; width: 100%; max-width: 210mm; min-height: 297mm;
  padding: 25mm; margin: auto; box-shadow: 0 6px 18px rgba(0,0,0,0.08);
  font-family: 'Noto Serif KR','Nanum Myeongjo',serif;
  color:#111; line-height:1.7; position:relative;
}
.doc-header {
  text-align:center; font-size:24pt; font-weight:900;
  border-bottom:2px solid #111; padding-bottom:10px; margin-bottom:18px;
  letter-spacing:1px;
}
.doc-info {
  font-size:11pt; border-bottom:1px solid #d1d5db;
  padding-bottom:10px; margin-bottom:22px;
}
.doc-info b { color:#111; }
.doc-body { font-size:12pt; text-align: justify; min-height: 430px; }
.doc-footer {
  text-align:center; font-size:20pt; font-weight:900;
  margin-top:90px; border-top:1px solid #111; padding-top:20px;
  letter-spacing:3px;
}
.stamp {
  position:absolute; bottom:90px; right:80px;
  border:3px solid #d32f2f; color:#d32f2f;
  padding:6px 12px; font-size:14pt; font-weight:900;
  transform:rotate(-12deg); opacity:0.85; border-radius:4px;
  font-family: 'Nanum Gothic', sans-serif;
}

/* Agent logs */
.agent-log {
  font-family: 'Pretendard', sans-serif; font-size: 0.92rem;
  padding: 8px 12px; border-radius: 8px; margin-bottom: 6px;
  background: white; border: 1px solid #e5e7eb;
}
.log-legal { border-left: 5px solid #3b82f6; }
.log-search { border-left: 5px solid #f97316; }
.log-strat { border-left: 5px solid #8b5cf6; }
.log-draft { border-left: 5px solid #ef4444; }
.log-sys   { border-left: 5px solid #9ca3af; }

.small-muted { color:#6b7280; font-size:12px; }

/* Evidence cards */
.ev-card{
  background:#fff; border:1px solid #e5e7eb; border-radius:12px;
  padding:12px 14px; margin:10px 0;
}
.ev-title{ font-weight:800; font-size:0.98rem; }
.ev-desc{ color:#374151; margin-top:6px; font-size:0.92rem; }
.ev-meta{ color:#6b7280; margin-top:6px; font-size:0.82rem; }

.badge{
  display:inline-block; padding:2px 8px; border-radius:999px;
  border:1px solid #e5e7eb; background:#fff; font-size:12px;
}
.badge-ok{ border-color:#bbf7d0; background:#f0fdf4; color:#166534; }
.badge-warn{ border-color:#fed7aa; background:#fff7ed; color:#9a3412; }
.badge-bad{ border-color:#fecaca; background:#fff1f2; color:#9f1239; }

</style>
""",
    unsafe_allow_html=True,
)

# =========================
# 2) Sanitizers (U+EA01 & Non-printable)
# =========================
_TAG_RE = re.compile(r"<[^>]+>")
_CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

# Private Use Area 포함 제거(문제의 U+EA01 같은 애들)
_PUA_RE = re.compile(r"[\uE000-\uF8FF]")

# 한자(표시용 제거)
_HANJA_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF]+")

def strip_pua(s: str) -> str:
    if not s:
        return ""
    return _PUA_RE.sub("", s)

def clean_text(value) -> str:
    """HTML 태그/제어문자/PUA 제거"""
    if value is None:
        return ""
    s = str(value)
    s = strip_pua(s)
    s = unescape(s)
    s = _TAG_RE.sub("", s)
    s = _CTRL_RE.sub("", s)
    return s.strip()

def safe_html(value) -> str:
    return escape(clean_text(value), quote=False).replace("\n", "<br>")

def normalize_whitespace(s: str) -> str:
    if not s:
        return ""
    s = strip_pua(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def strip_hanja_for_display(s: str) -> str:
    if not s:
        return ""
    s = strip_pua(s)
    s = _HANJA_RE.sub("", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def truncate_text(s: str, max_chars: int = 2800) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n...(내용 축소됨)"

def safe_json_dump(obj):
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return "{}"

def ensure_doc_shape(doc):
    fallback = {
        "title": "문 서 (생성 실패)",
        "receiver": "수신자 참조",
        "body_paragraphs": ["시스템 오류로 인해 문서 생성에 실패했습니다."],
        "department_head": "행정기관장",
    }
    if not isinstance(doc, dict):
        return fallback

    body = doc.get("body_paragraphs")
    if isinstance(body, str):
        body = [body]
    if not isinstance(body, list) or not body:
        body = fallback["body_paragraphs"]

    out = {
        "title": clean_text(doc.get("title") or fallback["title"]),
        "receiver": clean_text(doc.get("receiver") or fallback["receiver"]),
        "body_paragraphs": [clean_text(x) for x in body if clean_text(x)] or fallback["body_paragraphs"],
        "department_head": clean_text(doc.get("department_head") or fallback["department_head"]),
    }
    return out

def extract_keywords_kor(text: str, max_k: int = 10) -> List[str]:
    if not text:
        return []
    t = re.sub(r"[^가-힣A-Za-z0-9\s]", " ", text)
    words = re.findall(r"[가-힣A-Za-z0-9]{2,14}", t)
    stop = {
        "그리고","관련","문의","사항","대하여","대한","처리","요청","작성","안내","검토",
        "불편","민원","신청","발급","제출","가능","여부","조치","확인","통보","회신","결과","사유"
    }
    out = []
    for w in words:
        if w in stop: 
            continue
        if w.isdigit():
            continue
        if w not in out:
            out.append(w)
        if len(out) >= max_k:
            break
    return out


# =========================
# 3) Session State Init
# =========================
def ss_init():
    defaults = {
        "dept": "OO시청 OO과",
        "officer": "김주무관",
        "user_key": "local_user",

        "metrics": {"calls": {}, "tokens_total": 0},

        "result": None,

        # 클릭 UX용
        "law_candidates": [],
        "selected_candidate_idx": 0,
        "selected_law_pack": None,

        "case_struct": None,
        "strategy_md": "",

        "evidence_items": [],
        "example_items": [],  # (확장용) 향후 판례/사례 스크랩 넣을 자리
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

ss_init()


# =========================
# 4) Metrics
# =========================
def metrics_add(model_name: str, tokens_total: Optional[int] = None):
    m = st.session_state["metrics"]
    m["calls"][model_name] = m["calls"].get(model_name, 0) + 1
    if tokens_total is not None:
        try:
            m["tokens_total"] += int(tokens_total)
        except Exception:
            pass


# =========================
# 5) LLM Service (Dual Router)
# =========================
class LLMService:
    """
    [Model Hierarchy]
    1. Gemini 2.5 Flash
    2. Gemini 2.5 Flash Lite
    3. Gemini 2.0 Flash
    4. Groq (Llama 3 Backup)
    """
    def __init__(self):
        self.gemini_key = st.secrets["general"].get("GEMINI_API_KEY")
        self.groq_key = st.secrets["general"].get("GROQ_API_KEY")
        
        # [선생님 요청사항] 모델 리스트 원상복구 (2.5 포함)
        self.gemini_models = [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash"
        ]
        
        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)
            
        self.groq_client = Groq(api_key=self.groq_key) if self.groq_key else None

    def _try_gemini(self, prompt, is_json=False, schema=None):
        for model_name in self.gemini_models:
            try:
                # 모델 호출 (대소문자 이슈 방지 위해 lower 처리 등은 상황에 맞게)
                model = genai.GenerativeModel(model_name)
                config = genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=schema
                ) if is_json else None
                
                res = model.generate_content(prompt, generation_config=config)
                return res.text, model_name
            except Exception:
                continue # 다음 모델 시도
        raise Exception("All Gemini models failed")

    def generate_text(self, prompt):
        try:
            text, model_used = self._try_gemini(prompt, is_json=False)
            return text
        except Exception:
            if self.groq_client:
                return self._generate_groq(prompt)
            return "시스템 오류: AI 모델 연결 실패"

    def generate_json(self, prompt, schema=None):
        try:
            text, model_used = self._try_gemini(prompt, is_json=True, schema=schema)
            return json.loads(text)
        except Exception:
            # Fallback for Groq or Gemini without JSON mode
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


# =========================
# 6) LAW API (DRF)
# =========================
class LawAPIService:
    def __init__(self):
        self.oc = st.secrets.get("law", {}).get("LAW_API_ID")
        self.search_url = "https://www.law.go.kr/DRF/lawSearch.do"
        self.service_url = "https://www.law.go.kr/DRF/lawService.do"
        self.enabled = bool(requests and xmltodict and self.oc)

    def search_law(self, query: str, display: int = 10) -> List[Dict[str, str]]:
        if not self.enabled or not query:
            return []
        try:
            params = {"OC": self.oc, "target": "law", "type": "XML", "query": query, "display": display, "page": 1}
            r = requests.get(self.search_url, params=params, timeout=7)
            r.raise_for_status()
            data = xmltodict.parse(r.text)
            laws = data.get("LawSearch", {}).get("law", [])
            if isinstance(laws, dict):
                laws = [laws]

            out = []
            for it in laws:
                if not isinstance(it, dict):
                    continue
                out.append(
                    {
                        "lawNm": it.get("법령명한글") or it.get("lawNm") or it.get("법령명") or "",
                        "MST": it.get("법령일련번호") or it.get("MST") or it.get("mst") or "",
                        "link": it.get("법령상세링크") or it.get("link") or "",
                        "amend": it.get("개정일자") or "",
                    }
                )
            return [x for x in out if clean_text(x.get("lawNm")) and clean_text(x.get("MST"))]
        except Exception:
            return []

    def _extract_articles(self, law_obj: dict) -> List[dict]:
        articles = law_obj.get("Article", []) or []
        if isinstance(articles, dict):
            articles = [articles]
        return [a for a in articles if isinstance(a, dict)]

    def get_article_by_mst(self, mst: str, article_no: Optional[str] = None) -> Dict[str, Any]:
        if not self.enabled or not mst:
            return {}
        try:
            params = {"OC": self.oc, "target": "law", "type": "XML", "MST": mst}
            r = requests.get(self.service_url, params=params, timeout=10)
            r.raise_for_status()
            data = xmltodict.parse(r.text)

            law = data.get("Law") or data.get("law") or {}
            law_name = clean_text(law.get("법령명한글") or law.get("LawName") or law.get("법령명") or "")
            articles = self._extract_articles(law)

            index_titles = []
            for a in articles[:120]:
                at = clean_text(a.get("ArticleTitle") or "")
                an = clean_text(a.get("@조문번호") or "")
                if at:
                    index_titles.append(at)
                elif an:
                    index_titles.append(f"제{an}조")

            if not article_no:
                if articles:
                    return self._format_article(law_name, mst, articles[0], index_titles)
                return {"law_name": law_name, "mst": mst, "all_articles_index": index_titles}

            tgt = re.sub(r"[^0-9]", "", str(article_no))
            if not tgt:
                return {"law_name": law_name, "mst": mst, "all_articles_index": index_titles}

            for a in articles:
                an = clean_text(a.get("@조문번호") or "")
                at = clean_text(a.get("ArticleTitle") or "")
                if tgt == re.sub(r"[^0-9]", "", an) or (tgt and f"제{tgt}조" in at):
                    return self._format_article(law_name, mst, a, index_titles)

            return {"law_name": law_name, "mst": mst, "article_no": tgt, "all_articles_index": index_titles}
        except Exception:
            return {}

    def _format_article(self, law_name: str, mst: str, art: dict, idx: List[str]) -> Dict[str, Any]:
        at = clean_text(art.get("ArticleTitle") or "")
        an = clean_text(art.get("@조문번호") or "")
        content = clean_text(art.get("ArticleContent") or "")

        paras = art.get("Paragraph", [])
        if isinstance(paras, dict):
            paras = [paras]
        p_lines = []
        for p in paras:
            if not isinstance(p, dict):
                continue
            pc = clean_text(p.get("ParagraphContent") or "")
            if pc:
                p_lines.append(pc)

        text = "\n".join([x for x in [content] + p_lines if x]).strip()
        text = normalize_whitespace(text)

        # 표시용: 한자 제거
        text_disp = strip_hanja_for_display(text)

        return {
            "law_name": law_name,
            "mst": mst,
            "article_no": re.sub(r"[^0-9]", "", an) or "",
            "article_title": at or (f"제{an}조" if an else ""),
            "article_text": text_disp,
            "all_articles_index": idx,
        }

law_api = LawAPIService()


# =========================
# 7) NAVER Search
# =========================
class NaverSearchService:
    def __init__(self):
        n = st.secrets.get("naver", {})
        self.cid = n.get("CLIENT_ID")
        self.csec = n.get("CLIENT_SECRET")
        self.enabled = bool(requests and self.cid and self.csec)

    def search(self, query: str, cat: str = "news", display: int = 8):
        if not self.enabled or not query:
            return []
        try:
            url = f"https://openapi.naver.com/v1/search/{cat}.json"
            headers = {"X-Naver-Client-Id": self.cid, "X-Naver-Client-Secret": self.csec}
            params = {"query": query, "display": display, "sort": "sim", "start": 1}
            r = requests.get(url, headers=headers, params=params, timeout=7)
            r.raise_for_status()
            return r.json().get("items", []) or []
        except Exception:
            return []

naver = NaverSearchService()


# =========================
# 8) (Optional) Supabase
# =========================
class DatabaseService:
    def __init__(self):
        self.client = None
        s = st.secrets.get("supabase", {})
        url = s.get("SUPABASE_URL")
        key = s.get("SUPABASE_KEY")
        if create_client and url and key:
            try:
                self.client = create_client(url, key)
            except Exception:
                self.client = None

    def enabled(self) -> bool:
        return bool(self.client)

    def save_log(self, data: dict) -> str:
        if not self.client:
            return "DB 미연결"
        try:
            safe_data = json.loads(safe_json_dump(data))
            self.client.table("law_logs").insert(safe_data).execute()
            return "저장 성공"
        except Exception as e:
            return f"저장 실패: {e}"

db = DatabaseService()


# =========================
# 9) Core Agents
# =========================
def intake_schema(user_input: str) -> Dict[str, Any]:
    kw_fallback = extract_keywords_kor(user_input, max_k=10)

    prompt = f"""
다음 민원/업무 지시를 '행정 사실관계' 중심으로 구조화해라.
반드시 아래 JSON 스키마만 출력(키 추가 금지).

{{
  "task_type": "주기위반|무단방치|불법주정차|행정처분|정보공개|기타",
  "facts": {{
    "who": "대상(차량/건설기계/업체/개인 등)",
    "what": "무슨 일이 있었는지(핵심 1~2문장)",
    "where": "장소(모르면 빈문자열)",
    "when": "기간/일시(모르면 빈문자열)",
    "evidence": ["사진","영상","진술","기타(없으면 빈배열)"]
  }},
  "request": {{
    "user_wants": "민원인이 원하는 조치",
    "constraints": "기한/절차/이의제기 등(없으면 빈문자열)"
  }},
  "issues": ["쟁점1","쟁점2"],
  "keywords": ["키워드1","키워드2","키워드3","키워드4"]
}}

입력:
\"\"\"{user_input}\"\"\"

주의:
- 입력에 없는 사실을 만들지 마라. 없으면 '추가 확인 필요'로 표현.
- keywords는 사실 기반 핵심어.
"""
    js = llm.generate_json(prompt, prefer="fast", max_retry=2) or {}
    if not js:
        return {
            "task_type": "기타",
            "facts": {"who": "", "what": user_input[:140], "where": "", "when": "", "evidence": []},
            "request": {"user_wants": "", "constraints": ""},
            "issues": [],
            "keywords": kw_fallback[:4],
            "_input_quality": {"score": 60, "missing_fields": ["where", "when"]},
        }

    if not isinstance(js.get("keywords"), list) or not js["keywords"]:
        js["keywords"] = kw_fallback[:4]
    js["keywords"] = [clean_text(x) for x in js["keywords"] if clean_text(x)]
    if not js["keywords"]:
        js["keywords"] = kw_fallback[:4]

    if not isinstance(js.get("issues"), list):
        js["issues"] = []
    js["issues"] = [clean_text(x) for x in js["issues"] if clean_text(x)]

    facts = js.get("facts") if isinstance(js.get("facts"), dict) else {}
    missing = []
    if not clean_text(facts.get("where")):
        missing.append("where")
    if not clean_text(facts.get("when")):
        missing.append("when")
    score = max(40, 100 - 20 * len(missing))
    js["_input_quality"] = {"score": score, "missing_fields": missing}
    return js


def generate_law_candidates(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    task_type = clean_text(case.get("task_type"))
    facts = case.get("facts", {}) if isinstance(case.get("facts"), dict) else {}
    issues = case.get("issues", [])
    keywords = case.get("keywords", [])

    domain_hint = []
    if task_type == "주기위반":
        domain_hint += ["건설기계관리법", "건설기계관리법 시행령", "도로교통법"]
    elif task_type == "무단방치":
        domain_hint += ["자동차관리법", "도로교통법"]
    elif task_type == "불법주정차":
        domain_hint += ["도로교통법", "주차장법"]

    prompt = f"""
너는 대한민국 행정 실무 기준으로 '법령 후보'를 생성한다.
반드시 아래 JSON만 출력.

{{
  "candidates": [
    {{"law_name":"법령명(공식)","article_hint":"조번호(숫자만, 모르면 빈문자열)","reason":"처분/의무/근거 관점 1줄","confidence":0.0}}
  ]
}}

입력(요약):
- task_type: {task_type}
- what: {facts.get("what","")}
- issues: {issues}
- keywords: {keywords}

규칙:
- 3~6개 후보
- 확신 없으면 confidence 낮게
- article_hint는 추정 가능하면 넣되, 모르면 비워라
"""
    js = llm.generate_json(prompt, prefer="fast", max_retry=2) or {}
    cands = js.get("candidates", []) if isinstance(js.get("candidates"), list) else []

    out = []
    for x in domain_hint:
        out.append({"law_name": x, "article_hint": "", "reason": "도메인 규칙 후보", "confidence": 0.35})

    for c in cands:
        if not isinstance(c, dict):
            continue
        ln = clean_text(c.get("law_name"))
        if not ln:
            continue
        out.append({
            "law_name": ln,
            "article_hint": clean_text(c.get("article_hint") or ""),
            "reason": clean_text(c.get("reason") or ""),
            "confidence": float(c.get("confidence") or 0.0),
        })

    seen = set()
    uniq = []
    for c in out:
        k = c["law_name"]
        if k in seen:
            continue
        seen.add(k)
        uniq.append(c)
        if len(uniq) >= 8:
            break
    return uniq[:8]


def verifier_score(case: Dict[str, Any], article_title: str, article_text: str) -> Dict[str, Any]:
    keywords = case.get("keywords", []) or []
    issues = case.get("issues", []) or []
    facts = case.get("facts", {}) if isinstance(case.get("facts"), dict) else {}
    text = (article_title + "\n" + article_text).lower()

    pool = []
    for w in keywords[:8]:
        w2 = clean_text(w)
        if w2:
            pool.append(w2)
    for w in issues[:6]:
        w2 = clean_text(w)
        if w2:
            pool.append(w2)
    for w in extract_keywords_kor(clean_text(facts.get("what", "")), max_k=6):
        pool.append(w)
    pool = list(dict.fromkeys(pool))[:12]

    hits = sum(1 for w in pool if w and w.lower() in text)
    relevance = min(40, int((hits / max(1, len(pool))) * 40))

    # 권한 밖 단어 감점(대충 방지)
    out_of_scope = ["구속", "수사", "압수", "수색", "체포", "기소", "형사", "구금"]
    o_hits = sum(1 for w in out_of_scope if w in article_text)
    scope_fit = max(0, 25 - min(25, o_hits * 8))

    length_score = 0
    if len(article_text) >= 200:
        length_score = 20
    elif len(article_text) >= 120:
        length_score = 12
    elif len(article_text) >= 80:
        length_score = 6
    else:
        length_score = 0

    risk = 0
    if not article_text or len(article_text) < 80:
        risk += 10
    if "||" in article_text or ">>" in article_text:
        risk += 5
    risk = min(15, risk)

    total = relevance + scope_fit + length_score + (15 - risk)

    if total >= 75:
        verdict = "CONFIRMED"
    elif total >= 50:
        verdict = "WEAK"
    else:
        verdict = "FAIL"

    return {
        "score_total": int(total),
        "verdict": verdict,
        "breakdown": {
            "relevance": int(relevance),
            "scope_fit": int(scope_fit),
            "length_score": int(length_score),
            "risk": int(risk),
        },
        "notes": [f"키워드 매칭 {hits}/{max(1, len(pool))}", f"원문 길이 {len(article_text)}자"],
    }


def draft_strategy(case: Dict[str, Any], law_pack: Dict[str, Any], evidence_text: str) -> str:
    prefer = "strict" if law_pack.get("verdict") != "CONFIRMED" else "fast"
    prompt = f"""
[업무유형] {case.get("task_type")}
[사실 요약]
- who: {case.get("facts",{}).get("who","")}
- what: {case.get("facts",{}).get("what","")}
- where: {case.get("facts",{}).get("where","")}
- when: {case.get("facts",{}).get("when","")}
[요구] {case.get("request",{}).get("user_wants","")}
[쟁점] {case.get("issues",[])}

[법적근거(선택)]
- 법령: {law_pack.get("law_name","")}
- 조문: {law_pack.get("article_title","")}
- 원문(정리): {truncate_text(law_pack.get("article_text",""), 900)}

[사례/참고(네이버)]
{truncate_text(evidence_text, 700)}

아래 형식(마크다운)만 출력:
1) 처리 방향(현실 프로세스 중심, 6~10줄)
2) 체크리스트(불릿 10~14개)
3) 민원인 설명 문장(바로 복붙용 4~6줄)
"""
    return llm.generate_text(prompt, prefer=prefer, temp=0.1)


def draft_document_json(dept: str, officer: str, case: Dict[str, Any], law_pack: Dict[str, Any], strategy_md: str) -> Dict[str, Any]:
    today_str = datetime.now().strftime("%Y. %m. %d.")
    doc_num = f"행정-{datetime.now().strftime('%Y')}-{int(time.time()) % 10000:04d}호"

    prompt = f"""
아래 스키마로만 JSON 출력(키 추가 금지):
{{
  "title": "문서 제목",
  "receiver": "수신",
  "body_paragraphs": ["문단1","문단2","문단3","문단4"],
  "department_head": "발신 명의"
}}

작성 정보:
- 부서: {dept}
- 담당자: {officer}
- 시행일: {today_str}
- 문서번호: {doc_num}

사실관계(확정된 범위):
- who: {case.get("facts",{}).get("who","")}
- what: {case.get("facts",{}).get("what","")}
- where: {case.get("facts",{}).get("where","")}
- when: {case.get("facts",{}).get("when","")}
- 민원요구: {case.get("request",{}).get("user_wants","")}
- 제약/기한: {case.get("request",{}).get("constraints","")}

법적 근거(확보된 범위):
- 법령: {law_pack.get("law_name","")}
- 조문: {law_pack.get("article_title","")}
- 원문(정리): {truncate_text(law_pack.get("article_text",""), 1200)}

작성 원칙:
- 톤: 건조/정중, 단정/추측 금지
- 구조: [경위]→[법적 근거]→[조치/안내]→[권리구제/문의]
- 법령 원문이 약하면 "추가 확인 필요" 포함
"""
    js = llm.generate_json(prompt, prefer="strict", max_retry=3)
    out = ensure_doc_shape(js)
    out["_meta"] = {"doc_num": doc_num, "today": today_str, "dept": dept, "officer": officer}
    return out


def build_a4_html(doc: Dict[str, Any], meta: Dict[str, str]) -> str:
    body_html = "".join(
        [f"<p style='margin:0 0 15px 0; text-indent: 10px;'>{safe_html(p)}</p>" for p in doc.get("body_paragraphs", [])]
    )
    html = f"""
<div class="paper-sheet" id="printable-area">
  <div class="stamp">직인생략</div>
  <div class="doc-header">{safe_html(doc.get('title',''))}</div>
  <div class="doc-info">
    <div><b>문서번호:</b> {safe_html(meta.get('doc_num',''))}</div>
    <div><b>시행일자:</b> {safe_html(meta.get('today',''))}</div>
    <div style="margin-top:6px;"><b>수신:</b> {safe_html(doc.get('receiver',''))}</div>
  </div>
  <div class="doc-body">
    {body_html}
  </div>
  <div class="doc-footer">{safe_html(doc.get('department_head',''))}</div>
  <div style="font-size: 10pt; color: #666; margin-top: 18px;">
    담당자: {safe_html(meta.get('officer',''))} / 부서: {safe_html(meta.get('dept',''))}
  </div>
</div>
"""
    return html


def naver_case_query(case: Dict[str, Any], law_name: str, article_title: str) -> str:
    # "사례"를 더 잘 찾기 위한 쿼리 조합(실무형)
    kw = case.get("keywords", []) or []
    core = " ".join([k for k in kw[:3] if k])
    law = clean_text(law_name)
    art = clean_text(article_title)

    # 너무 길면 잘라
    base = " ".join([x for x in [core, law, art] if x]).strip()
    base = re.sub(r"\s{2,}", " ", base).strip()

    # 행정 실무에서 사례 찾기 단어
    return (base + " 행정처분 처분사례 과태료 통지").strip()


# =========================
# 10) Click-driven Law Pack Loader
# =========================
def load_law_pack_from_candidate(case: Dict[str, Any], cand: Dict[str, Any]) -> Dict[str, Any]:
    """
    후보 1개를 선택했을 때:
    - DRF search -> MST -> article fetch -> verifier 점수
    """
    q = clean_text(cand.get("law_name"))
    art_hint = clean_text(cand.get("article_hint") or "")

    laws = law_api.search_law(q, display=10)
    if not laws:
        return {"law_name": q, "verdict": "FAIL", "score": 0, "article_title": "", "article_text": "", "link": ""}

    chosen = laws[0]
    mst = clean_text(chosen.get("MST"))
    law_name = clean_text(chosen.get("lawNm"))
    link = clean_text(chosen.get("link"))

    pack = law_api.get_article_by_mst(mst, article_no=art_hint if art_hint else None)
    article_title = clean_text(pack.get("article_title", ""))
    article_text = clean_text(pack.get("article_text", ""))  # 이미 한자 제거된 표시용

    if not article_text:
        return {"law_name": law_name, "mst": mst, "link": link, "verdict": "FAIL", "score": 0, "article_title": article_title, "article_text": ""}

    v = verifier_score(case, article_title, article_text)
    return {
        "law_name": law_name,
        "mst": mst,
        "link": link,
        "article_title": article_title,
        "article_text": article_text,
        "verdict": v["verdict"],
        "score": v["score_total"],
        "verify": v,
        "cand": cand,
        "all_articles_index": pack.get("all_articles_index", []),
    }


# =========================
# 11) Main Workflow
# =========================
def run_workflow(user_input: str, dept: str, officer: str, user_key: str):
    log_area = st.empty()
    logs = []

    def add_log(msg: str, style: str = "sys"):
        logs.append(f"<div class='agent-log log-{style}'>{safe_html(msg)}</div>")
        log_area.markdown("".join(logs), unsafe_allow_html=True)
        time.sleep(0.02)

    started = datetime.now().isoformat()

    # 1) INTAKE
    add_log("🧾 [INTAKE] 사실관계 구조화(FAST)…", "sys")
    case = intake_schema(user_input)
    st.session_state["case_struct"] = case
    add_log(f"✅ [INTAKE] 완료 (quality={case.get('_input_quality',{}).get('score','?')})", "sys")

    # 2) 법령 후보 생성
    add_log("🧩 [LAW-CAND] 법령 후보 생성(FAST)…", "legal")
    candidates = generate_law_candidates(case)
    if not candidates:
        kws = case.get("keywords", []) or []
        candidates = [{"law_name": k, "article_hint": "", "reason": "fallback", "confidence": 0.2} for k in kws[:3]]
    st.session_state["law_candidates"] = candidates
    st.session_state["selected_candidate_idx"] = 0
    add_log("📌 후보 준비 완료 (우측에서 클릭/선택 가능)", "legal")

    # 3) 기본 후보 1개 로딩
    add_log("📚 [LAW] 기본 후보 원문 로딩 + 검증…", "legal")
    first_pack = load_law_pack_from_candidate(case, candidates[0])
    st.session_state["selected_law_pack"] = first_pack
    add_log(f"✅ [LAW] 기본 선택: {first_pack.get('law_name','')} / {first_pack.get('article_title','')} ({first_pack.get('verdict')}, score={first_pack.get('score',0)})", "legal")

    # 4) 사례(네이버) 기본 로딩
    add_log("🌍 [CASE] 사례/기사 수집(네이버)…", "search")
    ev_items = []
    evidence_text = ""
    q = naver_case_query(case, first_pack.get("law_name",""), first_pack.get("article_title",""))
    raw_news = naver.search(q, cat="news", display=8) if naver.enabled else []
    raw_web = naver.search(q, cat="webkr", display=8) if naver.enabled else []

    def _push(items, source: str):
        nonlocal evidence_text
        for it in items:
            title = clean_text(it.get("title"))
            desc = clean_text(it.get("description"))
            link = clean_text(it.get("link"))
            # naver 응답에 HTML b 태그 섞이므로 clean_text로 제거됨
            ev_items.append({"title": title, "desc": desc, "link": link, "source": source})
            evidence_text += f"- [{source}] {title}: {desc}\n"

    _push(raw_news, "NEWS")
    _push(raw_web, "WEB")

    st.session_state["evidence_items"] = ev_items
    add_log(f"✅ [CASE] {len(ev_items)}건", "search")

    # 5) 전략 작성
    add_log("🧠 [STRATEGY] 처리 전략 생성…", "strat")
    strategy = draft_strategy(case, first_pack, evidence_text)
    st.session_state["strategy_md"] = strategy

    # 6) 공문 작성(STRICT)
    add_log("✍️ [DRAFT] 공문 JSON 생성(STRICT)…", "draft")
    doc = draft_document_json(dept, officer, case, first_pack, strategy)
    doc_final = ensure_doc_shape(doc)
    meta = doc.get("_meta", {}) if isinstance(doc, dict) else {}
    doc_meta = {
        "doc_num": meta.get("doc_num", ""),
        "today": meta.get("today", ""),
        "dept": meta.get("dept", dept),
        "officer": meta.get("officer", officer),
    }

    # 7) DB 저장(옵션)
    add_log("💾 [SAVE] 로그 저장…", "sys")
    payload = {
        "created_at": started,
        "dept": dept,
        "officer": officer,
        "user_key": user_key,
        "input_text": clean_text(user_input),
        "case_json": safe_json_dump(case),
        "law_pack_json": safe_json_dump(first_pack),
        "strategy_md": strategy,
        "final_doc_json": safe_json_dump(doc_final),
        "evidence_json": safe_json_dump(ev_items),
        "metrics": safe_json_dump(st.session_state.get("metrics", {})),
        "model_last": llm.last_model,
    }
    db_msg = db.save_log(payload) if db.enabled() else "DB 미연결"
    add_log(f"✅ 완료 ({db_msg})", "sys")

    time.sleep(0.25)
    log_area.empty()

    return {
        "case": case,
        "law_pack": first_pack,
        "strategy": strategy,
        "doc": doc_final,
        "doc_meta": doc_meta,
        "evidence_items": ev_items,
        "db_msg": db_msg,
    }


# =========================
# 12) UI Renderers
# =========================
def verdict_badge(verdict: str) -> str:
    v = (verdict or "").upper()
    if v == "CONFIRMED":
        return "<span class='badge badge-ok'>CONFIRMED</span>"
    if v == "WEAK":
        return "<span class='badge badge-warn'>WEAK</span>"
    return "<span class='badge badge-bad'>FAIL</span>"

def render_a4(doc: Dict[str, Any], meta: Dict[str, str]):
    html_content = build_a4_html(doc, meta)
    components.html(html_content, height=980, scrolling=True)

    st.download_button(
        label="📥 공문 HTML로 내보내기",
        data=html_content,
        file_name=f"공문_{meta.get('doc_num','') or 'draft'}.html",
        mime="text/html",
        use_container_width=True
    )

def render_law_panel(case: Dict[str, Any]):
    """
    - 후보 리스트(선택) -> 선택 즉시 원문/검증/사례 재로딩
    """
    candidates = st.session_state.get("law_candidates", []) or []
    if not candidates:
        st.info("법령 후보가 없습니다. 먼저 왼쪽에서 실행하세요.")
        return

    # 후보 표시 문자열
    def fmt(i: int) -> str:
        c = candidates[i]
        ln = clean_text(c.get("law_name"))
        ah = clean_text(c.get("article_hint"))
        rs = clean_text(c.get("reason"))
        cf = c.get("confidence", 0.0)
        tail = f" / 조힌트:{ah}" if ah else ""
        return f"{ln}{tail}  (conf={cf:.2f}) — {rs}"

    idx = st.selectbox(
        "📌 법령 후보 선택(클릭)",
        options=list(range(len(candidates))),
        index=int(st.session_state.get("selected_candidate_idx", 0)),
        format_func=fmt,
    )

    if idx != st.session_state.get("selected_candidate_idx", 0):
        st.session_state["selected_candidate_idx"] = idx

        # 선택한 후보 로딩
        with st.spinner("선택한 후보의 원문/검증/사례를 불러오는 중..."):
            pack = load_law_pack_from_candidate(case, candidates[idx])
            st.session_state["selected_law_pack"] = pack

            # 사례도 법령+조문 기반으로 재검색
            ev_items = []
            q = naver_case_query(case, pack.get("law_name",""), pack.get("article_title",""))
            raw_news = naver.search(q, cat="news", display=8) if naver.enabled else []
            raw_web = naver.search(q, cat="webkr", display=8) if naver.enabled else []

            def _push(items, source: str):
                for it in items:
                    ev_items.append({
                        "title": clean_text(it.get("title")),
                        "desc": clean_text(it.get("description")),
                        "link": clean_text(it.get("link")),
                        "source": source
                    })

            _push(raw_news, "NEWS")
            _push(raw_web, "WEB")
            st.session_state["evidence_items"] = ev_items

    # 현재 선택 pack 표시
    pack = st.session_state.get("selected_law_pack") or {}
    law_name = pack.get("law_name", "")
    article_title = pack.get("article_title", "")
    verdict = pack.get("verdict", "FAIL")
    score = pack.get("score", 0)
    link = pack.get("link", "")

    st.markdown(
        f"""
<div style="padding:10px 12px; background:#fff; border:1px solid #e5e7eb; border-radius:12px;">
  <div style="font-weight:900; font-size:1.02rem;">선택 법령: {escape(clean_text(law_name))}</div>
  <div style="margin-top:6px;">조문: <b>{escape(clean_text(article_title))}</b></div>
  <div style="margin-top:6px;">검증: {verdict_badge(verdict)} &nbsp; <span class="badge">score={int(score)}</span></div>
</div>
""",
        unsafe_allow_html=True
    )

    cols = st.columns([1,1,1])
    with cols[0]:
        if link:
            st.link_button("🔗 law.go.kr 원문 열기", link, use_container_width=True)
        else:
            st.button("🔗 law.go.kr 원문 열기", disabled=True, use_container_width=True)
    with cols[1]:
        st.button("📌 이 법령으로 전략/공문 재생성", disabled=True, use_container_width=True)
        st.caption("※ 버튼은 확장용(현재는 선택 즉시 사례만 재로딩).")
    with cols[2]:
        st.caption("팁: 조문이 틀리면 후보를 바꾸세요.")

    # 원문 표시(정리본)
    st.markdown("### 📜 조문 원문(정리본)")
    txt = normalize_whitespace(pack.get("article_text","") or "")
    txt = strip_hanja_for_display(txt)

    if not txt:
        st.warning("조문 원문이 비어 있습니다. 다른 후보를 선택하세요.")
    else:
        st.code(txt, language="text")

    # verifier details
    v = pack.get("verify")
    if v:
        with st.expander("🧪 Verifier 점수(왜 이 법을 선택/배제했는지)", expanded=False):
            st.json(v)

    # 조문 인덱스(선택 확장용)
    idx_titles = pack.get("all_articles_index", []) or []
    if idx_titles:
        with st.expander("📚 이 법령의 조문 목록(일부)", expanded=False):
            st.write(idx_titles[:80])
            st.caption("※ 향후 '조문 클릭'으로 특정 조문 로딩 기능까지 확장 가능.")


def render_evidence():
    items = st.session_state.get("evidence_items", []) or []
    if not naver.enabled:
        st.warning("네이버 API 미설정: secrets.toml의 [naver] CLIENT_ID/SECRET 필요")
        return
    if not items:
        st.info("사례/기사 결과가 없습니다(키워드 또는 네이버 제한 가능).")
        return

    st.markdown("### 🧾 사례/기사 (클릭해서 원문 확인)")
    for it in items[:16]:
        title = clean_text(it.get("title"))
        desc = clean_text(it.get("desc"))
        link = clean_text(it.get("link"))
        src = clean_text(it.get("source"))
        meta = f"출처: {src}" if src else ""
        if link:
            st.markdown(
                f"""
<div class="ev-card">
  <div class="ev-title"><a href="{escape(link)}" target="_blank">{escape(title)}</a></div>
  <div class="ev-desc">{escape(desc)}</div>
  <div class="ev-meta">{escape(meta)}</div>
</div>
""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
<div class="ev-card">
  <div class="ev-title">{escape(title)}</div>
  <div class="ev-desc">{escape(desc)}</div>
  <div class="ev-meta">{escape(meta)}</div>
</div>
""",
                unsafe_allow_html=True
            )


# =========================
# 13) Main UI
# =========================
def main():
    col_l, col_r = st.columns([1, 1.25], gap="large")

    with col_l:
        st.title("AI 행정관 Pro")
        st.caption("v7.1 — 클릭 UX(법령 원문/사례) + A4 공문 + U+EA01 방어")
        st.markdown("---")

        with st.expander("🧩 부서/담당자 설정", expanded=False):
            st.text_input("부서명", key="dept")
            st.text_input("담당자", key="officer")
            st.text_input("사용자키(로그 구분용)", key="user_key")

        user_input = st.text_area(
            "업무 지시 사항(민원 상황 포함)",
            height=220,
            placeholder="예: 건설기계 차고지 외 장기간 주차(주기위반) 신고. 현장 확인 시 이동. 민원인은 상시 단속 요구. 담당자 권한 내 조치/답변 공문 작성.",
        )

        run_btn = st.button("🚀 실행(구조화→법령후보→원문/검증→사례→공문)", type="primary", use_container_width=True)

        if run_btn:
            if not user_input.strip():
                st.warning("내용을 입력하세요.")
            else:
                with st.spinner("실행 중..."):
                    try:
                        res = run_workflow(
                            clean_text(user_input),
                            st.session_state["dept"],
                            st.session_state["officer"],
                            st.session_state["user_key"],
                        )
                        st.session_state["result"] = res
                    except Exception as e:
                        st.error(f"치명적 오류: {e}")

        # Metrics
        st.markdown("---")
        st.subheader("📊 세션 사용량")
        m = st.session_state.get("metrics", {})
        calls = m.get("calls", {})
        tokens_total = m.get("tokens_total", 0)
        if calls:
            for k, v in sorted(calls.items(), key=lambda x: (-x[1], x[0])):
                st.write(f"- **{k}**: {v}회")
            st.caption(f"총 토큰(가능한 경우): {tokens_total}")
        else:
            st.info("대기 중...")

        st.markdown("<div class='small-muted'>TIP: 담당자는 우측에서 법령 후보를 바꿔가며 원문/사례를 보고 판단하세요.</div>", unsafe_allow_html=True)

    with col_r:
        tabs = st.tabs(["📄 공문(A4)", "⚖️ 법적 근거(클릭)", "🧾 사례(클릭)", "🧠 전략/구조화"])
        res = st.session_state.get("result")

        with tabs[0]:
            if not res:
                st.markdown(
                    """
<div style='text-align:center; padding:120px 20px; color:#9ca3af; border:2px dashed #e5e7eb; border-radius:14px; background:#fff;'>
  <h3 style='margin-bottom:8px;'>📄 A4 미리보기</h3>
  <p>왼쪽에서 민원 상황 입력 후 실행하세요.<br>공문을 A4 형태로 보여주고 HTML로 저장할 수 있습니다.</p>
</div>
""",
                    unsafe_allow_html=True,
                )
            else:
                render_a4(res["doc"], res["doc_meta"])

        with tabs[1]:
            if not res:
                st.info("실행 후 법령 후보를 클릭해서 원문을 보세요.")
            else:
                render_law_panel(res.get("case", {}))

        with tabs[2]:
            if not res:
                st.info("실행 후 사례(기사/웹문서)를 클릭해서 확인하세요.")
            else:
                render_evidence()

        with tabs[3]:
            if not res:
                st.info("실행 후 전략/구조화를 확인하세요.")
            else:
                st.success(f"DB: {res.get('db_msg','')}")
                st.markdown("### 1) 구조화된 사실관계(담당자 검토용)")
                st.json(res.get("case", {}))

                st.markdown("### 2) 처리 전략(복붙 가능)")
                st.markdown(res.get("strategy", ""))

if __name__ == "__main__":
    main()
