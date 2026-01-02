# app.py — AI 행정관 Pro (Stable / Dual-Model Router v7)
# Groq: qwen/qwen3-32b (FAST) + llama-3.3-70b-versatile (STRICT)
# LAWGO(DRF) + NAVER + Supabase(옵션) + "판단 UI(클릭형 원문/사례)"
#
# ✅ 핵심 UX: 담당자 판단용 브라우저
# - 법령 후보 리스트(3~8) -> [원문 보기] 클릭 -> 조문 이동(select) -> 원문 전문(expander)
# - [유사사례] 클릭 -> 웹문서/뉴스 리스트 링크로 확인
# - 공문은 A4 HTML로 렌더링
#
# ✅ 정확도 개선:
# 1) Intake(사실/요구/대상/시간/장소/증거) 구조화
# 2) 법령 후보 다중 생성 + DRF 원문 확보 + Verifier 점수(참고용)
# 3) 최종 "자동 선택"은 하되, UI에서 후보를 다 보여줘 담당자가 클릭으로 확정
#
# ⚠️ 복붙 주의: Private Use Character(U+E000대) 섞이면 에러날 수 있음.
# - 메모장(plain text)에서 저장 권장.

import streamlit as st
import streamlit.components.v1 as components

import json
import re
import time
from datetime import datetime
from html import escape, unescape
from typing import Any, Dict, List, Optional, Tuple

# =========================
# 0) Optional Imports (Safety)
# =========================
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
    page_title="AI 행정관 Pro (Dual v7.0)",
    page_icon="⚖️",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.stApp { background-color: #f8f9fa; }

.paper-sheet {
  background: #fff; width: 100%; max-width: 210mm; min-height: 297mm;
  padding: 25mm; margin: auto; box-shadow: 0 6px 18px rgba(0,0,0,0.08);
  font-family: 'Noto Serif KR','Nanum Myeongjo',serif;
  color:#111; line-height:1.65; position:relative;
}
.doc-header { text-align:center; font-size:24pt; font-weight:800; margin-bottom:30px; letter-spacing:1px; }
.doc-info {
  display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap;
  font-size:11pt; border-bottom:2px solid #111; padding-bottom:12px; margin-bottom:20px;
}
.doc-body { font-size:12pt; text-align: justify; }
.doc-footer { text-align:center; font-size:20pt; font-weight:800; margin-top:80px; letter-spacing:3px; }
.stamp {
  position:absolute; bottom:85px; right:80px; border:3px solid #d32f2f; color: #d32f2f;
  padding:6px 12px; font-size:14pt; font-weight:800; transform:rotate(-15deg);
  opacity:0.85; border-radius:4px; font-family: 'Nanum Gothic', sans-serif;
}

/* Agent logs */
.agent-log {
  font-family: 'Pretendard', sans-serif; font-size: 0.92rem; padding: 8px 12px;
  border-radius: 8px; margin-bottom: 6px; background: white; border: 1px solid #e5e7eb;
}
.log-legal { border-left: 5px solid #3b82f6; }
.log-search { border-left: 5px solid #f97316; }
.log-strat { border-left: 5px solid #8b5cf6; }
.log-draft { border-left: 5px solid #ef4444; }
.log-sys   { border-left: 5px solid #9ca3af; }

.small-muted { color:#6b7280; font-size:12px; }

/* Evidence card */
.ev-card{
  background:#fff; border:1px solid #e5e7eb; border-radius:10px;
  padding:10px 12px; margin:8px 0;
}
.ev-title{ font-weight:700; }
.ev-desc{ color:#374151; margin-top:4px; }

/* Candidate row */
.cand-row{
  background:#fff; border:1px solid #e5e7eb; border-radius:12px;
  padding:10px 12px; margin:10px 0;
}
.cand-sub{ color:#6b7280; font-size:12px; margin-top:4px; }
</style>
""",
    unsafe_allow_html=True,
)

_TAG_RE = re.compile(r"<[^>]+>")
_CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
# 표시용 한자 제거(원문 표시 UX 개선용)
_HANJA_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF]+")


# =========================
# 2) Helpers
# =========================
def clean_text(value) -> str:
    if value is None:
        return ""
    s = str(value)
    s = unescape(s)
    s = _TAG_RE.sub("", s)
    s = _CTRL_RE.sub("", s)
    return s.strip()


def safe_html(value) -> str:
    return escape(clean_text(value), quote=False).replace("\n", "<br>")


def truncate_text(s: str, max_chars: int = 2800) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n...(내용 축소됨)"


def strip_hanja_for_display(s: str) -> str:
    if not s:
        return ""
    s = _HANJA_RE.sub("", s)
    s = re.sub(r"\|\>+", "", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


def normalize_whitespace(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


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

    return {
        "title": clean_text(doc.get("title") or fallback["title"]),
        "receiver": clean_text(doc.get("receiver") or fallback["receiver"]),
        "body_paragraphs": [clean_text(x) for x in body if clean_text(x)] or fallback["body_paragraphs"],
        "department_head": clean_text(doc.get("department_head") or fallback["department_head"]),
    }


def safe_json_dump(obj):
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return "{}"


def extract_keywords_kor(text: str, max_k: int = 8) -> List[str]:
    if not text:
        return []
    t = re.sub(r"[^가-힣A-Za-z0-9\s]", " ", text)
    words = re.findall(r"[가-힣A-Za-z0-9]{2,14}", t)
    stop = {
        "그리고", "관련", "문의", "사항", "대하여", "대한", "처리", "요청",
        "작성", "안내", "검토", "불편", "민원", "신청", "발급", "제출",
        "가능", "여부", "조치", "확인", "통보", "회신", "결과", "사유"
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
# 3) Metrics
# =========================
def metrics_init():
    if "metrics" not in st.session_state:
        st.session_state["metrics"] = {"calls": {}, "tokens_total": 0}


def metrics_add(model_name: str, tokens_total: Optional[int] = None):
    metrics_init()
    m = st.session_state["metrics"]
    m["calls"][model_name] = m["calls"].get(model_name, 0) + 1
    if tokens_total is not None:
        try:
            m["tokens_total"] += int(tokens_total)
        except Exception:
            pass


metrics_init()


# =========================
# 4) LLM Service (Dual Router)
# =========================
class LLMService:
    def __init__(self):
        g = st.secrets.get("general", {})
        self.groq_key = g.get("GROQ_API_KEY")
        self.model_fast = g.get("GROQ_MODEL_FAST", "qwen/qwen3-32b")
        self.model_strict = g.get("GROQ_MODEL_STRICT", "llama-3.3-70b-versatile")
        self.client = None
        self.last_model = "N/A"

        if Groq and self.groq_key:
            try:
                self.client = Groq(api_key=self.groq_key)
            except Exception:
                self.client = None

    def _chat(self, model: str, messages, temp: float, json_mode: bool):
        if not self.client:
            raise RuntimeError("Groq client not ready (missing key/lib).")

        kwargs = {"model": model, "messages": messages, "temperature": temp}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self.client.chat.completions.create(**kwargs)
        self.last_model = model

        tokens_total = None
        try:
            usage = getattr(resp, "usage", None)
            if usage:
                tokens_total = getattr(usage, "total_tokens", None)
        except Exception:
            tokens_total = None

        metrics_add(model, tokens_total=tokens_total)
        return resp.choices[0].message.content or ""

    def _parse_json(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            cleaned = re.sub(r"```json|```", "", text).strip()
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return {}
            return {}

    def generate_text(self, prompt: str, prefer: str = "fast", temp: float = 0.1) -> str:
        if not self.client:
            return "Groq API Key가 없거나 라이브러리 미설치"

        model_first = self.model_fast if prefer == "fast" else self.model_strict
        messages = [
            {"role": "system", "content": "You are a Korean public-administration assistant. Be factual, structured, and practical."},
            {"role": "user", "content": prompt},
        ]

        try:
            return self._chat(model_first, messages, temp, json_mode=False)
        except Exception:
            if prefer == "fast":
                try:
                    return self._chat(self.model_strict, messages, temp, json_mode=False)
                except Exception as e2:
                    return f"LLM Error: {e2}"
            return "LLM Error"

    def generate_json(self, prompt: str, prefer: str = "fast", temp: float = 0.1, max_retry: int = 2) -> Dict[str, Any]:
        if not self.client:
            return {}

        sys_json = "Output JSON only. No markdown. No explanation. Follow the schema exactly."
        messages = [
            {"role": "system", "content": sys_json},
            {"role": "user", "content": prompt},
        ]
        model_first = self.model_fast if prefer == "fast" else self.model_strict

        for _ in range(max_retry):
            try:
                txt = self._chat(model_first, messages, temp, json_mode=True)
                js = self._parse_json(txt)
                if js:
                    return js
            except Exception:
                pass

        try:
            txt = self._chat(self.model_strict, messages, temp, json_mode=True)
            js = self._parse_json(txt)
            return js if js else {}
        except Exception:
            return {}


llm = LLMService()


# =========================
# 5) LAW API (DRF) — Search + Service (XML)
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
                        "promulgation": it.get("공포일자") or "",
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

            # 조문 인덱스(UI용)
            idx = []
            for a in articles[:120]:
                at = clean_text(a.get("ArticleTitle") or "")
                an = clean_text(a.get("@조문번호") or "")
                if at:
                    idx.append(at)
                elif an:
                    idx.append(f"제{an}조")

            # article_no 없으면 1조라도 반환(인덱스+샘플)
            if not article_no:
                if articles:
                    return self._format_article(law_name, mst, articles[0], idx)
                return {"law_name": law_name, "mst": mst, "all_articles_index": idx}

            tgt = re.sub(r"[^0-9]", "", str(article_no))
            if not tgt:
                return {"law_name": law_name, "mst": mst, "all_articles_index": idx}

            # 조문 매칭
            for a in articles:
                an = clean_text(a.get("@조문번호") or "")
                at = clean_text(a.get("ArticleTitle") or "")
                if tgt == re.sub(r"[^0-9]", "", an) or (tgt and f"제{tgt}조" in at):
                    return self._format_article(law_name, mst, a, idx)

            return {"law_name": law_name, "mst": mst, "article_no": tgt, "all_articles_index": idx}

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
# 6) NAVER Search
# =========================
class NaverSearchService:
    def __init__(self):
        n = st.secrets.get("naver", {})
        self.cid = n.get("CLIENT_ID")
        self.csec = n.get("CLIENT_SECRET")
        self.enabled = bool(requests and self.cid and self.csec)

    def search(self, query: str, cat: str = "webkr", display: int = 8):
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
# 7) Supabase (옵션)
# =========================
class DatabaseService:
    def __init__(self):
        self.client = None
        s = st.secrets.get("supabase", {})
        self.url = s.get("SUPABASE_URL")
        self.key = s.get("SUPABASE_KEY")
        if create_client and self.url and self.key:
            try:
                self.client = create_client(self.url, self.key)
            except Exception:
                self.client = None

    def enabled(self) -> bool:
        return bool(self.client)

    def insert_run(self, row: dict) -> Tuple[bool, str, Optional[str]]:
        if not self.client:
            return False, "DB 미연결", None
        try:
            safe_row = json.loads(safe_json_dump(row))
            resp = self.client.table("runs").insert(safe_row).execute()
            run_id = None
            try:
                data = getattr(resp, "data", None)
                if data and isinstance(data, list) and data:
                    run_id = data[0].get("run_id") or data[0].get("id")
            except Exception:
                run_id = None
            return True, "저장 성공", run_id
        except Exception as e:
            return False, f"저장 실패: {e}", None


db = DatabaseService()


# =========================
# 8) Core Logic
# =========================
def intake_schema(user_input: str) -> Dict[str, Any]:
    kw_fallback = extract_keywords_kor(user_input, max_k=10)

    prompt = f"""
다음 민원/업무 지시를 "행정 사실관계" 중심으로 구조화해라.
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
- 입력에 없는 사실은 "추가 확인 필요"로 처리.
- 장소/시간이 없으면 빈문자열.
- keywords는 사실 기반 핵심어로.
"""
    js = llm.generate_json(prompt, prefer="fast", max_retry=2) or {}

    if not js:
        js = {
            "task_type": "기타",
            "facts": {"who": "", "what": user_input[:120], "where": "", "when": "", "evidence": []},
            "request": {"user_wants": "", "constraints": ""},
            "issues": [],
            "keywords": kw_fallback[:4],
        }

    if not isinstance(js.get("keywords"), list) or not js["keywords"]:
        js["keywords"] = kw_fallback[:4]
    js["keywords"] = [clean_text(x) for x in js["keywords"] if clean_text(x)]
    if not js["keywords"]:
        js["keywords"] = kw_fallback[:4]

    if not isinstance(js.get("issues"), list):
        js["issues"] = []
    js["issues"] = [clean_text(x) for x in js["issues"] if clean_text(x)]

    missing = []
    facts = js.get("facts") if isinstance(js.get("facts"), dict) else {}
    if not clean_text(facts.get("where")):
        missing.append("where")
    if not clean_text(facts.get("when")):
        missing.append("when")
    score = 100 - 20 * len(missing)
    js["_input_quality"] = {"score": max(score, 40), "missing_fields": missing}
    return js


def generate_law_candidates(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    task_type = clean_text(case.get("task_type"))
    facts = case.get("facts") if isinstance(case.get("facts"), dict) else {}
    issues = case.get("issues", [])
    keywords = case.get("keywords", [])

    domain_hint = []
    if task_type == "주기위반":
        domain_hint += ["건설기계관리법", "건설기계관리법 시행령", "도로교통법"]
    if task_type == "무단방치":
        domain_hint += ["자동차관리법", "도로교통법"]
    if task_type == "불법주정차":
        domain_hint += ["도로교통법", "주차장법"]

    prompt = f"""
너는 '법령 후보 생성기'다. 반드시 아래 JSON만 출력.

{{
  "candidates": [
    {{"law_name":"법령명","article_hint":"조번호(숫자만, 모르면 빈문자열)","reason":"짧게","confidence":0.0}}
  ]
}}

입력(사실요약):
- task_type: {task_type}
- who: {facts.get("who","")}
- what: {facts.get("what","")}
- where: {facts.get("where","")}
- when: {facts.get("when","")}
- issues: {issues}
- keywords: {keywords}

규칙:
- candidates는 3~6개
- law_name은 공식 법령명 우선
- article_hint는 모르면 빈문자열
- 담당자가 "클릭으로 원문 확인"할 수 있게 넓게 뽑되 엉뚱한 분야는 제외
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

    # 중복 제거
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


def verifier_score(case: Dict[str, Any], law_name: str, article_title: str, article_text: str) -> Dict[str, Any]:
    keywords = case.get("keywords", [])
    issues = case.get("issues", [])
    facts = case.get("facts", {}) if isinstance(case.get("facts"), dict) else {}
    text = (article_title + "\n" + article_text).lower()

    hits = 0
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

    for w in pool:
        if w and w.lower() in text:
            hits += 1
    relevance = min(35, int((hits / max(1, len(pool))) * 35))

    out_of_scope = ["구속", "수사", "압수", "수색", "체포", "기소", "형사", "구금"]
    o_hits = sum(1 for w in out_of_scope if w in article_text)
    scope_fit = 25 - min(25, o_hits * 8)
    scope_fit = max(0, scope_fit)

    match = 10
    if len(article_text) >= 200:
        match += 10
    if any(k.lower() in (article_title.lower() if article_title else "") for k in keywords[:4] if k):
        match += 5
    article_match = min(25, match)

    risk = 0
    if not article_text or len(article_text) < 80:
        risk += 10
    if "||" in article_text or ">>" in article_text:
        risk += 5
    risk = min(15, risk)

    total = relevance + scope_fit + article_match + (15 - risk)
    verdict = "CONFIRMED" if total >= 75 else ("WEAK" if total >= 50 else "FAIL")

    return {
        "score_total": int(total),
        "score_breakdown": {
            "relevance": int(relevance),
            "scope_fit": int(scope_fit),
            "article_match": int(article_match),
            "hallucination_risk": int(risk),
        },
        "verdict": verdict,
        "reasons": [
            f"키워드 매칭 {hits}/{max(1,len(pool))}",
            f"원문 길이 {len(article_text)}자",
        ],
    }


def draft_strategy(case: Dict[str, Any], law_pack: Dict[str, Any], evidence_text: str) -> str:
    prefer = "strict" if law_pack.get("verdict") != "CONFIRMED" else "fast"
    prompt = f"""
[업무유형] {case.get("task_type")}
[사실(요약)]
- who: {case.get("facts",{}).get("who","")}
- what: {case.get("facts",{}).get("what","")}
- where: {case.get("facts",{}).get("where","")}
- when: {case.get("facts",{}).get("when","")}
[민원 요구] {case.get("request",{}).get("user_wants","")}
[쟁점] {case.get("issues",[])}

[법적근거(참고)]
- 법령: {law_pack.get("law_name","")}
- 조문: {law_pack.get("article_title","")}
- 원문(요약): {truncate_text(law_pack.get("article_text",""), 900)}

[참고(네이버)]
{truncate_text(evidence_text, 700)}

아래 형식(마크다운)만 출력:
1) 처리 방향(현실적인 행정 프로세스 중심, 5~8줄)
2) 체크리스트(불릿 8~12개, 확인/기록/통지/기한 포함)
3) 민원인 설명 포인트(오해 줄이는 문장 3~5개)
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

법적 근거(참고/확보된 범위):
- 법령: {law_pack.get("law_name","")}
- 조문: {law_pack.get("article_title","")}
- 원문: {truncate_text(law_pack.get("article_text",""), 1200)}

처리 전략(요약):
{truncate_text(strategy_md, 900)}

작성 원칙:
- 문서 톤: 건조/정중, 추측 금지
- 구조: [경위]→[법적 근거]→[조치/안내]→[권리구제/문의]
- 개인정보는 OOO로 마스킹
- 법령 원문이 약하면 "추가 확인 필요" 문구 포함
"""
    js = llm.generate_json(prompt, prefer="strict", max_retry=3)
    out = ensure_doc_shape(js)
    out["_meta"] = {"doc_num": doc_num, "today": today_str}
    return out


# =========================
# 9) Workflow
# =========================
def run_workflow(user_input: str, dept: str, officer: str, user_key: str):
    log_area = st.empty()
    logs = []

    def add_log(msg: str, style: str = "sys"):
        logs.append(f"<div class='agent-log log-{style}'>{safe_html(msg)}</div>")
        log_area.markdown("".join(logs), unsafe_allow_html=True)
        time.sleep(0.03)

    started = datetime.now().isoformat()

    add_log("🧾 [INTAKE] 사실관계 중심 구조화… (FAST)", "sys")
    case = intake_schema(user_input)
    add_log(f"✅ [INTAKE] 완료 (quality={case.get('_input_quality',{}).get('score','?')})", "sys")

    add_log("🧩 [LAW-CAND] 법령 후보 생성… (FAST)", "legal")
    candidates = generate_law_candidates(case)
    if not candidates:
        candidates = [{"law_name": k, "article_hint": "", "reason": "fallback", "confidence": 0.2} for k in case.get("keywords", [])[:3]]
    add_log("📌 후보: " + ", ".join([c['law_name'] for c in candidates[:6]]), "legal")

    add_log("📚 [LAW] DRF 원문 확보 + Verifier(참고용) 점수화…", "legal")
    best_pack = {
        "law_name": "",
        "mst": "",
        "link": "",
        "article_title": "",
        "article_text": "",
        "verdict": "FAIL",
        "score": 0,
        "verify": {},
    }
    loop_debug = []
    for i, cand in enumerate(candidates[:6], start=1):
        q = cand.get("law_name", "")
        art_hint = cand.get("article_hint", "")
        add_log(f"  - ({i}) {q} 검색 → 원문 확인", "legal")

        laws = law_api.search_law(q, display=10)
        if not laws:
            loop_debug.append({"cand": cand, "search": "no_result"})
            continue

        chosen = laws[0]
        mst = clean_text(chosen.get("MST"))
        law_name = clean_text(chosen.get("lawNm"))
        link = clean_text(chosen.get("link"))

        pack = law_api.get_article_by_mst(mst, article_no=art_hint if art_hint else None)
        article_title = clean_text(pack.get("article_title", ""))
        article_text = clean_text(pack.get("article_text", ""))
        if not article_text:
            loop_debug.append({"cand": cand, "mst": mst, "fetch": "empty"})
            continue

        v = verifier_score(case, law_name, article_title, article_text)
        score = v["score_total"]
        verdict = v["verdict"]

        loop_debug.append({
            "cand": cand,
            "selected": {"law_name": law_name, "mst": mst, "link": link, "article_title": article_title},
            "verify": v
        })

        if score > best_pack["score"]:
            best_pack = {
                "law_name": law_name,
                "mst": mst,
                "link": link,
                "article_title": article_title,
                "article_text": article_text,
                "verdict": verdict,
                "score": score,
                "verify": v,
            }

        if verdict == "CONFIRMED":
            break

    add_log(f"✅ [LAW] 자동선택(참고): {best_pack.get('law_name','(없음)')} / {best_pack.get('article_title','')} (score={best_pack.get('score',0)}, {best_pack.get('verdict')})", "legal")

    add_log("🌍 [EVIDENCE] 네이버 유사사례(선택) 수집…", "search")
    ev_items = []
    ev_text = ""
    kw = case.get("keywords", [])
    if kw:
        q = " ".join(kw[:2]) + " 행정처분"
        raw = naver.search(q, cat="webkr", display=8)
        for item in raw:
            title = clean_text(item.get("title"))
            desc = clean_text(item.get("description"))
            link = clean_text(item.get("link"))
            ev_items.append({"title": title, "desc": desc, "link": link})
            ev_text += f"- {title}: {desc}\n"
    add_log(f"✅ [EVIDENCE] {len(ev_items)}건", "search")

    add_log("🧠 [STRATEGY] 처리 전략… (FAST/STRICT)", "strat")
    strategy = draft_strategy(case, best_pack, ev_text)

    add_log("✍️ [DRAFT] 공문 JSON 생성… (STRICT)", "draft")
    doc = draft_document_json(dept, officer, case, best_pack, strategy)

    meta = doc.get("_meta", {})
    doc_num = meta.get("doc_num", "")
    today = meta.get("today", "")

    add_log("💾 [SAVE] (옵션) DB 저장…", "sys")
    db_msg = "DB 미연결"
    run_id = None
    if db.enabled():
        ok, msg, rid = db.insert_run({
            "user_id": user_key,
            "created_at": started,
            "task_type": clean_text(case.get("task_type","")),
            "input_text": user_input,
            "input_quality_score": int(case.get("_input_quality",{}).get("score", 0)),
            "final_verdict": best_pack.get("verdict"),
            "law_name": best_pack.get("law_name"),
            "law_mst": best_pack.get("mst"),
            "total_tokens": int(st.session_state.get("metrics",{}).get("tokens_total",0)),
            "status": "DONE",
            "result_json": safe_json_dump({
                "case": case, "best_law": best_pack, "strategy": strategy, "doc": ensure_doc_shape(doc), "candidates": candidates
            })
        })
        db_msg = msg
        run_id = rid

    add_log(f"✅ 완료 ({db_msg})", "sys")
    time.sleep(0.25)
    log_area.empty()

    return {
        "case": case,
        "candidates": candidates,     # ✅ 후보 리스트(클릭형 UI 핵심)
        "best_law": best_pack,        # ✅ 자동선택(참고용)
        "strategy": strategy,
        "doc": ensure_doc_shape(doc),
        "doc_meta": {"doc_num": doc_num, "today": today, "dept": dept, "officer": officer},
        "ev_items": ev_items,
        "loop_debug": loop_debug,
        "db_msg": db_msg,
        "run_id": run_id,
    }


# =========================
# 10) 판단 UI(클릭형 원문/사례)
# =========================
def ss_init():
    st.session_state.setdefault("selected_mst", "")
    st.session_state.setdefault("selected_law_name", "")
    st.session_state.setdefault("selected_article_no", "")
    st.session_state.setdefault("selected_article_title", "")
    st.session_state.setdefault("selected_article_text", "")
    st.session_state.setdefault("selected_law_link", "")
    st.session_state.setdefault("case_examples", [])
ss_init()


def build_law_link_fallback(mst: str) -> str:
    # DRF link가 없을 때도 최소한의 이동 경로 제공
    if not mst:
        return ""
    return f"https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq={mst}"


def ui_law_browser(case: dict, candidates: list):
    st.markdown("## ⚖️ 법령 후보 (원문/사례 클릭해서 판단)")
    st.caption("자동선택은 참고용입니다. 후보를 눌러 원문과 사례를 직접 보고 확정하세요.")

    if not candidates:
        st.warning("법령 후보가 없습니다. 입력 내용을 더 구체화하세요(대상/장소/기간/증거).")
        return

    for idx, c in enumerate(candidates[:8], start=1):
        law_name = clean_text(c.get("law_name",""))
        article_hint = clean_text(c.get("article_hint",""))
        reason = clean_text(c.get("reason",""))
        conf = c.get("confidence", 0.0)

        st.markdown(
            f"<div class='cand-row'><div><b>{idx}. {escape(law_name)}</b></div>"
            f"<div class='cand-sub'>힌트 조문: {escape(article_hint or '-')} · 신뢰도: {conf}</div>"
            f"<div class='cand-sub'>사유: {escape(reason or '')}</div></div>",
            unsafe_allow_html=True
        )

        colA, colB = st.columns([1, 1])
        with colA:
            if st.button("📜 원문 보기", key=f"btn_law_open_{idx}", use_container_width=True):
                laws = law_api.search_law(law_name, display=10)
                if not laws:
                    st.warning(f"'{law_name}' 검색 결과 없음")
                else:
                    chosen = laws[0]
                    mst = clean_text(chosen.get("MST"))
                    ln = clean_text(chosen.get("lawNm")) or law_name
                    link = clean_text(chosen.get("link")) or build_law_link_fallback(mst)

                    pack = law_api.get_article_by_mst(mst, article_no=article_hint if article_hint else None)
                    st.session_state["selected_mst"] = mst
                    st.session_state["selected_law_name"] = ln
                    st.session_state["selected_article_no"] = clean_text(pack.get("article_no",""))
                    st.session_state["selected_article_title"] = clean_text(pack.get("article_title",""))
                    st.session_state["selected_article_text"] = clean_text(pack.get("article_text",""))
                    st.session_state["selected_law_link"] = link

        with colB:
            if st.button("🧩 유사사례", key=f"btn_case_{idx}", use_container_width=True):
                kw = case.get("keywords", [])
                base = " ".join([clean_text(x) for x in kw[:2] if clean_text(x)])
                q = f"{law_name} {article_hint} {base}".strip()
                items = naver.search(q, cat="webkr", display=10)
                ex = []
                for it in items:
                    ex.append({
                        "title": clean_text(it.get("title")),
                        "desc": clean_text(it.get("description")),
                        "link": clean_text(it.get("link")),
                    })
                st.session_state["case_examples"] = ex

        st.markdown("---")


def ui_law_viewer():
    st.markdown("## 📜 선택한 법령 원문")
    mst = st.session_state.get("selected_mst","")
    if not mst:
        st.info("위 후보에서 **원문 보기**를 눌러주세요.")
        return

    law_name = st.session_state.get("selected_law_name","")
    art_title = st.session_state.get("selected_article_title","")
    art_text = st.session_state.get("selected_article_text","")
    link = st.session_state.get("selected_law_link","")

    st.markdown(f"**법령:** {law_name}")
    if link:
        st.markdown(f"**상세 링크:** [{link}]({link})")

    # 조문 인덱스 제공
    pack_idx = law_api.get_article_by_mst(mst, article_no=None) or {}
    idx_list = pack_idx.get("all_articles_index", []) if isinstance(pack_idx.get("all_articles_index"), list) else []

    if idx_list:
        pick = st.selectbox("조문 이동", ["(현재 조문 유지)"] + idx_list)
        if pick != "(현재 조문 유지)":
            m = re.search(r"제(\d+)조", pick)
            if m:
                art_no = m.group(1)
                pack2 = law_api.get_article_by_mst(mst, article_no=art_no) or {}
                st.session_state["selected_article_title"] = clean_text(pack2.get("article_title",""))
                st.session_state["selected_article_text"] = clean_text(pack2.get("article_text",""))
                art_title = st.session_state["selected_article_title"]
                art_text = st.session_state["selected_article_text"]

    st.markdown(f"### {art_title or '조문'}")
    if not art_text:
        st.warning("조문 텍스트를 가져오지 못했습니다. 다른 후보를 눌러보세요.")
        return

    with st.expander("원문 전문 펼치기", expanded=True):
        st.code(normalize_whitespace(strip_hanja_for_display(art_text)), language="text")


def ui_case_examples():
    st.markdown("## 🧩 유사사례(클릭해서 확인)")
    ex = st.session_state.get("case_examples", []) or []
    if not ex:
        st.info("법령 후보에서 **유사사례** 버튼을 누르면 여기에 뜹니다.")
        return

    for it in ex[:12]:
        title = clean_text(it.get("title",""))
        desc = clean_text(it.get("desc",""))
        link = clean_text(it.get("link",""))
        if link:
            st.markdown(f"- **[{title}]({link})**  \n  {desc}")
        else:
            st.markdown(f"- **{title}**  \n  {desc}")


# =========================
# 11) Renderers
# =========================
def render_a4(doc: Dict[str, Any], meta: Dict[str, str]):
    body_html = "".join([f"<p style='margin:0 0 14px 0;'>{safe_html(p)}</p>" for p in doc.get("body_paragraphs", [])])
    html = f"""
<div class="paper-sheet">
  <div class="stamp">직인생략</div>
  <div class="doc-header">{safe_html(doc.get('title',''))}</div>
  <div class="doc-info">
    <span>문서번호: {safe_html(meta.get('doc_num',''))}</span>
    <span>시행일자: {safe_html(meta.get('today',''))}</span>
    <span>수신: {safe_html(doc.get('receiver',''))}</span>
  </div>
  <div class="doc-body">
    {body_html}
  </div>
  <div class="doc-footer">{safe_html(doc.get('department_head',''))}</div>
</div>
"""
    components.html(html, height=920, scrolling=True)


# =========================
# 12) Main UI
# =========================
def main():
    st.session_state.setdefault("user_key", "local_user")
    st.session_state.setdefault("dept", "OO시청 OO과")
    st.session_state.setdefault("officer", "김주무관")

    col_l, col_r = st.columns([1, 1.25], gap="large")

    with col_l:
        st.title("AI 행정관 Pro")
        st.caption("Dual Router v7.0 — 클릭형 원문/사례 기반 '판단 UI' + A4 공문 렌더링")
        st.markdown("---")

        with st.expander("🧩 사용자/부서 설정", expanded=False):
            st.text_input("부서명", key="dept")
            st.text_input("담당자", key="officer")
            st.text_input("사용자 키(히스토리 구분용, 임의)", key="user_key")
            st.caption("※ Supabase 미설정이어도 정상 동작합니다.")

        user_input = st.text_area(
            "업무 지시 사항(민원 상황 포함)",
            height=240,
            placeholder="예: 건설기계가 차고지 외 장기간 주차(주기위반) 신고가 들어옴. 현장 확인했더니 이동한 상태. 민원인은 상시 단속을 요구. 담당자가 할 수 있는 조치와 공문 초안 작성.",
        )

        if st.button("🚀 실행(구조화→법령후보→원문확보→공문작성)", type="primary", use_container_width=True):
            if not user_input.strip():
                st.warning("내용을 입력하세요.")
            else:
                with st.spinner("실행 중..."):
                    try:
                        res = run_workflow(
                            user_input.strip(),
                            st.session_state["dept"],
                            st.session_state["officer"],
                            st.session_state["user_key"],
                        )
                        st.session_state["result"] = res

                        # 자동선택 법령을 우선 '선택 상태'에 로드(바로 원문탭에서 보이게)
                        best = res.get("best_law", {}) or {}
                        if best.get("mst"):
                            st.session_state["selected_mst"] = best.get("mst","")
                            st.session_state["selected_law_name"] = best.get("law_name","")
                            st.session_state["selected_article_title"] = best.get("article_title","")
                            st.session_state["selected_article_text"] = best.get("article_text","")
                            st.session_state["selected_law_link"] = best.get("link","") or build_law_link_fallback(best.get("mst",""))
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

        st.markdown(
            "<div class='small-muted'>핵심: 자동선택은 참고용. 담당자는 후보를 클릭해 원문·사례를 직접 확인하고 판단합니다.</div>",
            unsafe_allow_html=True
        )

    with col_r:
        tab_doc, tab_law, tab_case, tab_debug = st.tabs(["📄 공문(A4)", "⚖️ 법령 원문", "🧩 유사사례", "🧪 디버그"])
        res = st.session_state.get("result")

        with tab_doc:
            if not res:
                st.markdown(
                    """
<div style='text-align:center; padding:120px 20px; color:#9ca3af; border:2px dashed #e5e7eb; border-radius:14px; background:#fff;'>
  <h3 style='margin-bottom:8px;'>📄 A4 미리보기</h3>
  <p>왼쪽에서 민원 상황을 입력하고 실행을 누르세요.<br>공문이 A4 형태로 자동 렌더링됩니다.</p>
</div>
""",
                    unsafe_allow_html=True,
                )
            else:
                render_a4(res["doc"], res["doc_meta"])

        with tab_law:
            if not res:
                st.info("결과가 아직 없습니다. 왼쪽에서 실행하세요.")
            else:
                ui_law_browser(res.get("case", {}), res.get("candidates", []))
                ui_law_viewer()

        with tab_case:
            ui_case_examples()

        with tab_debug:
            if not res:
                st.info("결과가 아직 없습니다.")
            else:
                st.success(f"DB 저장: {res.get('db_msg','')}")
                st.markdown("### 1) 구조화된 케이스(case)")
                st.json(res.get("case", {}))

                st.markdown("### 2) 자동선택(참고용) best_law")
                st.json(res.get("best_law", {}))

                st.markdown("### 3) 전략(strategy)")
                st.markdown(res.get("strategy",""))

                st.markdown("### 4) 법령 후보 루프 디버그(loop_debug)")
                st.json(res.get("loop_debug", []))


if __name__ == "__main__":
    main()
```0
