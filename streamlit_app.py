# app.py — AI 행정관 Pro (Stable / Dual-Model Router v6.2)
# Groq: qwen/qwen3-32b (FAST) + llama-3.3-70b-versatile (STRICT)
# LAWGO(DRF) + NAVER + Supabase + A4 HTML Preview + Anti-crash
#
# ✅ FAST(default): CaseNormalizer / Planner / Strategy
# ✅ STRICT: JSON 생성(Planner 보정 실패/공문 Draft), 법령 불확실 시 Strategy 승급
# ✅ 법령: DRF JSON 우선 + "사람이 읽는 조문"으로 파싱(제목/본문/항/호)
# ✅ 한자 제거(가독성): 한자 범위 제거 옵션 기본 ON
# ✅ UI: A4 용지 스타일 HTML 렌더링(components.html)
# ✅ 성능: 캐싱(st.cache_data) + 프롬프트 축약 + 규칙 기반 후보 생성
# ✅ Metrics: 모델별 호출 + (가능하면) tokens_total 합산

import streamlit as st
import streamlit.components.v1 as components

import json
import re
import time
from datetime import datetime
from html import escape, unescape

# =========================
# 0) Optional Imports (Safety)
# =========================
try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import requests
except ImportError:
    requests = None

try:
    import xmltodict
except ImportError:
    xmltodict = None

try:
    from supabase import create_client
except ImportError:
    create_client = None


# =========================
# 1) Page & Style
# =========================
st.set_page_config(
    layout="wide",
    page_title="AI 행정관 Pro (Dual v6.2)",
    page_icon="⚖️",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.stApp { background-color: #f8f9fa; }

/* ===== A4 Paper Preview ===== */
.paper-wrap { display:flex; justify-content:center; }
.paper-sheet {
  background:#fff;
  width: 210mm;
  min-height: 297mm;
  padding: 22mm 20mm;
  margin: 14px 0;
  box-shadow: 0 8px 24px rgba(0,0,0,0.10);
  border-radius: 10px;
  font-family: 'Noto Serif KR','Nanum Myeongjo',serif;
  color:#111;
  line-height: 1.68;
  position: relative;
}
.doc-header {
  text-align:center;
  font-size: 24pt;
  font-weight: 800;
  letter-spacing: 4px;
  margin: 0 0 20mm 0;
}
.doc-meta {
  display:flex;
  justify-content:space-between;
  gap: 10px;
  flex-wrap:wrap;
  font-size: 11pt;
  border-bottom: 2px solid #222;
  padding-bottom: 8mm;
  margin-bottom: 10mm;
}
.doc-meta span { white-space: nowrap; }
.doc-body { font-size: 12pt; text-align: justify; }
.doc-body p { margin: 0 0 12px 0; }
.doc-footer {
  text-align:center;
  font-size: 20pt;
  font-weight: 800;
  letter-spacing: 6px;
  margin-top: 22mm;
}
.stamp {
  position:absolute;
  right: 18mm;
  bottom: 26mm;
  border: 3px solid #d32f2f;
  color: #d32f2f;
  padding: 6px 12px;
  font-size: 13pt;
  font-weight: 900;
  transform: rotate(-12deg);
  opacity: 0.82;
  border-radius: 6px;
  font-family: 'Nanum Gothic', sans-serif;
}

/* ===== Logs ===== */
.agent-log {
  font-family: 'Pretendard', sans-serif;
  font-size: 0.92rem;
  padding: 8px 12px;
  border-radius: 8px;
  margin-bottom: 6px;
  background: white;
  border: 1px solid #e5e7eb;
}
.log-legal { border-left: 4px solid #2563eb; color: #1e3a8a; }
.log-search { border-left: 4px solid #f97316; color: #9a3412; }
.log-strat { border-left: 4px solid #8b5cf6; color: #5b21b6; }
.log-draft { border-left: 4px solid #ef4444; color: #7f1d1d; }
.log-sys   { border-left: 4px solid #9ca3af; color: #374151; }

.small-muted { color:#6b7280; font-size:12px; }
.kpi { background:#fff; border:1px solid #e5e7eb; border-radius: 10px; padding: 10px 12px; }
.kpi h4 { margin:0 0 6px 0; font-size: 0.95rem; }
.kpi p { margin:0; color:#374151; font-size: 0.9rem; }

</style>
""",
    unsafe_allow_html=True,
)

_TAG_RE = re.compile(r"<[^>]+>")
# 제어문자 + Private Use Area(오류 유발) 제거
_CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
# 한자(중국어/한자) 범위 제거(가독성 위해)
_HANJA_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF]")


# =========================
# 2) Helpers
# =========================
def clean_text(value) -> str:
    """HTML 태그 + 제어문자 + PUA 제거"""
    if value is None:
        return ""
    s = str(value)
    s = unescape(s)
    s = _TAG_RE.sub("", s)
    s = _CTRL_RE.sub("", s)
    # PUA(Private Use Area) 제거: U+E000–U+F8FF
    s = re.sub(r"[\uE000-\uF8FF]", "", s)
    return s.strip()


def remove_hanja(s: str) -> str:
    """한자를 한글로 '변환'은 못하니(라이브러리 의존), 가독성 위해 제거(옵션)."""
    if not s:
        return ""
    return _HANJA_RE.sub("", s)


def safe_html(value) -> str:
    return escape(clean_text(value), quote=False).replace("\n", "<br>")


def truncate_text(s: str, max_chars: int = 1800) -> str:
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


def extract_keywords_kor(text: str, max_k: int = 6) -> list:
    if not text:
        return []
    t = re.sub(r"[^가-힣A-Za-z0-9\s]", " ", text)
    words = re.findall(r"[가-힣A-Za-z0-9]{2,12}", t)
    stop = set([
        "그리고","관련","문의","사항","대하여","대한","처리","요청","작성","안내","검토","불편","민원",
        "신청","발급","제출","통지","답변","회신","부탁","조치","확인","내용","사유","진행"
    ])
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

def metrics_add(model_name: str, tokens_total=None):
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
    """
    secrets.toml
    [general]
    GROQ_API_KEY = "..."
    GROQ_MODEL_FAST = "qwen/qwen3-32b"
    GROQ_MODEL_STRICT = "llama-3.3-70b-versatile"
    """
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
            raise RuntimeError("Groq client not ready")

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

    def _parse_json(self, text: str) -> dict:
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
            {"role": "system", "content": "Korean public administration assistant. Practical, correct, concise."},
            {"role": "user", "content": prompt},
        ]
        # 1차
        try:
            return self._chat(model_first, messages, temp, json_mode=False)
        except Exception:
            pass
        # 승급
        try:
            return self._chat(self.model_strict, messages, temp, json_mode=False)
        except Exception as e:
            return f"LLM Error: {e}"

    def generate_json(self, prompt: str, prefer: str = "fast", temp: float = 0.1, max_retry: int = 2) -> dict:
        if not self.client:
            return {}

        sys_json = "Output JSON only. No markdown. No extra keys. Follow schema exactly."
        messages = [
            {"role": "system", "content": sys_json},
            {"role": "user", "content": prompt},
        ]
        model_first = self.model_fast if prefer == "fast" else self.model_strict

        # 같은 모델 재시도
        for _ in range(max_retry):
            try:
                txt = self._chat(model_first, messages, temp, json_mode=True)
                js = self._parse_json(txt)
                if js:
                    return js
            except Exception:
                pass

        # strict 승급
        try:
            txt = self._chat(self.model_strict, messages, temp, json_mode=True)
            js = self._parse_json(txt)
            return js if js else {}
        except Exception:
            return {}

llm_service = LLMService()


# =========================
# 5) LAW API (DRF) — JSON 우선 + 조문 파싱
# =========================
class LawAPIService:
    """
    secrets.toml
    [law]
    LAW_API_ID = "OC값"
    """
    def __init__(self):
        self.oc = st.secrets.get("law", {}).get("LAW_API_ID")
        self.search_url = "https://www.law.go.kr/DRF/lawSearch.do"
        self.service_url = "https://www.law.go.kr/DRF/lawService.do"
        self.enabled = bool(requests and self.oc)

    def search_law(self, query: str, display: int = 10) -> list:
        if not self.enabled or not query:
            return []
        # JSON 우선
        try:
            params = {
                "OC": self.oc,
                "target": "law",
                "type": "JSON",
                "query": query,
                "display": display,
                "page": 1,
            }
            r = requests.get(self.search_url, params=params, timeout=7)
            r.raise_for_status()
            data = r.json()
            laws = data.get("LawSearch", {}).get("law", [])
            if isinstance(laws, dict):
                laws = [laws]
            out = []
            for it in laws:
                if not isinstance(it, dict):
                    continue
                out.append({
                    "law_name": clean_text(it.get("법령명한글") or it.get("lawNm") or it.get("법령명") or ""),
                    "mst": clean_text(it.get("법령일련번호") or it.get("MST") or it.get("mst") or ""),
                    "law_id": clean_text(it.get("법령ID") or it.get("lawId") or ""),
                    "link": clean_text(it.get("법령상세링크") or it.get("link") or ""),
                })
            return [x for x in out if x["law_name"] and x["mst"]]
        except Exception:
            # XML 백업
            if not xmltodict:
                return []
            try:
                params = {
                    "OC": self.oc,
                    "target": "law",
                    "type": "XML",
                    "query": query,
                    "display": display,
                    "page": 1,
                }
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
                    out.append({
                        "law_name": clean_text(it.get("법령명한글") or it.get("lawNm") or it.get("법령명") or ""),
                        "mst": clean_text(it.get("법령일련번호") or it.get("MST") or it.get("mst") or ""),
                        "law_id": clean_text(it.get("법령ID") or it.get("lawId") or ""),
                        "link": clean_text(it.get("법령상세링크") or it.get("link") or ""),
                    })
                return [x for x in out if x["law_name"] and x["mst"]]
            except Exception:
                return []

    def get_article_pretty(self, mst: str, article_no: str | None = None) -> dict:
        """
        return {
          ok: bool,
          law_name: str,
          article_no: "33" or "",
          text: "제33조(...)\\n본문\\n1. ...\\n- ..."
        }
        """
        if not self.enabled or not mst:
            return {"ok": False, "law_name": "", "article_no": "", "text": ""}

        tgt = re.sub(r"[^0-9]", "", str(article_no or ""))

        # JSON 우선
        try:
            params = {"OC": self.oc, "target": "law", "type": "JSON", "MST": mst}
            r = requests.get(self.service_url, params=params, timeout=9)
            r.raise_for_status()
            data = r.json()

            law = data.get("Law", {}) or {}
            law_name = clean_text(law.get("lawNm") or law.get("법령명한글") or "")

            articles = law.get("Article", []) or []
            if isinstance(articles, dict):
                articles = [articles]

            # 조문번호 없으면: 첫 조문 1개만
            if not tgt:
                if articles and isinstance(articles[0], dict):
                    at = clean_text(articles[0].get("ArticleTitle") or "")
                    ac = clean_text(articles[0].get("ArticleContent") or "")
                    txt = "\n".join([x for x in [at, ac] if x]).strip()
                    return {"ok": bool(txt), "law_name": law_name, "article_no": "", "text": txt}
                return {"ok": False, "law_name": law_name, "article_no": "", "text": ""}

            for art in articles:
                if not isinstance(art, dict):
                    continue
                an = clean_text(art.get("@조문번호") or art.get("joNo") or "")
                an_num = re.sub(r"[^0-9]", "", an)
                at = clean_text(art.get("ArticleTitle") or "")
                if tgt == an_num or (tgt and f"제{tgt}조" in at):
                    content = clean_text(art.get("ArticleContent") or "")

                    paras = art.get("Paragraph", []) or []
                    if isinstance(paras, dict):
                        paras = [paras]

                    lines = []
                    for p in paras:
                        if not isinstance(p, dict):
                            continue
                        pc = clean_text(p.get("ParagraphContent") or "")
                        if pc:
                            lines.append(pc)
                        items = p.get("Item", []) or []
                        if isinstance(items, dict):
                            items = [items]
                        for it in items:
                            if not isinstance(it, dict):
                                continue
                            ic = clean_text(it.get("ItemContent") or "")
                            if ic:
                                lines.append(f"- {ic}")

                    full = "\n".join([x for x in [at, content] if x] + lines).strip()
                    return {"ok": bool(full), "law_name": law_name, "article_no": tgt, "text": full}

            return {"ok": False, "law_name": law_name, "article_no": tgt, "text": ""}

        except Exception:
            # XML 백업(최후 수단)
            if not xmltodict:
                return {"ok": False, "law_name": "", "article_no": tgt, "text": ""}
            try:
                params = {"OC": self.oc, "target": "law", "type": "XML", "MST": mst}
                r = requests.get(self.service_url, params=params, timeout=9)
                r.raise_for_status()
                data = xmltodict.parse(r.text)
                law = data.get("Law") or {}
                law_name = clean_text(law.get("법령명한글") or law.get("lawNm") or "")
                articles = law.get("Article", []) or []
                if isinstance(articles, dict):
                    articles = [articles]

                if not tgt:
                    txt = clean_text(r.text)
                    return {"ok": bool(txt), "law_name": law_name, "article_no": "", "text": txt[:1200]}

                for art in articles:
                    if not isinstance(art, dict):
                        continue
                    an = clean_text(art.get("@조문번호") or "")
                    an_num = re.sub(r"[^0-9]", "", an)
                    at = clean_text(art.get("ArticleTitle") or "")
                    if tgt == an_num or (tgt and f"제{tgt}조" in at):
                        content = clean_text(art.get("ArticleContent") or "")
                        full = "\n".join([x for x in [at, content] if x]).strip()
                        return {"ok": bool(full), "law_name": law_name, "article_no": tgt, "text": full}

                return {"ok": False, "law_name": law_name, "article_no": tgt, "text": ""}
            except Exception:
                return {"ok": False, "law_name": "", "article_no": tgt, "text": ""}

law_api = LawAPIService()


# =========================
# 6) NAVER Search
# =========================
class NaverSearchService:
    """
    secrets.toml
    [naver]
    CLIENT_ID="..."
    CLIENT_SECRET="..."
    """
    def __init__(self):
        n = st.secrets.get("naver", {})
        self.cid = n.get("CLIENT_ID")
        self.csec = n.get("CLIENT_SECRET")
        self.enabled = bool(requests and self.cid and self.csec)

    def search(self, query: str, cat: str = "news", display: int = 5):
        if not self.enabled or not query:
            return []
        try:
            url = f"https://openapi.naver.com/v1/search/{cat}.json"
            headers = {"X-Naver-Client-Id": self.cid, "X-Naver-Client-Secret": self.csec}
            params = {"query": query, "display": display, "sort": "sim", "start": 1}
            r = requests.get(url, headers=headers, params=params, timeout=6)
            r.raise_for_status()
            return r.json().get("items", []) or []
        except Exception:
            return []

naver_search = NaverSearchService()


# =========================
# 7) Supabase
# =========================
class DatabaseService:
    """
    secrets.toml
    [supabase]
    SUPABASE_URL="..."
    SUPABASE_KEY="..."
    """
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

    def save_log(self, data: dict):
        if not self.client:
            return "DB 미연결"
        try:
            safe_data = json.loads(safe_json_dump(data))
            self.client.table("law_logs").insert(safe_data).execute()
            return "저장 성공"
        except Exception as e:
            return f"저장 실패: {str(e)}"

db_service = DatabaseService()


# =========================
# 8) Caching (성능)
# =========================
@st.cache_data(show_spinner=False, ttl=60 * 30)
def cached_law_search(query: str, display: int = 10):
    return law_api.search_law(query, display=display)

@st.cache_data(show_spinner=False, ttl=60 * 60)
def cached_law_article(mst: str, article_no: str):
    return law_api.get_article_pretty(mst, article_no)

@st.cache_data(show_spinner=False, ttl=60 * 20)
def cached_naver_news(query: str, display: int = 5):
    return naver_search.search(query, cat="news", display=display)


# =========================
# 9) Workflow 핵심 개선 포인트
# =========================
def normalize_case_fast(user_input: str) -> dict:
    """
    '민원 상황을 넣으면 이해 못함' 해결용:
    1) 입력을 먼저 '사실/요구/대상/장소/시간/증거/쟁점'으로 구조화
    2) 이후 Planner/LawSearch는 이 구조화 텍스트 기반으로만 진행
    """
    kw = extract_keywords_kor(user_input, max_k=6)
    prompt = f"""
아래 민원/업무 지시를 '사실 중심'으로 구조화해 JSON만 출력.

[원문]
{user_input}

[스키마]
{{
  "summary_one_line": "한 줄 요약(20~40자)",
  "facts": ["사실1","사실2","사실3"],
  "request": "민원인이 원하는 것(또는 처리 목표)",
  "targets": ["대상(차량/업체/사람/기관 등)"],
  "place_time": "장소/시간(없으면 빈문자열)",
  "evidence": ["증거/자료(사진/문서/녹취 등)"],
  "risk_points": ["쟁점/주의점(법적/민원/절차)"],
  "keywords": {kw}
}}
주의:
- 추측 금지(없으면 빈값/모름)
- 법령명 추정은 여기서 하지 말 것
"""
    js = llm_service.generate_json(prompt, prefer="fast", max_retry=2, temp=0.1)
    if not js:
        return {
            "summary_one_line": "",
            "facts": [user_input[:120]],
            "request": "",
            "targets": [],
            "place_time": "",
            "evidence": [],
            "risk_points": [],
            "keywords": kw[:4],
        }
    # 안전정리
    def _list(v):
        return v if isinstance(v, list) else []
    return {
        "summary_one_line": clean_text(js.get("summary_one_line") or ""),
        "facts": [clean_text(x) for x in _list(js.get("facts")) if clean_text(x)][:6],
        "request": clean_text(js.get("request") or ""),
        "targets": [clean_text(x) for x in _list(js.get("targets")) if clean_text(x)][:6],
        "place_time": clean_text(js.get("place_time") or ""),
        "evidence": [clean_text(x) for x in _list(js.get("evidence")) if clean_text(x)][:6],
        "risk_points": [clean_text(x) for x in _list(js.get("risk_points")) if clean_text(x)][:6],
        "keywords": [clean_text(x) for x in _list(js.get("keywords")) if clean_text(x)][:6] or kw[:4],
    }


def plan_law_and_keywords(case_pack: dict) -> dict:
    """
    Planner가 엉뚱한 법령 찍는 문제를 줄이려면:
    - '법령명 맞추기'를 1shot으로 하지 말고
    - 후보 법령명을 최대 3개만 제시하게 하고(확신 없으면 빈값),
    - 이후 실제 law.go.kr search 결과로 검증해서 채택.
    """
    base_text = f"""
[요약] {case_pack.get('summary_one_line','')}
[사실] {" / ".join(case_pack.get('facts',[]))}
[요구] {case_pack.get('request','')}
[대상] {", ".join(case_pack.get('targets',[]))}
[장소/시간] {case_pack.get('place_time','')}
[쟁점] {" / ".join(case_pack.get('risk_points',[]))}
[키워드] {", ".join(case_pack.get('keywords',[]))}
""".strip()

    prompt = f"""
다음 업무를 처리하기 위해 '가능성 높은 법령 후보'와 '검색 키워드'만 JSON으로 출력.

[업무정보]
{base_text}

[스키마]
{{
  "task_type": "업무유형(짧게)",
  "law_candidates": ["법령명 후보1","법령명 후보2","법령명 후보3"],
  "article_no_hint": "조번호 힌트(숫자만, 모르면 빈문자열)",
  "keywords": ["검색어1","검색어2","검색어3"]
}}

제약:
- 확신 없으면 law_candidates는 빈문자열로 채우지 말고 그냥 비워도 됨.
- 법령명은 '공식명' 우선(예: 자동차관리법, 건설기계관리법, 도로교통법 등)
- 조번호는 정말 확실할 때만(모르면 빈문자열)
"""
    js = llm_service.generate_json(prompt, prefer="fast", max_retry=2, temp=0.1)
    if not js:
        return {"task_type": "업무", "law_candidates": [], "article_no_hint": "", "keywords": case_pack.get("keywords", [])[:3]}

    cands = js.get("law_candidates") if isinstance(js.get("law_candidates"), list) else []
    cands = [clean_text(x) for x in cands if clean_text(x)]
    kws = js.get("keywords") if isinstance(js.get("keywords"), list) else []
    kws = [clean_text(x) for x in kws if clean_text(x)]
    if not kws:
        kws = case_pack.get("keywords", [])[:3]
    return {
        "task_type": clean_text(js.get("task_type") or "업무"),
        "law_candidates": cands[:3],
        "article_no_hint": clean_text(js.get("article_no_hint") or ""),
        "keywords": kws[:4],
        "base_text": base_text,
    }


def choose_best_law(law_queries: list, article_no_hint: str, add_log=None) -> dict:
    """
    실제 DRF 검색 결과로 '검증'해서 선정:
    - 후보 쿼리 순서대로 search -> top 결과 채택
    - 조문 파싱 성공하면 CONFIRMED
    """
    legal_status = "FAIL"
    legal_basis = "관련 법령 검색 실패"
    law_debug = {"queries": law_queries, "picked": None}
    chosen = None

    for q in law_queries[:5]:
        if add_log:
            add_log(f"법령검색 시도: {q}", "legal")
        cands = cached_law_search(q, display=10)
        if cands:
            chosen = cands[0]
            break

    if not chosen:
        return {"legal_status": "FAIL", "legal_basis": legal_basis, "law_debug": law_debug}

    law_name = clean_text(chosen.get("law_name") or "")
    mst = clean_text(chosen.get("mst") or "")
    link = clean_text(chosen.get("link") or "")
    law_debug["picked"] = {"law_name": law_name, "mst": mst, "link": link}

    # 조문번호가 있으면 조문 우선, 없으면 첫 조문 1개라도 사람 읽게
    art_no = re.sub(r"[^0-9]", "", article_no_hint or "")
    art_pack = cached_law_article(mst, art_no) if art_no else law_api.get_article_pretty(mst, None)

    if art_pack.get("ok") and art_pack.get("text"):
        legal_status = "CONFIRMED" if art_pack.get("article_no") else "WEAK"
        legal_basis = f"{art_pack.get('law_name','')}\n{art_pack.get('text','')}".strip()
    else:
        legal_status = "WEAK"
        legal_basis = f"{law_name}\n(조문 원문 파싱 실패 — 추가 확인 필요)"
    return {"legal_status": legal_status, "legal_basis": legal_basis, "law_debug": law_debug}


def build_strategy(case_pack: dict, plan_pack: dict, legal_basis: str, legal_status: str, ev_text: str) -> str:
    prefer = "strict" if legal_status != "CONFIRMED" else "fast"
    prompt = f"""
[업무유형] {plan_pack.get('task_type','업무')}
[업무요약] {case_pack.get('summary_one_line','')}
[사실] {" / ".join(case_pack.get('facts',[]))}
[요구] {case_pack.get('request','')}
[쟁점] {" / ".join(case_pack.get('risk_points',[]))}

[법적근거]
{truncate_text(legal_basis, 1200)}

[참고(네이버)]
{truncate_text(ev_text, 700)}

아래 형식 마크다운으로만:
1) 처리 방향 (3~6줄)
2) 핵심 체크리스트 (불릿 6~12개)
3) 예상 민원/반발 & 대응 (3~6줄)
4) '담당부서 한계'가 있으면 한 줄 명시(예: 주기위반만 가능 등)

원칙:
- 과장/추측 금지, 불확실하면 '추가 확인 필요' 명시
- 실제 행정 절차 관점(통지/계고/청문/이의신청 등)으로 작성
"""
    return llm_service.generate_text(prompt, prefer=prefer, temp=0.1)


def build_official_doc_json(
    dept: str,
    officer: str,
    case_pack: dict,
    legal_basis: str,
    legal_status: str,
    strategy_md: str,
    doc_num: str,
    today_str: str,
) -> dict:
    # STRICT 고정
    prompt = f"""
아래 스키마로만 JSON 출력(키 추가 금지):
{{
  "title": "문서 제목",
  "receiver": "수신",
  "body_paragraphs": ["문단1","문단2","문단3","문단4","문단5"],
  "department_head": "발신 명의"
}}

작성 정보:
- 부서: {dept}
- 담당자: {officer}
- 시행일: {today_str}
- 문서번호: {doc_num}

사건 요약:
- 한줄요약: {case_pack.get("summary_one_line","")}
- 사실: {" / ".join(case_pack.get("facts",[]))}
- 요구: {case_pack.get("request","")}
- 대상: {", ".join(case_pack.get("targets",[]))}
- 장소/시간: {case_pack.get("place_time","")}

법적 근거(확보된 범위 / 상태={legal_status}):
{truncate_text(legal_basis, 1200)}

처리 전략(요약):
{truncate_text(strategy_md, 900)}

작성 원칙:
- 문체: 건조/정중/명확
- 구조: [감사/요지] -> [사실관계] -> [법적근거] -> [조치/안내] -> [문의처]
- 법령이 불확실하면 '추가 확인 필요' 또는 '관련 규정 검토 후' 문구 포함
- 개인정보는 OOO로 마스킹
"""
    doc_json = llm_service.generate_json(prompt, prefer="strict", max_retry=2, temp=0.1)
    return ensure_doc_shape(doc_json)


def run_workflow(user_input: str, dept: str, officer: str, remove_hanja_on: bool = True):
    log_area = st.empty()
    logs = []

    def add_log(msg: str, style: str = "sys"):
        logs.append(f"<div class='agent-log log-{style}'>{safe_html(msg)}</div>")
        log_area.markdown("".join(logs), unsafe_allow_html=True)
        time.sleep(0.04)

    # 0) Normalize
    add_log("🧩 [Normalizer] 민원/업무 입력을 '사실 중심 구조'로 정리...", "sys")
    case_pack = normalize_case_fast(user_input)

    # 1) Planner
    add_log("🧭 [Planner] 법령 후보/키워드 산출 (FAST)...", "sys")
    plan_pack = plan_law_and_keywords(case_pack)

    # 2) Law Search (검증 기반)
    add_log("📚 [Law] 법령 검색 및 조문 파싱(사람이 읽는 형태)...", "legal")
    # law query 우선순위: 후보법령명 -> 키워드
    law_queries = []
    for x in plan_pack.get("law_candidates", [])[:3]:
        if x and x not in law_queries:
            law_queries.append(x)
    for k in plan_pack.get("keywords", [])[:3]:
        if k and k not in law_queries:
            law_queries.append(k)

    law_pick = choose_best_law(law_queries, plan_pack.get("article_no_hint", ""), add_log=add_log)
    legal_status = law_pick["legal_status"]
    legal_basis = law_pick["legal_basis"]
    law_debug = law_pick["law_debug"]

    if remove_hanja_on:
        legal_basis = remove_hanja(legal_basis)

    # 3) Naver Evidence
    add_log("🌍 [Search] 네이버 뉴스로 사실관계/리스크 점검...", "search")
    ev_items = []
    ev_text = ""
    if plan_pack.get("keywords"):
        q = " ".join(plan_pack["keywords"][:2])
        raw = cached_naver_news(q, display=5)
        for it in raw:
            t = clean_text(it.get("title"))
            d = clean_text(it.get("description"))
            link = clean_text(it.get("link"))
            if remove_hanja_on:
                t, d = remove_hanja(t), remove_hanja(d)
            ev_items.append({"title": t, "link": link, "desc": d})
            ev_text += f"- {t}: {d}\n"

    # 4) Strategy
    add_log("🧠 [Analyst] 처리 전략 수립(법령 불확실 시 STRICT 승급)...", "strat")
    strategy = build_strategy(case_pack, plan_pack, legal_basis, legal_status, ev_text)
    if remove_hanja_on:
        strategy = remove_hanja(strategy)

    # 5) Drafter (A4 문서용 JSON)
    add_log("✍️ [Drafter] 공문서(JSON) 생성 (STRICT)...", "draft")
    today_str = datetime.now().strftime("%Y. %m. %d.")
    doc_num = f"행정-{datetime.now().strftime('%Y')}-{int(time.time()) % 10000:04d}호"
    doc_final = build_official_doc_json(
        dept, officer, case_pack, legal_basis, legal_status, strategy, doc_num, today_str
    )

    # 6) Save
    add_log("💾 [System] 결과 저장...", "sys")
    payload = {
        "created_at": datetime.now().isoformat(),
        "dept": dept,
        "officer": officer,
        "input": user_input,
        "case_pack": safe_json_dump(case_pack),
        "task_type": plan_pack.get("task_type", ""),
        "keywords": safe_json_dump(plan_pack.get("keywords", [])),
        "legal_status": legal_status,
        "legal_basis": legal_basis,
        "final_doc": safe_json_dump(doc_final),
        "strategy": strategy,
        "provenance": safe_json_dump(ev_items),
        "model_last": llm_service.last_model,
        "metrics": safe_json_dump(st.session_state.get("metrics", {})),
        "law_debug": safe_json_dump(law_debug),
        "remove_hanja": remove_hanja_on,
    }
    db_msg = db_service.save_log(payload)
    add_log(f"✅ 완료 ({db_msg})", "sys")

    time.sleep(0.25)
    log_area.empty()

    return {
        "doc": doc_final,
        "meta": {"doc_num": doc_num, "today": today_str, "dept": dept, "officer": officer},
        "case_pack": case_pack,
        "legal_basis": legal_basis,
        "legal_status": legal_status,
        "strategy": strategy,
        "ev_items": ev_items,
        "db_msg": db_msg,
        "law_debug": law_debug,
        "plan_pack": plan_pack,
    }


# =========================
# 10) UI
# =========================
def render_a4_html(doc: dict, meta: dict) -> str:
    body_html = "".join([f"<p>{safe_html(p)}</p>" for p in doc.get("body_paragraphs", [])])
    html = f"""
<div class="paper-wrap">
  <div class="paper-sheet">
    <div class="stamp">직인생략</div>
    <div class="doc-header">{safe_html(doc.get('title',''))}</div>
    <div class="doc-meta">
      <span>문서번호: {safe_html(meta.get('doc_num',''))}</span>
      <span>시행일자: {safe_html(meta.get('today',''))}</span>
      <span>수신: {safe_html(doc.get('receiver',''))}</span>
    </div>
    <div class="doc-body">
      {body_html}
    </div>
    <div class="doc-footer">{safe_html(doc.get('department_head',''))}</div>
  </div>
</div>
"""
    return html


def main():
    st.session_state.setdefault("dept", "OO시청 OO과")
    st.session_state.setdefault("officer", "김주무관")
    st.session_state.setdefault("remove_hanja_on", True)

    col_l, col_r = st.columns([1, 1.25], gap="large")

    with col_l:
        st.title("AI 행정관 Pro")
        st.caption("Dual Router v6.2 — FAST(qwen/qwen3-32b) + STRICT(llama-3.3-70b) / 법령 조문 가독성 패치 완료")
        st.markdown("---")

        with st.expander("📝 사용자 정보 설정", expanded=False):
            st.text_input("부서명", key="dept")
            st.text_input("담당자", key="officer")
            st.checkbox("법령/전략 텍스트에서 한자 제거(가독성)", key="remove_hanja_on")

        user_input = st.text_area(
            "업무 지시 사항(민원 상황 포함 가능)",
            height=220,
            placeholder="예: 차고지 외 불법 방치된 건설기계에 대해 주기위반 여부 검토 후 안내 답변문 작성.",
        )

        if st.button("🚀 문서 생성 실행", type="primary", use_container_width=True):
            if not user_input.strip():
                st.warning("내용을 입력하세요.")
            else:
                with st.spinner("AI 에이전트 협업 중..."):
                    try:
                        res = run_workflow(
                            user_input.strip(),
                            st.session_state["dept"],
                            st.session_state["officer"],
                            remove_hanja_on=bool(st.session_state.get("remove_hanja_on", True)),
                        )
                        st.session_state["result"] = res
                    except Exception as e:
                        st.error(f"치명적 오류 발생: {e}")

        st.markdown("---")
        m = st.session_state.get("metrics", {})
        calls = m.get("calls", {})
        tokens_total = m.get("tokens_total", 0)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='kpi'><h4>모델 호출(세션)</h4>", unsafe_allow_html=True)
            if calls:
                for k, v in sorted(calls.items(), key=lambda x: (-x[1], x[0])):
                    st.write(f"- **{k}**: {v}회")
            else:
                st.write("- 대기 중")
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='kpi'><h4>토큰 합계(가능한 경우)</h4>", unsafe_allow_html=True)
            st.write(f"- **{tokens_total}**")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "<div class='small-muted'>TIP: 입력이 길어도 먼저 Normalizer가 사실관계를 구조화해서 Planner/법령검색의 '엉뚱함'을 줄입니다.</div>",
            unsafe_allow_html=True,
        )

    with col_r:
        res = st.session_state.get("result")

        if not res:
            st.markdown(
                """
<div style='text-align: center; padding: 120px 20px; color: #9ca3af; border: 2px dashed #e5e7eb; border-radius: 12px; background:#fff;'>
  <h3 style="margin:0 0 6px 0;">📄 A4 공문 미리보기</h3>
  <p style="margin:0;">왼쪽에서 업무를 입력하고 실행을 누르면<br>법령 검증 후 공문서 형태로 출력됩니다.</p>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            doc = res["doc"]
            meta = res["meta"]

            tab1, tab2 = st.tabs(["📄 공문서(A4)", "🔍 근거/분석/디버그"])

            with tab1:
                html = render_a4_html(doc, meta)
                components.html(html, height=920, scrolling=True)

            with tab2:
                st.success(f"DB: {res.get('db_msg','')}")
                st.markdown("### 🧩 사건 구조화(입력 이해 결과)")
                st.code(safe_json_dump(res.get("case_pack", {})), language="json")

                st.markdown("### 📜 법적 근거(가독성 조문)")
                st.info(f"상태: {res.get('legal_status')}")
                st.code(res.get("legal_basis", ""), language="text")

                st.markdown("### 💡 처리 전략")
                st.markdown(res.get("strategy", ""))

                st.markdown("### 📎 참고 자료 (Naver)")
                for item in res.get("ev_items", []):
                    title = clean_text(item.get("title"))
                    link = clean_text(item.get("link"))
                    desc = clean_text(item.get("desc"))
                    if link:
                        st.markdown(f"- [{title}]({link}) — {desc}")
                    else:
                        st.markdown(f"- {title} — {desc}")

                with st.expander("🛠️ Planner/법령 디버그", expanded=False):
                    st.markdown("**Planner 결과**")
                    st.code(safe_json_dump(res.get("plan_pack", {})), language="json")
                    st.markdown("**Law Debug**")
                    st.code(safe_json_dump(res.get("law_debug", {})), language="json")


if __name__ == "__main__":
    main()
