# app.py — AI 행정관 Pro (v6.0 / Agentic Dual-Model Router)
# Groq: qwen/qwen3-32b (FAST) + llama-3.3-70b-versatile (STRICT)
# LAWGO(DRF) + NAVER + Supabase
#
# 핵심 개선 (성능/정확도 급상승)
# ✅ (1) Extractor(슬롯추출) -> Candidate Law Search(후보풀) -> Law Selector(후보 중 선택) -> Verify(원문확보) -> Draft
# ✅ (2) 법령 튐 방지: LLM은 후보 목록에서만 선택
# ✅ (3) 조문 원문 확보 실패 시: "법령 단정 금지" 모드로 기안 프롬프트 강제
# ✅ (4) NAVER는 보조 (On/Off + 관련성 필터 + 전문성 필터)
# ✅ (5) 중간에 한자/비정상 문자(U+EA01 등) 제거/정리(입력/표시 모두)
# ✅ (6) Metrics: 모델별 호출 수 + total_tokens(가능하면) + 단계별 카운트
# ✅ (7) Anti-crash: optional deps, timeouts, JSON retry/승급, HTML sanitize, components.html 안정화

import streamlit as st
import streamlit.components.v1 as components

import json
import re
import time
from datetime import datetime
from html import escape, unescape
from typing import Any

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
    page_title="AI 행정관 Pro (Agentic v6.0)",
    page_icon="🏛️",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.stApp { background-color: #f8f9fa; }

.paper-sheet{
  background:#fff; width:100%; max-width:210mm; min-height:297mm;
  padding:25mm; margin:auto; box-shadow:0 4px 15px rgba(0,0,0,.08);
  font-family:'Noto Serif KR','Nanum Myeongjo',serif; color:#111; line-height:1.6; position:relative;
}
.doc-header{ text-align:center; font-size:24pt; font-weight:900; margin-bottom:28px; letter-spacing:2px; }
.doc-info{
  display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap;
  font-size:11pt; border-bottom:2px solid #333; padding-bottom:12px; margin-bottom:22px;
}
.doc-body{ font-size:12pt; text-align:justify; white-space:normal; }
.doc-footer{ text-align:center; font-size:22pt; font-weight:bold; margin-top:80px; letter-spacing:4px; }
.stamp{
  position:absolute; bottom:85px; right:80px; border:3px solid #d32f2f; color:#d32f2f;
  padding:6px 12px; font-size:14pt; font-weight:bold; transform:rotate(-15deg);
  opacity:.8; border-radius:4px; font-family:'Nanum Gothic', sans-serif;
}

.agent-log{
  font-family:'Pretendard',sans-serif; font-size:.9rem; padding:8px 12px;
  border-radius:8px; margin-bottom:6px; background:#fff; border:1px solid #e5e7eb;
}
.log-extract{ border-left:4px solid #0ea5e9; color:#0c4a6e; }
.log-law{ border-left:4px solid #3b82f6; color:#1e40af; }
.log-verify{ border-left:4px solid #22c55e; color:#166534; }
.log-search{ border-left:4px solid #f97316; color:#c2410c; }
.log-strat{ border-left:4px solid #8b5cf6; color:#6d28d9; }
.log-draft{ border-left:4px solid #ef4444; color:#991b1b; }
.log-sys{ border-left:4px solid #9ca3af; color:#374151; }

.small-muted{ color:#6b7280; font-size:12px; }
.badge{ display:inline-block; padding:3px 9px; border-radius:999px; font-size:12px; margin-right:6px; border:1px solid #e5e7eb; background:#fff; }
.badge-ok{ border-color:#bbf7d0; background:#f0fdf4; }
.badge-warn{ border-color:#fde68a; background:#fffbeb; }
.badge-bad{ border-color:#fecaca; background:#fef2f2; }

.item-card{ background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:12px 14px; margin-bottom:10px; }
.item-title{ font-weight:800; }
.item-meta{ color:#6b7280; font-size:12px; margin-top:4px; line-height:1.3; }
.item-desc{ margin-top:8px; white-space:pre-line; }
</style>
""",
    unsafe_allow_html=True,
)

_TAG_RE = re.compile(r"<[^>]+>")
# 제어문자 + Private Use Area(U+E000~U+F8FF) 제거(대표적으로 U+EA01 같은 것)
_CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_PUA_RE = re.compile(r"[\uE000-\uF8FF]")

# =========================
# 2) Helpers (Sanitize / Parse)
# =========================
def clean_text(value: Any) -> str:
    """HTML 태그/제어문자/PUA/이상 공백 제거"""
    if value is None:
        return ""
    s = str(value)
    s = unescape(s)
    s = _TAG_RE.sub("", s)
    s = _CTRL_RE.sub("", s)
    s = _PUA_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def safe_html(value: Any) -> str:
    return escape(clean_text(value), quote=False).replace("\n", "<br>")

def truncate_text(s: str, max_chars: int = 2500) -> str:
    s = s or ""
    return s if len(s) <= max_chars else s[:max_chars] + "\n...(내용 축소됨)"

def safe_json_dump(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return "{}"

def ensure_doc_shape(doc: Any) -> dict:
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

    body_clean = [clean_text(x) for x in body if clean_text(x)]
    if not body_clean:
        body_clean = fallback["body_paragraphs"]

    return {
        "title": clean_text(doc.get("title") or fallback["title"]),
        "receiver": clean_text(doc.get("receiver") or fallback["receiver"]),
        "body_paragraphs": body_clean,
        "department_head": clean_text(doc.get("department_head") or fallback["department_head"]),
    }

def extract_keywords_kor(text: str, max_k: int = 8) -> list[str]:
    """LLM 없이도 후보풀 넓히는 안전망"""
    if not text:
        return []
    t = re.sub(r"[^가-힣A-Za-z0-9\s]", " ", text)
    words = re.findall(r"[가-힣A-Za-z0-9]{2,14}", t)
    stop = set([
        "그리고","관련","문의","사항","대하여","대한","처리","요청","작성","안내","검토","불편","민원","신청","발급","제출",
        "위해","대한","습니다","합니다","입니다","가능","조치","대상","경우","확인"
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

def score_overlap(text: str, terms: list[str]) -> int:
    t = text or ""
    hit = 0
    for w in terms:
        if w and w in t:
            hit += 1
    return hit

# =========================
# 3) Metrics
# =========================
def metrics_init():
    if "metrics" not in st.session_state:
        st.session_state["metrics"] = {
            "calls": {},
            "tokens_total": 0,
            "steps": {"extract":0,"law_search":0,"law_select":0,"law_verify":0,"naver":0,"strategy":0,"draft":0}
        }

def metrics_add(model_name: str, tokens_total: int | None = None):
    metrics_init()
    m = st.session_state["metrics"]
    m["calls"][model_name] = m["calls"].get(model_name, 0) + 1
    if tokens_total is not None:
        try:
            m["tokens_total"] += int(tokens_total)
        except Exception:
            pass

def step_inc(step: str):
    metrics_init()
    st.session_state["metrics"]["steps"][step] = st.session_state["metrics"]["steps"].get(step, 0) + 1

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
        self.key = g.get("GROQ_API_KEY")
        self.model_fast = g.get("GROQ_MODEL_FAST", "qwen/qwen3-32b")
        self.model_strict = g.get("GROQ_MODEL_STRICT", "llama-3.3-70b-versatile")
        self.client = None
        self.last_model = "N/A"

        if Groq and self.key:
            try:
                self.client = Groq(api_key=self.key)
            except Exception:
                self.client = None

    def _chat(self, model: str, messages: list[dict], temp: float, json_mode: bool):
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
            {"role": "system", "content": "Korean public administration assistant. Be practical, concise, and correct."},
            {"role": "user", "content": prompt},
        ]
        # 1) first
        try:
            return self._chat(model_first, messages, temp, json_mode=False)
        except Exception:
            pass
        # 2) fallback strict
        try:
            return self._chat(self.model_strict, messages, temp, json_mode=False)
        except Exception as e:
            return f"LLM Error: {e}"

    def generate_json(self, prompt: str, prefer: str = "fast", temp: float = 0.1, max_retry: int = 2) -> dict:
        if not self.client:
            return {}

        sys_json = "Output JSON only. No markdown. No explanation. Follow the schema exactly."
        messages = [
            {"role": "system", "content": sys_json},
            {"role": "user", "content": prompt},
        ]
        model_first = self.model_fast if prefer == "fast" else self.model_strict

        # 1) same model retry
        for _ in range(max_retry):
            try:
                txt = self._chat(model_first, messages, temp, json_mode=True)
                js = self._parse_json(txt)
                if js:
                    return js
            except Exception:
                pass

        # 2) upgrade to strict
        try:
            txt = self._chat(self.model_strict, messages, temp, json_mode=True)
            js = self._parse_json(txt)
            return js if js else {}
        except Exception:
            return {}

llm = LLMService()

# =========================
# 5) LAW API (DRF)
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
        self.enabled = bool(requests and xmltodict and self.oc)

    def search_law(self, query: str, display: int = 10) -> list[dict]:
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
                nm = it.get("법령명한글") or it.get("lawNm") or it.get("법령명") or ""
                mst = it.get("법령일련번호") or it.get("MST") or it.get("mst") or ""
                link = it.get("법령상세링크") or it.get("link") or ""
                out.append({
                    "law_name": clean_text(nm),
                    "mst": clean_text(mst),
                    "link": clean_text(link),
                })
            out = [x for x in out if x["law_name"]]
            return out
        except Exception:
            return []

    def get_article_text_by_mst(self, mst: str, article_no: str | None = None) -> str:
        if not self.enabled or not mst:
            return ""
        try:
            params = {"OC": self.oc, "target": "law", "type": "XML", "MST": mst}
            r = requests.get(self.service_url, params=params, timeout=9)
            r.raise_for_status()
            data = xmltodict.parse(r.text)

            law = data.get("Law") or data.get("law") or {}
            articles = law.get("Article", []) or []
            if isinstance(articles, dict):
                articles = [articles]

            # 조문번호 없으면 일부 텍스트만 반환(표시/LLM참조용)
            if not article_no:
                raw = clean_text(r.text)
                return raw[:4000]

            tgt = re.sub(r"[^0-9]", "", str(article_no))
            if not tgt:
                return ""

            for art in articles:
                if not isinstance(art, dict):
                    continue
                an = clean_text(art.get("@조문번호") or "")
                at = clean_text(art.get("ArticleTitle") or "")
                an_num = re.sub(r"[^0-9]", "", an)

                if tgt == an_num or (tgt and f"제{tgt}조" in at):
                    content = clean_text(art.get("ArticleContent") or "")
                    paras = art.get("Paragraph", [])
                    if isinstance(paras, dict):
                        paras = [paras]
                    p_text = "\n".join([clean_text(p.get("ParagraphContent")) for p in paras if isinstance(p, dict)])
                    return "\n".join([x for x in [at, content, p_text] if x]).strip()

            return ""
        except Exception:
            return ""

law_api = LawAPIService()

# =========================
# 6) NAVER Search (Optional)
# =========================
class NaverSearchService:
    """
    secrets.toml
    [naver]
    CLIENT_ID="..."
    CLIENT_SECRET="..."
    """
    BASE = "https://openapi.naver.com/v1/search"
    _PRO = [
        "법령","시행령","시행규칙","조문","판례","행정심판","행정소송","과태료","처분","사전통지",
        "의견제출","이의신청","불복","유권해석","질의회신","고시","훈령","예규","지침","매뉴얼","가이드",
        "법제처","국가법령정보","행정절차법","개인정보","보호법","요건","기준"
    ]
    _NONPRO = ["후기","맛집","일상","여행","다이어트","브이로그","내돈내산","감성","연애","육아","리뷰"]

    def __init__(self):
        n = st.secrets.get("naver", {})
        self.cid = n.get("CLIENT_ID")
        self.csec = n.get("CLIENT_SECRET")
        self.enabled = bool(requests and self.cid and self.csec)

    def _call(self, endpoint: str, query: str, display: int = 8, sort: str = "sim"):
        if not self.enabled or not query:
            return None
        try:
            url = f"{self.BASE}/{endpoint}.json"
            headers = {"X-Naver-Client-Id": self.cid, "X-Naver-Client-Secret": self.csec}
            params = {"query": query, "display": display, "start": 1, "sort": sort}
            r = requests.get(url, headers=headers, params=params, timeout=7)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    @classmethod
    def professional_score(cls, title: str, desc: str, link: str) -> int:
        t = (title or "") + " " + (desc or "")
        score = 0
        for k in cls._PRO:
            if k in t:
                score += 2
        if re.search(r"제?\s*\d+\s*조", t):
            score += 4
        if len(desc or "") >= 80:
            score += 1
        for k in cls._NONPRO:
            if k in t:
                score -= 4
        if re.search(r"[😂🤣😍😅]|ㅋㅋ|ㅎㅎ|ㅠㅠ", t):
            score -= 2
        if any(dom in (link or "") for dom in ["law.go.kr", "go.kr", "ac.kr", "korea.kr"]):
            score += 3
        return score

    def parse_items(self, data: dict, source: str) -> list[dict]:
        out = []
        if not data:
            return out
        for it in (data.get("items") or [])[:15]:
            title = clean_text(it.get("title", "")) or "(제목 없음)"
            desc = clean_text(it.get("description", "")) or clean_text(it.get("snippet", ""))
            link = clean_text(it.get("link", ""))
            out.append({"source": source, "title": title, "desc": truncate_text(desc, 320), "link": link})
        # dedup
        uniq, seen = [], set()
        for x in out:
            key = x["link"] or (x["source"] + "|" + x["title"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(x)
        return uniq

    def search_bundle(self, terms: list[str], primary_law: str, display_news=8, display_web=8, display_blog=12, display_cafe=12) -> list[dict]:
        if not self.enabled:
            return []
        step_inc("naver")

        terms = [t for t in terms if t]
        terms_q = " ".join(terms[:6]) if terms else ""
        if not terms_q:
            return []

        q_news = f"{terms_q} 행정처분 사례"
        q_web  = f"{terms_q} 법령 해설"
        q_blog = f"{terms_q} 실무 해설"
        q_cafe = f"{terms_q} 질의회신"

        news = self._call("news", q_news, display=display_news)
        webkr = self._call("webkr", q_web, display=display_web)
        blog = self._call("blog", q_blog, display=display_blog)
        cafe = self._call("cafearticle", q_cafe, display=display_cafe)

        items = []
        items += self.parse_items(news, "news")
        items += self.parse_items(webkr, "webkr")
        items += self.parse_items(blog, "blog")
        items += self.parse_items(cafe, "cafe")

        # relevance: 최소 2개 키워드 히트
        scored = []
        for x in items:
            t = (x["title"] or "") + " " + (x["desc"] or "")
            rel_hit = score_overlap(t, terms)
            rel_score = rel_hit * 3 - (6 if rel_hit < 2 else 0)

            pro_score = 0
            if x["source"] in ("blog","cafe"):
                pro_score = self.professional_score(x["title"], x["desc"], x["link"])

            x2 = dict(x)
            x2["rel_score"] = rel_score
            x2["pro_score"] = pro_score
            scored.append(x2)

        filtered = []
        for x in scored:
            if x["source"] in ("news","webkr"):
                if x["rel_score"] >= 6:
                    filtered.append(x)
            else:
                if x["rel_score"] >= 6 and x["pro_score"] >= 8:
                    filtered.append(x)

        filtered.sort(key=lambda z: (z.get("rel_score",0) + z.get("pro_score",0)*0.35), reverse=True)

        # cap per source
        caps = {"news":5,"webkr":5,"blog":3,"cafe":3}
        cnt = {k:0 for k in caps}
        out = []
        for x in filtered:
            s = x["source"]
            if s in caps and cnt[s] >= caps[s]:
                continue
            cnt[s] += 1
            out.append(x)
        return out

naver = NaverSearchService()

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

    def save_log(self, table: str, payload: dict) -> str:
        if not self.client:
            return "DB 미연결"
        try:
            safe_payload = json.loads(safe_json_dump(payload))
            self.client.table(table).insert(safe_payload).execute()
            return "저장 성공"
        except Exception as e:
            return f"저장 실패: {e}"

db = DatabaseService()

# =========================
# 8) Agentic Pipeline (Extractor -> Candidates -> Selector -> Verify -> Strategy -> Draft)
# =========================
def extractor_slots(user_input: str) -> dict:
    """
    FAST: 민원 서사 -> 슬롯 구조화
    """
    step_inc("extract")
    user_input = clean_text(user_input)
    kw = extract_keywords_kor(user_input, max_k=10)

    prompt = f"""
너는 '민원/업무 상황'을 법률 검토에 유리한 구조로 분해한다.
아래 스키마를 정확히 지켜 JSON만 출력:

{{
  "object": "대상(예: 건설기계/자동차/도로/주차/영업/복지조사 등)",
  "act": "핵심 행위(예: 방치/불법주차/차고지 외 주차/미이행/불법영업/과태료 이의 등)",
  "place": "장소(모르면 빈문자열)",
  "time": "시간/기간/반복(모르면 빈문자열)",
  "request": "요청/목표(예: 시정요청/단속요청/처분취소/안내문 작성 등)",
  "agency_scope": "담당자(지자체)가 할 수 있는 범위/처리유형(모르면 빈문자열)",
  "keywords": ["키워드1","키워드2","키워드3","키워드4","키워드5"]
}}

규칙:
- 내용 모르면 빈문자열.
- keywords는 '명사 위주', 5개 이하.
- 법령명/조문번호는 여기서 쓰지 마.

[민원]
{user_input}

[키워드 힌트(룰 기반)]
{kw[:10]}
"""
    js = llm.generate_json(prompt, prefer="fast", max_retry=2)
    if not js:
        return {
            "object": "",
            "act": "",
            "place": "",
            "time": "",
            "request": "",
            "agency_scope": "",
            "keywords": kw[:5],
        }

    # sanitize fields
    out = {}
    out["object"] = clean_text(js.get("object",""))
    out["act"] = clean_text(js.get("act",""))
    out["place"] = clean_text(js.get("place",""))
    out["time"] = clean_text(js.get("time",""))
    out["request"] = clean_text(js.get("request",""))
    out["agency_scope"] = clean_text(js.get("agency_scope",""))
    kws = js.get("keywords", [])
    if not isinstance(kws, list):
        kws = []
    kws = [clean_text(x) for x in kws if clean_text(x)]
    if not kws:
        kws = kw[:5]
    out["keywords"] = kws[:5]
    return out

def build_law_queries(slots: dict, user_input: str) -> list[str]:
    """
    룰 기반으로 법령 검색 쿼리 생성 (노이즈 줄이기)
    """
    base = []
    for k in (slots.get("keywords") or []):
        if k and k not in base:
            base.append(k)
    obj_ = slots.get("object","")
    act_ = slots.get("act","")
    req_ = slots.get("request","")

    combos = []
    if obj_ and act_:
        combos.append(f"{obj_} {act_}")
    if act_:
        combos.append(act_)
    if obj_:
        combos.append(obj_)
    if obj_ and req_:
        combos.append(f"{obj_} {req_}")
    if act_ and req_:
        combos.append(f"{act_} {req_}")

    # 키워드 1~2개 조합
    if len(base) >= 2:
        combos.append(f"{base[0]} {base[1]}")
    if base:
        combos.append(base[0])

    # 마지막 안전망: 입력에서 추출
    fallback = extract_keywords_kor(user_input, max_k=6)
    for f in fallback:
        combos.append(f)

    # dedup + trim
    out, seen = [], set()
    for q in combos:
        q = clean_text(q)
        if not q:
            continue
        if q in seen:
            continue
        seen.add(q)
        out.append(q)
        if len(out) >= 10:
            break
    return out

def law_candidate_pool(user_input: str, slots: dict, pool_size: int = 25) -> list[dict]:
    """
    DRF로 후보풀 확장 (LLM 없이도 작동)
    """
    step_inc("law_search")
    queries = build_law_queries(slots, user_input)
    pool = []
    for q in queries[:8]:
        pool += law_api.search_law(q, display=10)

    # dedup (mst+law_name)
    uniq, seen = [], set()
    for x in pool:
        key = (x.get("mst","") + "|" + x.get("law_name","")).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(x)
        if len(uniq) >= pool_size:
            break
    return uniq

def law_selector_from_pool(user_input: str, slots: dict, pool: list[dict]) -> dict:
    """
    STRICT: 후보 목록에서만 대표법령/연관법령 선택 + 조문 '숫자만'(가능하면)
    """
    step_inc("law_select")
    if not pool:
        return {
            "primary": None,
            "related": [],
            "article_no": "",
            "status": "FAIL",
            "reason": "후보풀 없음",
        }

    # 후보 텍스트 (짧게)
    lines = []
    for i, it in enumerate(pool[:25], 1):
        lines.append(f"{i}. {it.get('law_name')} (MST={it.get('mst')})")
    pool_text = "\n".join(lines)

    prompt = f"""
너는 대한민국 행정 실무 '법령 선택기'다.
아래 [후보 목록] 중에서만 선택한다. 절대 목록 밖 법령을 만들지 마.

반드시 JSON만 출력:
{{
  "pick": {{
    "primary_idx": 1,
    "related_idx": [2,3],
    "article_no": "조번호(숫자만, 모르면 빈문자열)"
  }},
  "reason": "한 문장"
}}

[민원 원문]
{clean_text(user_input)}

[슬롯 요약]
- 대상(object): {slots.get("object","")}
- 행위(act): {slots.get("act","")}
- 요청(request): {slots.get("request","")}
- 처리범위(scope): {slots.get("agency_scope","")}
- 키워드: {slots.get("keywords",[])}

[후보 목록]
{pool_text}

규칙:
- primary_idx는 1~{min(len(pool),25)} 중 1개
- related_idx는 0~2개(없으면 [])
- article_no는 숫자만. 확신 없으면 빈문자열.
"""
    js = llm.generate_json(prompt, prefer="strict", max_retry=2) or {}
    pick = js.get("pick", {})
    if not isinstance(pick, dict):
        pick = {}

    pidx = pick.get("primary_idx")
    rids = pick.get("related_idx", [])
    art = clean_text(pick.get("article_no",""))

    # normalize indices
    def valid_idx(x):
        return isinstance(x, int) and 1 <= x <= len(pool)

    primary = pool[pidx-1] if valid_idx(pidx) else pool[0]
    related = []

    if isinstance(rids, list):
        for rid in rids:
            if valid_idx(rid):
                cand = pool[rid-1]
                if cand not in related and cand != primary:
                    related.append(cand)
            if len(related) >= 2:
                break

    # ensure at least 3 total when possible
    all_list = [primary] + related
    for it in pool:
        if it not in all_list:
            all_list.append(it)
        if len(all_list) >= 3:
            break

    # article number (digits only)
    art_num = re.sub(r"[^0-9]", "", art)

    return {
        "primary": all_list[0] if all_list else None,
        "related": all_list[:3],
        "article_no": art_num,
        "status": "OK" if all_list else "FAIL",
        "reason": clean_text(js.get("reason","")),
    }

def law_verify_fetch(primary: dict | None, article_no: str) -> dict:
    """
    원문 확보 기준으로 CONFIRMED/WEAK 결정
    """
    step_inc("law_verify")
    if not primary:
        return {"status": "FAIL", "legal_basis": "관련 법령 검색 실패(후보 없음).", "article_text": "", "link": ""}

    nm = clean_text(primary.get("law_name",""))
    mst = clean_text(primary.get("mst",""))
    link = clean_text(primary.get("link",""))

    # 조문 원문 시도
    article_text = ""
    if mst:
        article_text = law_api.get_article_text_by_mst(mst, article_no if article_no else None)

    article_text = clean_text(article_text)

    if article_no and article_text and len(article_text) >= 40:
        legal_basis = f"{nm} 제{article_no}조\n{truncate_text(article_text, 2600)}"
        return {"status": "CONFIRMED", "legal_basis": legal_basis, "article_text": article_text, "link": link, "mst": mst, "law_name": nm}
    if (not article_no) and article_text and len(article_text) >= 60:
        legal_basis = f"{nm}\n{truncate_text(article_text, 2600)}"
        return {"status": "WEAK", "legal_basis": legal_basis, "article_text": article_text, "link": link, "mst": mst, "law_name": nm}

    # 원문 확보 실패
    return {"status": "FAIL", "legal_basis": f"법령({nm})은 찾았으나 조문 원문 확보 실패.", "article_text": "", "link": link, "mst": mst, "law_name": nm}

def strategy_agent(user_input: str, slots: dict, law_pack: dict, naver_items: list[dict]) -> str:
    """
    FAST 기본. 법령이 FAIL이면 STRICT로 승급.
    """
    step_inc("strategy")
    prefer = "fast" if law_pack.get("status") == "CONFIRMED" else "strict"

    brief = []
    for it in (naver_items or [])[:8]:
        brief.append(f"- [{it.get('source')}] {it.get('title')}: {it.get('desc')}")
    brief_block = "\n".join(brief) if brief else "(검색 결과 없음)"

    prompt = f"""
[출력 제약]
- 인삿말/자기소개 금지. 바로 본문.
- 과도한 일반론 금지. 본 민원과 법령/절차에 직접 연결된 문장만.
- 아래 3개 항목만, 마크다운.

[민원]
{clean_text(user_input)}

[슬롯]
- 대상: {slots.get("object","")}
- 행위: {slots.get("act","")}
- 요청: {slots.get("request","")}
- 처리범위: {slots.get("agency_scope","")}
- 키워드: {slots.get("keywords",[])}

[법적 근거 상태]
{law_pack.get("status")}

[법적 근거(확보 범위)]
{law_pack.get("legal_basis")}

[네이버(보조)]
{truncate_text(brief_block, 1100)}

1. **처리 방향**
2. **핵심 체크리스트**
3. **예상 민원/반발 및 대응**
"""
    return llm.generate_text(prompt, prefer=prefer, temp=0.1)

def draft_agent(dept: str, officer: str, user_input: str, slots: dict, law_pack: dict, strategy: str) -> dict:
    """
    STRICT: 공문 JSON 생성
    법령 FAIL/WEAK이면 '추가 확인 필요' 문구 강제
    """
    step_inc("draft")
    today_str = datetime.now().strftime("%Y. %m. %d.")
    doc_num = f"행정-{datetime.now().strftime('%Y')}-{int(time.time()) % 10000:04d}호"

    law_status = law_pack.get("status","FAIL")
    caution = ""
    if law_status != "CONFIRMED":
        caution = "※ 본 문서의 법적 근거는 전산조회/원문확보 한계로 '추가 확인 필요'가 포함되어야 한다."

    prompt = f"""
아래 스키마로만 JSON 출력(키 추가 금지):
{{
  "title": "문서 제목",
  "receiver": "수신",
  "body_paragraphs": ["문단1","문단2","문단3","문단4"],
  "department_head": "발신 명의"
}}

작성 정보:
- 부서: {clean_text(dept)}
- 담당자: {clean_text(officer)}
- 시행일: {today_str}
- 문서번호: {doc_num}

민원/업무 상황:
{clean_text(user_input)}

슬롯 요약:
- 대상: {slots.get("object","")}
- 행위: {slots.get("act","")}
- 요청: {slots.get("request","")}
- 처리범위: {slots.get("agency_scope","")}
- 키워드: {slots.get("keywords",[])}

법적 근거 상태: {law_status}
법적 근거(확보 범위):
{law_pack.get("legal_basis")}

처리 전략(요약):
{truncate_text(clean_text(strategy), 1000)}

필수 원칙:
- 문서 톤: 건조/정중, 불필요한 수사 금지
- 본문 구조: [경위] -> [근거] -> [조치/안내] -> [권리구제/문의]
- 개인정보는 OOO로 마스킹(있으면)
- 법령 원문이 불확실하면 반드시 "추가 확인 필요" 또는 "전산 확인 결과" 표현을 포함
{caution}
"""
    js = llm.generate_json(prompt, prefer="strict", max_retry=2)
    doc = ensure_doc_shape(js)
    # 메타는 별도로 반환
    return {"doc": doc, "meta": {"doc_num": doc_num, "today": today_str}}

# =========================
# 9) Rendering
# =========================
def badge(text: str, kind: str = "ok") -> str:
    cls = "badge badge-ok" if kind == "ok" else ("badge badge-warn" if kind == "warn" else "badge badge-bad")
    return f"<span class='{cls}'>{escape(text)}</span>"

def render_precedents(items: list[dict]):
    if not items:
        st.info("관련 검색 결과가 없습니다.")
        return

    def src_label(src: str) -> str:
        return {"news":"뉴스","webkr":"웹문서","blog":"블로그(필터)","cafe":"카페(필터)"}.get(src, src or "검색")

    for it in items[:16]:
        src = clean_text(it.get("source",""))
        title = clean_text(it.get("title",""))
        desc = clean_text(it.get("desc",""))
        link = clean_text(it.get("link",""))
        rel = it.get("rel_score")
        pro = it.get("pro_score")

        st.markdown("<div class='item-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='item-title'>[{escape(src_label(src))}] {escape(title)}</div>", unsafe_allow_html=True)

        meta = []
        if isinstance(rel, int):
            meta.append(f"rel={rel}")
        if isinstance(pro, int) and src in ("blog","cafe"):
            meta.append(f"pro={pro}")
        if meta:
            st.markdown(f"<div class='item-meta'>{escape(' | '.join(meta))}</div>", unsafe_allow_html=True)

        st.markdown(f"<div class='item-desc'>{escape(desc)}</div>", unsafe_allow_html=True)
        if link.startswith("http"):
            st.link_button("열기", link, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

def render_metrics():
    m = st.session_state.get("metrics", {})
    calls = m.get("calls", {})
    steps = m.get("steps", {})
    tokens_total = m.get("tokens_total", 0)

    st.subheader("📊 사용량(세션 기준)")
    if calls:
        for k, v in sorted(calls.items(), key=lambda x: (-x[1], x[0])):
            st.write(f"- **{k}**: {v}회")
        st.caption(f"총 토큰(가능한 경우): {tokens_total}")
    else:
        st.info("호출 기록이 없습니다.")

    st.markdown("#### 🧩 단계별 실행 횟수")
    st.json(steps)

# =========================
# 10) Workflow Orchestrator
# =========================
def run_workflow(user_input: str, dept: str, officer: str, use_naver: bool):
    log_area = st.empty()
    logs = []

    def add_log(msg: str, style: str = "sys"):
        logs.append(f"<div class='agent-log log-{style}'>{safe_html(msg)}</div>")
        log_area.markdown("".join(logs), unsafe_allow_html=True)
        time.sleep(0.06)

    user_input = clean_text(user_input)

    # A) Extractor (FAST)
    add_log("🧠 [Extractor] 민원 서사를 슬롯으로 분해(FAST: qwen/qwen3-32b)...", "extract")
    slots = extractor_slots(user_input)

    # B) Candidate Law Search (DRF)
    add_log("📚 [LawSearch] DRF로 후보 법령 풀 생성(룰 기반 쿼리 + 다중 검색)...", "law")
    pool = law_candidate_pool(user_input, slots, pool_size=30)

    # C) Law Selector (STRICT, 후보 중 선택)
    add_log("🎯 [LawSelect] 후보 목록에서만 대표/연관 법령 선택(STRICT: llama-3.3-70b)...", "law")
    selection = law_selector_from_pool(user_input, slots, pool)

    primary = selection.get("primary")
    related = selection.get("related", []) or []
    article_no = selection.get("article_no","")

    # D) Verify / Fetch article text
    add_log("✅ [Verify] 대표 법령 원문(조문) 확보로 신뢰도 확정...", "verify")
    law_ver = law_verify_fetch(primary, article_no)

    # 최종 법령 팩
    law_pack = {
        "status": law_ver.get("status","FAIL"),
        "primary": primary,
        "related": related[:3],
        "article_no": article_no,
        "legal_basis": law_ver.get("legal_basis",""),
        "article_text": law_ver.get("article_text",""),
        "reason": selection.get("reason",""),
    }

    # E) Naver (optional)
    naver_items = []
    if use_naver and naver.enabled:
        add_log("🔎 [Naver] 유사 사례/해설 검색(보조, 필터 적용)...", "search")
        terms = []
        terms += [slots.get("object",""), slots.get("act",""), slots.get("request","")]
        terms += (slots.get("keywords") or [])
        terms = [t for t in [clean_text(x) for x in terms] if t]
        # dedup
        uniq = []
        seen = set()
        for t in terms:
            if t in seen:
                continue
            seen.add(t)
            uniq.append(t)
        naver_items = naver.search_bundle(uniq[:12], clean_text(primary.get("law_name","") if primary else ""))
    else:
        add_log("🔎 [Naver] OFF (비활성/키 없음/사용자 옵션)", "search")

    # F) Strategy
    add_log("🧭 [Strategy] 처리 방향/체크리스트/대응 수립...", "strat")
    strategy = strategy_agent(user_input, slots, law_pack, naver_items)

    # G) Draft (STRICT)
    add_log("✍️ [Draft] 공문서 JSON 생성(STRICT: llama-3.3-70b)...", "draft")
    drafted = draft_agent(dept, officer, user_input, slots, law_pack, strategy)
    doc = drafted["doc"]
    meta = drafted["meta"]

    # H) Save
    add_log("💾 [Save] Supabase 저장...", "sys")
    payload = {
        "created_at": datetime.now().isoformat(),
        "dept": clean_text(dept),
        "officer": clean_text(officer),
        "input": user_input,
        "slots": safe_json_dump(slots),
        "law_status": law_pack["status"],
        "law_primary": safe_json_dump(primary or {}),
        "law_related": safe_json_dump(related[:3]),
        "law_article_no": article_no,
        "legal_basis": law_pack["legal_basis"],
        "strategy": strategy,
        "naver_items": safe_json_dump(naver_items),
        "final_doc": safe_json_dump(doc),
        "model_last": llm.last_model,
        "metrics": safe_json_dump(st.session_state.get("metrics", {})),
    }
    db_msg = db.save_log("law_logs", payload)  # 테이블명: law_logs (원하는 걸로 바꾸면 됨)
    add_log(f"✅ 완료 ({db_msg})", "sys")

    time.sleep(0.35)
    log_area.empty()

    return {
        "slots": slots,
        "pool_count": len(pool),
        "law_pack": law_pack,
        "naver_items": naver_items,
        "strategy": strategy,
        "doc": doc,
        "meta": meta,
        "db_msg": db_msg,
    }

# =========================
# 11) UI
# =========================
def main():
    st.session_state.setdefault("dept", "OO시청 OO과")
    st.session_state.setdefault("officer", "김주무관")
    st.session_state.setdefault("use_naver", True)

    col_l, col_r = st.columns([1, 1.2], gap="large")

    with col_l:
        st.title("🏛️ AI 행정관 Pro")
        st.caption("Agentic v6.0 — Extractor → Candidate Pool → Selector(후보중선택) → Verify(원문확보) → Draft")
        st.markdown("---")

        with st.expander("📝 사용자 정보 / 옵션", expanded=False):
            st.text_input("부서명", key="dept")
            st.text_input("담당자", key="officer")
            st.checkbox("네이버 검색 사용(보조)", key="use_naver")

        user_input = st.text_area(
            "업무 지시 사항(민원 상황)",
            height=220,
            placeholder="예: 무단방치차량 강제처리 절차 안내 공문 작성\n예: 건설기계 차고지 외 주차(주기위반) 민원 답변문 작성",
        )

        if st.button("🚀 실행", type="primary", use_container_width=True):
            if not clean_text(user_input):
                st.warning("내용을 입력하세요.")
            else:
                with st.spinner("에이전트 파이프라인 실행 중..."):
                    try:
                        res = run_workflow(
                            user_input=user_input,
                            dept=st.session_state["dept"],
                            officer=st.session_state["officer"],
                            use_naver=st.session_state["use_naver"],
                        )
                        st.session_state["result"] = res
                    except Exception as e:
                        st.error(f"치명적 오류: {e}")

        st.markdown("---")
        render_metrics()

        st.markdown(
            "<div class='small-muted'>"
            "TIP: 성능이 튀면(법령 엉뚱) → 후보풀/선택/원문확보 3단계가 방어합니다. "
            "조문 원문 확보 실패 시 기안에 '추가 확인 필요'가 자동 포함됩니다."
            "</div>",
            unsafe_allow_html=True,
        )

    with col_r:
        res = st.session_state.get("result")

        if not res:
            st.markdown(
                """
<div style='text-align: center; padding: 120px 20px; color: #aaa; border: 2px dashed #ddd; border-radius: 12px; background:#fff;'>
  <h3>📄 Document Preview</h3>
  <p>왼쪽에서 민원 상황을 입력하고 실행하세요.<br>법령 후보풀/검증 후 공문이 생성됩니다.</p>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            doc = res["doc"]
            meta = res["meta"]
            law_pack = res["law_pack"]

            tab1, tab2 = st.tabs(["📄 공문서", "🔍 근거/분석"])

            with tab1:
                body_html = "".join([f"<p style='margin:0 0 14px 0;'>{safe_html(p)}</p>" for p in doc["body_paragraphs"]])
                html = f"""
<div class="paper-sheet">
  <div class="stamp">직인생략</div>
  <div class="doc-header">{safe_html(doc['title'])}</div>
  <div class="doc-info">
    <span>문서번호: {safe_html(meta['doc_num'])}</span>
    <span>시행일자: {safe_html(meta['today'])}</span>
    <span>수신: {safe_html(doc['receiver'])}</span>
  </div>
  <div class="doc-body">{body_html}</div>
  <div class="doc-footer">{safe_html(doc['department_head'])}</div>
</div>
"""
                components.html(html, height=880, scrolling=True)

            with tab2:
                # 상태 뱃지
                st.markdown(
                    badge(f"DB: {clean_text(res.get('db_msg',''))}", "ok" if "성공" in (res.get("db_msg","")) else "warn")
                    + badge(f"법령상태: {clean_text(law_pack.get('status'))}", "ok" if law_pack.get("status")=="CONFIRMED" else ("warn" if law_pack.get("status")=="WEAK" else "bad"))
                    + badge(f"후보풀: {res.get('pool_count',0)}건", "ok"),
                    unsafe_allow_html=True,
                )

                st.markdown("### 🧩 슬롯(Extractor 결과)")
                st.json(res.get("slots", {}))

                st.markdown("### 📜 법적 근거(확보 범위)")
                st.info(law_pack.get("legal_basis",""))

                st.markdown("### 🧭 처리 전략")
                st.markdown(res.get("strategy",""))

                if st.session_state.get("use_naver"):
                    st.markdown("### 🔎 네이버(보조) — 관련성/전문성 필터")
                    render_precedents(res.get("naver_items", []))
                else:
                    st.caption("네이버 검색 OFF")

                with st.expander("🛠️ 디버그(법령 선택/원문확보)", expanded=False):
                    dbg = {
                        "status": law_pack.get("status"),
                        "article_no": law_pack.get("article_no"),
                        "primary": law_pack.get("primary"),
                        "related": law_pack.get("related"),
                        "selector_reason": law_pack.get("reason"),
                    }
                    st.code(safe_json_dump(dbg), language="json")


if __name__ == "__main__":
    main()
