import streamlit as st
import streamlit.components.v1 as components

import json
import re
import time
from datetime import datetime, timedelta
from html import escape, unescape
from urllib.parse import urlparse

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
st.set_page_config(layout="wide", page_title="AI Bureau: Legal Glass (TGD Ops-Final v3)", page_icon="⚖️")

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
  position:absolute; bottom:85px; right:80px; border:3px solid #c00; color: #c00;
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
.small { font-size: 0.9rem; color:#6b7280; }
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

def truncate_text(s: str, max_chars: int = 3800) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n…(결과가 길어 일부 생략됨)"

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

    # 태그 잔재 제거
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
# 2) Metrics (세션 기준)
# =========================
MODEL_PRICES_PER_1M = {
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
            "calls": {},
            "tokens_est": {},
            "cost_est": {},
            "timing": [],
        }

def metrics_add(model_name: str, prompt: str, output: str):
    m = st.session_state["metrics"]
    m["calls"][model_name] = m["calls"].get(model_name, 0) + 1
    t = estimate_tokens(prompt) + estimate_tokens(output)
    m["tokens_est"][model_name] = m["tokens_est"].get(model_name, 0) + t
    price = MODEL_PRICES_PER_1M.get(model_name, 0.0)
    m["cost_est"][model_name] = m["cost_est"].get(model_name, 0.0) + (t / 1_000_000) * price


# =========================
# 3) JSON Parsing (강화)
# =========================
def parse_first_json_object(text: str) -> dict:
    """
    LLM이 JSON 앞뒤에 문구를 붙이는 상황을 최대한 안전하게 파싱.
    - 1) ``` 제거
    - 2) 가장 바깥 { ... } 범위를 find/rfind로 잡아 시도
    - 3) 실패 시 JSONDecoder.raw_decode로 스캔
    """
    if not text:
        return {}

    raw = re.sub(r"```(?:json)?\s*|\s*```", "", text, flags=re.IGNORECASE).strip()

    # 1) find/rfind로 가장 바깥 {}를 잡아 시도
    try:
        s = raw.find("{")
        e = raw.rfind("}")
        if s != -1 and e != -1 and e > s:
            candidate = raw[s:e+1].strip()
            return json.loads(candidate)
    except Exception:
        pass

    # 2) raw_decode 스캔 (중간에 JSON이 있어도 찾음)
    try:
        dec = json.JSONDecoder()
        for i in range(len(raw)):
            if raw[i] == "{":
                obj, end = dec.raw_decode(raw[i:])
                if isinstance(obj, dict):
                    return obj
    except Exception:
        pass

    return {}


# =========================
# 4) LLM Service (70B 고정)
# =========================
class LLMService:
    def __init__(self):
        self.groq_key = st.secrets.get("general", {}).get("GROQ_API_KEY")
        self.last_model_used = "N/A"
        self.groq_client = Groq(api_key=self.groq_key) if self.groq_key else None

    def generate_text(self, prompt: str, temperature: float = 0.1) -> str:
        last_err = None
        if self.groq_client:
            try:
                completion = self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
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
        obj = parse_first_json_object(raw)
        return obj if isinstance(obj, dict) else {}

llm_service = LLMService()


# =========================
# helpers
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

def make_law_query_candidates(law_name: str, keywords: list) -> list:
    """
    ✅ suffix 중복 방지 강화:
    - '법/령/규칙'으로 끝나면 추가 suffix 붙이지 않음
    """
    law_name = norm_space(law_name)
    keywords = keywords if isinstance(keywords, list) else []

    suffixes = ["법", "령", "규칙"]

    cands = []
    if law_name:
        cands += [law_name, law_name.replace(" ", "")]
        if not any(law_name.endswith(suf) for suf in suffixes):
            cands += [law_name + "법"]

    for kw in keywords[:6]:
        kw = norm_space(kw)
        if not kw:
            continue
        cands += [kw, kw.replace(" ", "")]
        if not any(kw.endswith(suf) for suf in suffixes) and len(kw) <= 10:
            cands += [kw + "법"]

    seen = set()
    out = []
    for x in cands:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out[:12]


# =========================
# 5) LAW API Service (enabled 2중 안전)
# =========================
class LawAPIService:
    def __init__(self):
        self.enabled = False
        try:
            self.oc = st.secrets["law"]["LAW_API_ID"]
            self.base_url = st.secrets["law"].get("BASE_URL", "https://www.law.go.kr/DRF/lawService.do")
            self.enabled = (requests is not None) and (xmltodict is not None) and bool(self.oc)
        except Exception:
            self.enabled = False
        self._law_xml_cache = {}

    def _call_xml(self, params: dict) -> dict:
        if not self.enabled or requests is None or xmltodict is None:
            return {}
        r = requests.get(self.base_url, params=params, timeout=15)
        r.raise_for_status()
        return xmltodict.parse(r.text)

    def _as_list(self, x):
        if x is None: return []
        if isinstance(x, list): return x
        return [x]

    def search_law_candidates(self, query: str, display: int = 20) -> list:
        if not self.enabled or not query:
            return []
        # 검색어 정제
        clean_query = re.sub(r"[^가-힣a-zA-Z0-9 ]", "", query).strip()
        data = self._call_xml({
            "OC": self.oc, 
            "target": "law", 
            "type": "XML",
            "query": clean_query, 
            "display": max(1, min(display, 50)),
        })
        try:
            law = data.get("LawSearch", {}).get("law")
            if not law: return []
            if isinstance(law, dict): law = [law]
            out = []
            for item in law:
                if not isinstance(item, dict): continue
                out.append({
                    "law_id": item.get("lawId", ""),
                    "law_name": item.get("lawNm", ""),
                    "law_type": item.get("lawType", ""),
                })
            return out
        except Exception:
            return []

    def choose_best_law(self, candidates: list, query: str) -> dict:
        """들여쓰기 오류가 나지 않도록 정렬된 메서드"""
        q = norm_space(query).replace(" ", "")
        if not candidates:
            return {}

        def score(item):
            name = norm_space(item.get("law_name", ""))
            n2 = name.replace(" ", "")
            s = 0
            if not name: return -999
            if q and q in n2: s += 50
            if "시행령" in name: s -= 2
            if "시행규칙" in name: s -= 2
            s -= max(0, len(name) - 12) * 0.2
            return s

        best = sorted(candidates, key=score, reverse=True)[0]
        return best if best.get("law_id") else {}

    def get_law_xml(self, law_id: str) -> dict:
        if not self.enabled or not law_id: return {}
        if law_id in self._law_xml_cache: return self._law_xml_cache[law_id]
        data = self._call_xml({"OC": self.oc, "target": "law", "type": "XML", "ID": law_id})
        self._law_xml_cache[law_id] = data
        return data

    def _article_no_display(self, art: dict) -> str:
        try:
            raw = clean_text(art.get("@조문번호", ""))
            d = re.sub(r"[^0-9]", "", raw)
            if not d:
                title = clean_text(art.get("ArticleTitle") or art.get("title") or "")
                m = re.search(r"제\s*([0-9]+)\s*조", title)
                return f"제{m.group(1)}조" if m else ""
            num = int(d[:4]) if len(d) >= 4 else int(d)
            return f"제{num}조"
        except Exception: return ""

    def build_article_fulltext(self, art: dict) -> str:
        try:
            title = clean_text(art.get("ArticleTitle") or art.get("title") or "")
            content = clean_text(art.get("ArticleContent") or art.get("content") or "")
            paragraphs = self._as_list(art.get("Paragraph"))
            p_texts = []
            for p in paragraphs:
                if not isinstance(p, dict): continue
                pno = clean_text(p.get("@항번호", "")) or clean_text(p.get("ParagraphNo", ""))
                ptxt = clean_text(p.get("ParagraphContent", ""))
                if not ptxt: continue
                prefix = (to_circled(pno) + " ") if pno else ""
                p_texts.append(f"{prefix}{ptxt}")
            joined = "\n".join([x for x in [title, content, "\n".join(p_texts)] if x])
            return clean_text(joined)
        except Exception: return ""

    def extract_article_text(self, law_xml: dict, article_no: str) -> str:
        if not self.enabled or not law_xml or not article_no: return ""
        try:
            target_num = only_digits(article_no)
            if not target_num: return ""
            articles = self._as_list(law_xml.get("Law", {}).get("Article", []))
            for art in articles:
                if not isinstance(art, dict): continue
                curr_art_no = clean_text(art.get("@조문번호", ""))
                title = clean_text(art.get("ArticleTitle") or art.get("title") or "")
                title_hit = (target_num in only_digits(title)) or (target_num in title)
                no_hit = (target_num in curr_art_no)
                if title_hit or no_hit:
                    return self.build_article_fulltext(art)
        except Exception: pass
        return ""

    def find_article_candidates(self, law_xml: dict, keywords: list, topk: int = 5) -> list:
        if not self.enabled or not law_xml: return []
        keywords = [clean_text(k) for k in (keywords or []) if clean_text(k)]
        if not keywords: return []
        articles = self._as_list(law_xml.get("Law", {}).get("Article", []))
        scored = []
        for art in articles:
            if not isinstance(art, dict): continue
            title = clean_text(art.get("ArticleTitle") or art.get("title") or "")
            fulltext = self.build_article_fulltext(art)
            if not fulltext: continue
            text_low = fulltext.lower()
            score = 0
            for kw in keywords:
                kw_low = kw.lower()
                if kw_low in title.lower(): score += 8
                if kw_low in text_low: score += 3
            if score <= 0: continue
            scored.append({
                "article_no": self._article_no_display(art),
                "title": title,
                "score": score,
                "text_excerpt": fulltext[:220] + "...",
                "fulltext": fulltext,
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:topk]

law_api = LawAPIService()

# =========================
# 6) Naver Search + Evidence Risk Tagging (enabled 2중 안전)
# =========================
class NaverSearchService:
    def __init__(self):
        self.client_id = st.secrets.get("naver", {}).get("CLIENT_ID")
        self.client_secret = st.secrets.get("naver", {}).get("CLIENT_SECRET")
        self.enabled = bool(self.client_id and self.client_secret and requests is not None)

    def _req(self, endpoint: str, params: dict):
        if not self.enabled or requests is None:
            return {}
        url = f"https://openapi.naver.com/v1/search/{endpoint}.json"
        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
        }
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def search(self, query: str, category: str = "news", display: int = 5, sort: str = "date") -> list:
        if not self.enabled or not query:
            return []
        category = (category or "news").strip()
        if category not in ["news", "blog", "kin", "webkr"]:
            category = "news"
        if category != "news" and sort == "date":
            sort = "sim"
        params = {"query": query, "display": max(1, min(int(display), 10)), "sort": sort}
        try:
            data = self._req(category, params)
            items = data.get("items", []) or []
            out = []
            for it in items:
                title = clean_text(it.get("title", ""))
                link = clean_text(it.get("link", ""))
                desc = clean_text(it.get("description", ""))
                pub = clean_text(it.get("pubDate", "")) if category == "news" else ""
                out.append({
                    "source": f"naver:{category}",
                    "title": title,
                    "description": desc,
                    "link": link,
                    "pubDate": pub,
                })
            return out
        except Exception:
            return []

naver_search = NaverSearchService()

def _norm_key(s: str) -> str:
    s = norm_space(s).lower()
    s = re.sub(r"[^0-9a-z가-힣 ]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

_AD_PATTERNS = ["협찬", "광고", "체험단", "원고료", "홍보", "제공받", "파트너", "프로모션"]
_PERSONAL_PATTERNS = ["후기", "경험", "일기", "리뷰", "내돈내산", "썰", "느낌", "개인", "주관"]

def tag_evidence_item(it: dict) -> dict:
    src = it.get("source", "")
    title = norm_space(it.get("title", ""))
    desc = norm_space(it.get("description", ""))
    link = norm_space(it.get("link", ""))

    score = 0.5
    tags = []

    if "naver:news" in src:
        score = 0.85
    elif "naver:blog" in src:
        score = 0.55
        tags.append("BLOG")
    elif "naver:kin" in src:
        score = 0.45
        tags.append("KIN")
    else:
        tags.append("OTHER")

    try:
        host = urlparse(link).netloc.lower()
        if "news.naver.com" in host:
            score = max(score, 0.9)
            tags.append("NAVER_NEWS")
        if "blog.naver.com" in host:
            tags.append("NAVER_BLOG")
        if "kin.naver.com" in host:
            tags.append("NAVER_KIN")
    except Exception:
        pass

    hay = f"{title} {desc}"
    if any(p in hay for p in _AD_PATTERNS):
        score -= 0.25
        tags.append("AD_RISK")

    if any(p in hay for p in _PERSONAL_PATTERNS):
        score -= 0.12
        tags.append("PERSONAL_RISK")

    if len(desc) < 25:
        score -= 0.05
        tags.append("THIN")

    score = max(0.05, min(0.95, score))

    if score >= 0.8:
        level = "HIGH"
    elif score >= 0.55:
        level = "MED"
    else:
        level = "LOW"

    it2 = dict(it)
    it2["quality_score"] = round(score, 2)
    it2["quality_level"] = level
    it2["risk_tags"] = sorted(list(set(tags)))
    return it2

def normalize_evidence(items: list, max_items: int = 12) -> list:
    if not isinstance(items, list):
        return []
    seen = set()
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        title = norm_space(it.get("title", ""))
        link = norm_space(it.get("link", ""))
        if not title or not link:
            continue
        key = _norm_key(title)
        if key in seen:
            continue
        seen.add(key)
        out.append(tag_evidence_item({
            "source": it.get("source", ""),
            "title": title,
            "description": norm_space(it.get("description", "")),
            "link": link,
            "pubDate": norm_space(it.get("pubDate", "")),
        }))
        if len(out) >= max_items:
            break
    return out

def evidence_to_text(evidence: list) -> str:
    lines = []
    for it in evidence:
        pub = f" ({it.get('pubDate')})" if it.get("pubDate") else ""
        q = f"[{it.get('quality_level')}/{it.get('quality_score')}]"
        tags = it.get("risk_tags", [])
        tag_txt = f" tags={','.join(tags)}" if tags else ""
        lines.append(
            f"- {q} [{it.get('source')}] {it.get('title')}{pub}{tag_txt}\n"
            f"  - {it.get('description')}\n"
            f"  - {it.get('link')}"
        )
    return "\n".join(lines) if lines else "검색 결과가 없습니다."

def evidence_quality_summary(evidence: list) -> dict:
    if not evidence:
        return {"count": 0, "high": 0, "med": 0, "low": 0, "avg_score": 0.0, "top_tags": []}
    cnt = len(evidence)
    high = sum(1 for x in evidence if x.get("quality_level") == "HIGH")
    med = sum(1 for x in evidence if x.get("quality_level") == "MED")
    low = sum(1 for x in evidence if x.get("quality_level") == "LOW")
    avg = sum(float(x.get("quality_score", 0.0)) for x in evidence) / max(1, cnt)
    tag_count = {}
    for x in evidence:
        for t in (x.get("risk_tags") or []):
            tag_count[t] = tag_count.get(t, 0) + 1
    top_tags = [k for k, _ in sorted(tag_count.items(), key=lambda kv: -kv[1])[:6]]
    return {"count": cnt, "high": high, "med": med, "low": low, "avg_score": round(avg, 2), "top_tags": top_tags}


# =========================
# 7) Supabase (스키마 불일치 자동 축소 저장)
# =========================
class DatabaseService:
    def __init__(self):
        self.is_active = False
        self.client = None
        if create_client is None:
            return
        try:
            self.url = st.secrets["supabase"]["SUPABASE_URL"]
            self.key = st.secrets["supabase"]["SUPABASE_KEY"]
            self.client = create_client(self.url, self.key)
            self.is_active = True
        except Exception:
            self.is_active = False

    def _attempt_insert(self, data: dict):
        return self.client.table("law_logs").insert(data).execute()

    def save_log(self, payload: dict):
        """
        ✅ 개선 포인트:
        - 확장 컬럼이 DB에 없으면 PostgREST 에러가 나므로
        - 에러 메시지에 따라 "unknown column"로 추정되는 필드를 순차 제거하고 재시도
        - 최종 실패 시 최소 컬럼(slim)로 저장 시도
        """
        if not self.is_active:
            return "DB 미연결 (저장 건너뜀)"

        base = dict(payload or {})
        base["created_at"] = base.get("created_at") or datetime.now().isoformat()

        # 1) 전체 insert 시도
        try:
            self._attempt_insert(base)
            return "DB 저장 성공"
        except Exception as e1:
            msg = str(e1)

        # 2) 스키마 불일치 추정: 안전 축소(몇 개 필드 제거)
        #   - 확장 필드 목록(없어도 무방)
        shrink_order = [
            "evidence_quality", "provenance", "legal_status",
            "task_type", "dept", "officer",
            "model_usage", "timing",
        ]
        temp = dict(base)
        for k in shrink_order:
            if k in temp:
                try:
                    temp.pop(k, None)
                    self._attempt_insert(temp)
                    return f"DB 저장 성공(축소 모드: {k} 제거)"
                except Exception:
                    continue

        # 3) 최후: 최소 컬럼만
        try:
            slim = {
                "input_text": base.get("input_text"),
                "legal_basis": base.get("legal_basis"),
                "strategy": base.get("strategy"),
                "final_doc": base.get("final_doc"),
                "created_at": base.get("created_at"),
            }
            self._attempt_insert(slim)
            return "DB 저장 성공(최소 모드)"
        except Exception as e2:
            return f"DB 저장 실패: {e2}"

    def fetch_since(self, days: int = 30, max_rows: int = 5000):
        if not self.is_active:
            return []
        since_dt = datetime.now() - timedelta(days=days)
        since_iso = since_dt.isoformat()
        try:
            res = (
                self.client.table("law_logs")
                .select("created_at, task_type, dept, officer")
                .gte("created_at", since_iso)
                .order("created_at", desc=True)
                .limit(max_rows)
                .execute()
            )
            return res.data or []
        except Exception:
            try:
                res = (
                    self.client.table("law_logs")
                    .select("created_at")
                    .gte("created_at", since_iso)
                    .order("created_at", desc=True)
                    .limit(max_rows)
                    .execute()
                )
                return res.data or []
            except Exception:
                return []

    def daily_usage(self, days: int = 30):
        rows = self.fetch_since(days=days)
        daily = {}
        for r in rows:
            ca = r.get("created_at")
            if not ca:
                continue
            day = str(ca)[:10]
            daily[day] = daily.get(day, 0) + 1
        return daily, sum(daily.values())

    def breakdown(self, days: int = 30):
        rows = self.fetch_since(days=days)
        by_task, by_dept, by_officer = {}, {}, {}
        for r in rows:
            t = norm_space(r.get("task_type", "")) or "(미분류)"
            d = norm_space(r.get("dept", "")) or "(미지정)"
            o = norm_space(r.get("officer", "")) or "(미지정)"
            by_task[t] = by_task.get(t, 0) + 1
            by_dept[d] = by_dept.get(d, 0) + 1
            by_officer[o] = by_officer.get(o, 0) + 1
        return by_task, by_dept, by_officer

db_service = DatabaseService()


# =========================
# 8) Agents
# =========================
class TGD_Agents:
    @staticmethod
    def planner(user_input: str) -> dict:
        prompt = f"""
상황: "{user_input}"
너는 행정업무 '플래너'다.

[중요 규칙]
- law_name: 법령의 '공식 명칭'만 적어라 (예: '도로교통법', '자동차관리법'). 조문번호는 넣지 마.
- article_no: '제OO조' 또는 'OO' 숫자만 적어라.
- keywords: 법령 검색에 도움이 될 핵심 단어 3~5개.

JSON ONLY:
{{
  "task_type": "",
  "law_hint": {{"law_name":"", "article_no":"", "keywords":[]}},
  "naver_queries": {{"news":[], "blog":[], "kin":[]}}
}}
        obj = llm_service.generate_json(prompt)
        if not isinstance(obj, dict):
            return {"task_type": "", "law_hint": {"law_name":"", "article_no":"", "keywords":[]}, "naver_queries": {"news":[], "blog":[], "kin":[]}}

        lh = obj.get("law_hint", {}) if isinstance(obj.get("law_hint", {}), dict) else {}
        nq = obj.get("naver_queries", {}) if isinstance(obj.get("naver_queries", {}), dict) else {}

        kws = lh.get("keywords", [])
        if not isinstance(kws, list):
            kws = []
        kws = [norm_space(x) for x in kws if norm_space(x)][:7]

        def list3(x):
            if not isinstance(x, list):
                return []
            return [norm_space(i) for i in x if norm_space(i)][:3]

        return {
            "task_type": norm_space(obj.get("task_type","")),
            "law_hint": {
                "law_name": norm_space(lh.get("law_name","")),
                "article_no": norm_space(lh.get("article_no","")),
                "keywords": kws,
            },
            "naver_queries": {
                "news": list3(nq.get("news", [])),
                "blog": list3(nq.get("blog", [])),
                "kin":  list3(nq.get("kin", [])),
            }
        }

    @staticmethod
    def analyst(user_input: str, legal_basis: str, evidence_text: str, evidence_summary: dict) -> str:
        prompt = f"""
당신은 행정업무 베테랑 주무관이다.

[민원/업무 상황]
{user_input}

[법적 근거(원문/상태 포함)]
{legal_basis}

[현실 근거(네이버: 뉴스/블로그/지식iN)]
{evidence_text}

[근거 품질 요약]
- 총 {evidence_summary.get("count",0)}건 / HIGH {evidence_summary.get("high",0)} / MED {evidence_summary.get("med",0)} / LOW {evidence_summary.get("low",0)}
- 평균 점수: {evidence_summary.get("avg_score",0.0)}
- 주요 태그: {", ".join(evidence_summary.get("top_tags",[]) or [])}

[가중치 규칙]
- HIGH(뉴스/공식성 강함)는 사실관계/시점/사회적 쟁점 파악에 강하게 반영
- BLOG/KIN/AD_RISK/PERSONAL_RISK/THIN 태그가 있는 항목은 참고만(단정 근거 금지)
- 법령이 PENDING 또는 SEMI_CONFIRMED이면 처분 강도를 낮추고 확인/자료요청/안내 중심으로

아래를 마크다운으로 작성:
1) 처리 방향(실무 단계별)
2) 핵심 주의사항(증거/절차/기한/통지 방식)
3) 예상 반발 & 대응
4) 추가 확인 체크리스트
"""
        return llm_service.generate_text(prompt, temperature=0.1).strip()

    @staticmethod
    def clerk_deadline(user_input: str, legal_status: str) -> dict:
        today = datetime.now()
        default_days = 15 if legal_status != "PENDING" else 10
        prompt = f"""
오늘: {today.strftime('%Y-%m-%d')}
업무: {user_input}
법령상태: {legal_status}
'의견제출/이행 기한(일수)'를 숫자만 출력. 불명확하면 {default_days}.
"""
        try:
            res = llm_service.generate_text(prompt, temperature=0.1)
            days = int(re.sub(r"[^0-9]", "", res)) if res else default_days
            if days <= 0:
                days = default_days
        except Exception:
            days = default_days

        deadline = today + timedelta(days=days)
        return {
            "today_str": today.strftime("%Y. %m. %d."),
            "deadline_str": deadline.strftime("%Y. %m. %d."),
            "days_added": days,
            "doc_num": f"행정-{today.strftime('%Y')}-{int(time.time())%1000:03d}호",
        }

    @staticmethod
    def drafter(user_input: str, legal_basis: str, legal_status: str, meta: dict, strategy_md: str, dept: str, officer: str) -> dict:
        prompt = f"""
당신은 행정기관의 베테랑 서기다. 아래 정보를 바탕으로 완결 공문서를 작성한다.

[입력]
- 부서: {dept}
- 담당자: {officer}
- 상황: {user_input}
- 법적 근거(원문 유지): {legal_basis}
- 법적 근거 상태: {legal_status}  # CONFIRMED / SEMI_CONFIRMED / PENDING
- 시행일자: {meta.get("today_str")}
- 기한: {meta.get("deadline_str")} ({meta.get("days_added")}일)
- 처리전략:
{strategy_md}

[핵심 규칙]
- 절대 새로운 법조문/없는 법령을 만들지 말 것.
- PENDING: "관련 법령 검토 중/조항 확인 필요"로 처리, 안내/자료요청/확인 중심.
- SEMI_CONFIRMED: "자동매칭(최종 확인 필요)" 표기.
- CONFIRMED: 원문 확정, 인용 가능.

[출력 규칙]
- HTML/태그/마크다운/코드블록 금지
- 본문은 순수 텍스트 문단
- 구조: [경위] -> [근거] -> [처분 내용] -> [권리구제 절차]
- JSON ONLY

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
# 9) Workflow (Law Resolver + Evidence)
# =========================
def resolve_law_with_status(law_hint: dict) -> dict:
    if not law_api.enabled:
        return {
            "legal_status": "PENDING",
            "legal_basis": "⚠️ LAW API OFF (requests/xmltodict/secrets 확인 필요)",
            "law_debug": {"source": "LAW_API_OFF"}
        }

    law_name = law_hint.get("law_name", "")
    article_no = law_hint.get("article_no", "")
    keywords = law_hint.get("keywords", [])

    queries = make_law_query_candidates(law_name, keywords)
    best_law, best_from_query = {}, ""
    traces = []

    for q in queries:
        law_cands = law_api.search_law_candidates(q, display=20)
        chosen = law_api.choose_best_law(law_cands, q)
        traces.append({"query": q, "count": len(law_cands), "top": (law_cands[0].get("law_name","") if law_cands else "")})
        if chosen.get("law_id"):
            best_law = chosen
            best_from_query = q
            break

    if not best_law.get("law_id"):
        return {
            "legal_status": "PENDING",
            "legal_basis": "⚠️ LAW API에서 법령을 찾지 못했습니다. (법령명/키워드 보강 필요)",
            "law_debug": {"source": "LAW_API_NO_LAW", "queries": queries, "traces": traces}
        }

    law_xml = law_api.get_law_xml(best_law["law_id"])

    # 1) 조문번호 직접
    article_text = law_api.extract_article_text(law_xml, article_no) if article_no else ""

    # 2) 실패 시 키워드 자동탐색
    auto_candidates = []
    chosen_auto = None
    if not article_text:
        auto_candidates = law_api.find_article_candidates(law_xml, keywords, topk=5)
        if auto_candidates:
            chosen_auto = auto_candidates[0]
            article_text = chosen_auto.get("fulltext", "")
            article_no = chosen_auto.get("article_no", "") or article_no

    # 3) 상태 결정
    if article_text and chosen_auto is None and article_no:
        return {
            "legal_status": "CONFIRMED",
            "legal_basis": f"[{best_law['law_name']} {article_no}]\n\n{article_text}",
            "law_debug": {
                "source": "LAW_API_SUCCESS",
                "law_id": best_law.get("law_id"),
                "law_name": best_law.get("law_name"),
                "article_no": article_no,
                "query_used": best_from_query,
                "auto_candidates": [],
                "traces": traces,
            }
        }

    if article_text and chosen_auto is not None:
        return {
            "legal_status": "SEMI_CONFIRMED",
            "legal_basis": f"[{best_law['law_name']} {article_no}] (키워드 기반 조문 자동매칭: 최종 확인 필요)\n\n{article_text}",
            "law_debug": {
                "source": "LAW_API_AUTO_ARTICLE",
                "law_id": best_law.get("law_id"),
                "law_name": best_law.get("law_name"),
                "article_no": article_no,
                "query_used": best_from_query,
                "auto_candidates": auto_candidates,
                "traces": traces,
            }
        }

    return {
        "legal_status": "PENDING",
        "legal_basis": (
            "⚠️ LAW API로 '법령(공식명)'은 확인했으나, 조문 원문을 확정하지 못했습니다.\n"
            f"- 법령명(공식): {best_law.get('law_name','')}\n"
            f"- 조문(입력/추정): {article_no or '(미지정)'}\n"
            f"- 사용 쿼리: {best_from_query}\n"
            "→ 조문번호를 명확히 지정하거나, 키워드를 보강해 재시도 필요"
        ),
        "law_debug": {
            "source": "LAW_API_PARTIAL",
            "law_id": best_law.get("law_id"),
            "law_name": best_law.get("law_name"),
            "article_no": article_no,
            "query_used": best_from_query,
            "auto_candidates": auto_candidates,
            "traces": traces,
        }
    }

def collect_naver_evidence(naver_queries: dict, fallback_query: str) -> dict:
    if not naver_search.enabled:
        return {
            "evidence_items": [],
            "evidence_text": "⚠️ 네이버 API 미설정 또는 requests 미설치: Naver 검색 생략",
            "raw_counts": {},
            "summary": evidence_quality_summary([]),
        }

    news_qs = naver_queries.get("news", []) if isinstance(naver_queries, dict) else []
    blog_qs = naver_queries.get("blog", []) if isinstance(naver_queries, dict) else []
    kin_qs  = naver_queries.get("kin", []) if isinstance(naver_queries, dict) else []

    if not news_qs: news_qs = [fallback_query]
    if not blog_qs: blog_qs = [fallback_query]
    if not kin_qs:  kin_qs  = [fallback_query]

    raw = []
    cnt = {"news":0, "blog":0, "kin":0}

    for q in news_qs[:2]:
        items = naver_search.search(q, category="news", display=5, sort="date")
        cnt["news"] += len(items)
        raw += items

    for q in blog_qs[:2]:
        items = naver_search.search(q, category="blog", display=4, sort="sim")
        cnt["blog"] += len(items)
        raw += items

    for q in kin_qs[:1]:
        items = naver_search.search(q, category="kin", display=4, sort="sim")
        cnt["kin"] += len(items)
        raw += items

    evidence_items = normalize_evidence(raw, max_items=12)
    summary = evidence_quality_summary(evidence_items)
    evidence_text = truncate_text(evidence_to_text(evidence_items), 4300)

    return {"evidence_items": evidence_items, "evidence_text": evidence_text, "raw_counts": cnt, "summary": summary}


def run_workflow(user_input: str, dept: str, officer: str):
    log_placeholder = st.empty()
    logs = []
    model_usage = {}
    timing = {}
    provenance = {}

    def add_log(msg, style="sys"):
        style = style if style in ["legal", "search", "strat", "calc", "draft", "sys"] else "sys"
        logs.append(f"<div class='agent-log log-{style}'>{escape(msg)}</div>")
        log_placeholder.markdown("".join(logs), unsafe_allow_html=True)
        time.sleep(0.05)

    def tick():
        return time.perf_counter()

    add_log("🧭 Phase A (Planner): 상황 구조화 + 검색 설계", "sys")
    t0 = tick()
    plan = TGD_Agents.planner(user_input)
    timing["Planner(ms)"] = int((tick() - t0) * 1000)
    model_usage["Planner"] = llm_service.last_model_used
    provenance["plan"] = plan

    task_type = plan.get("task_type", "") or "(미분류)"

    add_log("📚 Phase B (Law Resolver): LAW API 확정(상태 머신)", "legal")
    t0 = tick()
    law_res = resolve_law_with_status(plan.get("law_hint", {}))
    timing["LawResolver(ms)"] = int((tick() - t0) * 1000)

    legal_status = law_res["legal_status"]
    legal_basis = law_res["legal_basis"]
    law_debug = law_res["law_debug"]
    provenance["law_debug"] = law_debug
    add_log(f"✅ 법령 상태: {legal_status} / 소스: {law_debug.get('source')}", "legal")

    add_log("🌍 Phase C (Evidence): 네이버 근거 수집 + 리스크 태깅", "search")
    t0 = tick()
    ev = collect_naver_evidence(plan.get("naver_queries", {}), user_input)
    timing["Evidence(ms)"] = int((tick() - t0) * 1000)

    evidence_items = ev.get("evidence_items", [])
    evidence_text = ev.get("evidence_text", "")
    evidence_summary = ev.get("summary", {})

    provenance["evidence_items"] = evidence_items
    provenance["evidence_summary"] = evidence_summary
    provenance["evidence_raw_counts"] = ev.get("raw_counts", {})

    # 법령 후보 선택 UI는 main()에서 탭으로 노출 (세션 오버라이드)
    if st.session_state.get("override_legal"):
        ov = st.session_state["override_legal"]
        legal_status = ov.get("status", legal_status)
        legal_basis = ov.get("basis", legal_basis)
        provenance["override_legal"] = ov
        add_log("🔁 사용자 선택으로 법령 상태 CONFIRMED 격상", "legal")

    add_log("🧠 Phase D (Analyst): 근거 가중치 반영 전략 수립", "strat")
    t0 = tick()
    strategy = TGD_Agents.analyst(user_input, legal_basis, evidence_text, evidence_summary)
    timing["Analyst(ms)"] = int((tick() - t0) * 1000)
    model_usage["Analyst"] = llm_service.last_model_used

    add_log("📅 Phase E: 기한 산정 + 공문(JSON) 작성", "calc")
    meta = TGD_Agents.clerk_deadline(user_input, legal_status)

    add_log("✍️ 공문(JSON) 생성 중...", "draft")
    t0 = tick()
    doc = TGD_Agents.drafter(user_input, legal_basis, legal_status, meta, strategy, dept, officer)
    timing["Drafter(ms)"] = int((tick() - t0) * 1000)
    model_usage["Drafter"] = llm_service.last_model_used
    doc = ensure_doc_shape(doc)

    add_log("💾 Supabase 저장(업무유형/부서/담당자/근거품질 포함) ...", "sys")
    t0 = tick()
    payload = {
        "task_type": task_type,
        "dept": dept,
        "officer": officer,
        "input_text": user_input,
        "legal_basis": legal_basis,
        "legal_status": legal_status,
        "strategy": strategy,
        "final_doc": json.dumps(doc, ensure_ascii=False),
        "model_usage": json.dumps(model_usage, ensure_ascii=False),
        "timing": json.dumps(timing, ensure_ascii=False),
        "provenance": json.dumps(provenance, ensure_ascii=False),
        "evidence_quality": json.dumps(evidence_summary, ensure_ascii=False),
        "created_at": datetime.now().isoformat(),
    }
    save_result = db_service.save_log(payload)
    timing["SupabaseSave(ms)"] = int((tick() - t0) * 1000)

    add_log(f"✅ 완료 ({save_result})", "sys")
    time.sleep(0.1)
    log_placeholder.empty()

    m = st.session_state["metrics"]
    m["runs"] += 1
    m["timing"].append(timing)

    return {
        "doc": doc,
        "meta": meta,
        "legal_basis": legal_basis,
        "legal_status": legal_status,
        "law_debug": law_debug,
        "strategy": strategy,
        "evidence_text": evidence_text,
        "evidence_summary": evidence_summary,
        "task_type": task_type,
        "model_usage": model_usage,
        "timing": timing,
    }


# =========================
# 10) Dashboard
# =========================
@st.cache_data(ttl=60)
def _cached_db_usage(days: int):
    if not db_service.is_active:
        return {}, 0
    return db_service.daily_usage(days=days)

@st.cache_data(ttl=60)
def _cached_breakdown(days: int):
    if not db_service.is_active:
        return {}, {}, {}
    return db_service.breakdown(days=days)

def render_dashboard():
    st.markdown("## 📊 운영 계기판")

    st.markdown("### 🗓️ 일일 사용량 (Supabase 기준)")
    if not db_service.is_active:
        st.info("DB(Supabase)가 OFF라서 일일 사용량/분해를 집계할 수 없습니다.")
    else:
        daily_7, total_7 = _cached_db_usage(7)
        daily_30, total_30 = _cached_db_usage(30)

        today_key = datetime.now().strftime("%Y-%m-%d")
        today_cnt = daily_30.get(today_key, 0)

        c1, c2, c3 = st.columns(3)
        c1.metric("오늘 실행", f"{today_cnt} 건")
        c2.metric("최근 7일 합계", f"{total_7} 건")
        c3.metric("최근 30일 합계", f"{total_30} 건")

        daily_14, _ = _cached_db_usage(14)
        items = sorted(daily_14.items(), key=lambda x: x[0])
        if items:
            st.text_area(
                "최근 14일 일별 실행 건수",
                value="\n".join([f"{d} : {cnt}건" for d, cnt in items]),
                height=220,
                disabled=True,
            )

        st.markdown("### 🧩 사용량 분해 (최근 30일)")
        by_task, by_dept, by_officer = _cached_breakdown(30)

        def top_lines(d: dict, k: int = 10):
            if not d:
                return "(데이터 없음 / 컬럼 미구성 가능)"
            items = sorted(d.items(), key=lambda kv: -kv[1])[:k]
            return "\n".join([f"- {name}: {cnt}건" for name, cnt in items])

        c1, c2, c3 = st.columns(3)
        with c1:
            st.text_area("업무유형 TOP", value=top_lines(by_task), height=260, disabled=True)
        with c2:
            st.text_area("부서 TOP", value=top_lines(by_dept), height=260, disabled=True)
        with c3:
            st.text_area("담당자 TOP", value=top_lines(by_officer), height=260, disabled=True)

    st.markdown("---")

    m = st.session_state["metrics"]
    calls = m.get("calls", {})
    tokens = m.get("tokens_est", {})
    cost = m.get("cost_est", {})
    runs = m.get("runs", 0)

    if not calls:
        st.info("세션 기준 모델 호출 기록이 없습니다. (실행 후 표시됩니다)")
        return

    total_calls = sum(calls.values()) or 1

    st.markdown("### ✅ 모델별 호출 횟수/비율 (세션 기준)")
    for k, v in sorted(calls.items(), key=lambda x: -x[1]):
        st.write(f"- **{k}**: {v}회 ({(v/total_calls)*100:.1f}%)")

    st.markdown("### 💸 모델별 예상 비용(근사, 세션 기준)")
    for k in sorted(tokens.keys(), key=lambda x: -tokens[x]):
        st.write(f"- **{k}**: 토큰≈{tokens.get(k,0):,} / 비용≈${cost.get(k,0):.6f}")

    st.markdown("### ⏱️ 최근 1회 단계별 시간(ms)")
    if m.get("timing"):
        last = m["timing"][-1]
        for step, ms in last.items():
            st.write(f"- {step}: {ms} ms")

    st.caption(f"Session Runs: {runs}, Session Total model calls: {total_calls}")


# =========================
# 11) UI (Tabs 적용)
# =========================
def main():
    metrics_init()

    # 세션 기본값
    st.session_state.setdefault("dept", "충주시청 ○○과")
    st.session_state.setdefault("officer", "○○○")

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.title("⚖️ AI 행정관 Pro (TGD Ops-Final v3)")
        st.caption("LAW API + Naver(근거 품질/리스크 태그) + Groq 70B + Supabase(사용량/분해) + JSON 파싱 강화 + Tabs UI")
        st.markdown("---")

        with st.expander("🧩 운영 메타(저장/집계용)", expanded=True):
            st.text_input("부서명", key="dept")
            st.text_input("담당자", key="officer")

        user_input = st.text_area(
            "업무 내용",
            height=180,
            placeholder="예시:\n- 자동차 정비업체 불법영업(작업범위 초과) 민원 회신\n- 불법 주정차 관련 과태료 부과 예고 통지\n- 건설기계 차고지 외 주기위반 관련 사전통지",
        )

        c1, c2 = st.columns([1, 1])
        run_btn = c1.button("⚡ 실행", type="primary", use_container_width=True)
        clear_btn = c2.button("🧹 초기화", use_container_width=True)

        if clear_btn:
            for k in [
                "final", "override_legal",
                "metrics"
            ]:
                st.session_state.pop(k, None)
            st.rerun()

        st.markdown("### ⚙️ 상태")
        st.write(f"- LAW API: {'ON' if law_api.enabled else 'OFF'}")
        st.write(f"- Naver 검색: {'ON' if naver_search.enabled else 'OFF'}")
        st.write(f"- Groq 70B: {'ON' if bool(st.secrets.get('general', {}).get('GROQ_API_KEY')) else 'OFF'}")
        st.write(f"- DB(Supabase): {'ON' if db_service.is_active else 'OFF'}")

        if run_btn:
            if not user_input.strip():
                st.warning("내용을 입력해주세요.")
            else:
                try:
                    with st.spinner("TGD 에이전트 처리 중..."):
                        out = run_workflow(user_input, dept=st.session_state["dept"], officer=st.session_state["officer"])
                        st.session_state["final"] = out
                except Exception as e:
                    st.error(f"시스템 오류 발생: {e}")

        st.markdown("---")
        render_dashboard()

    with col_right:
        final = st.session_state.get("final")
        if not final:
            st.markdown(
                """
<div style='text-align: center; padding: 100px; color: #aaa; background: white; border-radius: 10px; border: 2px dashed #ddd;'>
  <h3>📄 Document Preview</h3>
  <p>왼쪽에서 업무를 지시하면<br>완성된 공문서가 여기에 나타납니다.</p>
</div>
""",
                unsafe_allow_html=True,
            )
            return

        doc = ensure_doc_shape(final["doc"])
        meta = final["meta"]
        legal_basis = final["legal_basis"]
        legal_status = final["legal_status"]
        law_debug = final.get("law_debug", {})
        strategy = final.get("strategy", "")
        evidence_text = final.get("evidence_text", "")
        evsum = final.get("evidence_summary", {})
        task_type = final.get("task_type", "(미분류)")

        tab1, tab2, tab3 = st.tabs(["📄 공문서 프리뷰", "🔍 법리/증거 분석", "🧩 조문 후보/디버그"])

        with tab1:
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

with tab2:
            st.subheader("⚖️ 법적 근거 및 처리 전략")
            st.info(f"법령 상태: {legal_status} / 소스: {law_debug.get('source')}")
            st.text_area("법령 원문", value=legal_basis, height=200, disabled=True)
            st.markdown(strategy)

            st.markdown("---")
            st.subheader("🧾 네이버 검색 근거 (클릭 시 원문 이동)")
            
            # workflow에서 저장한 evidence_items 리스트를 가져옵니다.
            evidence_items = final.get("provenance", {}).get("evidence_items", [])
            
            if not evidence_items:
                st.info("수집된 네이버 검색 근거가 없습니다.")
            else:
                for it in evidence_items:
                    lvl = it.get("quality_level", "LOW")
                    color = "#1e40af" if lvl == "HIGH" else "#c2410c" if lvl == "MED" else "#6b7280"
                    
                    # HTML을 사용한 클릭 가능한 카드형 UI
                    st.markdown(f"""
                    <div style="border-left: 5px solid {color}; padding: 10px 15px; margin-bottom: 15px; background-color: white; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <div style="font-size: 0.8rem; color: {color}; font-weight: bold; margin-bottom: 5px;">
                            [{lvl} / {it.get('quality_score')}] {it.get('source')}
                        </div>
                        <a href="{it.get('link')}" target="_blank" style="text-decoration: none; color: #1e3a8a; font-size: 1.1rem; font-weight: bold;">
                            {it.get('title')} <span style="font-size: 0.9rem;">🔗</span>
                        </a>
                        <div style="font-size: 0.95rem; color: #374151; margin-top: 5px; line-height: 1.5;">
                            {it.get('description')}
                        </div>
                        <div style="font-size: 0.8rem; color: #9ca3af; margin-top: 8px;">
                            {it.get('pubDate')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)            # --- 교체 끝 ---

        with tab3:
            st.subheader("🧩 조문 후보(자동탐색) → 사람 선택으로 CONFIRMED 격상")
            auto_cands = law_debug.get("auto_candidates", []) if isinstance(law_debug.get("auto_candidates", []), list) else []
            if not auto_cands:
                st.info("자동탐색 후보가 없습니다. (이미 CONFIRMED이거나, 키워드 매칭 실패)")
            else:
                options = [f"{i+1}) {c.get('article_no','')} | 점수:{c.get('score','')} | {c.get('title','')}" for i, c in enumerate(auto_cands)]
                sel = st.selectbox("조문 후보 선택", options=options, index=0)
                if st.button("✅ 선택한 조문으로 확정(CONFIRMED) 후 재작성", use_container_width=True):
                    idx = max(0, options.index(sel))
                    picked = auto_cands[idx]
                    st.session_state["override_legal"] = {
                        "status": "CONFIRMED",
                        "basis": f"[{law_debug.get('law_name','')} {picked.get('article_no','')}]\n\n{picked.get('fulltext','')}",
                        "picked": picked,
                        "law_name": law_debug.get("law_name",""),
                        "law_id": law_debug.get("law_id",""),
                    }
                    st.rerun()

            st.markdown("---")
            st.subheader("🔧 LAW API 디버깅 정보")
            traces = law_debug.get("traces", [])
            if not traces:
                st.warning("API 호출 시도 기록이 없습니다. API ID(OC) 설정을 확인하세요.")
            else:
                for t in traces:
                    status_icon = "✅" if t.get('count', 0) > 0 else "❌"
                    st.write(f"{status_icon} **쿼리**: `{t.get('query')}` → **검색결과**: {t.get('count')}건")
                    if t.get('top'):
                        st.caption(f"   └ 가장 유사한 법령: {t.get('top')}")
            
            st.info(f"현재 사용 중인 API Key(OC) 존재 여부: {'예' if law_api.oc else '아니오'}")

if __name__ == "__main__":
    main()
