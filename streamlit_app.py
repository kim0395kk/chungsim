# app.py — AI 행정관 Pro (Stable / Dual-Model Router v6)
# Groq: qwen/qwen3-32b (FAST) + llama-3.3-70b-versatile (STRICT)
# LAWGO(DRF) + NAVER + Supabase (옵션: 로그인/이력)
#
# ✅ 정확도 개선 포인트(핵심)
# 1) Intake(사실/요구/대상/시간/장소/증거) 강제 구조화
# 2) 법령 후보 3~6개 생성 -> DRF로 원문 확보 -> Verifier 점수로 선택(루프)
# 3) "법적근거"는 '원문 텍스트'만 정리해서 보여줌(XML/잡문/한자 제거)
# 4) 공문(JSON)은 STRICT 모델 고정 + JSON 재시도 + 품질체크(QA)
#
# ⚠️ U+EA01(비표시 문자) 에러 방지:
# - 이 파일은 "메모장(plain text)"로 붙여넣고 저장하세요.
# - 한글 워드/웹에서 복붙하면 종종 Private Use Character가 섞입니다.
#
# -------------------------------
# secrets.toml 예시 (Streamlit Cloud)
# -------------------------------
# [general]
# GROQ_API_KEY = "..."
# GROQ_MODEL_FAST = "qwen/qwen3-32b"
# GROQ_MODEL_STRICT = "llama-3.3-70b-versatile"
#
# [law]
# LAW_API_ID = "..."  # law.go.kr DRF OC 값
#
# [naver]
# CLIENT_ID = "..."
# CLIENT_SECRET = "..."
#
# [supabase]  # 옵션(로그/히스토리)
# SUPABASE_URL = "https://xxxx.supabase.co"
# SUPABASE_KEY = "service_role_or_anon_key"
#
# -------------------------------
# requirements.txt (권장)
# -------------------------------
# streamlit
# groq
# requests
# xmltodict
# supabase
# python-dateutil

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
    page_title="AI 행정관 Pro (Dual v6.0)",
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
</style>
""",
    unsafe_allow_html=True,
)

_TAG_RE = re.compile(r"<[^>]+>")
_CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
# 한자(CJK Unified Ideographs) 제거 (원문에 섞여 나오면 보기 힘들어서 "표시용"에서 제거)
_HANJA_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF]+")


# =========================
# 2) Helpers
# =========================
def clean_text(value) -> str:
    """HTML 태그 및 제어문자 제거"""
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
    """표시용: 한자 제거 + 이상한 구분자(스크린샷 같은 ||> 등) 정리"""
    if not s:
        return ""
    s = _HANJA_RE.sub("", s)
    # DRF/가공 과정에서 섞이는 잡문 패턴 정리
    s = re.sub(r"\|\>+", "", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = s.replace("  ", " ")
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
    """간이 키워드: LLM 실패시 fallback"""
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
    """
    - FAST: qwen/qwen3-32b
    - STRICT: llama-3.3-70b-versatile
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

        # 1) same model retries
        for _ in range(max_retry):
            try:
                txt = self._chat(model_first, messages, temp, json_mode=True)
                js = self._parse_json(txt)
                if js:
                    return js
            except Exception:
                pass

        # 2) strict escalate
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
        out = []
        for a in articles:
            if isinstance(a, dict):
                out.append(a)
        return out

    def get_article_by_mst(self, mst: str, article_no: Optional[str] = None) -> Dict[str, Any]:
        """
        반환:
        {
          "law_name": "...",
          "mst": "...",
          "article_no": "33",
          "article_title": "...",
          "article_text": "정리된 본문",
          "all_articles_index": ["제1조", "제2조", ...] (최대 80개)
        }
        """
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

            # 인덱스 (UI용)
            idx = []
            for a in articles[:80]:
                at = clean_text(a.get("ArticleTitle") or "")
                an = clean_text(a.get("@조문번호") or "")
                if at:
                    idx.append(at)
                elif an:
                    idx.append(f"제{an}조")
            # article_no 없으면, 일부라도 보여줄 수 있게 1조 반환
            if not article_no:
                # 첫 조문 구성
                if articles:
                    a0 = articles[0]
                    return self._format_article(law_name, mst, a0, idx)
                return {"law_name": law_name, "mst": mst, "all_articles_index": idx}

            tgt = re.sub(r"[^0-9]", "", str(article_no))
            if not tgt:
                return {"law_name": law_name, "mst": mst, "all_articles_index": idx}

            # 조문 매칭: 조문번호 또는 제목 "제NN조"
            for a in articles:
                an = clean_text(a.get("@조문번호") or "")
                at = clean_text(a.get("ArticleTitle") or "")
                if tgt == re.sub(r"[^0-9]", "", an) or (tgt and f"제{tgt}조" in at):
                    return self._format_article(law_name, mst, a, idx)

            # 못 찾으면 빈값
            return {"law_name": law_name, "mst": mst, "article_no": tgt, "all_articles_index": idx}

        except Exception:
            return {}

    def _format_article(self, law_name: str, mst: str, art: dict, idx: List[str]) -> Dict[str, Any]:
        at = clean_text(art.get("ArticleTitle") or "")
        an = clean_text(art.get("@조문번호") or "")
        content = clean_text(art.get("ArticleContent") or "")

        # 항/호 문단 합치기
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
        text_disp = strip_hanja_for_display(text)  # 보기 좋게 한자 제거

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


naver = NaverSearchService()


# =========================
# 7) Supabase (로그/히스토리)
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
        """runs 테이블 insert, run_id 리턴(가능하면)"""
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

    def insert_step(self, row: dict) -> None:
        if not self.client:
            return
        try:
            safe_row = json.loads(safe_json_dump(row))
            self.client.table("run_steps").insert(safe_row).execute()
        except Exception:
            return

    def insert_artifact(self, row: dict) -> None:
        if not self.client:
            return
        try:
            safe_row = json.loads(safe_json_dump(row))
            self.client.table("artifacts").insert(safe_row).execute()
        except Exception:
            return

    def list_runs(self, user_key: str, limit: int = 20):
        """간단 유저키 기반 히스토리(진짜 Auth 대신)"""
        if not self.client:
            return []
        try:
            resp = (
                self.client.table("runs")
                .select("run_id, created_at, task_type, law_name, article_no, final_verdict")
                .eq("user_id", user_key)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return getattr(resp, "data", []) or []
        except Exception:
            return []

    def load_run_detail(self, run_id: str):
        if not self.client or not run_id:
            return None
        try:
            r1 = (
                self.client.table("runs")
                .select("*")
                .eq("run_id", run_id)
                .limit(1)
                .execute()
            )
            data = getattr(r1, "data", None)
            if not data:
                return None
            run_row = data[0]

            art = (
                self.client.table("artifacts")
                .select("kind, content, created_at")
                .eq("run_id", run_id)
                .order("created_at", desc=True)
                .execute()
            )
            art_data = getattr(art, "data", []) or []
            run_row["_artifacts"] = art_data
            return run_row
        except Exception:
            return None


db = DatabaseService()


# =========================
# 8) Core Logic (Agentic-ish)
# =========================
def intake_schema(user_input: str) -> Dict[str, Any]:
    """
    민원 상황을 '사실/요구/대상/시간/장소/증거/쟁점/키워드'로 강제 구조화.
    이게 정확도 핵심(법령 엉뚱함 방지).
    """
    kw_fallback = extract_keywords_kor(user_input, max_k=10)

    prompt = f"""
다음 민원/업무 지시를 "행정사실관계" 중심으로 구조화해라.
반드시 아래 JSON 스키마만 출력(키 추가 금지).

{{
  "task_type": "주기위반|무단방치|불법주정차|행정처분|정보공개|기타",
  "authority_scope": {{
    "my_role": "주기위반 단속 담당",
    "can_do": ["현장확인","계도","통지","안내","이관"],
    "cannot_do": ["형사수사","강제집행","압수수색","구금"]
  }},
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
- 소설 금지. 입력에 없는 사실은 '추가 확인 필요'로 처리.
- 장소/시간이 없으면 빈문자열.
- keywords는 '사실 기반' 핵심어로.
"""
    js = llm.generate_json(prompt, prefer="fast", max_retry=2) or {}
    # 보정
    if not js:
        return {
            "task_type": "기타",
            "authority_scope": {"my_role": "주기위반 단속 담당", "can_do": ["현장확인", "계도", "통지", "안내", "이관"], "cannot_do": ["형사수사", "강제집행", "압수수색", "구금"]},
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

    # input quality (룰)
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
    """
    법령 후보는 1개가 아니라 여러 개!
    여기서 '엉뚱한 법령' 확률이 확 줄어듦.
    """
    task_type = clean_text(case.get("task_type"))
    facts = case.get("facts") if isinstance(case.get("facts"), dict) else {}
    issues = case.get("issues", [])
    keywords = case.get("keywords", [])
    # rule hint (업무 도메인)
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
    {{"law_name":"법령명","article_hint":"조번호(숫자만, 모르면 빈문자열)","reason":"짧게","confidence":0.0}},
    {{"law_name":"...","article_hint":"","reason":"...","confidence":0.0}}
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
- law_name은 "공식 법령명" 우선
- 확신 없으면 confidence 낮게
- article_hint는 모르면 빈문자열
- '내 권한(주기위반 단속 담당)' 범위에서 다룰 가능성이 큰 법령 우선
"""
    js = llm.generate_json(prompt, prefer="fast", max_retry=2) or {}
    cands = js.get("candidates", []) if isinstance(js.get("candidates"), list) else []
    out = []
    # 룰 기반 보강
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

    # 중복 제거(법령명 기준)
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
    """
    점수화 Verifier (0~100).
    - relevance: 키워드/쟁점 vs 조문 텍스트
    - scope_fit: 주기위반 담당자 권한 범위에 맞는지
    - article_match: 제목/내용이 직접 연결되는지
    - hallucination_risk: 원문이 빈약하거나 추측성
    """
    keywords = case.get("keywords", [])
    issues = case.get("issues", [])
    facts = case.get("facts", {}) if isinstance(case.get("facts"), dict) else {}
    text = (article_title + "\n" + article_text).lower()

    # relevance (룰)
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

    # scope_fit (룰)
    # 권한 밖 키워드가 조문에 많으면 감점
    out_of_scope = ["구속", "수사", "압수", "수색", "체포", "기소", "형사", "구금"]
    o_hits = sum(1 for w in out_of_scope if w in article_text)
    scope_fit = 25 - min(25, o_hits * 8)
    scope_fit = max(0, scope_fit)

    # article_match (룰)
    # 제목이 명확하면 가점, 조문이 너무 짧으면 감점
    match = 10
    if len(article_text) >= 200:
        match += 10
    if any(k.lower() in (article_title.lower() if article_title else "") for k in keywords[:4] if k):
        match += 5
    article_match = min(25, match)

    # hallucination_risk (룰)
    risk = 0
    if not article_text or len(article_text) < 80:
        risk += 10
    if "추가 확인 필요" in article_text:
        risk += 2
    # display 텍스트가 너무 깨져 있으면
    if "||" in article_text or ">>" in article_text:
        risk += 5
    risk = min(15, risk)

    total = relevance + scope_fit + article_match + (15 - risk)
    if total >= 75:
        verdict = "CONFIRMED"
    elif total >= 50:
        verdict = "WEAK"
    else:
        verdict = "FAIL"

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

[법적근거(선택)]
- 법령: {law_pack.get("law_name","")}
- 조문: {law_pack.get("article_title","")}
- 원문(요약): {truncate_text(law_pack.get("article_text",""), 900)}

[참고(네이버)]
{truncate_text(evidence_text, 700)}

아래 형식(마크다운)만 출력:
1) 처리 방향(현실적인 행정 프로세스 중심, 5~8줄)
2) 체크리스트(불릿 8~12개, "확인/기록/통지/기한" 포함)
3) 권한범위(내가 할 수 있는 것/없는 것 각 3~5개)
4) 민원인 설명 포인트(오해 줄이는 문장 3~5개)
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

법적 근거(선택/확보된 범위):
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


def qa_guardrails(doc: Dict[str, Any], law_pack: Dict[str, Any]) -> Dict[str, Any]:
    """필수요소/금지요소 간단 검사 후 보정 힌트"""
    issues = []
    if not doc.get("title"):
        issues.append("title_missing")
    if not doc.get("receiver"):
        issues.append("receiver_missing")
    if not isinstance(doc.get("body_paragraphs"), list) or len(doc.get("body_paragraphs")) < 2:
        issues.append("body_weak")

    # '단정/추측' 완화: 너무 공격적/추측적 문구 제거는 LLM 재작성까지는 안하고 경고만
    forbidden = ["확실히", "반드시", "100%", "무조건", "무차별"]
    body = "\n".join(doc.get("body_paragraphs", []))
    if any(x in body for x in forbidden):
        issues.append("overconfident_language")

    # 법령이 FAIL인데 법령 단정하면 문제
    if law_pack.get("verdict") == "FAIL" and ("법령" in body or "제" in body):
        issues.append("law_claim_without_confidence")

    doc["_qa"] = {"issues": issues}
    return doc


def run_workflow(user_input: str, dept: str, officer: str, user_key: str):
    log_area = st.empty()
    logs = []

    def add_log(msg: str, style: str = "sys"):
        logs.append(f"<div class='agent-log log-{style}'>{safe_html(msg)}</div>")
        log_area.markdown("".join(logs), unsafe_allow_html=True)
        time.sleep(0.03)

    started = datetime.now().isoformat()

    # STEP: INTAKE
    add_log("🧾 [INTAKE] 민원/업무 내용을 사실관계 중심으로 구조화… (FAST)", "sys")
    t0 = time.time()
    case = intake_schema(user_input)
    db_step_payload = {"case": case}
    # STEP LOG
    if db.enabled():
        db.insert_step({
            "run_id": None,  # run_id는 나중에 insert 후 업데이트가 이상적이지만, 단순화
            "step_name": "INTAKE",
            "model_used": llm.model_fast,
            "tokens": 0,
            "cost": 0,
            "payload_json": db_step_payload
        })
    add_log(f"✅ [INTAKE] 완료 (quality={case.get('_input_quality',{}).get('score','?')})", "sys")

    # STEP: LAW CANDIDATES
    add_log("🧩 [LAW-CAND] 법령 후보 3~6개 생성… (FAST)", "legal")
    candidates = generate_law_candidates(case)
    if not candidates:
        candidates = [{"law_name": k, "article_hint": "", "reason": "fallback", "confidence": 0.2} for k in case.get("keywords", [])[:3]]
    add_log(f"📌 후보: " + ", ".join([c['law_name'] for c in candidates[:6]]), "legal")

    # STEP: LAW FETCH + VERIFY LOOP
    add_log("📚 [LAW] DRF로 원문 확보 + 검증 점수화…", "legal")
    best_pack = {
        "law_name": "",
        "mst": "",
        "article_title": "",
        "article_text": "",
        "verdict": "FAIL",
        "score": 0,
        "debug": {}
    }

    loop_debug = []
    for i, cand in enumerate(candidates[:6], start=1):
        q = cand.get("law_name", "")
        art_hint = cand.get("article_hint", "")
        add_log(f"  - ({i}) {q} 검색 → 원문 확인", "legal")

        # 1) search
        laws = law_api.search_law(q, display=10)
        if not laws:
            loop_debug.append({"cand": cand, "search": "no_result"})
            continue

        chosen = laws[0]
        mst = clean_text(chosen.get("MST"))
        law_name = clean_text(chosen.get("lawNm"))
        link = clean_text(chosen.get("link"))

        # 2) fetch (조문 힌트가 있으면 그 조문, 없으면 1조라도)
        pack = law_api.get_article_by_mst(mst, article_no=art_hint if art_hint else None)
        article_title = clean_text(pack.get("article_title", ""))
        article_text = clean_text(pack.get("article_text", ""))
        if not article_text:
            loop_debug.append({"cand": cand, "mst": mst, "fetch": "empty"})
            continue

        # 3) verify score
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
                "debug": {"cand": cand, "selected": chosen, "loop": loop_debug[-1]}
            }

        # 확정이면 바로 종료(성능↑)
        if verdict == "CONFIRMED":
            break

    add_log(f"✅ [LAW] 선택: {best_pack.get('law_name','(없음)')} / {best_pack.get('article_title','')} (score={best_pack.get('score',0)}, {best_pack.get('verdict')})", "legal")

    # STEP: NAVER EVIDENCE
    add_log("🌍 [EVIDENCE] 네이버 참고자료(선택) 수집…", "search")
    ev_items = []
    ev_text = ""
    kw = case.get("keywords", [])
    if kw:
        q = " ".join(kw[:2]) + " 행정처분"
        raw = naver.search(q, cat="news", display=5)
        for item in raw:
            title = clean_text(item.get("title"))
            desc = clean_text(item.get("description"))
            link = clean_text(item.get("link"))
            ev_items.append({"title": title, "desc": desc, "link": link})
            ev_text += f"- {title}: {desc}\n"
    add_log(f"✅ [EVIDENCE] {len(ev_items)}건", "search")

    # STEP: STRATEGY
    add_log("🧠 [STRATEGY] 처리 전략 요약… (FAST/STRICT 자동)", "strat")
    strategy = draft_strategy(case, best_pack, ev_text)

    # STEP: DRAFT
    add_log("✍️ [DRAFT] 공문 JSON 생성… (STRICT)", "draft")
    doc = draft_document_json(dept, officer, case, best_pack, strategy)
    doc = qa_guardrails(doc, best_pack)

    # A4 HTML 생성
    meta = doc.get("_meta", {})
    doc_num = meta.get("doc_num", "")
    today = meta.get("today", "")

    # STEP: SAVE
    add_log("💾 [SAVE] 이력 저장…", "sys")
    run_id = None
    db_msg = "DB 미연결"
    if db.enabled():
        ok, msg, rid = db.insert_run({
            "user_id": user_key,               # 간이 user_key (실제 Auth 대신)
            "created_at": started,
            "task_type": clean_text(case.get("task_type","")),
            "input_text": user_input,
            "input_quality_score": int(case.get("_input_quality",{}).get("score", 0)),
            "final_verdict": best_pack.get("verdict"),
            "law_name": best_pack.get("law_name"),
            "law_mst": best_pack.get("mst"),
            "article_no": best_pack.get("verify",{}).get("score_breakdown",{}),  # 테이블 설계에 맞춰 수정 가능
            "total_tokens": int(st.session_state.get("metrics",{}).get("tokens_total",0)),
            "total_cost": 0,
            "status": "DONE"
        })
        db_msg = msg
        run_id = rid

        # artifacts 저장(선택)
        if run_id:
            db.insert_artifact({"run_id": run_id, "kind": "case_json", "content": safe_json_dump(case)})
            db.insert_artifact({"run_id": run_id, "kind": "law_pack_json", "content": safe_json_dump(best_pack)})
            db.insert_artifact({"run_id": run_id, "kind": "strategy_md", "content": strategy})
            db.insert_artifact({"run_id": run_id, "kind": "draft_json", "content": safe_json_dump(doc)})
            # A4 html도 저장하고 싶으면:
            # db.insert_artifact({"run_id": run_id, "kind": "draft_html", "content": "..."})

    add_log(f"✅ 완료 ({db_msg})", "sys")
    time.sleep(0.25)
    log_area.empty()

    # 반환
    return {
        "case": case,
        "law": best_pack,
        "strategy": strategy,
        "doc": ensure_doc_shape(doc),
        "doc_meta": {"doc_num": doc_num, "today": today, "dept": dept, "officer": officer},
        "ev_items": ev_items,
        "loop_debug": loop_debug,
        "db_msg": db_msg,
        "run_id": run_id
    }


# =========================
# 9) UI
# =========================
def render_a4(doc: Dict[str, Any], meta: Dict[str, str]):
    body_html = "".join(
        [f"<p style='margin:0 0 14px 0;'>{safe_html(p)}</p>" for p in doc.get("body_paragraphs", [])]
    )
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
    components.html(html, height=900, scrolling=True)


def render_law(law_pack: Dict[str, Any]):
    law_name = law_pack.get("law_name", "")
    article_title = law_pack.get("article_title", "")
    verdict = law_pack.get("verdict", "")
    score = law_pack.get("score", 0)
    link = law_pack.get("link", "")

    st.markdown(f"**선택 법령:** {law_name}  \n**조문:** {article_title}  \n**검증:** {verdict} / score={score}")
    if link:
        st.markdown(f"- 상세 링크: {link}")

    txt = law_pack.get("article_text", "") or ""
    txt = normalize_whitespace(txt)
    txt = strip_hanja_for_display(txt)

    if not txt:
        st.warning("조문 원문을 표시할 수 없습니다(빈 텍스트). 후보를 바꿔야 합니다.")
        return

    # 보기 좋은 형태: 제목 + 조문을 코드블록/프리텍스트로
    st.markdown("### 조문 원문(정리본)")
    st.code(txt, language="text")

    v = law_pack.get("verify") or {}
    if v:
        st.markdown("### Verifier 점수")
        st.json(v)


def main():
    # 간이 사용자 키(로그인 대신): 조직/사용자 구분용
    st.session_state.setdefault("user_key", "local_user")

    st.session_state.setdefault("dept", "OO시청 OO과")
    st.session_state.setdefault("officer", "김주무관")

    col_l, col_r = st.columns([1, 1.2], gap="large")

    with col_l:
        st.title("AI 행정관 Pro")
        st.caption("Dual Router v6.0 — FAST(qwen/qwen3-32b) + STRICT(llama-3.3-70b) + Law Verifier Loop")
        st.markdown("---")

        with st.expander("🧩 사용자/부서 설정", expanded=False):
            st.text_input("부서명", key="dept")
            st.text_input("담당자", key="officer")
            st.text_input("사용자 키(히스토리 구분용, 임의)", key="user_key")
            st.caption("※ Supabase Auth를 붙이려면 여기 user_key 대신 auth.uid()를 넣는 구조로 확장하세요.")

        user_input = st.text_area(
            "업무 지시 사항(민원 상황 포함)",
            height=220,
            placeholder="예: 건설기계가 차고지 외 장기간 주차(주기위반) 신고가 들어왔고, 현장 확인했더니 이동한 상태. 민원인은 상시 단속을 요구. 담당자가 할 수 있는 조치와 답변 공문 작성.",
        )

        if st.button("🚀 문서 생성 실행", type="primary", use_container_width=True):
            if not user_input.strip():
                st.warning("내용을 입력하세요.")
            else:
                with st.spinner("에이전트(구조화→법령후보→원문확보→검증→공문작성) 실행 중..."):
                    try:
                        res = run_workflow(
                            user_input.strip(),
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

        st.markdown("<div class='small-muted'>핵심: 법령은 1번 찍지 않고, 후보→원문→검증점수 루프를 돌려서 엉뚱한 법령을 줄입니다.</div>", unsafe_allow_html=True)

        # History (옵션)
        if db.enabled():
            st.markdown("---")
            st.subheader("🕘 히스토리(최근)")
            runs = db.list_runs(st.session_state["user_key"], limit=15)
            if runs:
                opts = [f"{r.get('created_at','')} | {r.get('task_type','')} | {r.get('final_verdict','')}" for r in runs]
                idx = st.selectbox("불러올 실행 기록 선택", range(len(opts)), format_func=lambda i: opts[i])
                if st.button("불러오기", use_container_width=True):
                    rid = runs[idx].get("run_id")
                    detail = db.load_run_detail(rid)
                    if detail:
                        st.session_state["history_detail"] = detail
            else:
                st.caption("runs 테이블에 저장된 기록이 없습니다(테이블/권한 확인).")

    with col_r:
        tab_main, tab_debug, tab_history = st.tabs(["📄 공문서(A4)", "🔍 근거/전략", "🧾 히스토리 상세"])

        with tab_main:
            res = st.session_state.get("result")
            if not res:
                st.markdown(
                    """
<div style='text-align:center; padding:120px 20px; color:#9ca3af; border:2px dashed #e5e7eb; border-radius:14px; background:#fff;'>
  <h3 style='margin-bottom:8px;'>📄 A4 미리보기</h3>
  <p>왼쪽에서 민원 상황을 입력하고 실행을 누르세요.<br>자동으로 법령을 확보/검증 후 공문을 작성합니다.</p>
</div>
""",
                    unsafe_allow_html=True,
                )
            else:
                render_a4(res["doc"], res["doc_meta"])

        with tab_debug:
            res = st.session_state.get("result")
            if not res:
                st.info("결과가 아직 없습니다.")
            else:
                st.success(f"DB 저장: {res.get('db_msg','')}")
                st.markdown("## 1) 구조화된 민원(사실관계)")
                st.json(res.get("case", {}))

                st.markdown("## 2) 법적 근거(정리본)")
                render_law(res.get("law", {}))

                st.markdown("## 3) 처리 전략")
                st.markdown(res.get("strategy", ""))

                st.markdown("## 4) 네이버 참고(옵션)")
                ev = res.get("ev_items", [])
                if not ev:
                    st.caption("참고자료 없음(키/요청 제한/네이버 API 미설정 가능).")
                for item in ev:
                    title = clean_text(item.get("title"))
                    desc = clean_text(item.get("desc"))
                    link = clean_text(item.get("link"))
                    if link:
                        st.markdown(
                            f"<div class='ev-card'><div class='ev-title'><a href='{link}' target='_blank'>{escape(title)}</a></div><div class='ev-desc'>{escape(desc)}</div></div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<div class='ev-card'><div class='ev-title'>{escape(title)}</div><div class='ev-desc'>{escape(desc)}</div></div>",
                            unsafe_allow_html=True
                        )

                with st.expander("🛠️ 법령 후보 루프 디버그", expanded=False):
                    st.json(res.get("loop_debug", []))

        with tab_history:
            if not db.enabled():
                st.info("Supabase 미연결입니다(secrets.toml 확인).")
            else:
                detail = st.session_state.get("history_detail")
                if not detail:
                    st.caption("왼쪽 히스토리에서 실행 기록을 선택 후 '불러오기'를 누르세요.")
                else:
                    st.markdown("### runs row")
                    st.json(detail)

                    arts = detail.get("_artifacts", [])
                    st.markdown("### artifacts")
                    if not arts:
                        st.caption("artifacts 없음")
                    else:
                        # kind별 보기
                        kinds = list(dict.fromkeys([a.get("kind") for a in arts if a.get("kind")]))
                        ksel = st.selectbox("artifact kind", kinds)
                        for a in arts:
                            if a.get("kind") == ksel:
                                st.code(a.get("content", "")[:12000], language="text")


if __name__ == "__main__":
    main()
