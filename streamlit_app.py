# app.py — AI 행정관 Pro (Stable / Dual-Model Router)
# Groq: qwen/qwen3-32b (FAST) + llama-3.3-70b-versatile (STRICT)
# LAWGO(DRF) + NAVER + Supabase + Anti-crash patches
#
# ✅ FAST(default): Planner/Strategy
# ✅ STRICT(fallback/critical): Drafter(JSON), Planner JSON fail, Strategy when law not confirmed
# ✅ JSON 안정화: 재시도 + STRICT 승급
# ✅ UI 튐 방지: HTML sanitize + components.html 버그 수정
# ✅ Metrics: 모델별 호출 + (가능하면) total_tokens 합산

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
    page_title="AI 행정관 Pro (Dual v5.0)",
    page_icon="⚖️",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.stApp { background-color: #f8f9fa; }
/* 문서 스타일 */
.paper-sheet {
  background: #fff; width: 100%; max-width: 210mm; min-height: 297mm;
  padding: 25mm; margin: auto; box-shadow: 0 4px 15px rgba(0,0,0,0.08);
  font-family: 'Noto Serif KR','Nanum Myeongjo',serif;
  color:#111; line-height:1.6; position:relative;
}
.doc-header { text-align:center; font-size:24pt; font-weight:800; margin-bottom:35px; letter-spacing:2px; }
.doc-info {
  display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap;
  font-size:11pt; border-bottom:2px solid #333; padding-bottom:12px; margin-bottom:25px;
}
.doc-body { font-size:12pt; text-align: justify; }
.doc-footer { text-align:center; font-size:22pt; font-weight:bold; margin-top:80px; letter-spacing:4px; }
.stamp {
  position:absolute; bottom:85px; right:80px; border:3px solid #d32f2f; color: #d32f2f;
  padding:6px 12px; font-size:14pt; font-weight:bold; transform:rotate(-15deg);
  opacity:0.8; border-radius:4px; font-family: 'Nanum Gothic', sans-serif;
}
/* 로그 스타일 */
.agent-log {
  font-family: 'Pretendard', sans-serif; font-size: 0.9rem; padding: 8px 12px;
  border-radius: 6px; margin-bottom: 6px; background: white; border: 1px solid #e5e7eb;
}
.log-legal { border-left: 4px solid #3b82f6; color: #1e40af; }
.log-search { border-left: 4px solid #f97316; color: #c2410c; }
.log-strat { border-left: 4px solid #8b5cf6; color: #6d28d9; }
.log-draft { border-left: 4px solid #ef4444; color: #991b1b; }
.log-sys   { border-left: 4px solid #9ca3af; color: #4b5563; }
.small-muted { color:#6b7280; font-size:12px; }
</style>
""",
    unsafe_allow_html=True,
)

_TAG_RE = re.compile(r"<[^>]+>")


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
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", s)
    return s.strip()


def safe_html(value) -> str:
    return escape(clean_text(value), quote=False).replace("\n", "<br>")


def truncate_text(s: str, max_chars: int = 2500) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n...(내용 축소됨)"


def ensure_doc_shape(doc):
    """LLM 응답이 깨졌을 때 기본값 보장"""
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
    """Supabase 저장 시 터지지 않게 직렬화"""
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return "{}"


def extract_keywords_kor(text: str, max_k: int = 6) -> list[str]:
    if not text:
        return []
    t = re.sub(r"[^가-힣A-Za-z0-9\s]", " ", text)
    words = re.findall(r"[가-힣A-Za-z0-9]{2,12}", t)
    stop = set(["그리고", "관련", "문의", "사항", "대하여", "대한", "처리", "요청", "작성", "안내", "검토", "불편", "민원", "신청", "발급", "제출"])
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


def metrics_add(model_name: str, tokens_total: int | None = None):
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

        # ✅ 모델명: 사용자 스샷 기준
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

        # metrics
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

        # 1) 1차
        try:
            return self._chat(model_first, messages, temp, json_mode=False)
        except Exception:
            pass

        # 2) fast 실패면 strict 승급
        if prefer == "fast":
            try:
                return self._chat(self.model_strict, messages, temp, json_mode=False)
            except Exception as e2:
                return f"LLM Error: {e2}"

        return "LLM Error"

    def generate_json(self, prompt: str, prefer: str = "fast", temp: float = 0.1, max_retry: int = 2) -> dict:
        if not self.client:
            return {}

        sys_json = "Output JSON only. No markdown. No explanation. Follow the schema exactly."
        messages = [
            {"role": "system", "content": sys_json},
            {"role": "user", "content": prompt},
        ]

        model_first = self.model_fast if prefer == "fast" else self.model_strict

        # 1) 같은 모델 재시도
        for _ in range(max_retry):
            try:
                txt = self._chat(model_first, messages, temp, json_mode=True)
                js = self._parse_json(txt)
                if js:
                    return js
            except Exception:
                pass

        # 2) strict 승급
        try:
            txt = self._chat(self.model_strict, messages, temp, json_mode=True)
            js = self._parse_json(txt)
            return js if js else {}
        except Exception:
            return {}


llm_service = LLMService()


# =========================
# 5) LAW API (DRF) — Search + Service (XML)
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

            # LawSearch > law
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
                        "lawId": it.get("법령ID") or it.get("lawId") or it.get("id") or "",
                        "link": it.get("법령상세링크") or it.get("link") or "",
                    }
                )
            # 기본 필터
            return [x for x in out if clean_text(x.get("lawNm"))]
        except Exception:
            return []

    def get_article_text_by_mst(self, mst: str, article_no: str | None = None) -> str:
        """
        DRF lawService는 MST로 가져오는 게 안정적.
        article_no는 "숫자"로 넣으면 최대한 매칭 시도.
        """
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

            if not article_no:
                # 특정 조문 없이도 반환(일부라도)
                # 너무 길면 컷
                raw = clean_text(r.text)
                return raw[:4000]

            tgt = re.sub(r"[^0-9]", "", str(article_no))
            if not tgt:
                return ""

            # 조문 찾기
            for art in articles:
                if not isinstance(art, dict):
                    continue
                an = clean_text(art.get("@조문번호") or "")
                at = clean_text(art.get("ArticleTitle") or "")
                # 매칭
                if tgt == re.sub(r"[^0-9]", "", an) or (tgt and f"제{tgt}조" in at):
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
# 8) Workflow
# =========================
def run_workflow(user_input: str, dept: str, officer: str):
    log_area = st.empty()
    logs = []

    def add_log(msg: str, style: str = "sys"):
        logs.append(f"<div class='agent-log log-{style}'>{safe_html(msg)}</div>")
        log_area.markdown("".join(logs), unsafe_allow_html=True)
        time.sleep(0.05)

    # ---- Phase 0: cheap keyword extraction (no LLM)
    kw_fallback = extract_keywords_kor(user_input, max_k=6)

    # 1) Planner (FAST/Qwen)
    add_log("🧭 [Planner] 업무 분석 및 법령/검색어 추출 (FAST: qwen/qwen3-32b)...", "sys")
    prompt_plan = f"""
입력: "{user_input}"

아래 스키마를 정확히 지켜 JSON만 출력:
{{
  "task_type": "업무유형(짧게)",
  "law_hint": {{
    "law_name": "법령명(가능하면 공식명)",
    "article_no": "조번호(숫자만, 모르면 빈문자열)"
  }},
  "keywords": ["검색어1","검색어2","검색어3"]
}}

주의:
- law_name이 확신 없으면 빈 문자열로.
- keywords는 상황핵심 위주.
"""
    plan = llm_service.generate_json(prompt_plan, prefer="fast", max_retry=2)
    if not plan:
        plan = {"task_type": "업무", "law_hint": {"law_name": "", "article_no": ""}, "keywords": kw_fallback[:3]}

    # Plan 보정
    task_type = clean_text(plan.get("task_type") or "업무")
    law_hint = plan.get("law_hint") if isinstance(plan.get("law_hint"), dict) else {}
    law_name = clean_text(law_hint.get("law_name") or "")
    art_no = clean_text(law_hint.get("article_no") or "")
    keywords = plan.get("keywords") if isinstance(plan.get("keywords"), list) else []
    keywords = [clean_text(x) for x in keywords if clean_text(x)]
    if not keywords:
        keywords = kw_fallback[:3]

    # 2) Law Search (Rule-first: if law_name empty -> try keywords)
    add_log("📚 [Law] 법령 검색 및 조문 확인...", "legal")
    legal_basis = "법령 정보를 찾을 수 없습니다."
    legal_status = "PENDING"
    law_debug = {}

    # 후보 쿼리 구성
    law_queries = []
    if law_name:
        law_queries.append(law_name)
    # fallback: keywords-based
    for k in keywords[:3]:
        if k and k not in law_queries:
            law_queries.append(k)

    chosen = None
    for q in law_queries[:4]:
        candidates = law_api.search_law(q, display=10)
        if candidates:
            chosen = candidates[0]
            break

    if chosen:
        nm = clean_text(chosen.get("lawNm") or "")
        mst = clean_text(chosen.get("MST") or "")
        link = clean_text(chosen.get("link") or "")
        # 조문 텍스트
        full_text = law_api.get_article_text_by_mst(mst, art_no if art_no else None)
        if full_text and len(full_text) >= 20:
            # 대표 문구
            if art_no:
                legal_basis = f"{nm} 제{re.sub(r'[^0-9]', '', art_no)}조\n{truncate_text(full_text, 2500)}"
            else:
                legal_basis = f"{nm}\n{truncate_text(full_text, 2500)}"
            legal_status = "CONFIRMED"
            law_debug = {"mst": mst, "name": nm, "link": link}
        else:
            legal_basis = f"법령({nm})은 찾았으나 조문 원문 확보 실패."
            legal_status = "WEAK"
            law_debug = {"mst": mst, "name": nm, "link": link}
    else:
        legal_basis = "관련 법령 검색 실패(후보 없음)."
        legal_status = "FAIL"

    # 3) Naver Evidence (fast, no LLM)
    add_log("🌍 [Search] 사실관계 및 리스크 점검 (Naver)...", "search")
    ev_text = ""
    ev_items = []

    if keywords:
        q = " ".join(keywords[:2]) + " 행정처분"
        raw_items = naver_search.search(q, cat="news", display=5)
        for item in raw_items:
            clean_t = clean_text(item.get("title"))
            clean_d = clean_text(item.get("description"))
            link = clean_text(item.get("link"))
            ev_items.append({"title": clean_t, "link": link, "desc": clean_d})
            ev_text += f"- {clean_t}: {clean_d}\n"

    # 4) Strategy (FAST by default, STRICT if law weak)
    prefer_strat = "strict" if legal_status != "CONFIRMED" else "fast"
    add_log(f"🧠 [Analyst] 처리 전략 수립 ({'STRICT: llama-3.3-70b' if prefer_strat=='strict' else 'FAST: qwen/qwen3-32b'})...", "strat")
    prompt_strat = f"""
[업무유형] {task_type}
[상황] {user_input}

[법적근거]
{legal_basis}

[참고(네이버)]
{truncate_text(ev_text, 900)}

아래 형식으로만 작성(마크다운):
1) 처리 방향 (3~6줄)
2) 핵심 체크리스트 (불릿 5~10개)
3) 예상 민원/반발 & 대응 (3~6줄)

주의:
- 과도한 일반론 금지
- 모르면 "추가 확인 필요"를 명시
"""
    strategy = llm_service.generate_text(prompt_strat, prefer=prefer_strat, temp=0.1)

    # 5) Drafter (STRICT always)
    add_log("✍️ [Drafter] 공문서 초안 작성 (STRICT: llama-3.3-70b-versatile)...", "draft")
    today_str = datetime.now().strftime("%Y. %m. %d.")
    doc_num = f"행정-{datetime.now().strftime('%Y')}-{int(time.time()) % 10000:04d}호"

    prompt_draft = f"""
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

민원/업무 상황:
{user_input}

법적 근거(확보된 범위):
{legal_basis}

처리 전략(요약):
{truncate_text(strategy, 900)}

작성 원칙:
- 문서 톤: 건조/정중, 불필요한 수사 금지
- 본문 구조: [경위] -> [근거] -> [조치/안내] -> [권리구제/문의]
- 개인정보는 OOO로 마스킹(있으면)
- 법령 원문이 불확실하면 '추가 확인 필요' 문구를 포함
"""
    doc_json = llm_service.generate_json(prompt_draft, prefer="strict", max_retry=2)
    doc_final = ensure_doc_shape(doc_json)

    # 6) Save to DB
    add_log("💾 [System] 결과 저장 중...", "sys")
    payload = {
        "created_at": datetime.now().isoformat(),
        "dept": dept,
        "officer": officer,
        "task_type": task_type,
        "keywords": safe_json_dump(keywords),
        "input": user_input,
        "legal_status": legal_status,
        "legal_basis": legal_basis,
        "final_doc": safe_json_dump(doc_final),
        "strategy": strategy,
        "provenance": safe_json_dump(ev_items),
        "model_last": llm_service.last_model,
        "metrics": safe_json_dump(st.session_state.get("metrics", {})),
        "law_debug": safe_json_dump(law_debug),
    }
    db_msg = db_service.save_log(payload)
    add_log(f"✅ 완료 ({db_msg})", "sys")

    time.sleep(0.35)
    log_area.empty()

    return {
        "doc": doc_final,
        "meta": {"doc_num": doc_num, "today": today_str, "dept": dept, "officer": officer},
        "legal_basis": legal_basis,
        "legal_status": legal_status,
        "strategy": strategy,
        "ev_items": ev_items,
        "task_type": task_type,
        "keywords": keywords,
        "db_msg": db_msg,
        "law_debug": law_debug,
    }


# =========================
# 9) UI
# =========================
def main():
    st.session_state.setdefault("dept", "OO시청 OO과")
    st.session_state.setdefault("officer", "김주무관")

    col_l, col_r = st.columns([1, 1.2], gap="large")

    with col_l:
        st.title("AI 행정관 Pro")
        st.caption("Dual Router v5.0 — FAST(qwen/qwen3-32b) + STRICT(llama-3.3-70b)")
        st.markdown("---")

        with st.expander("📝 사용자 정보 설정", expanded=False):
            st.text_input("부서명", key="dept")
            st.text_input("담당자", key="officer")

        user_input = st.text_area(
            "업무 지시 사항",
            height=200,
            placeholder="예: 불법주정차 과태료 부과에 대한 이의신청 기각 통지서 작성해줘.",
        )

        if st.button("🚀 문서 생성 실행", type="primary", use_container_width=True):
            if not user_input.strip():
                st.warning("내용을 입력하세요.")
            else:
                with st.spinner("AI 에이전트 협업 중..."):
                    try:
                        res = run_workflow(user_input.strip(), st.session_state["dept"], st.session_state["officer"])
                        st.session_state["result"] = res
                    except Exception as e:
                        st.error(f"치명적 오류 발생: {e}")

        # Metrics Dashboard
        st.markdown("---")
        st.subheader("📊 사용량(세션 기준)")
        m = st.session_state.get("metrics", {})
        calls = m.get("calls", {})
        tokens_total = m.get("tokens_total", 0)

        if calls:
            for k, v in sorted(calls.items(), key=lambda x: (-x[1], x[0])):
                st.write(f"- **{k}**: {v}회")
            st.caption(f"총 토큰(가능한 경우): {tokens_total}")
        else:
            st.info("대기 중...")

        st.markdown("<div class='small-muted'>TIP: Planner/Strategy는 FAST, 공문(JSON)은 STRICT로 고정되어 품질-속도 균형을 맞춥니다.</div>", unsafe_allow_html=True)

    with col_r:
        res = st.session_state.get("result")

        if not res:
            st.markdown(
                """
<div style='text-align: center; padding: 120px 20px; color: #aaa; border: 2px dashed #ddd; border-radius: 12px; background:#fff;'>
  <h3>📄 Document Preview</h3>
  <p>왼쪽에서 업무를 입력하고 실행 버튼을 누르세요.<br>자동으로 법령을 검토하고 공문을 작성합니다.</p>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            doc = res["doc"]
            meta = res["meta"]

            tab1, tab2 = st.tabs(["📄 공문서 결과", "🔍 근거 및 분석"])

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
  <div class="doc-body">
    {body_html}
  </div>
  <div class="doc-footer">{safe_html(doc['department_head'])}</div>
</div>
"""
                # ✅ components.html 버그 수정: 불필요한 <head> CSS 삽입 제거
                components.html(html, height=880, scrolling=True)

            with tab2:
                st.success(f"DB: {res.get('db_msg','')}")
                st.info(f"📜 법적 근거 상태: {res.get('legal_status')}")
                st.info(f"📜 법적 근거:\n{res['legal_basis']}")

                st.markdown("### 💡 처리 전략")
                st.markdown(res["strategy"])

                st.markdown("### 🔎 키워드")
                st.write(res.get("keywords", []))

                st.markdown("### 📎 참고 자료 (Naver)")
                for item in res["ev_items"]:
                    title = clean_text(item.get("title"))
                    link = clean_text(item.get("link"))
                    desc = clean_text(item.get("desc"))
                    if link:
                        st.markdown(f"- [{title}]({link}) — {desc}")
                    else:
                        st.markdown(f"- {title} — {desc}")

                with st.expander("🛠️ 디버그(법령)", expanded=False):
                    st.code(safe_json_dump(res.get("law_debug", {})), language="json")


if __name__ == "__main__":
    main()
```0
