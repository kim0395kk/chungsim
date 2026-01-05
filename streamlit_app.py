import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from html import escape as _escape
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st

# Optional deps (없어도 안 죽게)
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


# =========================================================
# 1) Page & Style
# =========================================================
st.set_page_config(layout="wide", page_title="AI Bureau: The Legal Glass", page_icon="⚖️")

st.markdown(
    """
<style>
    .stApp { background-color: #f3f4f6; }

    .paper-sheet {
        background-color: white;
        width: 100%;
        max-width: 210mm;
        min-height: 297mm;
        padding: 25mm;
        margin: auto;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        font-family: 'Batang', serif;
        color: #111;
        line-height: 1.6;
        position: relative;
    }

    .doc-header { text-align: center; font-size: 22pt; font-weight: 900; margin-bottom: 30px; letter-spacing: 2px; }
    .doc-info { display: flex; justify-content: space-between; font-size: 11pt; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; gap: 12px; flex-wrap: wrap; }
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

    .badge {
        display:inline-block; padding: 2px 8px; border-radius: 999px;
        font-size: 0.75rem; border: 1px solid #ddd; background: #fff;
        margin-right: 6px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# 2) Utilities
# =========================================================
def _safe_get(d: dict, path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def mask_pii(s: str) -> str:
    """아주 단순한 마스킹: 전화/주민/차량번호/이메일 등 흔한 패턴 완화."""
    if not s:
        return s

    # 전화번호
    s = re.sub(r"\b(01[016789])[-.\s]?\d{3,4}[-.\s]?\d{4}\b", "010-OOOO-OOOO", s)
    s = re.sub(r"\b(0\d{1,2})[-.\s]?\d{3,4}[-.\s]?\d{4}\b", "0OO-OOOO-OOOO", s)

    # 이메일
    s = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "OOO@OOO.OOO", s)

    # 주민등록번호(대략)
    s = re.sub(r"\b\d{6}[-\s]?\d{7}\b", "OOOOOO-OOOOOOO", s)

    # 차량번호(대략: 12가3456 / 123가4567 등)
    s = re.sub(r"\b\d{2,3}\s?[가-힣]\s?\d{4}\b", "OO가OOOO", s)

    return s


def first_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# =========================================================
# 3) Services
# =========================================================
@dataclass
class ServiceConfig:
    gemini_key: Optional[str]
    groq_key: Optional[str]
    serpapi_key: Optional[str]
    naver_id: Optional[str]
    naver_secret: Optional[str]
    law_oc: Optional[str]
    supabase_url: Optional[str]
    supabase_key: Optional[str]


def load_config() -> ServiceConfig:
    g = st.secrets.get("general", {})
    s = st.secrets.get("supabase", {})
    return ServiceConfig(
        gemini_key=g.get("GEMINI_API_KEY"),
        groq_key=g.get("GROQ_API_KEY"),
        serpapi_key=g.get("SERPAPI_KEY"),
        naver_id=g.get("NAVER_CLIENT_ID"),
        naver_secret=g.get("NAVER_CLIENT_SECRET"),
        law_oc=g.get("LAW_OC"),  # 법령 OpenAPI OC 값 (이메일 ID) 2
        supabase_url=s.get("SUPABASE_URL"),
        supabase_key=s.get("SUPABASE_KEY"),
    )


CFG = load_config()


class LLMService:
    """
    모델 우선순위:
    - Gemini (가능하면 JSON schema)
    - Groq (llama 3.3) fallback
    """

    def __init__(self, cfg: ServiceConfig):
        self.cfg = cfg
        self.gemini_models = [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
        ]

        self.gemini_ok = bool(cfg.gemini_key and genai is not None)
        if self.gemini_ok:
            genai.configure(api_key=cfg.gemini_key)

        self.groq_ok = bool(cfg.groq_key and Groq is not None)
        self.groq_client = Groq(api_key=cfg.groq_key) if self.groq_ok else None

    def _try_gemini(self, prompt: str, is_json: bool = False, schema: Optional[dict] = None) -> Tuple[str, str]:
        if not self.gemini_ok:
            raise RuntimeError("Gemini not available")

        last_err = None
        for model_name in self.gemini_models:
            try:
                model = genai.GenerativeModel(model_name)
                config = None
                if is_json:
                    # Gemini JSON mode
                    config = genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                    )
                res = model.generate_content(prompt, generation_config=config)
                return (res.text or "").strip(), model_name
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"All Gemini models failed: {last_err}")

    def _groq(self, prompt: str) -> str:
        if not self.groq_ok:
            raise RuntimeError("Groq not available")
        completion = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return (completion.choices[0].message.content or "").strip()

    def generate_text(self, prompt: str) -> Tuple[str, str]:
        # returns (text, model_used)
        if self.gemini_ok:
            try:
                t, m = self._try_gemini(prompt, is_json=False)
                return t, m
            except Exception:
                pass
        if self.groq_ok:
            try:
                return self._groq(prompt), "groq:llama-3.3-70b-versatile"
            except Exception:
                pass
        return "시스템 오류: LLM 연결 실패", "none"

    def generate_json(self, prompt: str, schema: Optional[dict] = None) -> Tuple[Optional[dict], str]:
        if self.gemini_ok:
            try:
                t, m = self._try_gemini(prompt, is_json=True, schema=schema)
                return json.loads(t), m
            except Exception:
                pass

        # fallback: strict JSON 요구 후 regex 파싱
        text, model = self.generate_text(prompt + "\n\n반드시 JSON만 출력하세요. JSON 외 텍스트 금지.")
        obj = first_json_object(text)
        return obj, model


class GoogleSearchService:
    """
    SerpApi를 requests로 직접 호출 (패키지 import 불필요)
    - 정밀도가 높고 관련성이 좋음 (사용자 피드백)
    """

    def __init__(self, cfg: ServiceConfig):
        self.api_key = cfg.serpapi_key

    def search(self, query: str, num: int = 5) -> str:
        if not self.api_key:
            return "⚠️ SERPAPI_KEY가 없어 Google 유사 사례를 조회할 수 없습니다."

        q = normalize_text(query)
        # 공공/법령/판례 쪽으로 끌어당기는 “정밀 쿼리 템플릿”
        refined = (
            f"({q}) (판례 OR 행정심판 OR 행정처분 OR 민원답변) "
            f"(site:go.kr OR site:law.go.kr OR site:scourt.go.kr OR site:moj.go.kr OR site:moleg.go.kr)"
        )

        url = "https://serpapi.com/search.json"
        params = {
            "engine": "google",
            "q": refined,
            "api_key": self.api_key,
            "num": max(3, min(num, 10)),
            "hl": "ko",
            "gl": "kr",
        }

        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            items = data.get("organic_results", []) or []

            lines = ["**[Google 유사사례 - 정밀]**"]
            if not items:
                return "Google 검색 결과가 없습니다."

            for it in items[:num]:
                title = it.get("title", "제목 없음")
                snippet = it.get("snippet", "내용 없음")
                link = it.get("link", "#")
                lines.append(f"- **[{title}]({link})**: {snippet}")
            return "\n".join(lines)

        except Exception as e:
            return f"Google 검색 오류: {e}"


class NaverSearchService:
    """
    네이버 Web+News 검색 (보조)
    - '뻘소리' 줄이기 위한 강한 정제 적용
      1) 도메인 우선 가중치(go.kr/law.go.kr/...) 키워드 포함
      2) 불필요 매체/블로그류 강한 배제
      3) title/desc HTML 제거
    """

    def __init__(self, cfg: ServiceConfig):
        self.client_id = cfg.naver_id
        self.client_secret = cfg.naver_secret
        self.web_url = "https://openapi.naver.com/v1/search/webkr.json"
        self.news_url = "https://openapi.naver.com/v1/search/news.json"

    def _headers(self):
        return {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
        }

    @staticmethod
    def _clean_html(s: str) -> str:
        if not s:
            return ""
        s = re.sub(r"</?b>", "", s)
        s = re.sub(r"<[^>]+>", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _naver_search(self, url: str, query: str, display: int = 5) -> dict:
        params = {"query": query, "display": display, "start": 1, "sort": "sim"}
        r = requests.get(url, headers=self._headers(), params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _is_noise(link: str, title: str, desc: str) -> bool:
        blob = f"{link} {title} {desc}".lower()

        # 블로그/카페/마케팅성 강한 도메인 배제(필요하면 더 추가)
        bad = ["blog.", "cafe.", "post.naver", "tistory.com", "brunch.co.kr", "velog.io"]
        if any(b in blob for b in bad):
            return True

        # 너무 홍보성/무관성 패턴(경험 기반 최소)
        if any(k in blob for k in ["쿠폰", "광고", "협찬", "홍보", "이벤트"]):
            return True

        return False

    @staticmethod
    def _score(link: str, title: str, desc: str) -> float:
        score = 0.0
        blob = f"{link} {title} {desc}".lower()

        # 정부/법령/판례 도메인 가중치
        if "law.go.kr" in blob:
            score += 5
        if "go.kr" in blob:
            score += 3
        if "scourt.go.kr" in blob:
            score += 4
        if "moleg.go.kr" in blob:
            score += 4

        # 실무 키워드 가중치
        for kw, w in [("판례", 2), ("행정심판", 2), ("행정처분", 1.5), ("민원", 1.0), ("과태료", 1.0), ("법령", 1.2)]:
            if kw in title or kw in desc:
                score += w

        return score

    def search(self, situation: str, display_each: int = 5) -> str:
        if not self.client_id or not self.client_secret:
            return "⚠️ NAVER_CLIENT_ID / NAVER_CLIENT_SECRET이 없어 Naver 유사 사례를 조회할 수 없습니다."

        q = normalize_text(situation)
        if len(q) > 80:
            q = q[:80] + "…"

        # “정제된 쿼리 템플릿”: 네이버는 이게 중요함
        # - 법령/판례 키워드를 강하게
        # - 도메인 힌트를 텍스트로라도 넣어줌(네이버는 site:가 완전 동일 동작은 아니어도 힌트로 작동)
        refined = f"{q} 행정처분 판례 행정심판 민원 답변 law.go.kr go.kr"

        try:
            web = self._naver_search(self.web_url, refined, display=display_each)
            news = self._naver_search(self.news_url, refined, display=display_each)

            items: List[Tuple[float, str]] = []

            for src_name, payload in [("웹문서", web), ("뉴스", news)]:
                for it in (payload.get("items", []) or []):
                    title = self._clean_html(it.get("title", "제목 없음"))
                    desc = self._clean_html(it.get("description", "내용 없음"))
                    link = it.get("link", "#") or "#"

                    if self._is_noise(link, title, desc):
                        continue

                    sc = self._score(link, title, desc)
                    line = f"- <span class='badge'>{src_name}</span> **[{_escape(title)}]({link})**: {_escape(desc)}"
                    items.append((sc, line))

            if not items:
                return "Naver 검색 결과(정제 후)가 없습니다."

            items.sort(key=lambda x: x[0], reverse=True)
            top = [it[1] for it in items[: min(6, len(items))]]

            return "**[Naver 유사사례 - 정제]**\n" + "\n".join(top)

        except Exception as e:
            return f"Naver 검색 오류: {e}"


class LawResolverService:
    """
    Law Resolver (존재 검증 핵심)
    - LLM이 찍어낸 '법령/조문'을 그대로 믿지 않도록
    - 국가법령정보센터 Open API(DRF)를 통해
      1) 법령명 → 법령ID 탐색
      2) 법령 상세 링크 확보

    가이드에 명시된 목록 조회:
      http://www.law.go.kr/DRF/lawSearch.do?OC=...&target=law&type=JSON&query=... 3
    """

    def __init__(self, cfg: ServiceConfig):
        self.oc = cfg.law_oc
        self.base_search = "http://www.law.go.kr/DRF/lawSearch.do"

    @staticmethod
    def parse_law_basis(text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        '자동차관리법 제26조(…)' 같은 문자열에서
        - law_name: 자동차관리법
        - article: 제26조
        """
        t = normalize_text(text)
        # law name: 앞쪽 한글/공백/영문/괄호 일부 허용
        m = re.search(r"^(.+?)\s+제\s*\d+\s*조", t)
        law_name = m.group(1).strip() if m else None

        m2 = re.search(r"(제\s*\d+\s*조)", t)
        article = m2.group(1).replace(" ", "") if m2 else None
        return law_name, article

    def resolve(self, law_basis_text: str) -> Dict[str, Any]:
        """
        return:
          {
            ok: bool,
            law_name: str,
            article: str|None,
            best_match: {law_id, name, link, ...}|None,
            candidates: [...]
          }
        """
        law_name, article = self.parse_law_basis(law_basis_text)

        if not self.oc:
            return {
                "ok": False,
                "reason": "LAW_OC가 없어 법령 OpenAPI 검증을 수행할 수 없습니다.",
                "law_name": law_name,
                "article": article,
                "best_match": None,
                "candidates": [],
            }

        if not law_name:
            return {
                "ok": False,
                "reason": "법령명 파싱 실패(형식 불명확)",
                "law_name": None,
                "article": article,
                "best_match": None,
                "candidates": [],
            }

        params = {
            "OC": self.oc,
            "target": "law",
            "type": "JSON",
            "query": law_name,
            "display": 5,
            "page": 1,
            "sort": "lasc",
        }

        try:
            r = requests.get(self.base_search, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()

            # 응답 구조는 문서/상황에 따라 다를 수 있어서 방어적으로 처리
            # 보통 data["law"]가 리스트로 내려오는 케이스가 많음.
            raw_candidates = []
            if isinstance(data, dict):
                if isinstance(data.get("law"), list):
                    raw_candidates = data.get("law")
                elif isinstance(_safe_get(data, ["LawSearch", "law"]), list):
                    raw_candidates = _safe_get(data, ["LawSearch", "law"])
                elif isinstance(_safe_get(data, ["lawSearch", "law"]), list):
                    raw_candidates = _safe_get(data, ["lawSearch", "law"])

            candidates = []
            for it in raw_candidates[:5]:
                name = it.get("법령명한글") or it.get("lawName") or it.get("법령명") or ""
                link = it.get("법령상세링크") or it.get("detailLink") or ""
                law_id = it.get("법령ID") or it.get("lawId") or it.get("법령일련번호") or it.get("법령일련번호")
                dept = it.get("소관부처명") or it.get("deptName") or ""
                eff = it.get("시행일자") or it.get("effectiveDate") or ""
                candidates.append(
                    {
                        "name": str(name),
                        "law_id": law_id,
                        "link": str(link),
                        "dept": str(dept),
                        "effective": str(eff),
                    }
                )

            if not candidates:
                return {
                    "ok": False,
                    "reason": f"OpenAPI에서 '{law_name}' 검색 결과 없음",
                    "law_name": law_name,
                    "article": article,
                    "best_match": None,
                    "candidates": [],
                }

            # best match: 가장 이름이 유사한 것(단순)
            best = candidates[0]
            for c in candidates:
                if normalize_text(c["name"]) == normalize_text(law_name):
                    best = c
                    break

            return {
                "ok": True,
                "law_name": law_name,
                "article": article,
                "best_match": best,
                "candidates": candidates,
            }

        except Exception as e:
            return {
                "ok": False,
                "reason": f"법령 OpenAPI 조회 실패: {e}",
                "law_name": law_name,
                "article": article,
                "best_match": None,
                "candidates": [],
            }


class DatabaseService:
    def __init__(self, cfg: ServiceConfig):
        self.is_active = False
        self.client = None
        if not (cfg.supabase_url and cfg.supabase_key and create_client is not None):
            return
        try:
            self.client = create_client(cfg.supabase_url, cfg.supabase_key)
            self.is_active = True
        except Exception:
            self.is_active = False

    def save_log(self, payload: dict) -> str:
        if not self.is_active:
            return "DB 미연결 (저장 건너뜀)"
        try:
            self.client.table("law_reports").insert(payload).execute()
            return "DB 저장 성공"
        except Exception as e:
            return f"DB 저장 실패: {e}"


# Singletons
llm = LLMService(CFG)
google_search = GoogleSearchService(CFG)
naver_search = NaverSearchService(CFG)
law_resolver = LawResolverService(CFG)
db = DatabaseService(CFG)


# =========================================================
# 4) Agents
# =========================================================
class Agents:
    @staticmethod
    def researcher(situation: str) -> Tuple[str, str]:
        situation = mask_pii(situation)

        prompt = f"""
[역할] 대한민국 행정 법률 실무용 법령 식별기
[목표] 아래 상황에 적용될 '법령명 + 조항'을 1개만 제시.

[출력 규칙]
- 인삿말/사족 금지
- 반드시 다음 한 줄 형식:
  법령명 제N조(조문명)
- 모르면 가장 가능성 높은 1개만

[상황]
{situation}
"""
        text, model = llm.generate_text(prompt)
        # 한 줄로 정리(모델이 길게 쓰는 경우 컷)
        line = normalize_text(text).split("\n")[0].strip()
        # 최소 형태 보정
        if "제" not in line or "조" not in line:
            line = normalize_text(text)
        return line, model

    @staticmethod
    def strategist(situation: str, legal_basis_verified: str, search_bundle: str) -> Tuple[str, str]:
        situation = mask_pii(situation)
        prompt = f"""
당신은 행정 실무 베테랑 주무관이다. 아래를 근거로 '처리 전략'을 세운다.

[민원 상황]
{situation}

[법적 근거(검증됨)]
{legal_basis_verified}

[유사 사례/근거 링크]
{search_bundle}

아래 3개 항목을 '마크다운'으로 간결하게 작성:
1) 처리 방향(계도/처분/반려/이첩 등)
2) 핵심 주의사항(절차, 입증, 관할, 기한)
3) 예상 반발 및 대응 논리(짧게)
"""
        return llm.generate_text(prompt)

    @staticmethod
    def clerk_deadline_days(situation: str, legal_basis_verified: str) -> Dict[str, Any]:
        # 기본 15일, 상황에 따라 LLM이 숫자만 출력하도록
        today = datetime.now()

        situation = mask_pii(situation)
        prompt = f"""
오늘: {today.strftime('%Y-%m-%d')}
상황: {situation}
법령: {legal_basis_verified}

질문: 사전통지/이행명령의 의견제출/이행기간을 통상 며칠 부여?
규칙: 숫자만 출력(예: 10). 애매하면 15.
"""
        txt, _ = llm.generate_text(prompt)
        try:
            days = int(re.sub(r"[^0-9]", "", txt) or "15")
        except Exception:
            days = 15

        # 너무 극단값 방지
        if days < 7:
            days = 7
        if days > 30:
            days = 30

        deadline = today + timedelta(days=days)
        return {
            "today_str": today.strftime("%Y. %m. %d."),
            "deadline_str": deadline.strftime("%Y. %m. %d."),
            "days_added": days,
            "doc_num": f"행정-{today.strftime('%Y')}-{int(time.time())%1000:03d}호",
        }

    @staticmethod
    def drafter_document(
        situation: str,
        legal_basis_verified: str,
        meta: Dict[str, Any],
        strategy: str,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        situation = mask_pii(situation)

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
당신은 행정기관 서기다. 아래 정보로 '완결된 공문서'를 작성한다.

[입력]
- 민원상황: {situation}
- 법적근거(검증됨): {legal_basis_verified}
- 시행일자: {meta['today_str']}
- 의견/이행기한: {meta['deadline_str']} ({meta['days_added']}일)
- 처리전략:
{strategy}

[작성 규칙]
- JSON만 출력
- 본문은 문단 배열로
- 본문 구성: (경위)->(근거)->(조치내용/이행요구)->(미이행 시 후속조치)->(권리구제 안내)
- 개인정보는 OOO로 마스킹
"""
        obj, model = llm.generate_json(prompt, schema=schema)
        return obj, model


# =========================================================
# 5) Workflow
# =========================================================
def run_workflow(user_input: str, use_google: bool, use_naver: bool) -> Dict[str, Any]:
    log_placeholder = st.empty()
    logs: List[str] = []

    def add_log(msg: str, style: str = "sys"):
        logs.append(f"<div class='agent-log log-{style}'>{_escape(msg)}</div>")
        log_placeholder.markdown("".join(logs), unsafe_allow_html=True)
        time.sleep(0.2)

    user_input = user_input.strip()

    add_log("🔍 Phase 1: 법령 후보 추출(LLM) ...", "legal")
    raw_basis, model_research = Agents.researcher(user_input)
    add_log(f"📌 후보 법령(LLM): {raw_basis}", "legal")

    add_log("🧷 Phase 1-2: 법령 존재 검증(OpenAPI) ...", "legal")
    verify = law_resolver.resolve(raw_basis)

    if verify.get("ok"):
        best = verify["best_match"] or {}
        # 검증된 표현을 강제 생성: (법령명 + 조항) + 링크
        legal_verified = f"{verify['law_name']} {verify.get('article') or ''}".strip()
        if best.get("link"):
            legal_verified += f"  (참조: {best['link']})"
        add_log("✅ 법령 검증 완료(OpenAPI)", "legal")
    else:
        legal_verified = f"{raw_basis}\n\n⚠️ 검증 실패: {verify.get('reason')}"
        add_log("⚠️ 법령 검증 실패(그래도 진행)", "legal")

    add_log("🟩 Phase 2: 유사사례 검색 ...", "search")
    search_parts = []
    if use_google:
        search_parts.append(google_search.search(user_input, num=5))
    if use_naver:
        search_parts.append(naver_search.search(user_input, display_each=5))

    search_bundle = "\n\n---\n\n".join(search_parts) if search_parts else "검색 사용 안함"
    add_log("✅ 유사사례 수집 완료", "search")

    add_log("🧠 Phase 3: 처리 전략 수립 ...", "strat")
    strategy, model_strat = Agents.strategist(user_input, legal_verified, search_bundle)
    add_log("✅ 전략 수립 완료", "strat")

    add_log("📅 Phase 4: 기한 산정 ...", "calc")
    meta = Agents.clerk_deadline_days(user_input, legal_verified)

    add_log("✍️ Phase 5: 공문 JSON 작성 ...", "draft")
    doc, model_doc = Agents.drafter_document(user_input, legal_verified, meta, strategy)

    add_log("💾 Phase 6: 저장(Supabase) ...", "sys")
    payload = {
        "created_at": datetime.now().isoformat(),
        "situation": mask_pii(user_input),
        "law_name": legal_verified,
        "summary": json.dumps(
            {
                "raw_basis": raw_basis,
                "verify": verify,
                "strategy": strategy,
                "search_bundle": search_bundle,
                "doc": doc,
                "models": {
                    "research": model_research,
                    "strategy": model_strat,
                    "doc": model_doc,
                },
            },
            ensure_ascii=False,
        ),
    }
    save_msg = db.save_log(payload)

    add_log(f"✅ 완료 ({save_msg})", "sys")
    time.sleep(0.6)
    log_placeholder.empty()

    return {
        "raw_basis": raw_basis,
        "verify": verify,
        "law_verified": legal_verified,
        "search_bundle": search_bundle,
        "strategy": strategy,
        "meta": meta,
        "doc": doc,
        "save_msg": save_msg,
        "models": {
            "research": model_research,
            "strategy": model_strat,
            "doc": model_doc,
        },
    }


# =========================================================
# 6) UI
# =========================================================
def render_doc_preview(doc: dict, meta: dict):
    title = _escape(doc.get("title", "공 문 서"))
    receiver = _escape(doc.get("receiver", "수신자 참조"))
    dept_head = _escape(doc.get("department_head", "행정기관장"))

    html = f"""
<div class="paper-sheet">
  <div class="stamp">직인생략</div>
  <div class="doc-header">{title}</div>
  <div class="doc-info">
    <span>문서번호: {_escape(meta.get('doc_num','-'))}</span>
    <span>시행일자: {_escape(meta.get('today_str','-'))}</span>
    <span>수신: {receiver}</span>
  </div>
  <hr style="border: 1px solid black; margin-bottom: 30px;">
  <div class="doc-body">
"""

    paragraphs = doc.get("body_paragraphs", [])
    if isinstance(paragraphs, str):
        paragraphs = [paragraphs]
    if not isinstance(paragraphs, list):
        paragraphs = ["(본문 생성 실패)"]

    for p in paragraphs:
        html += f"<p style='margin-bottom: 15px;'>{_escape(str(p))}</p>"

    html += f"""
  </div>
  <div class="doc-footer">{dept_head}</div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


def main():
    col_left, col_right = st.columns([1, 1.2], vertical_alignment="top")

    with col_left:
        st.title("🏢 AI 행정관 Pro (Legal Glass)")
        st.caption("Gemini/Groq + 법령 OpenAPI 검증 + Google/Naver 유사사례 + 공문 JSON + Supabase(옵션)")

        st.markdown("---")

        # 옵션
        with st.expander("⚙️ 검색 옵션", expanded=True):
            use_google = st.toggle("Google 유사사례(정밀, 추천)", value=True)
            use_naver = st.toggle("Naver 유사사례(보조, 정제 적용)", value=False)
            st.caption("※ 네이버는 정제/필터 적용했지만, Google이 기본적으로 더 안정적입니다.")

        st.markdown("### 🗣️ 업무 지시")
        user_input = st.text_area(
            "업무 내용",
            height=160,
            placeholder="예시:\n- 아파트 단지 내 소방차 전용구역 불법 주차 차량에 대해 과태료 부과 사전통지 공문 작성\n- 건설기계 차고지 외 주기위반 민원 답변서 작성",
            label_visibility="collapsed",
        )

        run = st.button("⚡ 스마트 분석 시작", type="primary", use_container_width=True)

        if run:
            if not user_input.strip():
                st.warning("내용을 입력해주세요.")
            else:
                try:
                    with st.spinner("AI 에이전트 팀이 협업 중입니다..."):
                        st.session_state["workflow_result"] = run_workflow(user_input, use_google, use_naver)
                except Exception as e:
                    st.error(f"시스템 오류 발생: {e}")

        if "workflow_result" in st.session_state:
            res = st.session_state["workflow_result"]
            st.markdown("---")

            if "성공" in res.get("save_msg", ""):
                st.success(f"✅ {res['save_msg']}")
            else:
                st.info(f"ℹ️ {res.get('save_msg','(저장 메시지 없음)')}")

            # 법령 및 검증
            with st.expander("✅ [검토] 법령(후보/검증) & 유사사례", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**📜 법령 후보(LLM)**")
                    st.code(res.get("raw_basis", ""), language="text")
                    st.markdown("**🧷 법령 검증(OpenAPI 기반)**")
                    st.code(res.get("law_verified", ""), language="text")

                    v = res.get("verify", {}) or {}
                    if v.get("candidates"):
                        st.caption("OpenAPI 후보(상위)")
                        st.json(v.get("candidates")[:3])

                with c2:
                    st.markdown("**🔎 유사사례**")
                    st.markdown(res.get("search_bundle", "검색 결과 없음"), unsafe_allow_html=True)

            # 전략
            with st.expander("🧭 [방향] 업무 처리 가이드라인", expanded=True):
                st.markdown(res.get("strategy", ""))

    with col_right:
        if "workflow_result" not in st.session_state:
            st.markdown(
                """<div style='text-align: center; padding: 100px; color: #aaa; background: white; border-radius: 10px; border: 2px dashed #ddd;'>
<h3>📄 Document Preview</h3><p>왼쪽에서 업무를 지시하면<br>완성된 공문서가 여기에 나타납니다.</p></div>""",
                unsafe_allow_html=True,
            )
            return

        res = st.session_state["workflow_result"]
        doc = res.get("doc")
        meta = res.get("meta") or {}

        if not doc:
            st.error("문서 생성 실패: doc_data가 비어있습니다.")
            st.json(res)
            return

        render_doc_preview(doc, meta)

        with st.expander("🧾 [JSON] 생성된 문서 데이터", expanded=False):
            st.json(doc)

        with st.expander("🧠 [모델 사용] 추적", expanded=False):
            st.json(res.get("models", {}))


if __name__ == "__main__":
    main()
