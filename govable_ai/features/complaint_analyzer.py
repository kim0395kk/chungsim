# -*- coding: utf-8 -*-
"""
민원 분석기 (Complaint Analyzer) — UI-free workflow.

Extracted from ``streamlit_app_legacy.run_complaint_analyzer_workflow`` (~L2297).

Pipeline (5 phases):
  1) Claim decomposition — LLM splits 민원 텍스트 into MVC + claim units.
  2) Hallucination/MVC completeness signals (rule-based).
  3) Law citation verification via 국가법령정보센터 official API.
  4) Per-claim safe verdict (LLM, conservative — INSUFFICIENT by default).
  5) Reply draft assembly (단정 최소 형태).

Dependencies are injected — no streamlit imports.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers (extracted from legacy module-level utilities)
# =============================================================================
def mask_sensitive(text: str) -> str:
    """전화/주민/차량번호 등 명백한 민감정보 패턴을 마스킹."""
    if not text:
        return ""
    t = text
    t = re.sub(r"\b0\d{1,2}-\d{3,4}-\d{4}\b", "0**-****-****", t)
    t = re.sub(r"\b\d{6}-\d{7}\b", "******-*******", t)
    t = re.sub(r"\b\d{2,3}[가-힣]\d{4}\b", "***(차량번호)", t)
    return t


def _mvc_completion(mvc: dict) -> Tuple[int, int]:
    keys = ["time", "place", "target", "request"]
    filled = sum(1 for k in keys if (mvc.get(k) or "").strip())
    ev = mvc.get("evidence")
    if isinstance(ev, list) and len(ev) > 0:
        filled += 1
    elif isinstance(ev, str) and ev.strip():
        filled += 1
    return filled, 5


def _normalize_article(article_val) -> Optional[dict]:
    if article_val is None:
        return None
    s = str(article_val).strip()
    if not s:
        return None
    digits = re.sub(r"\D", "", s)
    return {"raw": s, "digits": (digits or None)}


def _dedup(lines: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in lines:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _clerk_meta() -> dict:
    today = datetime.now()
    return {
        "today_str": today.strftime("%Y. %m. %d."),
        "doc_num": f"행정-{today.strftime('%Y')}-{int(time.time()) % 1000:03d}호",
    }


# =============================================================================
# Prompt builders
# =============================================================================
def _build_claim_decompose_prompt(masked_text: str) -> str:
    return f"""
너는 '민원 입력 품질 분석관'이다.
아래 민원 텍스트를 **주장 단위로 쪼개고**, 사실요건(MVC) 충족 여부를 구조화하라.
- 환각/추정 가능성이 있는 문장은 LEGAL/FACT로 구분하되, '단정'하지 마라.
- 법령/조문이 등장하면 citations에 넣되, **확실하지 않으면 null/빈값으로 남겨라.**
- 출력은 JSON만.

[민원 텍스트]
{masked_text}

[출력 JSON 스키마]
{{
  "mvc": {{
    "time": "언제(모르면 빈문자)",
    "place": "어디(모르면 빈문자)",
    "target": "대상(기관/사람/차량/시설 등, 모르면 빈문자)",
    "request": "민원인이 원하는 것(모르면 빈문자)",
    "evidence": ["사진/영상/문서/링크 등(없으면 빈배열)"]
  }},
  "claims": [
    {{
      "id": "C1",
      "type": "FACT|LEGAL|REQUEST|OPINION",
      "text": "주장 내용",
      "citations": [{{"law_name": "정식 법령명", "article": "조문(예: 26 또는 57-2, 없으면 빈문자)"}}],
      "notes": "모순/추정/감정적 수사 등 메모(없으면 빈문자)"
    }}
  ],
  "possible_hallucination_signals": ["환각 가능 신호(없으면 빈배열)"]
}}
"""


def _build_judge_prompt(cid: str, ctype: str, ctext: str, cnotes: str, rel_text: str) -> str:
    return f"""
너는 '민원 주장 검증 보조관'이다.
중요: 너는 사실을 새로 만들면 안 된다. 아래 근거가 부족하면 반드시 INSUFFICIENT로 판단한다.
REFUTED(반박)은 근거가 명확할 때만 선택하며, 불확실하면 INSUFFICIENT로 둔다.

[주장]
- id: {cid}
- type: {ctype}
- text: {ctext}
- notes: {cnotes}

[가용 근거(법령 발췌/검증 상태)]
{rel_text if rel_text else "(관련 근거 없음)"}

[출력 JSON]
{{
  "verdict": "SUPPORTED|INSUFFICIENT|REFUTED",
  "confidence": 0.0,
  "safe_statement": "공무원이 책임질 수 있는 안전한 문장(단정 최소)",
  "needed": ["추가 제출/확인 항목 3~7개"]
}}
"""


# =============================================================================
# Phase 1 — Claim decomposition
# =============================================================================
def _phase1_decompose(llm_service: Any, masked_text: str) -> Tuple[dict, List[dict], List[str]]:
    parsed = llm_service.generate_json(_build_claim_decompose_prompt(masked_text)) or {}
    if not isinstance(parsed, dict):
        parsed = {}

    mvc = parsed.get("mvc") if isinstance(parsed.get("mvc"), dict) else {}
    claims = parsed.get("claims") if isinstance(parsed.get("claims"), list) else []
    halluc_signals = parsed.get("possible_hallucination_signals")
    if not isinstance(halluc_signals, list):
        halluc_signals = []

    if not claims:
        claims = [{
            "id": "C1",
            "type": "FACT",
            "text": masked_text[:500],
            "citations": [],
            "notes": "자동 주장 분해 실패(원문 요약)",
        }]
    return mvc, claims, halluc_signals


# =============================================================================
# Phase 2 — MVC completeness + grade preliminary
# =============================================================================
def _phase2_signals(mvc: dict, claims: List[dict], halluc_signals: List[str]):
    filled, total = _mvc_completion(mvc)
    verifiability_score = round(filled / max(total, 1), 2)

    citation_items: List[dict] = []
    for c in claims:
        cits = c.get("citations") or []
        if isinstance(cits, dict):
            cits = [cits]
        if isinstance(cits, list):
            for it in cits:
                if not isinstance(it, dict):
                    continue
                law_name = (it.get("law_name") or "").strip()
                art = (it.get("article") or "").strip()
                if law_name:
                    citation_items.append({
                        "law_name": law_name,
                        "article": _normalize_article(art) if art else None,
                        "claim_id": c.get("id") or "",
                    })

    noise_grade = "GREEN"
    grade_reasons: List[str] = []
    if verifiability_score <= 0.4:
        noise_grade = "YELLOW"
        grade_reasons.append("필수 사실요소(언제/어디/대상/요청/증거) 중 다수가 누락됨")
    if len(halluc_signals) >= 3 and noise_grade != "RED":
        noise_grade = "YELLOW"
        grade_reasons.append("환각/추정 신호가 다수 감지됨(보완요구 권장)")

    return verifiability_score, citation_items, noise_grade, grade_reasons


# =============================================================================
# Phase 3 — Law citation verification (official API)
# =============================================================================
_ERROR_KEYWORDS = ["검색 결과가 없습니다", "API ID", "오류", "실패", "찾지 못했습니다", "No results"]
_PARTIAL_KEYWORDS = ["자동 추출 실패", "조문번호 미지정", "법령일련번호(MST) 추출 실패"]


def _phase3_verify_citations(law_api: Any, citation_items: List[dict]) -> Tuple[List[dict], int]:
    verified: List[dict] = []
    invalid_count = 0

    if not citation_items or law_api is None:
        return verified, invalid_count

    cache: Dict[Tuple[str, str], Tuple[str, str, str]] = {}
    for it in citation_items[:12]:
        law_name = it["law_name"]
        art = it.get("article")
        digits = art.get("digits") if isinstance(art, dict) else None
        cache_key = (law_name, digits or "")

        if cache_key in cache:
            law_text, link, status = cache[cache_key]
        else:
            article_num = digits if digits else None
            try:
                result = law_api.get_law_text(law_name, article_num, return_link=True)
                if isinstance(result, tuple):
                    law_text, link = result
                else:
                    law_text, link = result, ""
            except Exception as e:
                logger.warning("law_api.get_law_text failed: %s", e)
                law_text, link = f"[오류] {e}", ""

            txt = law_text or ""
            if any(k in txt for k in _ERROR_KEYWORDS):
                status = "INVALID"
            elif any(k in txt for k in _PARTIAL_KEYWORDS):
                status = "PARTIAL"
            else:
                status = "VALID"
            cache[cache_key] = (law_text, link, status)

        if status == "INVALID":
            invalid_count += 1

        verified.append({
            "claim_id": it.get("claim_id"),
            "law_name": law_name,
            "article_raw": (art.get("raw") if isinstance(art, dict) else None),
            "article_digits": digits,
            "status": status,
            "link": link,
            "excerpt": (law_text or "")[:900],
        })
    return verified, invalid_count


def _law_pack_markdown(verified_citations: List[dict]) -> str:
    lines = ["##### ⚖️ 법령 인용 검증 결과", "---"]
    if verified_citations:
        for v in verified_citations:
            nm = v["law_name"]
            link = v.get("link")
            status = v.get("status")
            art_raw = v.get("article_raw") or ""
            title = f"[{nm}]({link})" if link else nm
            badge = "✅" if status == "VALID" else ("🟨" if status == "PARTIAL" else "❌")
            lines.append(f"- {badge} **{title}** {('(' + art_raw + ')') if art_raw else ''}  \n  - 상태: {status}")
    else:
        lines.append("- (민원 텍스트에서 명시적 법령 인용이 없거나 추출하지 못했습니다.)")
    return "\n".join(lines)


# =============================================================================
# Phase 4 — Safe verdicts per claim
# =============================================================================
def _phase4_verdicts(
    llm_service: Any,
    claims: List[dict],
    verified_citations: List[dict],
    noise_grade: str,
) -> List[dict]:
    verdicts: List[dict] = []
    for c in claims[:12]:
        cid = c.get("id") or ""
        ctype = (c.get("type") or "FACT").strip().upper()
        ctext = (c.get("text") or "").strip()
        cnotes = (c.get("notes") or "").strip()

        rel = [v for v in verified_citations if v.get("claim_id") == cid]
        rel_text = ""
        for v in rel[:2]:
            rel_text += f"- {v.get('law_name')} ({v.get('article_raw') or ''}) [{v.get('status')}]\n"
            rel_text += f"  EXCERPT: {v.get('excerpt', '')[:400]}\n"

        vj = llm_service.generate_json(_build_judge_prompt(cid, ctype, ctext, cnotes, rel_text)) or {}
        if not isinstance(vj, dict):
            vj = {}
        verdict = (vj.get("verdict") or "INSUFFICIENT").strip().upper()
        if verdict not in ("SUPPORTED", "INSUFFICIENT", "REFUTED"):
            verdict = "INSUFFICIENT"
        # 노이즈 등급이 높을수록 REFUTED 단정을 회피.
        if noise_grade in ("YELLOW", "RED") and verdict == "REFUTED":
            verdict = "INSUFFICIENT"

        needed = vj.get("needed")
        if not isinstance(needed, list):
            needed = []
        safe_stmt = (vj.get("safe_statement") or "").strip() or \
            "제출된 자료 범위 내에서는 해당 주장에 대해 단정하기 어렵습니다."

        verdicts.append({
            "claim_id": cid,
            "type": ctype,
            "text": ctext,
            "verdict": verdict,
            "confidence": float(vj.get("confidence") or 0.5),
            "safe_statement": safe_stmt,
            "needed": needed[:10],
        })
    return verdicts


# =============================================================================
# Phase 5 — Reply draft assembly (단정 최소)
# =============================================================================
_GRADE_TITLES = {
    "GREEN": "민원 검토 결과 안내(초안)",
    "YELLOW": "민원 처리 관련 추가자료 요청(보완요구) (초안)",
    "RED": "민원 내용 확인 및 절차 안내(요건/관할 검토) (초안)",
}


def _phase5_draft(mvc: dict, verdicts: List[dict], noise_grade: str) -> Tuple[dict, List[str], List[str]]:
    required_facts: List[str] = []
    required_evidence: List[str] = []
    if not (mvc.get("time") or "").strip():
        required_facts.append("발생 일시(연월일·시간)")
    if not (mvc.get("place") or "").strip():
        required_facts.append("발생 장소(주소/시설명/위치)")
    if not (mvc.get("target") or "").strip():
        required_facts.append("대상 특정(차량/시설/업체/담당부서 등)")
    if not (mvc.get("request") or "").strip():
        required_facts.append("요청사항(원하는 조치/결과)")
    ev = mvc.get("evidence") if isinstance(mvc.get("evidence"), (list, str)) else []
    if (isinstance(ev, list) and len(ev) == 0) or (isinstance(ev, str) and not ev.strip()):
        required_evidence.append("사진/영상/문서/링크 등 객관적 자료(가능한 범위)")

    title = _GRADE_TITLES.get(noise_grade, "민원 검토 결과 안내(초안)")

    verified_lines: List[str] = []
    unverified_lines: List[str] = []
    needed_lines: List[str] = []

    for k, label in [("time", "발생 일시"), ("place", "발생 장소"), ("target", "대상"), ("request", "요청사항")]:
        v = (mvc.get(k) or "").strip()
        if v:
            verified_lines.append(f"- {label}: {v}")
    if isinstance(ev, list) and ev:
        verified_lines.append(f"- 제출 자료: {', '.join([str(x) for x in ev[:5]])}")
    elif isinstance(ev, str) and ev.strip():
        verified_lines.append(f"- 제출 자료: {ev.strip()}")

    for vd in verdicts:
        if vd["verdict"] == "SUPPORTED":
            verified_lines.append(f"- (주장 {vd['claim_id']}) {vd['safe_statement']}")
        else:
            unverified_lines.append(f"- (주장 {vd['claim_id']}) {vd['safe_statement']}")
        for n in vd.get("needed", [])[:3]:
            needed_lines.append(f"- {n}")

    verified_lines = _dedup(verified_lines) or ["- (기관이 확인 가능한 범위의 사실이 부족합니다.)"]
    unverified_lines = _dedup(unverified_lines) or ["- (미확인 주장 없음)"]
    needed_lines = _dedup(
        needed_lines + [f"- {x}" for x in required_facts] + [f"- {x}" for x in required_evidence]
    ) or ["- (추가 제출 요청 없음)"]

    next_step = "제출된 자료 범위 내에서만 판단이 가능하며, 필요 시 추가 확인 후 처리합니다."
    if noise_grade == "YELLOW":
        next_step = "추가자료 제출 시 재검토 예정이며, 미제출 시 현 단계에서 사실확정이 어렵습니다."
    elif noise_grade == "RED":
        next_step = "제출된 자료만으로 특정/판단이 어려워 요건·관할 기준으로 정형 안내드립니다. 추가자료 제출 시 재검토합니다."

    body_paragraphs = [
        "**1. 확인된 사실(제출 자료 기준)**",
        *verified_lines,
        "",
        "**2. 미확인 주장(현 단계에서 단정 불가)**",
        *unverified_lines,
        "",
        "**3. 확인을 위한 추가 자료/사실 요청**",
        *needed_lines,
        "",
        "**4. 절차 및 안내**",
        f"- {next_step}",
        "- 본 회신(초안)은 민원인이 제출한 내용 및 기관이 확인 가능한 범위에 한하여 작성됩니다.",
    ]

    doc = {
        "title": title,
        "receiver": "민원인 귀하",
        "body_paragraphs": body_paragraphs,
        "department_head": "행정기관장",
    }
    return doc, required_facts, required_evidence


# =============================================================================
# Main entry
# =============================================================================
def run_complaint_analyzer_workflow(
    user_input: str,
    llm_service: Any = None,
    law_api: Any = None,
    services: Optional[dict] = None,
    on_progress: Optional[Callable[[str, str], None]] = None,
) -> dict:
    """민원 텍스트를 분석하여 5단계 결과 dict를 반환.

    Args:
        user_input: 민원 원문.
        llm_service: ``generate_json(prompt)`` 가능한 LLM 서비스.
        law_api: 국가법령정보센터 클라이언트(``get_law_text(name, num, return_link=True)``).
        services: 대안적 의존성 주입. ``{"llm": ..., "law_api": ...}``.
        on_progress: 단계 알림 콜백. ``on_progress(stage_key, message)``.

    Returns:
        legacy 호환 dict — keys: situation, analysis, law, strategy, doc, meta,
        complaint_pack(mvc/claims/noise_grade/verifiability_score/citations/verdicts/...),
        token_usage, execution_time, model_used, app_mode.
    """
    if services and not llm_service:
        llm_service = services.get("llm") or services.get("llm_service")
    if services and not law_api:
        law_api = services.get("law_api")

    if llm_service is None:
        raise ValueError("llm_service is required (직접 또는 services dict 로 전달).")

    def _progress(stage: str, msg: str) -> None:
        if on_progress is not None:
            try:
                on_progress(stage, msg)
            except Exception:  # progress callback should never break the workflow
                logger.debug("on_progress callback raised", exc_info=True)

    start_time = time.time()
    s_masked = mask_sensitive(user_input or "")

    # Phase 1
    _progress("phase1", "민원 텍스트에서 주장/요건 요소를 분해 중...")
    mvc, claims, halluc_signals = _phase1_decompose(llm_service, s_masked)

    # Phase 2
    _progress("phase2", "요건 충족/환각 신호 점검 중...")
    verifiability_score, citation_items, noise_grade, grade_reasons = \
        _phase2_signals(mvc, claims, halluc_signals)

    # Phase 3
    _progress("phase3", f"법령 인용을 공식 API로 검증 중... ({len(citation_items)}건)")
    verified_citations, invalid_count = _phase3_verify_citations(law_api, citation_items)

    if invalid_count >= 2 and noise_grade == "GREEN":
        noise_grade = "YELLOW"
        grade_reasons.append("법령 인용 중 확인 불가 항목이 다수 존재함")
    if invalid_count >= 4:
        noise_grade = "RED"
        grade_reasons.append("법령/조문 인용이 다수 확인 불가(허위/환각 가능성 높음)")

    law_md = _law_pack_markdown(verified_citations)

    # Phase 4
    _progress("phase4", "주장별 안전한 결론을 산출 중...")
    verdicts = _phase4_verdicts(llm_service, claims, verified_citations, noise_grade)

    # Phase 5
    _progress("phase5", "회신 초안 조립 중...")
    doc, required_facts, required_evidence = _phase5_draft(mvc, verdicts, noise_grade)

    strategy_lines = [
        f"- 처리 등급: **{noise_grade}** (검증가능성 {verifiability_score * 100:.0f}%)",
        *(f"- 사유: {r}" for r in (grade_reasons or [])),
        "",
        "#### 운영 권고",
        "- **단정형 반박** 대신, `확인된 사실/미확인 주장/보완요구/절차` 구조로 회신",
        "- 법령 인용은 **공식 API로 확인된 범위만** 사용하고, 불확실한 인용은 '미확인' 처리",
        "- 동일·유사 민원은 사건ID로 **병합**하여 반복 대응 비용을 낮출 것",
    ]
    strategy = "\n".join(strategy_lines)

    analysis = {
        "case_type": "민원 분석",
        "core_issue": ["입력 품질(요건) 점검", "법령 인용 검증", "안전한 회신 조립"],
        "required_facts": _dedup(required_facts)[:10],
        "required_evidence": _dedup(required_evidence)[:10],
        "risk_flags": _dedup((halluc_signals or []) + (grade_reasons or []))[:10],
        "recommended_next_action": ["보완요구 또는 정형 안내 후 재검토"][:10],
        "summary": f"민원 분석기 결과: 등급 {noise_grade}, 검증가능성 {verifiability_score * 100:.0f}%",
    }

    full_res_text = f"{mvc}{verified_citations}{verdicts}{doc}"
    estimated_tokens = int(len(full_res_text) * 0.7)
    model_used = getattr(llm_service, "last_model_used", None)
    meta = _clerk_meta()

    _progress("done", "분석 완료")

    return {
        "situation": user_input,
        "analysis": analysis,
        "law_pack": {},
        "law": law_md,
        "search": "",
        "strategy": strategy,
        "objections": [],
        "procedure": {"timeline": [], "checklist": [], "templates": []},
        "meta": meta,
        "doc": doc,
        # legacy alias (UI 가 ``draft_doc`` 키도 종종 참조)
        "draft_doc": doc,
        "lawbot_pack": {},
        "followups": [],
        "app_mode": "complaint_analyzer",
        "token_usage": estimated_tokens,
        "execution_time": round(time.time() - start_time, 2),
        "search_count": 0,
        "model_used": model_used,
        # Legacy keys callers expect at the top level for convenience.
        "claims": claims,
        "verified_citations": verified_citations,
        "verdicts": verdicts,
        "complaint_pack": {
            "mvc": mvc,
            "claims": claims,
            "noise_grade": noise_grade,
            "verifiability_score": verifiability_score,
            "hallucination_signals": halluc_signals,
            "citations": verified_citations,
            "verdicts": verdicts,
            "grade_reasons": grade_reasons,
        },
    }
