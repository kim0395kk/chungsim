# -*- coding: utf-8 -*-
"""
Govable AI - 유틸리티 함수
UI 의존성 없음 (streamlit import 금지)
"""
import json
import re
import urllib.parse
from datetime import datetime
from html import escape as _escape
from typing import Any, Optional

from govable_ai.config import LAW_BOT_SEARCH_URL


def make_lawbot_url(query: str) -> str:
    """국가법령정보센터 Lawbot 검색 URL 생성"""
    return LAW_BOT_SEARCH_URL + urllib.parse.quote((query or "").strip())


def shorten_one_line(text: str, max_len: int = 28) -> str:
    """텍스트를 한 줄로 줄이고 길이 제한"""
    s = re.sub(r"\s+", " ", (text or "").strip())
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def estimate_tokens(text: str) -> int:
    """텍스트의 토큰 수 추정 (한글 기준 0.7배)"""
    if not text:
        return 0
    return int(len(text) * 0.7)


def safe_now_utc_iso() -> str:
    """현재 UTC 시간을 ISO 형식으로 반환"""
    return datetime.utcnow().isoformat() + "Z"


def safe_json_loads(text: str) -> Optional[Any]:
    """
    안전한 JSON 파싱.
    실패 시 정규식으로 JSON 객체/배열 추출 시도.
    """
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        m = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception:
        return None
    return None


def strip_html(text: str) -> str:
    """HTML 태그 제거"""
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return text


def md_bold_to_html_safe(text: str) -> str:
    """마크다운 bold(**text**)를 HTML <b>로 변환 (XSS 방지 포함)"""
    s = text or ""
    out = []
    pos = 0
    for m in re.finditer(r"\*\*(.+?)\*\*", s):
        out.append(_escape(s[pos:m.start()]))
        out.append(f"<b>{_escape(m.group(1))}</b>")
        pos = m.end()
    out.append(_escape(s[pos:]))
    html = "".join(out).replace("\n", "<br>")
    return html


def mask_sensitive(text: str) -> str:
    """개인정보 마스킹 (전화번호, 주민번호, 차량번호)"""
    if not text:
        return ""
    t = text
    t = re.sub(r"\b0\d{1,2}-\d{3,4}-\d{4}\b", "0**-****-****", t)
    t = re.sub(r"\b\d{6}-\d{7}\b", "******-*******", t)
    t = re.sub(r"\b\d{2,3}[가-힣]\d{4}\b", "***(차량번호)", t)
    return t


def short_for_context(s: str, limit: int = 2500) -> str:
    """컨텍스트 길이 제한 (LLM 프롬프트용)"""
    s = (s or "").strip()
    if len(s) <= limit:
        return s
    return s[:limit] + "\n...(생략)"
