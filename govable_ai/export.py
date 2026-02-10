# -*- coding: utf-8 -*-
"""DOCX export helpers used by Streamlit UI.

This module is intentionally lightweight and tolerant to missing optional
libraries, so importing it never breaks app boot.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, Iterable, List

try:
    from docx import Document
except Exception:  # pragma: no cover - optional fallback
    Document = None


def _to_paragraphs(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            out.append(str(item))
        return out
    return [str(value)]


def _fallback_text_bytes(title: str, lines: List[str]) -> bytes:
    content = [title, "=" * len(title), ""] + lines
    return "\n".join(content).encode("utf-8")


def generate_official_docx(doc_data: Dict[str, Any]) -> bytes:
    """Build an official document as DOCX bytes."""
    doc_data = doc_data or {}
    title = str(doc_data.get("title") or "공문서")
    receiver = str(doc_data.get("receiver") or "수신자 참조")
    department_head = str(doc_data.get("department_head") or "행정기관장")
    body_lines = _to_paragraphs(doc_data.get("body_paragraphs"))

    if Document is None:
        lines = [f"수신: {receiver}", ""] + body_lines + ["", department_head]
        return _fallback_text_bytes(title, lines)

    buffer = BytesIO()
    document = Document()
    document.add_heading(title, level=1)
    document.add_paragraph(f"수신: {receiver}")
    document.add_paragraph("")
    for line in body_lines:
        document.add_paragraph(line)
    document.add_paragraph("")
    document.add_paragraph(department_head)
    document.save(buffer)
    return buffer.getvalue()


def generate_guide_docx(workflow_result: Dict[str, Any]) -> bytes:
    """Build a handling guide/report as DOCX bytes."""
    workflow_result = workflow_result or {}
    analysis = workflow_result.get("analysis") or {}
    procedure = workflow_result.get("procedure") or {}

    summary_lines = _to_paragraphs(analysis.get("summary") or analysis.get("core_issue"))
    checklist = _to_paragraphs(procedure.get("checklist"))
    templates = _to_paragraphs(procedure.get("templates"))

    if Document is None:
        lines = ["[핵심 요약]"] + summary_lines + ["", "[체크리스트]"] + checklist + ["", "[필요 서식]"] + templates
        return _fallback_text_bytes("처리가이드", lines)

    buffer = BytesIO()
    document = Document()
    document.add_heading("처리가이드", level=1)

    document.add_heading("핵심 요약", level=2)
    if summary_lines:
        for line in summary_lines:
            document.add_paragraph(line)
    else:
        document.add_paragraph("요약 정보가 없습니다.")

    document.add_heading("체크리스트", level=2)
    if checklist:
        for item in checklist:
            document.add_paragraph(item, style="List Bullet")
    else:
        document.add_paragraph("체크리스트 정보가 없습니다.")

    document.add_heading("필요 서식", level=2)
    if templates:
        for item in templates:
            document.add_paragraph(item, style="List Bullet")
    else:
        document.add_paragraph("서식 정보가 없습니다.")

    document.save(buffer)
    return buffer.getvalue()
