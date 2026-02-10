# -*- coding: utf-8 -*-
"""DOCX export helpers used by Streamlit UI.

This module is intentionally lightweight and tolerant to missing optional
libraries, so importing it never breaks app boot.
"""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile
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


def _build_fallback_docx(title: str, paragraphs: List[str]) -> bytes:
    """Create a minimal valid DOCX package without python-docx."""

    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    text_lines = [title, ""] + [line for line in paragraphs if line is not None]
    runs = []
    for line in text_lines:
        escaped = escape(str(line))
        if escaped:
            runs.append(f"<w:p><w:r><w:t xml:space=\"preserve\">{escaped}</w:t></w:r></w:p>")
        else:
            runs.append("<w:p/>")
    body_xml = "".join(runs)

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
""".strip()

    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
""".strip()

    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas"
 xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
 xmlns:o="urn:schemas-microsoft-com:office:office"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"
 xmlns:v="urn:schemas-microsoft-com:vml"
 xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"
 xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
 xmlns:w10="urn:schemas-microsoft-com:office:word"
 xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"
 xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup"
 xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk"
 xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml"
 xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
 mc:Ignorable="w14 wp14">
  <w:body>
    {body_xml}
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>
      <w:cols w:space="708"/>
      <w:docGrid w:linePitch="360"/>
    </w:sectPr>
  </w:body>
</w:document>
""".strip()

    core_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{escape(title)}</dc:title>
  <dc:creator>govable-ai</dc:creator>
  <cp:lastModifiedBy>govable-ai</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:modified>
</cp:coreProperties>
""".strip()

    app_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>govable-ai</Application>
</Properties>
""".strip()

    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("docProps/core.xml", core_xml)
        archive.writestr("docProps/app.xml", app_xml)
    return buffer.getvalue()


def generate_official_docx(doc_data: Dict[str, Any]) -> bytes:
    """Build an official document as DOCX bytes."""
    doc_data = doc_data or {}
    title = str(doc_data.get("title") or "공문서")
    receiver = str(doc_data.get("receiver") or "수신자 참조")
    department_head = str(doc_data.get("department_head") or "행정기관장")
    body_lines = _to_paragraphs(doc_data.get("body_paragraphs"))

    if Document is None:
        lines = [f"수신: {receiver}", ""] + body_lines + ["", department_head]
        return _build_fallback_docx(title, lines)

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
        return _build_fallback_docx("처리가이드", lines)

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
