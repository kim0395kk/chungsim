# -*- coding: utf-8 -*-
"""Core Services Package"""
from govable_ai.core.llm_service import LLMService
from govable_ai.core.law_api import LawOfficialService
from govable_ai.core.search_api import SearchService
from govable_ai.core.db_client import SupabaseClient
from govable_ai.core.doc_generator import (
    HWPXGenerator,
    OfficialDocumentGenerator,
    ReportDocumentGenerator,
    generate_official_doc,
    generate_report_doc,
)

__all__ = [
    "LLMService",
    "LawOfficialService",
    "SearchService",
    "SupabaseClient",
    "HWPXGenerator",
    "OfficialDocumentGenerator",
    "ReportDocumentGenerator",
    "generate_official_doc",
    "generate_report_doc",
]
