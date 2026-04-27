# -*- coding: utf-8 -*-
"""Smoke tests for govable_ai/ui/pages/*.

각 페이지 모듈이:
  1) import 가능한가
  2) ``render_xxx_page`` 함수가 정의되어 있고 callable 인가
  3) 표준 인터페이스(``llm_service|llm|db|services|**kwargs``) 중 최소 하나를 받는가

UI 자체를 실제로 렌더링하지는 않는다 — 그건 streamlit 런타임이 필요하므로 별도.
"""
from __future__ import annotations

import inspect
import sys
import unittest
import warnings
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")


# (module suffix, expected entry function name)
_PAGES: list[tuple[str, str]] = [
    ("civil_engineering", "render_civil_engineering_page"),
    ("master_dashboard", "render_master_dashboard_page"),
    ("service_health", "render_service_health_page"),
    ("duty_manual", "render_duty_manual_page"),
    ("document_revision", "render_document_revision_page"),
    ("archive_history", "render_archive_history_page"),
    ("complaint_analyzer", "render_complaint_analyzer_page"),
    ("hallucination_check", "render_hallucination_check_page"),
]

_STANDARD_PARAMS = frozenset({"llm_service", "llm", "db", "services"})


class PageContractTest(unittest.TestCase):
    """모든 페이지가 동일한 표준 인터페이스를 따르는지 확인."""

    def test_all_pages_import_and_expose_callable(self) -> None:
        for suffix, fname in _PAGES:
            with self.subTest(page=suffix):
                mod = import_module(f"govable_ai.ui.pages.{suffix}")
                fn = getattr(mod, fname, None)
                self.assertIsNotNone(fn, f"{suffix}: {fname} not exported")
                self.assertTrue(callable(fn), f"{suffix}: {fname} is not callable")

    def test_all_pages_accept_standard_interface(self) -> None:
        for suffix, fname in _PAGES:
            with self.subTest(page=suffix):
                mod = import_module(f"govable_ai.ui.pages.{suffix}")
                fn = getattr(mod, fname)
                sig = inspect.signature(fn)
                params = sig.parameters
                has_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                )
                accepts_standard = bool(set(params) & _STANDARD_PARAMS)
                self.assertTrue(
                    has_var_kw or accepts_standard,
                    f"{suffix}: {fname}{sig} accepts none of {sorted(_STANDARD_PARAMS)} "
                    f"and has no **kwargs",
                )


class FeatureModuleTest(unittest.TestCase):
    """페이지가 의존하는 feature 계층의 핵심 함수가 import 가능한지."""

    def test_complaint_analyzer_workflow_exists(self) -> None:
        from govable_ai.features.complaint_analyzer import run_complaint_analyzer_workflow
        self.assertTrue(callable(run_complaint_analyzer_workflow))

    def test_document_revision_workflow_exists(self) -> None:
        from govable_ai.features.document_revision import run_revision_workflow
        self.assertTrue(callable(run_revision_workflow))


if __name__ == "__main__":
    unittest.main(verbosity=2)
