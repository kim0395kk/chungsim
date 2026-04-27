# -*- coding: utf-8 -*-
"""Contract tests for govable_ai.ui.pages.registry.

다루는 항목:
  - all_pages / get_page / pages_by_group / validate_all 의 기본 계약
  - render() 의 분기 (미존재 키, workflow 단락, llm 별칭)
  - 에러 바운더리: 페이지 함수가 raise 해도 render() 가 swallow 하는지
"""
from __future__ import annotations

import sys
import types
import unittest
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from govable_ai.ui.pages import registry  # noqa: E402


class _StreamlitStub:
    """registry._render_error_ui 가 import 하는 streamlit 의 최소 더미."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.captions: list[str] = []
        self.codes: list[str] = []
        self.infos: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def caption(self, msg: str) -> None:
        self.captions.append(msg)

    def info(self, msg: str) -> None:
        self.infos.append(msg)

    def code(self, body: str, language: str = "") -> None:  # noqa: ARG002
        self.codes.append(body)

    def expander(self, *_args, **_kwargs):  # noqa: ANN001
        class _Ctx:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

        return _Ctx()


class RegistryContractTest(unittest.TestCase):
    def test_all_pages_nonempty(self) -> None:
        self.assertGreater(len(registry.all_pages()), 0)

    def test_workflow_is_first_entry(self) -> None:
        first = registry.all_pages()[0]
        self.assertEqual(first.key, "workflow")

    def test_get_page_known_and_unknown(self) -> None:
        self.assertIsNotNone(registry.get_page("workflow"))
        self.assertIsNone(registry.get_page("__nope__"))

    def test_pages_by_group_keys(self) -> None:
        groups = registry.pages_by_group()
        for g in registry.GROUP_LABELS:
            self.assertIn(g, groups, f"group {g} missing")
        # 등록된 모든 페이지가 어떤 그룹에든 들어가야 한다
        flat = sum((entries for entries in groups.values()), [])
        self.assertEqual(len(flat), len(registry.all_pages()))

    def test_validate_all_passes(self) -> None:
        errors = registry.validate_all()
        self.assertEqual(errors, [], f"unexpected validation errors: {errors}")


class RegistryRenderDispatchTest(unittest.TestCase):
    def test_unknown_key_returns_false(self) -> None:
        self.assertFalse(registry.render("__no_such_page__"))

    def test_workflow_short_circuits(self) -> None:
        # workflow 는 main.py 에서 직접 그려지므로 registry.render 는 False 를 반환해야 한다
        self.assertFalse(registry.render("workflow"))

    def test_dispatch_filters_unaccepted_kwargs(self) -> None:
        captured: dict = {}

        def _fake_page(llm_service):
            captured["llm_service"] = llm_service

        fake_module = types.ModuleType("govable_ai.ui.pages.__test_only__")
        fake_module.render_test_page = _fake_page

        entry = registry.PageEntry(
            key="__test_only__",
            label="test",
            module="govable_ai.ui.pages.__test_only__",
            func="render_test_page",
            group="meta",
        )
        registry._REGISTRY.append(entry)
        try:
            with patch.dict(sys.modules, {"govable_ai.ui.pages.__test_only__": fake_module}):
                ok = registry.render(
                    "__test_only__",
                    llm_service="LLM",
                    db="DB",
                    services={"x": 1},
                )
        finally:
            registry._REGISTRY.pop()

        self.assertTrue(ok)
        # llm_service 만 받아야 하고, 받지 못한 kwargs 로 인해 raise 되지 않아야 한다
        self.assertEqual(captured, {"llm_service": "LLM"})

    def test_dispatch_maps_llm_alias(self) -> None:
        captured: dict = {}

        def _fake_page(llm):  # 일부 페이지는 'llm' 이라는 이름을 쓴다
            captured["llm"] = llm

        fake_module = types.ModuleType("govable_ai.ui.pages.__alias_only__")
        fake_module.render_test_page = _fake_page

        entry = registry.PageEntry(
            key="__alias_only__",
            label="alias",
            module="govable_ai.ui.pages.__alias_only__",
            func="render_test_page",
            group="meta",
        )
        registry._REGISTRY.append(entry)
        try:
            with patch.dict(sys.modules, {"govable_ai.ui.pages.__alias_only__": fake_module}):
                ok = registry.render("__alias_only__", llm_service="LLM-instance")
        finally:
            registry._REGISTRY.pop()

        self.assertTrue(ok)
        self.assertEqual(captured, {"llm": "LLM-instance"})


class RegistryErrorBoundaryTest(unittest.TestCase):
    def test_page_exception_does_not_propagate(self) -> None:
        def _broken_page(**_kwargs):
            raise RuntimeError("boom")

        fake_module = types.ModuleType("govable_ai.ui.pages.__boom__")
        fake_module.render_test_page = _broken_page

        entry = registry.PageEntry(
            key="__boom__",
            label="boom",
            module="govable_ai.ui.pages.__boom__",
            func="render_test_page",
            group="meta",
        )
        registry._REGISTRY.append(entry)

        st_stub = _StreamlitStub()
        try:
            with patch.dict(
                sys.modules,
                {
                    "govable_ai.ui.pages.__boom__": fake_module,
                    "streamlit": st_stub,
                },
            ):
                ok = registry.render("__boom__")
        finally:
            registry._REGISTRY.pop()

        self.assertTrue(ok, "render() should return True even when page raised")
        # 사용자가 실제로 오류 메시지를 볼 수 있어야 한다
        self.assertTrue(any("boom" in e or "오류" in e for e in st_stub.errors))
        self.assertTrue(any("RuntimeError" in c for c in st_stub.captions))


if __name__ == "__main__":
    unittest.main(verbosity=2)
