# -*- coding: utf-8 -*-
"""Contract tests for streamlit_page_registry."""
from __future__ import annotations

import sys
import types
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

warnings.filterwarnings("ignore")

from streamlit_page_registry import PageEntry, PageRegistry  # noqa: E402


class _StreamlitStub:
    """Minimal stub of the Streamlit API needed by the error renderer."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.captions: list[str] = []
        self.codes: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def caption(self, msg: str) -> None:
        self.captions.append(msg)

    def code(self, body: str, language: str = "") -> None:  # noqa: ARG002
        self.codes.append(body)

    def expander(self, *_args, **_kwargs):  # noqa: ANN001
        class _Ctx:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

        return _Ctx()


def _make_module(name: str, fn) -> types.ModuleType:
    m = types.ModuleType(name)
    setattr(m, fn.__name__, fn)
    return m


# ---------------------------------------------------------------------------
# PageEntry
# ---------------------------------------------------------------------------
class PageEntryTest(unittest.TestCase):
    def test_function_name_default(self) -> None:
        e = PageEntry(key="foo", label="Foo", module="x.y")
        self.assertEqual(e.function_name, "render_foo_page")

    def test_function_name_explicit(self) -> None:
        e = PageEntry(
            key="foo", label="Foo", module="x.y", func="render_my_thing"
        )
        self.assertEqual(e.function_name, "render_my_thing")

    def test_load_missing_callable_raises(self) -> None:
        e = PageEntry(key="foo", label="Foo", module="streamlit_page_registry")
        # streamlit_page_registry has no `render_foo_page`
        with self.assertRaises(RuntimeError):
            e.load()


# ---------------------------------------------------------------------------
# PageRegistry — basic contract
# ---------------------------------------------------------------------------
class PageRegistryContractTest(unittest.TestCase):
    def test_empty_registry(self) -> None:
        r = PageRegistry()
        self.assertEqual(r.all_pages(), [])
        self.assertIsNone(r.get_page("foo"))

    def test_add_and_get(self) -> None:
        r = PageRegistry()
        e = PageEntry(key="home", label="Home", module="x.y")
        r.add(e)
        self.assertEqual(r.all_pages(), [e])
        self.assertEqual(r.get_page("home"), e)
        self.assertIsNone(r.get_page("__unknown__"))

    def test_groups_default(self) -> None:
        r = PageRegistry()
        self.assertIn("default", r.groups)

    def test_groups_custom(self) -> None:
        r = PageRegistry(groups={"a": "Alpha", "b": "Beta"})
        self.assertEqual(list(r.groups.keys()), ["a", "b"])

    def test_pages_by_group_preserves_order_and_buckets(self) -> None:
        r = PageRegistry(groups={"core": "Core", "meta": "Meta"})
        r.add(PageEntry(key="a", label="A", module="x", group="core"))
        r.add(PageEntry(key="b", label="B", module="x", group="meta"))
        r.add(PageEntry(key="c", label="C", module="x", group="core"))
        grouped = r.pages_by_group()
        self.assertEqual([p.key for p in grouped["core"]], ["a", "c"])
        self.assertEqual([p.key for p in grouped["meta"]], ["b"])


# ---------------------------------------------------------------------------
# Dispatch logic
# ---------------------------------------------------------------------------
class PageRegistryDispatchTest(unittest.TestCase):
    def test_unknown_key_returns_false(self) -> None:
        r = PageRegistry()
        self.assertFalse(r.dispatch("__nope__"))

    def test_skip_keys_short_circuit(self) -> None:
        r = PageRegistry(skip_keys={"workflow"})
        r.add(PageEntry(key="workflow", label="W", module="x"))
        self.assertFalse(r.dispatch("workflow"))

    def test_dispatch_filters_unaccepted_kwargs(self) -> None:
        captured: dict = {}

        def render_x_page(llm_service):
            captured["llm_service"] = llm_service

        mod = _make_module("__pr_test_x__", render_x_page)
        r = PageRegistry()
        r.add(PageEntry(key="x", label="X", module="__pr_test_x__"))

        with patch.dict(sys.modules, {"__pr_test_x__": mod}):
            ok = r.dispatch("x", llm_service="LLM", db="DB", services={"a": 1})

        self.assertTrue(ok)
        # Only the accepted kwarg made it in.
        self.assertEqual(captured, {"llm_service": "LLM"})

    def test_dispatch_maps_llm_alias(self) -> None:
        captured: dict = {}

        def render_y_page(llm):
            captured["llm"] = llm

        mod = _make_module("__pr_test_y__", render_y_page)
        r = PageRegistry()
        r.add(PageEntry(key="y", label="Y", module="__pr_test_y__"))

        with patch.dict(sys.modules, {"__pr_test_y__": mod}):
            ok = r.dispatch("y", llm_service="LLM-instance")

        self.assertTrue(ok)
        self.assertEqual(captured, {"llm": "LLM-instance"})

    def test_dispatch_passes_all_kwargs_when_var_kw(self) -> None:
        captured: dict = {}

        def render_z_page(**kwargs):
            captured.update(kwargs)

        mod = _make_module("__pr_test_z__", render_z_page)
        r = PageRegistry()
        r.add(PageEntry(key="z", label="Z", module="__pr_test_z__"))

        with patch.dict(sys.modules, {"__pr_test_z__": mod}):
            ok = r.dispatch("z", llm_service="L", db="D", services={"x": 1})

        self.assertTrue(ok)
        self.assertEqual(set(captured.keys()), {"llm_service", "db", "services"})


# ---------------------------------------------------------------------------
# Error boundary
# ---------------------------------------------------------------------------
class PageRegistryErrorBoundaryTest(unittest.TestCase):
    def test_page_exception_does_not_propagate(self) -> None:
        def render_boom_page(**_kwargs):
            raise RuntimeError("boom")

        mod = _make_module("__pr_test_boom__", render_boom_page)
        r = PageRegistry()
        r.add(PageEntry(key="boom", label="Boom", module="__pr_test_boom__"))

        st_stub = _StreamlitStub()
        with patch.dict(
            sys.modules,
            {"__pr_test_boom__": mod, "streamlit": st_stub},
        ):
            ok = r.dispatch("boom")

        self.assertTrue(ok, "dispatch should swallow exceptions")
        self.assertTrue(any("Boom" in e or "failed" in e for e in st_stub.errors))
        self.assertTrue(any("RuntimeError" in c for c in st_stub.captions))

    def test_custom_error_handler_invoked(self) -> None:
        captured = {}

        def custom_handler(entry, error):
            captured["entry_key"] = entry.key
            captured["error_type"] = type(error).__name__

        def render_bang_page(**_kwargs):
            raise ValueError("bang")

        mod = _make_module("__pr_test_bang__", render_bang_page)
        r = PageRegistry(error_handler=custom_handler)
        r.add(PageEntry(key="bang", label="Bang", module="__pr_test_bang__"))

        with patch.dict(sys.modules, {"__pr_test_bang__": mod}):
            ok = r.dispatch("bang")

        self.assertTrue(ok)
        self.assertEqual(captured, {"entry_key": "bang", "error_type": "ValueError"})


# ---------------------------------------------------------------------------
# validate_all
# ---------------------------------------------------------------------------
class ValidateAllTest(unittest.TestCase):
    def test_clean_pages_pass(self) -> None:
        def render_ok_page(llm_service=None):  # noqa: ARG001
            return None

        mod = _make_module("__pr_test_ok__", render_ok_page)
        r = PageRegistry()
        r.add(PageEntry(key="ok", label="OK", module="__pr_test_ok__"))

        with patch.dict(sys.modules, {"__pr_test_ok__": mod}):
            errors = r.validate_all()

        self.assertEqual(errors, [])

    def test_missing_module_reported(self) -> None:
        r = PageRegistry()
        r.add(PageEntry(key="missing", label="M", module="__nope_no_such_module__"))
        errors = r.validate_all()
        self.assertTrue(any("missing" in e for e in errors))

    def test_unconventional_signature_warned(self) -> None:
        def render_weird_page(some_unrelated_arg):  # noqa: ARG001
            return None

        mod = _make_module("__pr_test_weird__", render_weird_page)
        r = PageRegistry()
        r.add(PageEntry(key="weird", label="W", module="__pr_test_weird__"))

        with patch.dict(sys.modules, {"__pr_test_weird__": mod}):
            errors = r.validate_all()

        self.assertTrue(
            any("accepts none" in e for e in errors),
            f"expected signature warning, got: {errors}",
        )

    def test_skip_keys_excluded_from_validation(self) -> None:
        r = PageRegistry(skip_keys={"workflow"})
        r.add(
            PageEntry(
                key="workflow", label="W", module="__nope_doesnt_exist__"
            )
        )
        errors = r.validate_all()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
