# -*- coding: utf-8 -*-
"""Data models for streamlit-page-registry."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable


@dataclass(frozen=True)
class PageEntry:
    """A single registered page.

    Attributes:
        key: The page identifier used in ``st.session_state.current_page``.
        label: The button label shown in the sidebar (Markdown allowed).
        module: Dotted module path containing the render function, e.g.
            ``"myapp.pages.home"``.
        func: Render function name. Defaults to ``"render_<key>_page"``.
        group: Group identifier for sidebar organization. Defaults to
            ``"default"``.
    """

    key: str
    label: str
    module: str
    func: str = ""
    group: str = "default"

    @property
    def function_name(self) -> str:
        """Resolved render-function name."""
        return self.func or f"render_{self.key}_page"

    def load(self) -> Callable[..., None]:
        """Lazy-import and return the render callable.

        Raises:
            RuntimeError: If the module exposes no callable with the expected name.
        """
        mod = import_module(self.module)
        fn = getattr(mod, self.function_name, None)
        if fn is None or not callable(fn):
            raise RuntimeError(
                f"Page entry '{self.key}' is missing callable "
                f"{self.module}:{self.function_name}"
            )
        return fn
