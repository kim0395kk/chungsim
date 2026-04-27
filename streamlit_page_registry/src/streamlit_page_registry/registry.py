# -*- coding: utf-8 -*-
"""The :class:`PageRegistry` class — main entry point of the package."""
from __future__ import annotations

import inspect
import logging
from typing import Any, Iterable, Mapping, Optional

from streamlit_page_registry.dispatch import call_with_filtered_kwargs
from streamlit_page_registry.errors import ErrorHandler, render_default_error
from streamlit_page_registry.models import PageEntry

logger = logging.getLogger(__name__)


# Sensible default — applications are expected to override via the constructor.
DEFAULT_GROUPS: dict[str, str] = {"default": "Pages"}

# Standard parameters considered "well-known". Used by ``validate_all`` to
# warn about pages whose signatures don't match the conventional interface.
_STANDARD_PARAMS = frozenset({"llm_service", "llm", "db", "services"})


class PageRegistry:
    """Holds page registrations, dispatches renders, and renders the sidebar.

    Args:
        groups: ordered mapping of ``group_id -> sidebar_label``. The order
            controls the rendering order of group expanders. If omitted, a
            single ``"default"`` group is used.
        skip_keys: Page keys whose renders are handled by the host app rather
            than dispatched by the registry (e.g. a default landing page).
            Calls to :meth:`dispatch` for these keys return ``False``.
        error_handler: Optional callback invoked when a page raises during
            render. Defaults to :func:`render_default_error`.
        session_state_key: Name of the ``st.session_state`` field that holds
            the current page key. Used by :meth:`render_sidebar`.
    """

    def __init__(
        self,
        groups: Optional[Mapping[str, str]] = None,
        *,
        skip_keys: Iterable[str] = (),
        error_handler: Optional[ErrorHandler] = None,
        session_state_key: str = "current_page",
    ) -> None:
        self._registry: list[PageEntry] = []
        self.groups: dict[str, str] = dict(groups) if groups else dict(DEFAULT_GROUPS)
        self._skip_keys: frozenset[str] = frozenset(skip_keys)
        self._error_handler: ErrorHandler = error_handler or render_default_error
        self._session_state_key = session_state_key

    # ------------------------------------------------------------------ register
    def add(self, entry: PageEntry) -> None:
        """Register a new page."""
        self._registry.append(entry)

    def extend(self, entries: Iterable[PageEntry]) -> None:
        """Register many pages at once."""
        for e in entries:
            self.add(e)

    def set_error_handler(self, handler: ErrorHandler) -> None:
        """Override the default error handler."""
        self._error_handler = handler

    # ------------------------------------------------------------------ inspect
    def all_pages(self) -> list[PageEntry]:
        """Return registered pages in registration order."""
        return list(self._registry)

    def get_page(self, key: str) -> Optional[PageEntry]:
        """Return the entry for ``key`` if registered, else ``None``."""
        return next((p for p in self._registry if p.key == key), None)

    def pages_by_group(self) -> dict[str, list[PageEntry]]:
        """Group registered pages by ``group``, preserving group ordering."""
        out: dict[str, list[PageEntry]] = {g: [] for g in self.groups}
        for p in self._registry:
            out.setdefault(p.group, []).append(p)
        return out

    # ------------------------------------------------------------------ validate
    def validate_all(self) -> list[str]:
        """Validate every page can be lazy-imported and has a sane signature.

        Returns a list of error strings. An empty list means all pages are OK.
        Pages listed in ``skip_keys`` are not validated (the host renders them).
        """
        errors: list[str] = []
        for p in self._registry:
            if p.key in self._skip_keys:
                continue
            try:
                fn = p.load()
                sig = inspect.signature(fn)
                params = sig.parameters
                has_var_kw = any(
                    v.kind == inspect.Parameter.VAR_KEYWORD for v in params.values()
                )
                if not has_var_kw and not (set(params) & _STANDARD_PARAMS):
                    errors.append(
                        f"{p.key}: {p.module}:{p.function_name} signature "
                        f"({', '.join(params)}) accepts none of "
                        f"{sorted(_STANDARD_PARAMS)} and has no **kwargs"
                    )
            except Exception as e:
                errors.append(f"{p.key}: {type(e).__name__}: {e}")
        return errors

    # ------------------------------------------------------------------ dispatch
    def dispatch(self, key: Optional[str] = None, **kwargs: Any) -> bool:
        """Render the page identified by ``key`` (or current_page).

        Returns:
            ``True`` if a page was rendered. ``False`` if the key was unknown
            or belongs to ``skip_keys`` (host renders it).
        """
        if key is None:
            key = self._read_current_page()
        if key is None:
            return False
        entry = self.get_page(key)
        if entry is None or entry.key in self._skip_keys:
            return False

        try:
            fn = entry.load()
            call_with_filtered_kwargs(fn, kwargs)
        except Exception as e:
            logger.exception(
                "Page render failed: key=%s module=%s", entry.key, entry.module
            )
            self._error_handler(entry, e)
        return True

    # ------------------------------------------------------------------ sidebar
    def render_sidebar(self, *, default_key: Optional[str] = None) -> None:
        """Render grouped sidebar buttons; one click sets ``current_page``.

        Streamlit is imported lazily so this module remains testable in
        environments without a Streamlit runtime.
        """
        import streamlit as st  # local import — keeps unit tests light

        if default_key is None and self._registry:
            default_key = self._registry[0].key
        if self._session_state_key not in st.session_state and default_key:
            st.session_state[self._session_state_key] = default_key

        current_key = st.session_state.get(self._session_state_key)
        grouped = self.pages_by_group()

        for group_id, group_label in self.groups.items():
            entries = grouped.get(group_id) or []
            if not entries:
                continue
            is_active_group = any(e.key == current_key for e in entries)
            with st.sidebar.expander(group_label, expanded=is_active_group):
                for entry in entries:
                    if st.button(
                        entry.label,
                        use_container_width=True,
                        key=f"nav_{entry.key}",
                        type=(
                            "primary" if current_key == entry.key else "secondary"
                        ),
                    ):
                        st.session_state[self._session_state_key] = entry.key
                        st.rerun()

        current = self.get_page(current_key) if current_key else None
        st.sidebar.caption(f"📍 {current.label if current else '?'}")

    # ------------------------------------------------------------------ helpers
    def _read_current_page(self) -> Optional[str]:
        try:
            import streamlit as st

            return st.session_state.get(self._session_state_key)
        except Exception:
            return None
