# -*- coding: utf-8 -*-
"""Error-boundary helpers.

The default boundary renders a friendly Streamlit message with an expandable
traceback. Apps can plug a custom handler via
:meth:`PageRegistry.set_error_handler`.
"""
from __future__ import annotations

import logging
import traceback
from typing import TYPE_CHECKING, Callable

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from streamlit_page_registry.models import PageEntry


# Public type alias.
ErrorHandler = Callable[["PageEntry", BaseException], None]


def render_default_error(entry: "PageEntry", error: BaseException) -> None:
    """Render a friendly error UI for a failing page render.

    Streamlit is imported lazily so the package itself does not require a
    Streamlit runtime (useful for unit tests and CI).
    """
    try:
        import streamlit as st
    except Exception:  # pragma: no cover - streamlit absent in unit tests
        logger.exception(
            "Page '%s' raised %s but Streamlit is unavailable for UI fallback.",
            entry.key,
            type(error).__name__,
        )
        return

    st.error(
        f"⚠️ Page '{entry.label}' failed to render. "
        f"Try another menu item or refresh the page."
    )
    st.caption(f"`{type(error).__name__}: {error}`")
    with st.expander("🛠️ Traceback (developers)", expanded=False):
        st.code("".join(traceback.format_exception(error)), language="text")
