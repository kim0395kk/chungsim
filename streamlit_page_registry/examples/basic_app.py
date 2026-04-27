# -*- coding: utf-8 -*-
"""streamlit-page-registry — End-to-end demo.

Run with::

    streamlit run examples/basic_app.py

What this demonstrates:

1. Group expander sidebar (Core / Tools / Settings).
2. Heterogeneous page signatures dispatched safely
   (one page takes ``llm``, another takes ``services``, another takes ``**kwargs``).
3. Error isolation — the "Broken" page intentionally raises every render,
   but the rest of the app keeps working.
4. Pre-flight ``validate_all()`` shown in the sidebar.
"""
from __future__ import annotations

import streamlit as st

from streamlit_page_registry import PageEntry, PageRegistry


# ---------------------------------------------------------------------------
# Page implementations — one file would normally hold one page; we inline
# them here so the demo is self-contained.
# ---------------------------------------------------------------------------
def render_home_page(llm=None):  # noqa: ARG001
    st.title("🏠 Home")
    st.markdown(
        "Welcome to the **streamlit-page-registry** demo. Use the sidebar to "
        "navigate. Try clicking the **💥 Broken** page — the rest of the app "
        "stays alive."
    )
    st.success("This page accepts an optional `llm`.")


def render_reports_page(services=None):
    st.title("📊 Reports")
    st.write("This page accepts a `services` dict.")
    st.json(services or {"info": "no services injected"})


def render_data_page(**kwargs):
    st.title("🔍 Data Explorer")
    st.write("This page accepts `**kwargs` and forwards them.")
    st.json({k: type(v).__name__ for k, v in kwargs.items()} or {"kwargs": "empty"})


def render_settings_page(db=None):
    st.title("⚙️ Settings")
    st.write("This page accepts a `db` reference.")
    st.code(repr(db) or "None")


def render_about_page():
    st.title("ℹ️ About")
    st.markdown(
        """
        - Package: [streamlit-page-registry](https://github.com/kim0395kk/streamlit-page-registry)
        - Demo file: `examples/basic_app.py`
        - Try the broken page to see error isolation in action!
        """
    )


def render_broken_page():
    """Intentionally raises to demonstrate the error boundary."""
    raise RuntimeError("This page is intentionally broken — but the app survives.")


# Stash render funcs in the demo module so the registry can resolve them
# via the ``module=__name__`` reference below.
_THIS_MODULE = __name__


# ---------------------------------------------------------------------------
# Registry setup
# ---------------------------------------------------------------------------
registry = PageRegistry(
    groups={
        "core": "🧠 Core",
        "tools": "🔧 Tools",
        "meta": "⚙️ Settings",
    },
)

registry.extend(
    [
        PageEntry(key="home", label="🏠 Home", module=_THIS_MODULE, group="core"),
        PageEntry(key="reports", label="📊 Reports", module=_THIS_MODULE, group="core"),
        PageEntry(key="data", label="🔍 Data Explorer", module=_THIS_MODULE, group="tools"),
        PageEntry(key="broken", label="💥 Broken", module=_THIS_MODULE, group="tools"),
        PageEntry(key="settings", label="⚙️ Settings", module=_THIS_MODULE, group="meta"),
        PageEntry(key="about", label="ℹ️ About", module=_THIS_MODULE, group="meta"),
    ]
)


# ---------------------------------------------------------------------------
# App body
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="page-registry demo", page_icon="🧭", layout="wide")

    # Pre-flight validation — surface misconfiguration in the sidebar.
    errors = registry.validate_all()
    if errors:
        st.sidebar.error("⚠️ Page misconfiguration:\n" + "\n".join(errors))
    else:
        st.sidebar.success("✅ All pages registered & valid.")

    registry.render_sidebar()

    # Fake "services" so dispatch has something to forward.
    fake_services = {"llm": "<LLM-instance>", "db": "<DB-handle>"}

    registry.dispatch(
        llm=fake_services["llm"],
        services=fake_services,
        db=fake_services["db"],
    )


if __name__ == "__main__":
    main()
else:
    main()
