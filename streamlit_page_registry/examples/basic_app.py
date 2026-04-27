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

Each page lives in its own file under ``examples/pages/``. This is the
canonical pattern: one module per page, registered by dotted import path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

from streamlit_page_registry import PageEntry, PageRegistry

# Make the sibling ``pages`` package importable when launched via
# ``streamlit run`` from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))


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
        PageEntry(key="home",     label="🏠 Home",          module="pages.home",     group="core"),
        PageEntry(key="reports",  label="📊 Reports",       module="pages.reports",  group="core"),
        PageEntry(key="data",     label="🔍 Data Explorer", module="pages.data",     group="tools"),
        PageEntry(key="broken",   label="💥 Broken",        module="pages.broken",   group="tools"),
        PageEntry(key="settings", label="⚙️ Settings",      module="pages.settings", group="meta"),
        PageEntry(key="about",    label="ℹ️ About",         module="pages.about",    group="meta"),
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
