# -*- coding: utf-8 -*-
"""streamlit-page-registry — Zero-config dynamic routing for production Streamlit apps.

Public API:

* :class:`PageRegistry` — register, group, dispatch, and validate pages
* :class:`PageEntry` — single page record (key/label/module/func/group)
* :func:`render_default_error` — standalone error-boundary UI helper

Quickstart::

    from streamlit_page_registry import PageRegistry, PageEntry

    registry = PageRegistry(groups={"core": "Core", "meta": "Settings"})
    registry.add(PageEntry(key="home", label="Home",
                           module="myapp.pages.home", group="core"))
    registry.render_sidebar()
    registry.dispatch(llm=my_llm)
"""
from streamlit_page_registry.errors import render_default_error
from streamlit_page_registry.models import PageEntry
from streamlit_page_registry.registry import PageRegistry

__all__ = ["PageEntry", "PageRegistry", "render_default_error"]
__version__ = "0.1.0a1"
