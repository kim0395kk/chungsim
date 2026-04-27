# -*- coding: utf-8 -*-
"""Demo home page — accepts an optional ``llm``."""
import streamlit as st


def render_home_page(llm=None):  # noqa: ARG001 — argument shows introspection works
    st.title("🏠 Home")
    st.markdown(
        "Welcome to the **streamlit-page-registry** demo. Use the sidebar to "
        "navigate. Try clicking the **💥 Broken** page — the rest of the app "
        "stays alive."
    )
    st.success("This page accepts an optional `llm`.")
