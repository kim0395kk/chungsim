# -*- coding: utf-8 -*-
"""Demo reports page — accepts a ``services`` dict."""
import streamlit as st


def render_reports_page(services=None):
    st.title("📊 Reports")
    st.write("This page accepts a `services` dict.")
    st.json(services or {"info": "no services injected"})
