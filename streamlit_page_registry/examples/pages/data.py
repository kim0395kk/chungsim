# -*- coding: utf-8 -*-
"""Demo data-explorer page — accepts ``**kwargs`` to show any kwargs work."""
import streamlit as st


def render_data_page(**kwargs):
    st.title("🔍 Data Explorer")
    st.write("This page accepts `**kwargs` and forwards them.")
    st.json({k: type(v).__name__ for k, v in kwargs.items()} or {"kwargs": "empty"})
