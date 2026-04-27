# -*- coding: utf-8 -*-
"""Demo about page — no kwargs (introspection drops everything cleanly)."""
import streamlit as st


def render_about_page():
    st.title("ℹ️ About")
    st.markdown(
        """
        - Package: [streamlit-page-registry](https://github.com/kim0395kk/streamlit-page-registry)
        - Demo file: `examples/basic_app.py`
        - Try the broken page to see error isolation in action!
        """
    )
