# -*- coding: utf-8 -*-
"""Demo settings page — accepts a ``db`` reference."""
import streamlit as st


def render_settings_page(db=None):
    st.title("⚙️ Settings")
    st.write("This page accepts a `db` reference.")
    st.code(repr(db) or "None")
