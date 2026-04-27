# -*- coding: utf-8 -*-
"""Demo broken page — intentionally raises to demonstrate the error boundary."""


def render_broken_page():
    """Intentionally raises to demonstrate the error boundary."""
    raise RuntimeError("This page is intentionally broken — but the app survives.")
