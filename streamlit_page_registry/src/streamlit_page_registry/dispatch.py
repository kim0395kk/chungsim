# -*- coding: utf-8 -*-
"""Introspection-based dispatch.

The dispatch layer inspects each page's signature and only forwards keyword
arguments the function actually accepts. This lets registered pages have
heterogeneous signatures without forcing every page to declare ``**kwargs``.

A common alias is also supported: if the call site provides ``llm_service``
but the page only accepts ``llm``, the value is automatically remapped.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, Mapping


def filter_kwargs(fn: Callable[..., Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the kwargs that ``fn`` accepts.

    If ``fn`` declares ``**kwargs``, all input kwargs pass through unchanged.
    The well-known ``llm_service`` ↔ ``llm`` alias is auto-mapped.
    """
    sig = inspect.signature(fn)
    params = sig.parameters
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    if has_var_kw:
        accepted = dict(kwargs)
    else:
        accepted = {k: v for k, v in kwargs.items() if k in params}

    # Common alias: function accepts `llm`, caller passed `llm_service`.
    if "llm" in params and "llm_service" not in params and "llm_service" in kwargs:
        accepted["llm"] = kwargs["llm_service"]
        accepted.pop("llm_service", None)

    return accepted


def call_with_filtered_kwargs(fn: Callable[..., Any], kwargs: Mapping[str, Any]) -> Any:
    """Invoke ``fn`` after filtering kwargs to its signature."""
    return fn(**filter_kwargs(fn, kwargs))
