# streamlit-page-registry

> **Zero-config dynamic routing for production Streamlit apps** — error isolation, introspection dispatch, and pre-flight validation.

[![PyPI](https://img.shields.io/badge/pypi-0.1.0a1-blue)](https://pypi.org/project/streamlit-page-registry/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)

When your Streamlit app grows past 5–6 pages, the official `st.navigation` becomes painful: one bad page kills the whole app, every page must declare the same signature, and there's no way to validate registrations before runtime.

`streamlit-page-registry` is a tiny (≈ 250 LOC, single dep) **enhancement layer** on top of Streamlit that solves all three problems.

---

## 30-second quickstart

```bash
pip install streamlit-page-registry
```

```python
# app.py
import streamlit as st
from streamlit_page_registry import PageRegistry, PageEntry

registry = PageRegistry(groups={
    "core": "🧠 Core",
    "meta": "⚙️ Settings",
})

registry.extend([
    PageEntry(key="home",     label="🏠 Home",      module="myapp.pages.home",     group="core"),
    PageEntry(key="reports",  label="📊 Reports",   module="myapp.pages.reports",  group="core"),
    PageEntry(key="settings", label="⚙️ Settings",  module="myapp.pages.settings", group="meta"),
])

# (Optional) catch broken signatures *before* a user clicks the page.
errors = registry.validate_all()
if errors:
    st.error("Page misconfiguration: " + "; ".join(errors))

registry.render_sidebar()
registry.dispatch(llm=my_llm_service)  # any kwargs — only what each page accepts is forwarded
```

Each page module just exposes a `render_<key>_page` function:

```python
# myapp/pages/home.py
def render_home_page(llm=None):
    import streamlit as st
    st.title("Home")
```

---

## Why use this over `st.navigation`?

| Feature | `st.navigation` | **streamlit-page-registry** |
|---|:---:|:---:|
| Dynamic page registration | ✅ | ✅ |
| **Error boundary** (one page crash ≠ app crash) | ❌ | ✅ |
| **Introspection dispatch** (heterogeneous signatures) | ❌ | ✅ |
| **Pre-flight `validate_all()`** | ❌ | ✅ |
| **Group expanders** in sidebar | ❌ | ✅ (auto-active group expand) |
| Custom error handler | ❌ | ✅ pluggable |
| `llm_service` ↔ `llm` alias mapping | ❌ | ✅ |

`streamlit-page-registry` does **not** replace `st.navigation` — it solves the operational pains of running 10+ page apps in production.

---

## Killer feature 1 — Error isolation

```python
def render_home_page():
    import streamlit as st
    st.title("Home")

def render_broken_page():
    raise RuntimeError("oops, this page is broken")
```

In stock Streamlit, clicking the broken page tears down the whole app and shows a Python traceback to the user. With `streamlit-page-registry`, only the broken page is replaced with a friendly fallback — every other page keeps working.

You can also plug a **custom error handler** for logging / Sentry:

```python
def my_handler(entry, error):
    sentry_sdk.capture_exception(error)
    st.error(f"'{entry.label}' is temporarily unavailable.")

registry = PageRegistry(error_handler=my_handler)
```

---

## Killer feature 2 — Introspection dispatch

Different pages need different services, and you don't want to either:
- Force every page to accept the same `**kwargs`
- Manually wire each page's call site

```python
def render_home_page(llm):                       # only LLM
    ...

def render_dashboard_page(db, services):         # DB + service map
    ...

def render_admin_page(llm_service, db):          # uses 'llm_service' name
    ...

# One call site — registry auto-picks per-page
registry.dispatch(llm=llm, llm_service=llm, db=db, services=services)
```

The registry inspects each function's signature, forwards only the kwargs it
actually accepts, and applies the well-known `llm_service` ↔ `llm` alias.

---

## Killer feature 3 — Pre-flight validation

```python
errors = registry.validate_all()
# [] when all pages: import successfully, expose render_<key>_page,
#     and accept at least one of {llm_service, llm, db, services} or **kwargs.
```

Run this in a CI job, in `if __name__ == "__main__"` of your app, or in a startup health check — and stop shipping broken pages.

---

## API

```python
PageRegistry(
    groups: Mapping[str, str] | None = None,
    *,
    skip_keys: Iterable[str] = (),       # pages whose render is handled by host
    error_handler: Callable | None = None,
    session_state_key: str = "current_page",
)
```

Methods:

- `add(entry)` / `extend(entries)` — register pages
- `set_error_handler(fn)` — override the default error UI
- `all_pages()` / `get_page(key)` / `pages_by_group()` — introspection
- `validate_all()` → `list[str]` — pre-flight check
- `render_sidebar(default_key=None)` — grouped expander UI
- `dispatch(key=None, **kwargs)` → `bool` — render current page

`PageEntry`:

```python
PageEntry(
    key: str,            # e.g. "home"  (becomes session_state.current_page value)
    label: str,          # e.g. "🏠 Home"
    module: str,         # e.g. "myapp.pages.home"  (lazy-imported)
    func: str = "",      # default: f"render_{key}_page"
    group: str = "default",
)
```

---

## Try the demo

```bash
git clone https://github.com/kim0395kk/streamlit-page-registry
cd streamlit-page-registry
pip install -e .
streamlit run examples/basic_app.py
```

The demo renders 5 working pages plus 1 deliberately broken page so you can see error isolation live.

---

## Compatibility

- Python 3.10+
- Streamlit ≥ 1.28
- Single runtime dependency: `streamlit`

---

## License

MIT — see [LICENSE](LICENSE).

---

## Roadmap

- v0.2 — `entry_points` plugin discovery (let other packages register pages)
- v0.3 — i18n helpers for group labels
- v0.4 — page permissions (per-role visibility)
- v1.0 — stable public API + integration with `st.Page` introduced in 2024
