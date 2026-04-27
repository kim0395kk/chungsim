# Changelog

## 0.1.0a1 (2026-04-27)

Initial alpha release, extracted from a production Streamlit app
(["Govable AI"](https://github.com/kim0395kk/chungsim)) where the pattern
was battle-tested across 10 pages.

### Added
- `PageRegistry` — register, group, dispatch, and validate pages.
- `PageEntry` — frozen dataclass for a single page record.
- `dispatch.filter_kwargs` — introspection-based kwargs filtering.
- `errors.render_default_error` — default Streamlit error boundary.
- Group expanders in `render_sidebar()` with automatic active-group expansion.
- Pre-flight `validate_all()` that lazy-imports every page and reports issues.
- Well-known `llm_service` ↔ `llm` alias auto-mapping.
- 19 unit tests (no Streamlit runtime required) including error-boundary
  contract tests via a minimal `_StreamlitStub`.
