# feat: streamlit-page-registry 패키지 추출 (v0.1.0a1)

## Summary

`govable_ai/ui/pages/registry.py` 의 검증된 페이지 라우팅 패턴을 별도 PyPI 후보 패키지로 추출했습니다. 도메인 무관 일반화 + 향상된 API + 19개 단위 테스트 포함.

## What's in this PR

### 새 디렉토리: `streamlit_page_registry/`
```
streamlit_page_registry/
├─ src/streamlit_page_registry/
│   ├─ __init__.py     # 공개 API: PageRegistry, PageEntry, render_default_error
│   ├─ models.py       # PageEntry frozen dataclass
│   ├─ registry.py     # PageRegistry 메인 클래스
│   ├─ dispatch.py     # introspection-based kwargs filtering
│   └─ errors.py       # default error boundary UI
├─ tests/test_registry.py  # 19 unit tests
├─ examples/basic_app.py   # 5 working + 1 broken page demo
├─ pyproject.toml      # hatchling, src layout, MIT
├─ README.md           # 30s quickstart + vs st.navigation 비교
├─ CHANGELOG.md
└─ LICENSE             # MIT
```

### 추출 + 일반화 핵심
- **GROUP_LABELS 한국어 하드코딩 제거** → `PageRegistry(groups={...})` 생성자 주입
- **introspection dispatch + `llm_service↔llm` 별칭** → `dispatch.py` 분리
- **에러 바운더리** → `errors.py` 분리, **커스텀 핸들러 플러그 가능** (Sentry/로깅 통합용)
- **새 옵션**: `skip_keys` (호스트가 직접 그리는 페이지), `session_state_key` (세션 키 이름 자유), `set_error_handler()`

### vs `st.navigation` (공식)
| Feature | `st.navigation` | **streamlit-page-registry** |
|---|:---:|:---:|
| 동적 등록 | ✅ | ✅ |
| **에러 격리** (한 페이지 크래시 ≠ 앱 크래시) | ❌ | ✅ |
| **Introspection dispatch** (시그니처 자유) | ❌ | ✅ |
| **`validate_all()`** 사전 검증 | ❌ | ✅ |
| **그룹 expander** + 활성 그룹 자동 펼침 | ❌ | ✅ |
| 커스텀 에러 핸들러 | ❌ | ✅ |

## Test plan
- [x] `python -m unittest tests.test_registry` → **19/19 pass in 0.002s**
- [x] `python -m py_compile src/streamlit_page_registry/*.py` → exit 0
- [x] `python -c "from streamlit_page_registry import PageRegistry, PageEntry"` → OK
- [ ] `streamlit run examples/basic_app.py` (수동, 시각 검증) — 6개 페이지 + 5개 작동 + 1개 의도적 깨짐
- [ ] PyPI test upload (`twine upload --repository testpypi dist/*`) — 후속 단계

## What's NOT in this PR (다음 PR 들)
- `govable_ai/ui/pages/registry.py` 를 새 패키지로 갈아끼는 마이그레이션 (별도 PR)
- PyPI 정식 게시
- GitHub Actions CI 매트릭스 (Python 3.10/3.11/3.12)
- 별도 `streamlit-page-registry` 레포로 분리 (현재는 chungsim 안에 같이 보존)

## Background — 왜 이런 게 필요했나?
이 패키지는 `govable_ai` (5,300줄 Streamlit 모놀리스) 정리 작업 중 **multi-agent 병렬 마이그레이션 시 main.py 충돌 회피 인프라**로 만들어졌습니다. Gemini 시니어 리뷰에서 *"엔터프라이즈 인핸스먼트 레이어로 추출 적극 추천 (High ROI)"* 평가를 받아 별도 패키지화 진행.

상세 분석/플랜:
- [재사용 자산 분석 보고서](https://github.com/kim0395kk/chungsim/blob/main/.claude/plans/purrfect-doodling-torvalds.md) (private 가능 — 로컬 파일)
- [4개 트랙 실행 플랜](https://github.com/kim0395kk/chungsim/blob/main/.claude/plans/reusable-assets-execution-plan.md)

## Roadmap
- v0.2 — `entry_points` 플러그인 디스커버리 (외부 패키지가 페이지 등록)
- v0.3 — i18n 헬퍼
- v0.4 — 페이지 권한 (역할별 visibility)
- v1.0 — 안정 API + `st.Page` (2024 도입) 통합

🤖 Generated with [Claude Code](https://claude.com/claude-code)
