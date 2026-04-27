# -*- coding: utf-8 -*-
"""
Pages Registry — 사이드바 라우팅 단일 진입점.

이곳에 새 페이지를 등록하면 govable_ai.main 은 변경 없이 라우트가 추가된다.
병렬 마이그레이션 시 main.py 충돌을 방지하기 위한 핵심 인프라.

[등록 방법]
1. ``govable_ai/ui/pages/<your_page>.py`` 작성:
   - ``def render_xxx_page(llm_service, db=None, services=None) -> None``
2. 아래 ``_REGISTRY`` 리스트 끝에 ``PageEntry`` 한 줄 추가.

[Page 인터페이스]
모든 페이지는 다음 인터페이스를 따른다::

    def render_<name>_page(
        llm_service: Any,
        db: Optional[Any] = None,
        services: Optional[dict] = None,
    ) -> None: ...

services 딕셔너리에는 ``llm``, ``law_api``, ``search``, ``db`` 등이 들어있고,
페이지가 필요로 하는 항목만 꺼내 쓰면 된다. ``db`` 는 흔히 쓰이므로 1급 키워드로 유지.
"""
from __future__ import annotations

import inspect
import logging
import traceback
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageEntry:
    """단일 페이지 등록 정보."""

    key: str                 # session_state.current_page 값
    label: str               # 사이드바 버튼 라벨
    module: str              # 'govable_ai.ui.pages.civil_engineering'
    func: str = ""           # 진입 함수명. 비우면 'render_<key>_page' 자동 추론.
    group: str = "core"      # 사이드바 그룹: 'core' | 'search' | 'meta'

    @property
    def function_name(self) -> str:
        return self.func or f"render_{self.key}_page"

    def load(self) -> Callable[..., None]:
        """진입 함수를 lazy import 한다. 누락된 의존성으로 인한 부팅 실패 방지."""
        mod = import_module(self.module)
        fn = getattr(mod, self.function_name, None)
        if fn is None or not callable(fn):
            raise RuntimeError(
                f"Page entry '{self.key}' is missing callable "
                f"{self.module}:{self.function_name}"
            )
        return fn


# 사이드바에 표시할 그룹 라벨 (등록부 순서를 보존하기 위해 dict 순서 사용).
GROUP_LABELS: dict[str, str] = {
    "core": "🧠 핵심 업무",
    "search": "🔎 검색 / RAG",
    "meta": "⚙️ 운영 / 메타",
}


# =============================================================================
# 페이지 등록부 — 새 페이지는 여기 한 줄만 추가하면 됨.
# =============================================================================
_REGISTRY: list[PageEntry] = [
    # --- Core workflows ---
    PageEntry(
        key="workflow",
        label="🧠 업무 처리",
        module="govable_ai.main",
        func="_render_workflow_page",  # main.py 의 기본 워크플로우 (특수 케이스)
        group="core",
    ),
    PageEntry(
        key="compiler",
        label="📋 공문 컴파일",
        module="govable_ai.ui.doc_compiler_page",
        func="render_doc_compiler_page",
        group="core",
    ),
    PageEntry(
        key="revision",
        label="📝 기안문 수정",
        module="govable_ai.ui.pages.document_revision",
        func="render_document_revision_page",
        group="core",
    ),
    PageEntry(
        key="complaint",
        label="🧾 민원 분석기",
        module="govable_ai.ui.pages.complaint_analyzer",
        func="render_complaint_analyzer_page",
        group="core",
    ),
    PageEntry(
        key="hallucination",
        label="🔍 환각 검사",
        module="govable_ai.ui.pages.hallucination_check",
        func="render_hallucination_check_page",
        group="core",
    ),
    # --- Search / RAG ---
    PageEntry(
        key="civil",
        label="👷 토목 RAG",
        module="govable_ai.ui.pages.civil_engineering",
        func="render_civil_engineering_page",
        group="search",
    ),
    PageEntry(
        key="duty",
        label="📚 업무편람",
        module="govable_ai.ui.pages.duty_manual",
        func="render_duty_manual_page",
        group="search",
    ),
    # --- Meta / 운영 ---
    PageEntry(
        key="history",
        label="🗂️ 작업 이력",
        module="govable_ai.ui.pages.archive_history",
        func="render_archive_history_page",
        group="meta",
    ),
    PageEntry(
        key="health",
        label="🩺 시스템 상태",
        module="govable_ai.ui.pages.service_health",
        func="render_service_health_page",
        group="meta",
    ),
    PageEntry(
        key="dashboard",
        label="🏛️ 마스터 대시보드",
        module="govable_ai.ui.pages.master_dashboard",
        func="render_master_dashboard_page",
        group="meta",
    ),
]


def pages_by_group() -> dict[str, list[PageEntry]]:
    """그룹별로 페이지 항목을 묶어 반환 (사이드바 렌더링용)."""
    out: dict[str, list[PageEntry]] = {g: [] for g in GROUP_LABELS}
    for p in _REGISTRY:
        out.setdefault(p.group, []).append(p)
    return out


def validate_all() -> list[str]:
    """모든 등록 페이지의 lazy import + 시그니처를 검증한다.

    개발/배포 시 한 번만 호출하면 IDE 가 잡지 못한 시그니처 오류를 조기에 발견한다.
    워크플로우 페이지는 main.py 안에서 직접 그려지므로 검증 대상에서 제외.

    Returns:
        에러 메시지 리스트. 비어 있으면 모두 정상.
    """
    errors: list[str] = []
    for p in _REGISTRY:
        if p.key == "workflow":
            continue
        try:
            fn = p.load()
            sig = inspect.signature(fn)
            params = sig.parameters
            has_var_kw = any(
                v.kind == inspect.Parameter.VAR_KEYWORD for v in params.values()
            )
            if not has_var_kw:
                # 표준 인터페이스 권장 인자 중 적어도 하나는 받아야 한다.
                expected = {"llm_service", "llm", "db", "services"}
                if not (set(params) & expected):
                    errors.append(
                        f"{p.key}: {p.module}:{p.function_name} signature "
                        f"({', '.join(params)}) accepts none of {sorted(expected)}"
                    )
        except Exception as e:
            errors.append(f"{p.key}: {type(e).__name__}: {e}")
    return errors


def all_pages() -> list[PageEntry]:
    """등록된 모든 페이지를 순서대로 반환."""
    return list(_REGISTRY)


def get_page(key: str) -> Optional[PageEntry]:
    """key 로 페이지 항목 조회."""
    return next((p for p in _REGISTRY if p.key == key), None)


def _render_error_ui(entry: PageEntry, error: BaseException) -> None:
    """페이지 렌더링 중 예외가 발생했을 때 사용자에게 친화적으로 표시한다.

    streamlit 은 호출 시점에만 import — registry 가 streamlit 의존성을 끌어오지
    않도록 lazy import.
    """
    import streamlit as st

    st.error(f"⚠️ '{entry.label}' 페이지를 그리는 중 오류가 발생했습니다.")
    st.caption(f"`{type(error).__name__}: {error}`")
    with st.expander("🛠️ 자세한 오류 정보 (개발자용)", expanded=False):
        st.code("".join(traceback.format_exception(error)), language="text")
    st.info(
        "다른 메뉴로 이동하거나 새로고침해보세요. 같은 오류가 반복되면 "
        "🩺 시스템 상태 페이지에서 서비스 연결을 점검해주세요."
    )


def render(key: str, **kwargs: Any) -> bool:
    """주어진 key 의 페이지를 렌더링한다.

    각 페이지 함수의 시그니처가 다양해도 안전하도록 introspection 으로
    수용 가능한 파라미터만 골라 전달한다. 한 페이지의 예외가 앱 전체를 멈추지
    않도록 전역 에러 바운더리로 감싼다.

    Returns:
        True 면 페이지가 렌더링되어 호출 측은 종료해야 함. False 면 매칭 페이지 없음.
    """
    entry = get_page(key)
    if entry is None or entry.key == "workflow":
        # workflow 는 main.py 가 기본 흐름으로 직접 그리므로 이 helper 로 분기시키지 않는다.
        return False

    try:
        fn = entry.load()

        sig = inspect.signature(fn)
        params = sig.parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

        if has_var_kw:
            accepted = kwargs
        else:
            accepted = {k: v for k, v in kwargs.items() if k in params}

        # 흔한 별칭 처리: 함수가 ``llm`` 만 받는 경우, llm_service 를 그쪽으로 매핑
        if "llm" in params and "llm_service" not in params and "llm_service" in kwargs:
            accepted["llm"] = kwargs["llm_service"]
            accepted.pop("llm_service", None)

        fn(**accepted)
    except Exception as e:
        # 로그는 운영 모니터링용으로 남기고, UI 에는 친화적으로 보여준다.
        logger.exception("Page render failed: key=%s module=%s", entry.key, entry.module)
        _render_error_ui(entry, e)

    return True
