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
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class PageEntry:
    """단일 페이지 등록 정보."""

    key: str                 # session_state.current_page 값
    label: str               # 사이드바 버튼 라벨
    module: str              # 'govable_ai.ui.pages.civil_engineering'
    func: str = ""           # 진입 함수명. 비우면 'render_<key>_page' 자동 추론.

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


# =============================================================================
# 페이지 등록부 — 새 페이지는 여기 한 줄만 추가하면 됨.
# =============================================================================
_REGISTRY: list[PageEntry] = [
    PageEntry(
        key="workflow",
        label="🧠 업무 처리",
        module="govable_ai.main",
        func="_render_workflow_page",  # main.py 의 기본 워크플로우 (특수 케이스)
    ),
    PageEntry(
        key="compiler",
        label="📋 공문 컴파일",
        module="govable_ai.ui.doc_compiler_page",
        func="render_doc_compiler_page",
    ),
    PageEntry(
        key="civil",
        label="👷 토목 RAG",
        module="govable_ai.ui.pages.civil_engineering",
        func="render_civil_engineering_page",
    ),
]


def all_pages() -> list[PageEntry]:
    """등록된 모든 페이지를 순서대로 반환."""
    return list(_REGISTRY)


def get_page(key: str) -> Optional[PageEntry]:
    """key 로 페이지 항목 조회."""
    return next((p for p in _REGISTRY if p.key == key), None)


def render(key: str, **kwargs: Any) -> bool:
    """주어진 key 의 페이지를 렌더링한다.

    각 페이지 함수의 시그니처가 다양해도 안전하도록 introspection 으로
    수용 가능한 파라미터만 골라 전달한다. 단, ``llm_service`` 가 첫 위치
    인자인 경우(가장 흔한 패턴)는 위치 인자로 매핑해준다.

    Returns:
        True 면 페이지가 렌더링되어 호출 측은 종료해야 함. False 면 매칭 페이지 없음.
    """
    entry = get_page(key)
    if entry is None or entry.key == "workflow":
        # workflow 는 main.py 가 기본 흐름으로 직접 그리므로 이 helper 로 분기시키지 않는다.
        return False
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
    return True
