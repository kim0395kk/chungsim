# -*- coding: utf-8 -*-
"""Unit tests for govable_ai.features.complaint_analyzer.run_complaint_analyzer_workflow.

LLM/law_api 를 mock 으로 주입해, 워크플로우가 5단계를 모두 통과하고 legacy 호환
반환 키 셋을 산출하는지 확인한다.
"""
from __future__ import annotations

import json
import sys
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from govable_ai.features.complaint_analyzer import (  # noqa: E402
    mask_sensitive,
    run_complaint_analyzer_workflow,
)


# =============================================================================
# Mock LLM — 워크플로우 단계마다 다른 JSON 응답을 돌려준다.
# =============================================================================
class MockLLM:
    """generate_json(prompt) 호출의 prompt 내용으로 단계를 식별해 응답한다."""

    model_name = "mock-model"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate_json(self, prompt: str, **_kwargs):  # noqa: ANN001
        self.calls.append(prompt[:80])

        # phase 1 — claim decomposition prompt 는 "주장(Claim)" / "MVC" 단어 포함
        if "MVC" in prompt or "claim" in prompt.lower() or "주장" in prompt:
            return {
                "mvc": {
                    "time": "2024-03-15",
                    "place": "충주시 ○○동",
                    "target": "도로 포장 균열",
                    "request": "보수 요청",
                    "evidence": ["사진 1장"],
                },
                "claims": [
                    {
                        "id": "c1",
                        "type": "fact",
                        "text": "도로에 균열이 있다",
                        "notes": "민원인 직접 관찰",
                        "law_refs": [
                            {"law": "도로법", "article": "31"},
                        ],
                    },
                ],
                "hallucination_signals": [],
            }

        # phase 4 — judge prompt (per claim)
        if "verdict" in prompt.lower() or "판정" in prompt:
            return {
                "verdict": "INSUFFICIENT",
                "reason": "추가 조사가 필요",
                "next_actions": ["현장 확인"],
            }

        # default
        return {}

    def generate_text(self, prompt: str, **_kwargs):  # noqa: ANN001
        return "Mock generated text"

    def get_last_usage(self) -> dict:
        return {"input_tokens": 0, "output_tokens": 0, "model_used": "mock-model"}


class MockLawAPI:
    def __init__(self, available: bool = True) -> None:
        self._available = available

    def get_law_text(self, law_name: str, article: str = "", return_link: bool = False):
        if not self._available:
            return None
        return {
            "name": law_name,
            "article": article,
            "text": f"제{article or '?'}조 (mock 본문)",
            "link": f"https://law.go.kr/?l={law_name}",
        }


# =============================================================================
# Tests
# =============================================================================
class MaskSensitiveTest(unittest.TestCase):
    def test_phone_pattern(self) -> None:
        self.assertIn("****", mask_sensitive("연락처 010-1234-5678 입니다"))

    def test_rrn_pattern(self) -> None:
        self.assertIn("******", mask_sensitive("주민번호 900101-1234567"))

    def test_empty(self) -> None:
        self.assertEqual(mask_sensitive(""), "")
        self.assertEqual(mask_sensitive(None), "")


class RunComplaintWorkflowTest(unittest.TestCase):
    sample_input = (
        "2024년 3월 15일 충주시 ○○동 도로 포장에 균열이 발생했습니다.\n"
        "도로법 제31조에 따라 보수 요청 드립니다."
    )

    def test_requires_llm(self) -> None:
        with self.assertRaises(ValueError):
            run_complaint_analyzer_workflow(self.sample_input, llm_service=None)

    def test_returns_legacy_compatible_keys(self) -> None:
        llm = MockLLM()
        law = MockLawAPI(available=True)
        result = run_complaint_analyzer_workflow(
            self.sample_input,
            llm_service=llm,
            law_api=law,
        )

        # 최상위 키 — legacy contract
        for key in (
            "situation",
            "analysis",
            "law",
            "strategy",
            "doc",
            "meta",
            "complaint_pack",
            "execution_time",
            "model_used",
        ):
            self.assertIn(key, result, f"missing key: {key}")

        # complaint_pack 내부 핵심 필드
        pack = result["complaint_pack"]
        for sub_key in (
            "mvc",
            "claims",
            "noise_grade",
            "verifiability_score",
            "citations",
            "verdicts",
        ):
            self.assertIn(sub_key, pack, f"complaint_pack missing: {sub_key}")

        # MVC 가 LLM 응답대로 채워졌는지
        self.assertEqual(pack["mvc"]["target"], "도로 포장 균열")
        self.assertEqual(len(pack["claims"]), 1)
        self.assertEqual(pack["claims"][0]["id"], "c1")

        # noise_grade 는 GREEN/YELLOW/RED 셋 중 하나
        self.assertIn(pack["noise_grade"], {"GREEN", "YELLOW", "RED"})

        # 실행 시간 기록 됨
        self.assertGreaterEqual(result["execution_time"], 0.0)

        # 5 phase 진행되며 LLM 이 호출됐는지 (phase1 분해 + phase4 판정)
        self.assertGreaterEqual(len(llm.calls), 2)

    def test_services_dict_alternative_injection(self) -> None:
        """services dict 로 의존성 주입해도 동작해야 함."""
        llm = MockLLM()
        law = MockLawAPI()
        result = run_complaint_analyzer_workflow(
            self.sample_input,
            services={"llm": llm, "law_api": law},
        )
        self.assertIn("complaint_pack", result)

    def test_progress_callback_invoked(self) -> None:
        llm = MockLLM()
        law = MockLawAPI()
        stages: list[str] = []

        def _on_progress(stage: str, msg: str) -> None:  # noqa: ARG001
            stages.append(stage)

        run_complaint_analyzer_workflow(
            self.sample_input,
            llm_service=llm,
            law_api=law,
            on_progress=_on_progress,
        )

        # 최소 phase1 ~ phase3 는 호출돼야 함
        for s in ("phase1", "phase2", "phase3"):
            self.assertIn(s, stages, f"progress stage {s} not reported")

    def test_progress_callback_exception_is_swallowed(self) -> None:
        """progress 콜백이 raise 해도 워크플로우는 끝까지 돌아야 한다."""
        llm = MockLLM()
        law = MockLawAPI()

        def _bad_callback(*_args, **_kwargs):
            raise RuntimeError("callback exploded")

        result = run_complaint_analyzer_workflow(
            self.sample_input,
            llm_service=llm,
            law_api=law,
            on_progress=_bad_callback,
        )
        self.assertIn("complaint_pack", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
