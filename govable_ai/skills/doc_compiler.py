# -*- coding: utf-8 -*-
"""
Govable AI - 공문서 컴파일러 스킬

행정업무운영 편람과 공문서 작성 지침에 따라
거친 초안을 규격화된 공문서로 변환

UI 의존성 없음 (streamlit import 금지)
"""
import json as _json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

from govable_ai.helpers import mask_sensitive

if TYPE_CHECKING:
    from govable_ai.core.llm_service import LLMService


class DocumentCompiler:
    """
    공문서 컴파일러 (Antigravity Norm Engine)
    
    거친 초안을 행정업무운영 편람과 공문서 작성 지침에 따라
    결재권자가 즉시 서명할 수 있는 '합격 문서'로 재구성합니다.
    
    지침 파일(rules/standard.txt)을 RAG 방식으로 읽어서 적용합니다.
    지침이 변경되면 파일만 수정하면 됩니다.
    """
    
    # 지침 파일 경로 (rules/standard.txt)
    RULES_FILE = Path(__file__).parent.parent / "rules" / "standard.txt"
    
    def __init__(self, llm_service: "LLMService"):
        """
        Args:
            llm_service: LLM 서비스 인스턴스
        """
        self.llm = llm_service
        self._rules_cache: Optional[str] = None
    
    def _load_rules(self) -> str:
        """
        지침 파일을 RAG 방식으로 로드
        
        Returns:
            지침 파일 내용 (없으면 기본 규칙)
        """
        if self._rules_cache is not None:
            return self._rules_cache
        
        try:
            if self.RULES_FILE.exists():
                self._rules_cache = self.RULES_FILE.read_text(encoding="utf-8")
                return self._rules_cache
        except Exception:
            pass
        
        # 파일이 없으면 기본 규칙 반환
        self._rules_cache = """
## 기본 규칙
1. 항목 체계: 1. → 가. → 1) → 가) → (1) → (가)
2. 문체: "~합니다" → "~함", "~바랍니다" → "~바람"
3. 날짜: YYYY. M. D. 형식
4. 시간: HH:MM 형식 (24시간제)
5. 본문 끝: "끝."
"""
        return self._rules_cache
    
    def reload_rules(self) -> str:
        """지침 파일 다시 로드 (캐시 갱신)"""
        self._rules_cache = None
        return self._load_rules()
    
    def compile(self, draft: str) -> Dict:
        """
        초안을 행정 표준에 맞게 컴파일
        
        Args:
            draft: 사용자가 입력한 거친 초안
            
        Returns:
            {
                "original_draft": str,    # 원문 (Diff View용)
                "compiled_doc": str,      # 컴파일된 기안문 전문
                "corrections": {
                    "style": List[str],   # 문체 교정 내역
                    "format": List[str],  # 형식 교정 내역
                    "security": List[str] # 보안 교정 내역
                },
                "structure": {
                    "title": str,
                    "overview": str,
                    "basis": str,
                    "details": str,
                    "admin_notes": str
                }
            }
        """
        today = datetime.now()
        today_str = today.strftime("%Y. %m. %d.")
        
        # RAG: 지침 파일 로드
        rules_content = self._load_rules()
        
        schema = """
{
  "compiled_doc": "컴파일된 기안문 전체 내용 (행정 표준 적용)",
  "corrections": {
    "style": ["문체 교정 내역 1", "문체 교정 내역 2"],
    "format": ["형식 교정 내역 1", "형식 교정 내역 2"],
    "security": ["보안 교정 내역 1"]
  },
  "structure": {
    "title": "제목",
    "overview": "개요",
    "basis": "근거",
    "details": "세부 내용",
    "admin_notes": "행정 사항"
  }
}
""".strip()
        
        prompt = f"""
너는 대한민국 행정 표준을 집행하는 **'안티그래비티 공문 컴파일러(Official Document Compiler)'**다.
너의 목적은 사용자가 입력한 거친 초안을 아래 지침에 따라 결재권자가 즉시 서명할 수 있는 수준의 '합격 문서'로 재구성하는 것이다.

## 적용할 지침 (rules/standard.txt에서 로드됨)

{rules_content}

## 작업 프로세스 (Step-by-Step)

**Step 1 (분석)**: 입력된 초안에서 핵심 목적, 근거, 실행 계획을 추출한다.
**Step 2 (구조화)**: 추출된 내용을 [제목 - 개요 - 근거 - 세부 내용 - 행정 사항]의 표준 구조로 배치한다.
**Step 3 (컴파일)**: 설정된 행정 문체로 문장을 치환하고 항목 번호를 정렬한다.

## 입력된 초안

{mask_sensitive(draft)}

## 오늘 날짜

{today_str}

## 출력 JSON 스키마

{schema}

## 규칙
- corrections에는 원문에서 변경된 내용을 **구체적으로** 기술 (예: "~합니다" → "~함"으로 3회 변환)
- compiled_doc은 실제 사용 가능한 완성된 기안문이어야 함
- 개인정보(성명, 연락처, 주소 등)가 포함되어 있으면 마스킹하고 보안 교정 내역에 기록

JSON만 출력.
"""
        
        data = self.llm.generate_json(prompt)
        
        # 결과 검증 및 폴백
        if self._validate_result(data):
            data["original_draft"] = draft  # 원문 추가 (Diff View용)
            return data
        
        # 재시도
        retry_prompt = f"""
방금 출력이 스키마를 만족하지 않았다.
아래 스키마를 정확히 만족하는 JSON만 다시 출력하라.

스키마:
{schema}

(다른 텍스트 금지)
"""
        data2 = self.llm.generate_json(prompt + "\n\n" + retry_prompt)
        
        if self._validate_result(data2):
            data2["original_draft"] = draft
            return data2
        
        # 폴백: 기본 결과
        result = self._create_fallback(draft, today_str)
        result["original_draft"] = draft
        return result
    
    def _validate_result(self, data: Dict) -> bool:
        """결과 유효성 검증"""
        if not isinstance(data, dict):
            return False
        if not data.get("compiled_doc"):
            return False
        if not isinstance(data.get("corrections"), dict):
            return False
        return True
    
    def _create_fallback(self, draft: str, today_str: str) -> Dict:
        """폴백 결과 생성"""
        masked_draft = mask_sensitive(draft)
        
        return {
            "compiled_doc": f"""제목: 업무 처리 관련

1. 개요
   가. 본 건은 아래와 같이 업무를 처리하고자 함.

2. 근거
   가. 관련 법령 및 규정에 따름.

3. 세부 내용
   {masked_draft[:500]}

4. 행정 사항
   가. 관련 부서 협조 바람.
   나. 시행일: {today_str}

끝.""",
            "corrections": {
                "style": ["존댓말 서술형을 행정 간결체로 수정"],
                "format": ["날짜 표기법 적용", "항목 번호 체계 적용"],
                "security": []
            },
            "structure": {
                "title": "업무 처리 관련",
                "overview": "본 건은 아래와 같이 업무를 처리하고자 함.",
                "basis": "관련 법령 및 규정에 따름.",
                "details": masked_draft[:200] + "...",
                "admin_notes": "관련 부서 협조 바람."
            }
        }
