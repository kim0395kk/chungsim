# -*- coding: utf-8 -*-
"""
Govable AI - 공문서 작성 에이전트

UI 의존성 없음 (streamlit import 금지)
"""
import json as _json
import time
from datetime import datetime
from typing import List, TYPE_CHECKING

from govable_ai.helpers import mask_sensitive

if TYPE_CHECKING:
    from govable_ai.core.llm_service import LLMService


class DocumentDrafter:
    """
    공문서 작성 에이전트
    
    의존성 주입으로 LLM 서비스를 전달받습니다.
    UI(streamlit) 의존성이 없어 다른 환경에서도 사용 가능합니다.
    
    사용법:
        ```python
        from govable_ai.skills.drafter import DocumentDrafter
        from govable_ai.core import LLMService
        
        llm = LLMService(gemini_key="...")
        drafter = DocumentDrafter(llm)
        meta = drafter.generate_meta()
        doc = drafter.draft(situation, legal_basis, meta, strategy, procedure, [])
        print(doc)  # {"title": ..., "body_paragraphs": [...], ...}
        ```
    """
    
    def __init__(self, llm_service: "LLMService"):
        """
        Args:
            llm_service: LLM 서비스 인스턴스
        """
        self.llm = llm_service
    
    @staticmethod
    def generate_meta() -> dict:
        """
        공문 메타데이터 생성
        
        Returns:
            {"today_str": str, "doc_num": str}
        """
        today = datetime.now()
        return {
            "today_str": today.strftime("%Y. %m. %d."),
            "doc_num": f"행정-{today.strftime('%Y')}-{int(time.time()) % 1000:03d}호",
        }
    
    def draft(
        self,
        situation: str,
        legal_basis_md: str,
        meta: dict,
        strategy: str,
        procedure: dict,
        objections: List[dict] = None,
    ) -> dict:
        """
        공문서 초안 작성
        
        Args:
            situation: 민원 상황 설명
            legal_basis_md: 법령 근거 마크다운
            meta: generate_meta() 결과
            strategy: Strategist.plan_strategy() 결과
            procedure: ProcedurePlanner.plan() 결과
            objections: 예상 반발 목록 (선택)
            
        Returns:
            공문서 딕셔너리:
            {
                "title": str,
                "receiver": str,
                "body_paragraphs": List[str],
                "department_head": str,
            }
        """
        objections = objections or []
        
        schema = """
{
  "title": "제목",
  "receiver": "수신",
  "body_paragraphs": ["문단1", "문단2", "..."],
  "department_head": "OOO과장"
}
""".strip()
        
        prompt = f"""
당신은 행정기관의 베테랑 서기이다. 아래 정보를 바탕으로 완결된 공문서를 JSON으로 작성하라.

[입력]
- 민원: {mask_sensitive(situation)}
- 시행일자: {meta.get('today_str')}
- 문서번호: {meta.get('doc_num')}

[법령 근거(필수 인용)]
{legal_basis_md}

[처리방향]
{strategy}

[절차 플랜(반영)]
{_json.dumps(procedure, ensure_ascii=False)}

[예상 반발(반영)]
{_json.dumps(objections, ensure_ascii=False)}

[원칙]
- 본문에 법 조항/근거를 문장으로 인용할 것
- 구조: 경위 -> 법적 근거 -> 조치/안내 -> 이의제기/문의
- 개인정보는 OOO로 마스킹
- 문단 내에 **1** 같은 번호는 **볼드**로 표시해도 됨(마크다운 허용)

[출력 JSON 스키마]
{schema}

JSON만 출력.
"""
        data = self.llm.generate_json(prompt)
        
        if isinstance(data, dict) and data.get("title") and data.get("body_paragraphs"):
            return data
        
        # 재시도
        retry = f"""
방금 출력이 스키마를 만족하지 않았다.
아래 스키마를 정확히 만족하는 JSON만 다시 출력하라.

스키마:
{schema}

(다른 텍스트 금지)
"""
        data2 = self.llm.generate_json(prompt + "\n\n" + retry)
        
        if isinstance(data2, dict) and data2.get("title") and data2.get("body_paragraphs"):
            return data2
        
        # 폴백: 기본 공문 템플릿
        return {
            "title": "민원 처리 결과 회신(안)",
            "receiver": "수신자 참조",
            "body_paragraphs": [
                "**1**. 경위",
                f"- 민원 요지: {mask_sensitive(situation[:200])}",
                "",
                "**2**. 법적 근거",
                "- 관련 법령 및 조문 근거에 따라 절차를 진행합니다.",
                "",
                "**3**. 조치 내용",
                "- 사실 확인 및 필요 절차를 단계적으로 이행 예정입니다.",
                "",
                "**4**. 이의제기/문의",
                "- 추가 의견이 있는 경우 의견제출 절차로 제출 바랍니다.",
            ],
            "department_head": "OOO과장",
        }
