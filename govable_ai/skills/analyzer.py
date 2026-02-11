# -*- coding: utf-8 -*-
"""
Govable AI - 민원 분석 에이전트

UI 의존성 없음 (streamlit import 금지)
"""
from typing import TYPE_CHECKING

from govable_ai.helpers import mask_sensitive

if TYPE_CHECKING:
    from govable_ai.core.llm_service import LLMService


class CaseAnalyzer:
    """
    민원/업무 케이스 분석 에이전트
    
    의존성 주입으로 LLM 서비스를 전달받습니다.
    UI(streamlit) 의존성이 없어 다른 환경에서도 사용 가능합니다.
    
    사용법:
        ```python
        from govable_ai.skills.analyzer import CaseAnalyzer
        from govable_ai.core.llm_service import LLMService
        
        llm = LLMService(gemini_key="...")
        analyzer = CaseAnalyzer(llm)
        result = analyzer.analyze("무단방치 차량 처리 문의")
        print(result)
        ```
    """
    
    def __init__(self, llm_service: "LLMService"):
        """
        Args:
            llm_service: LLM 서비스 인스턴스
        """
        self.llm = llm_service
    
    def analyze(self, situation: str) -> dict:
        """
        민원 상황 분석
        
        Args:
            situation: 민원 상황 설명
            
        Returns:
            분석 결과 딕셔너리:
            {
                "case_type": str,           # 케이스 유형
                "core_issue": List[str],    # 핵심 쟁점
                "required_facts": List[str], # 추가 필요 사실
                "required_evidence": List[str], # 필요 증빙
                "risk_flags": List[str],    # 리스크 플래그
                "recommended_next_action": List[str], # 권장 다음 행동
            }
        """
        s = mask_sensitive(situation)
        
        prompt = f"""
너는 '민원/업무 케이스 분석관'이다.
한국어로 응답하되, 법률 용어나 고유명사 등 필요한 경우 영어는 사용 가능하다. 단, 베트남어/중국어/일본어 등 기타 외국어는 사용하지 마라.

[입력]
{s}

[출력 JSON]
{{
  "case_type": "예: 무단방치/번호판훼손/불법주정차/건설기계/기타",
  "core_issue": ["핵심 쟁점 3~6개 (한국어만)"],
  "required_facts": ["추가로 필요한 사실확인 질문 5개"],
  "required_evidence": ["필요 증빙 5개"],
  "risk_flags": ["절차상 리스크 3개(예: 통지 누락, 증거 부족...)"],
  "recommended_next_action": ["즉시 다음 행동 3개"]
}}
JSON만 출력. 반드시 한국어로.
"""
        data = self.llm.generate_json(prompt)
        
        if isinstance(data, dict) and data.get("case_type"):
            return data
        
        # 폴백: 키워드 기반 기본 분석
        t = "기타"
        if "무단방치" in situation:
            t = "무단방치"
        if "번호판" in situation:
            t = "번호판훼손"
        if "주정차" in situation:
            t = "불법주정차"
        
        return {
            "case_type": t,
            "core_issue": ["사실관계 확정", "증빙 확보", "절차적 정당성 확보"],
            "required_facts": [
                "장소/시간?",
                "증빙(사진/영상)?",
                "소유자 특정 가능?",
                "반복/상습 여부?",
                "요청사항(처분/계도/회신)?",
            ],
            "required_evidence": [
                "현장 사진",
                "위치/시간 기록",
                "신고내용 원문",
                "소유자 확인 자료",
                "조치/통지 기록",
            ],
            "risk_flags": [
                "통지/의견제출 기회 누락",
                "증거 부족",
                "법적 근거 불명확",
            ],
            "recommended_next_action": [
                "증빙 정리",
                "소유자/점유자 확인",
                "절차 플로우 확정",
            ],
        }
