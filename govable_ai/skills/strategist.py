# -*- coding: utf-8 -*-
"""
Govable AI - 전략 및 절차 계획 에이전트

UI 의존성 없음 (streamlit import 금지)
"""
import json as _json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from govable_ai.core.llm_service import LLMService


class Strategist:
    """
    처리 전략 수립 에이전트
    
    의존성 주입으로 LLM 서비스를 전달받습니다.
    UI(streamlit) 의존성이 없어 다른 환경에서도 사용 가능합니다.
    """
    
    def __init__(self, llm_service: "LLMService"):
        """
        Args:
            llm_service: LLM 서비스 인스턴스
        """
        self.llm = llm_service
    
    def plan_strategy(
        self,
        situation: str,
        legal_basis_md: str,
        search_results: str = "",
    ) -> str:
        """
        처리 전략 수립
        
        Args:
            situation: 민원 상황 설명
            legal_basis_md: 법령 근거 마크다운
            search_results: 검색 결과 (뉴스/판례)
            
        Returns:
            처리 전략 마크다운 문자열
        """
        prompt = f"""
당신은 행정 업무 베테랑 '주무관'입니다.

[민원 상황]: {situation}
[확보된 법적 근거]:
{legal_basis_md}

[유사 사례/판례]: {search_results or "(없음)"}

위 정보를 종합하여 민원 처리 방향(Strategy)을 수립하세요.
서론(인사말/공감/네 알겠습니다 등) 금지.

1. 처리 방향
2. 핵심 주의사항
3. 예상 반발 및 대응
"""
        return self.llm.generate_text(prompt)


class ProcedurePlanner:
    """
    행정 절차 플래너 에이전트
    
    의존성 주입으로 LLM 서비스를 전달받습니다.
    UI(streamlit) 의존성이 없어 다른 환경에서도 사용 가능합니다.
    """
    
    def __init__(self, llm_service: "LLMService"):
        """
        Args:
            llm_service: LLM 서비스 인스턴스
        """
        self.llm = llm_service
    
    def plan(
        self,
        situation: str,
        legal_basis_summary: str,
        analysis: dict,
    ) -> dict:
        """
        절차 계획 수립
        
        Args:
            situation: 민원 상황 설명
            legal_basis_summary: 법령 근거 요약
            analysis: CaseAnalyzer.analyze() 결과
            
        Returns:
            절차 계획 딕셔너리:
            {
                "timeline": List[dict],  # 단계별 계획
                "checklist": List[str],  # 체크리스트
                "templates": List[str],  # 필요 서식 목록
            }
        """
        prompt = f"""
너는 '행정 절차 플래너'이다.

[상황]
{situation}

[분석]
{_json.dumps(analysis, ensure_ascii=False)}

[법적 근거(요약)]
{legal_basis_summary}

[출력 JSON]
{{
  "timeline": [
    {{"step": 1, "name": "단계명", "goal": "목표", "actions": ["행동1","행동2"], "records": ["기록/증빙"], "legal_note": "근거/유의"}}
  ],
  "checklist": ["담당자가 체크할 항목 10개"],
  "templates": ["필요 서식/문서 이름 5개"]
}}
JSON만.
"""
        data = self.llm.generate_json(prompt)
        
        if isinstance(data, dict) and data.get("timeline"):
            return data
        
        # 폴백: 기본 절차 계획
        return {
            "timeline": [
                {
                    "step": 1,
                    "name": "사실확인",
                    "goal": "사실관계 확정",
                    "actions": ["현장 확인", "증빙 확보"],
                    "records": ["사진/위치/시간"],
                    "legal_note": "기록이 절차 정당성 핵심",
                },
                {
                    "step": 2,
                    "name": "대상 특정",
                    "goal": "소유자/점유자 특정",
                    "actions": ["등록정보 조회", "연락/안내"],
                    "records": ["조회 로그", "통화/안내 기록"],
                    "legal_note": "통지/연락 시도 기록",
                },
                {
                    "step": 3,
                    "name": "통지/계고",
                    "goal": "자진 조치 유도",
                    "actions": ["계고/안내", "기한 부여"],
                    "records": ["통지문", "발송/수령 증빙"],
                    "legal_note": "행정절차상 통지 누락 주의",
                },
                {
                    "step": 4,
                    "name": "불이행 시 조치",
                    "goal": "강제/처분 검토",
                    "actions": ["불이행 확인", "처분/강제 조치"],
                    "records": ["확인서", "처분문"],
                    "legal_note": "처분 사유/근거 명확화",
                },
            ],
            "checklist": [
                "증빙 확보",
                "법령 근거 확인",
                "통지/의견제출 기회",
                "문서번호/기한",
                "기록 남김",
            ],
            "templates": [
                "회신 공문",
                "계고/통지",
                "의견제출 안내",
                "공시송달 공고",
                "처분서",
            ],
        }
