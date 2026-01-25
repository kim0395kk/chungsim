# -*- coding: utf-8 -*-
"""
Govable AI - 법령 탐색 에이전트

UI 의존성 없음 (streamlit import 금지)
"""
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from govable_ai.core.llm_service import LLMService
    from govable_ai.core.law_api import LawOfficialService


class LegalResearcher:
    """
    법령 탐색 에이전트
    
    의존성 주입으로 LLM 서비스와 법령 API를 전달받습니다.
    UI(streamlit) 의존성이 없어 다른 환경에서도 사용 가능합니다.
    
    사용법:
        ```python
        from govable_ai.skills.researcher import LegalResearcher
        from govable_ai.core import LLMService, LawOfficialService
        
        llm = LLMService(gemini_key="...")
        law_api = LawOfficialService(api_id="...")
        researcher = LegalResearcher(llm, law_api)
        result = researcher.research("무단방치 차량", {"case_type": "무단방치"})
        print(result)  # 마크다운 문자열
        ```
    """
    
    def __init__(
        self,
        llm_service: "LLMService",
        law_api: "LawOfficialService",
    ):
        """
        Args:
            llm_service: LLM 서비스 인스턴스
            law_api: 법령 API 서비스 인스턴스
        """
        self.llm = llm_service
        self.law_api = law_api
    
    def research(self, situation: str, analysis: dict) -> str:
        """
        법령 탐색 및 조문 수집
        
        Args:
            situation: 민원 상황 설명
            analysis: CaseAnalyzer.analyze() 결과
            
        Returns:
            법령 탐색 결과 마크다운 문자열
        """
        # 1. LLM으로 관련 법령/조문 추출
        prompt_extract = f"""
상황: "{situation}"

위 민원 처리를 위해 법적 근거로 삼아야 할 핵심 대한민국 법령과 조문 번호를
**중요도 순으로 최대 3개까지** JSON 리스트로 추출하시오.

형식: [{{"law_name": "도로교통법", "article_num": 32}}, ...]
* 법령명은 정식 명칭 사용. 조문 번호 불명확하면 null.
"""
        search_targets = []
        try:
            extracted = self.llm.generate_json(prompt_extract)
            if isinstance(extracted, list):
                search_targets = extracted
            elif isinstance(extracted, dict):
                search_targets = [extracted]
        except Exception:
            search_targets = [{"law_name": "도로교통법", "article_num": None}]
        
        if not search_targets:
            search_targets = [{"law_name": "도로교통법", "article_num": None}]
        
        # 2. 법령 API로 조문 조회
        report_lines = []
        api_success_count = 0
        
        report_lines.append(f"##### 🔍 AI가 식별한 핵심 법령 ({len(search_targets)}건)")
        report_lines.append("---")
        
        for idx, item in enumerate(search_targets):
            law_name = item.get("law_name", "관련법령")
            article_num = item.get("article_num")
            
            law_text, current_link = self.law_api.get_law_text(
                law_name, article_num, return_link=True
            )
            
            error_keywords = ["검색 결과가 없습니다", "오류", "API ID", "실패"]
            is_success = not any(k in (law_text or "") for k in error_keywords)
            
            if is_success:
                api_success_count += 1
                law_title = f"[{law_name}]({current_link})" if current_link else law_name
                art_str = f" 제{article_num}조" if article_num else ""
                header = f"✅ **{idx+1}. {law_title}{art_str} (확인됨)**"
                content = law_text
            else:
                art_str = f" 제{article_num}조" if article_num else ""
                header = f"⚠️ **{idx+1}. {law_name}{art_str} (API 조회 실패)**"
                content = "(국가법령정보센터에서 해당 조문을 찾지 못했습니다. 법령명이 정확한지 확인이 필요합니다.)"
            
            report_lines.append(f"{header}\n{content}\n")
        
        final_report = "\n".join(report_lines)
        
        # 3. API 실패 시 LLM 폴백
        if api_success_count == 0:
            prompt_fallback = f"""
Role: 행정 법률 전문가
Task: 아래 상황에 적용될 법령과 조항을 찾아 설명하시오.
상황: "{situation}"

* 경고: 현재 외부 법령 API 연결이 원활하지 않습니다.
반드시 상단에 [AI 추론 결과]임을 명시하고 환각 가능성을 경고하시오.
"""
            ai_fallback_text = self.llm.generate_text(prompt_fallback).strip()
            
            return f"""⚠️ **[시스템 경고: API 조회 실패]**
(국가법령정보센터 연결 실패로 AI 지식 기반 답변입니다. **환각 가능성** 있으니 법제처 확인 필수)

--------------------------------------------------
{ai_fallback_text}"""
        
        return final_report
    
    def extract_law_keywords(self, situation: str, analysis: dict) -> List[str]:
        """
        Lawbot 검색용 키워드 추출
        
        Args:
            situation: 민원 상황 설명
            analysis: CaseAnalyzer.analyze() 결과
            
        Returns:
            키워드 리스트
        """
        import json as _json
        
        prompt = f"""
상황: "{situation[:100]}"
분석: {_json.dumps(analysis, ensure_ascii=False)}
국가법령정보센터 Lawbot 검색창에 넣을 핵심 키워드 3~7개를 JSON 배열로만 출력.
예: ["무단방치","자동차관리법","공시송달","직권말소"]
"""
        kws = self.llm.generate_json(prompt) or []
        if not isinstance(kws, list):
            kws = []
        return [str(x).strip() for x in kws if str(x).strip()][:10]
