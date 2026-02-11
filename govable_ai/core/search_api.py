# -*- coding: utf-8 -*-
"""
Govable AI - 네이버 검색 API 클라이언트

UI 의존성 없음 (streamlit import 금지)
"""
import logging
import re
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from govable_ai.core.llm_service import LLMService

logger = logging.getLogger(__name__)

# Optional requests import
try:
    import requests
except ImportError:
    requests = None


class SearchService:
    """
    네이버 검색 API 클라이언트 (뉴스 중심)
    
    의존성 주입으로 API 키와 LLM 서비스를 전달받습니다.
    UI(streamlit) 의존성이 없어 다른 환경에서도 사용 가능합니다.
    """
    
    NEWS_API_URL = "https://openapi.naver.com/v1/search/news.json"
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        llm_service: Optional["LLMService"] = None,
    ):
        """
        Args:
            client_id: 네이버 API 클라이언트 ID
            client_secret: 네이버 API 클라이언트 시크릿
            llm_service: 키워드 추출용 LLM 서비스 (선택)
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.llm = llm_service
    
    def is_available(self) -> bool:
        """API 사용 가능 여부"""
        return bool(self.client_id and self.client_secret) and requests is not None
    
    def _headers(self) -> dict:
        """API 요청 헤더"""
        return {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
        }
    
    def _clean_html(self, s: str) -> str:
        """HTML 태그 제거"""
        if not s:
            return ""
        s = re.sub(r"<b>|</b>", "", s)
        s = re.sub(r"&quot;", '"', s)
        s = re.sub(r"&amp;", "&", s)
        s = re.sub(r"&lt;", "<", s)
        s = re.sub(r"&gt;", ">", s)
        return s.strip()
    
    def _extract_keywords_llm(self, situation: str) -> List[str]:
        """LLM으로 검색 키워드 추출"""
        if not self.llm:
            # LLM 없으면 간단한 토큰화
            words = re.findall(r"[가-힣]+", situation)
            return [w for w in words if len(w) >= 2][:5]
        
        prompt = f"""
상황: "{situation}"
위 상황에서 뉴스 검색에 사용할 핵심 키워드 3~5개를 JSON 배열로만 출력하세요.
예: ["무단방치", "자동차", "과태료"]
"""
        result = self.llm.generate_json(prompt)
        if isinstance(result, list):
            return [str(kw).strip() for kw in result if str(kw).strip()][:5]
        return []
    
    def search_news(self, query: str, top_k: int = 3) -> List[dict]:
        """
        뉴스 검색
        
        Args:
            query: 검색어
            top_k: 최대 결과 수
            
        Returns:
            검색 결과 리스트 [{"title": str, "description": str, "link": str, "pubDate": str}, ...]
        """
        if not self.is_available():
            logger.warning("Search API not available")
            return []
        
        try:
            params = {
                "query": query,
                "display": min(top_k, 10),
                "sort": "sim",  # 관련도순
            }
            resp = requests.get(
                self.NEWS_API_URL,
                headers=self._headers(),
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            
            results = []
            for item in data.get("items", [])[:top_k]:
                results.append({
                    "title": self._clean_html(item.get("title", "")),
                    "description": self._clean_html(item.get("description", "")),
                    "link": item.get("link", ""),
                    "pubDate": item.get("pubDate", ""),
                })
            
            return results
            
        except Exception as e:
            logger.error(f"News search error: {e}")
            return []
    
    def search_precedents(self, situation: str, top_k: int = 3) -> List[dict]:
        """
        상황 기반 판례/뉴스 검색 (키워드 자동 추출)
        
        Args:
            situation: 민원 상황 설명
            top_k: 최대 결과 수
            
        Returns:
            검색 결과 리스트
        """
        keywords = self._extract_keywords_llm(situation)
        if not keywords:
            return []
        
        query = " ".join(keywords)
        return self.search_news(query, top_k)
