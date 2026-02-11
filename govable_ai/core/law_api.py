# -*- coding: utf-8 -*-
"""
Govable AI - 국가법령정보센터 API 클라이언트

UI 의존성 없음 (streamlit import 금지)
"""
import logging
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Optional requests import
try:
    import requests
except ImportError:
    requests = None


class LawOfficialService:
    """
    국가법령정보센터(law.go.kr) 공식 API 연동
    
    의존성 주입으로 API ID를 전달받습니다.
    UI(streamlit) 의존성이 없어 다른 환경에서도 사용 가능합니다.
    """
    
    SERVICE_URL = "https://www.law.go.kr/DRF/lawService.do"
    
    def __init__(self, api_id: Optional[str] = None):
        """
        Args:
            api_id: 국가법령정보센터 API ID (OC 파라미터)
        """
        self.api_id = api_id
    
    def is_available(self) -> bool:
        """API 사용 가능 여부"""
        return bool(self.api_id) and requests is not None
    
    def _make_current_link(self, mst_id: str) -> str:
        """현행 법령 링크 생성"""
        if not mst_id:
            return ""
        return f"https://www.law.go.kr/법령/{mst_id}"
    
    def ai_search(self, query: str, top_k: int = 6) -> List[dict]:
        """
        법령 검색
        
        Args:
            query: 검색어
            top_k: 최대 결과 수
            
        Returns:
            검색 결과 리스트 [{"law_name": str, "mst_id": str, "link": str}, ...]
        """
        if not self.is_available():
            logger.warning("Law API not available")
            return []
        
        try:
            params = {
                "OC": self.api_id,
                "target": "law",
                "type": "XML",
                "query": query,
                "display": top_k,
            }
            resp = requests.get(self.SERVICE_URL, params=params, timeout=10)
            resp.raise_for_status()
            
            root = ET.fromstring(resp.text)
            results = []
            
            for law in root.findall(".//law"):
                name = law.findtext("법령명한글", "")
                mst_id = law.findtext("법령일련번호", "") or law.findtext("법령MST", "")
                if name:
                    results.append({
                        "law_name": name,
                        "mst_id": mst_id,
                        "link": self._make_current_link(urllib.parse.quote(name)),
                    })
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Law API search error: {e}")
            return []
    
    def get_law_text(
        self,
        law_name: str,
        article_num: Optional[int] = None,
        return_link: bool = False,
    ) -> Union[str, Tuple[str, str]]:
        """
        법령 조문 텍스트 조회
        
        Args:
            law_name: 법령명
            article_num: 조문 번호 (None이면 전체)
            return_link: True면 (text, link) 튜플 반환
            
        Returns:
            조문 텍스트 또는 (텍스트, 링크) 튜플
        """
        if not self.is_available():
            msg = "[API 오류] LAW_API_ID가 설정되지 않았습니다."
            return (msg, "") if return_link else msg
        
        try:
            # 1. 법령 검색으로 MST ID 획득
            params = {
                "OC": self.api_id,
                "target": "law",
                "type": "XML",
                "query": law_name,
                "display": 5,
            }
            resp = requests.get(self.SERVICE_URL, params=params, timeout=10)
            root = ET.fromstring(resp.text)
            
            mst_id = None
            found_name = None
            for law in root.findall(".//law"):
                name = law.findtext("법령명한글", "")
                if law_name in name or name in law_name:
                    mst_id = law.findtext("법령일련번호", "") or law.findtext("법령MST", "")
                    found_name = name
                    break
            
            if not mst_id:
                msg = f"검색 결과가 없습니다: {law_name}"
                return (msg, "") if return_link else msg
            
            current_link = self._make_current_link(urllib.parse.quote(found_name or law_name))
            
            # 2. 상세 조문 조회
            detail_params = {
                "OC": self.api_id,
                "target": "law",
                "type": "XML",
                "MST": mst_id,
            }
            detail_resp = requests.get(self.SERVICE_URL, params=detail_params, timeout=15)
            detail_root = ET.fromstring(detail_resp.text)
            
            # 조문 파싱
            articles = detail_root.findall(".//조문")
            if not articles:
                articles = detail_root.findall(".//Article")
            
            lines = []
            for art in articles:
                art_no = art.findtext("조문번호", "") or art.findtext("Number", "")
                art_title = art.findtext("조문제목", "") or art.findtext("Title", "")
                art_content = art.findtext("조문내용", "") or art.findtext("Content", "")
                
                # 특정 조문만 필터링
                if article_num:
                    try:
                        if int(art_no.replace("조", "").strip()) != article_num:
                            continue
                    except (ValueError, AttributeError):
                        pass
                
                if art_no or art_content:
                    line = f"**제{art_no}조** {art_title}\n{art_content}" if art_title else f"**제{art_no}조**\n{art_content}"
                    lines.append(line.strip())
            
            if not lines:
                msg = f"조문을 찾지 못했습니다: {law_name}"
                return (msg, current_link) if return_link else msg
            
            text = "\n\n".join(lines[:5])  # 최대 5개 조문
            return (text, current_link) if return_link else text
            
        except Exception as e:
            msg = f"법령 조회 오류: {e}"
            logger.error(msg)
            return (msg, "") if return_link else msg
