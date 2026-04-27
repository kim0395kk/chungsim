# civil_engineering/data_parser.py
"""
산업단지 교육자료 MD 파일 파싱 및 구조화
"""

import os
import re
import json
from typing import Dict, List, Optional
from datetime import datetime



def parse_manual_md(md_content: str, filename: str = "") -> Dict:
    """
    일반 업무 매뉴얼/지침 파싱
    """
    # 1. 제목 추출
    title = ""
    
    # (1) 첫 번째 헤더(# Title)를 찾되, "Extracted from" 같은 자동 생성 메타데이터는 무시
    header_matches = re.finditer(r'^#\s+(.+)$', md_content, re.MULTILINE)
    for match in header_matches:
        candidate = match.group(1).strip()
        # "Extracted from" 으로 시작하면 무시 (대소문자 구분 없이)
        if not candidate.lower().startswith("extracted from"):
            title = candidate
            break
            
    if not title:
        try:
            # (2) 헤더가 없으면 파일명 사용 (확장자 및 불필요한 경로 제거)
            # 예: "c:\...\8.9.도로보수팀-안정수 - 제목.pptx.md" -> "8.9.도로보수팀-안정수 - 제목"
            base_name = os.path.basename(filename)
            # .md, .pptx, .hwp 등 확장자 연쇄 제거
            current_title = base_name
            while os.path.splitext(current_title)[1] in ['.md', '.txt', '.hwp', '.hwpx', '.pdf', '.pptx', '.xls', '.xlsx', '.doc', '.docx']:
                current_title = os.path.splitext(current_title)[0]
            title = current_title
                
            # (3) 특수문자 및 불필요한 접두어 정리
            # 예: "20.[돌덩이] 개발업무..." -> "개발업무 매뉴얼" (선택적)
            # 지금은 파일명 그대로 쓰는게 식별에 도움됨
            # name_clean = re.sub(r'^[\d\-\.]+\s*', '', title) # 숫자+점+공백 패턴 제거 (예: "6-7-1. ", "1. ")
            # title = name_clean.replace('_', ' ').strip() # 언더바(_)를 공백으로 변환
        except Exception:
            title = "알 수 없음 (파일명 오류)"
        
    return {
        "name": title,
        "type": "manual",
        "category": "업무지침",
        "raw_text": md_content,
        "filename": filename
    }

def parse_industrial_complex_md(md_content: str) -> Optional[Dict]:
    """
    산업단지 데이터 파싱 (실패 시 None 반환)
    """
    # 산업단지명 추출 (엄격한 패턴)
    # 반드시 "산업단지", "산단", "일반산업단지" 등 산업단지 맥락 키워드가 함께 있어야 함
    has_complex_context = bool(re.search(r'(산업단지|산단|일반산단|국가산단|조성사업|조성계획|실시계획)', md_content))
    if not has_complex_context:
        return None  # 산업단지 관련 문서가 아님
    
    name_match = re.search(r'(동충주산업단지|바이오헬스|드림파크|비즈코어시티|법현|엄정|금가)', md_content)
    if not name_match:
        return None # 산업단지 데이터가 아님
        
    name = name_match.group(1)
    
    # ... (기존 추출 로직 유지) ...
    
    # 위치 추출
    location_match = re.search(r'위\s*치\s*(.+?)(?:기\s*간|$)', md_content)
    location = location_match.group(1).strip() if location_match else ""
    
    # 기간 추출
    period_match = re.search(r'기\s*간\s*(\d{4})년?\s*~\s*(\d{4})년?', md_content)
    if period_match:
        period = {
            "start": period_match.group(1),
            "end": period_match.group(2),
            "duration": int(period_match.group(2)) - int(period_match.group(1))
        }
    else:
        period = {"start": "", "end": "", "duration": 0}
    
    # 규모 추출 (면적, 예산)
    area_match = re.search(r'규\s*모\s*([\d,]+)㎡\s*/\s*([\d,]+)억원', md_content)
    if area_match:
        area = int(area_match.group(1).replace(',', ''))
        budget = int(area_match.group(2).replace(',', '')) * 100000000  # 억원 → 원
    else:
        area = 0
        budget = 0
    
    # 시행자 추출
    developer_match = re.search(r'시\s*행\s*자\s*(.+?)(?:유치업종|$)', md_content)
    developer = developer_match.group(1).strip() if developer_match else ""
    
    # 유치업종 추출
    industries_match = re.search(r'유치업종\s*(.+?)(?:추진|$)', md_content)
    if industries_match:
        industries_text = industries_match.group(1)
        industries = [ind.strip() for ind in re.split(r'[,·]', industries_text) if ind.strip()]
    else:
        industries = []
    
    # 추진 현황/계획 추출 (마일스톤)
    milestones = []
    milestone_pattern = r'▶\s*(.+?)\s*:\s*(\d{4})\.?\s*(\d{1,2})?\.?'
    for match in re.finditer(milestone_pattern, md_content):
        event = match.group(1).strip()
        year = match.group(2)
        month = match.group(3) if match.group(3) else "01"
        
        milestones.append({
            "event": event,
            "date": f"{year}-{month.zfill(2)}",
            "year": int(year),
            "month": int(month) if month else 1
        })
    
    # 사업 상태 판단
    if "준공" in md_content and any("2023" in m["date"] or "2024" in m["date"] for m in milestones if "준공" in m["event"]):
        status = "조성완료"
    elif "착공" in md_content:
        status = "조성중"
    else:
        status = "조성계획"
    
    # 사업 유형 판단
    if "공영개발" in md_content or "충주시" in developer or "한국토지주택공사" in developer:
        dev_type = "공영개발"
    elif "민관합동" in md_content:
        dev_type = "민관합동개발"
    else:
        dev_type = "민간개발"
    
    return {
        "name": name,
        "type": "complex", # 타입 명시
        "location": location,
        "period": period,
        "area_sqm": area,
        "budget_krw": budget,
        "developer": developer,
        "industries": industries,
        "milestones": milestones,
        "status": status,
        "development_type": dev_type,
        "raw_text": md_content
    }


def parse_all_md_files(md_files: List[str]) -> List[Dict]:
    """
    여러 MD 파일을 한번에 파싱
    """
    complexes = []
    
    for filepath in md_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 1. 산업단지 파싱 시도
            data = parse_industrial_complex_md(content)
            
            # 2. 실패 시 일반 매뉴얼로 파싱
            if not data:
                data = parse_manual_md(content, filepath)
            
            data['source_file'] = filepath
            complexes.append(data)
            
        except Exception as e:
            print(f"파싱 오류 ({filepath}): {e}")
    
    return complexes


def create_search_chunks(data: Dict) -> List[Dict]:
    """
    검색을 위한 텍스트 청크 생성 (Dual-Track)
    """
    chunks = []
    
    # [Case A] 산업단지 데이터
    if data.get('type') == 'complex':
        # 청크 1: 기본 정보
        period_str = f"{data['period']['start']}년 ~ {data['period']['end']}년"
        duration_str = f"({data['period']['duration']}년)" if data['period']['duration'] else ""
        
        basic_info = f"""
산업단지명: {data['name']}
위치: {data['location']}
사업기간: {period_str} {duration_str}
면적: {data['area_sqm']:,}㎡
예산: {data['budget_krw']:,}원 ({data['budget_krw']//100000000}억원)
시행자: {data['developer']}
사업상태: {data['status']}
개발유형: {data['development_type']}
        """.strip()
        
        chunks.append({
            "type": "basic_info",
            "complex_name": data['name'],
            "display_name": data['name'], # 화면 표시용 이름
            "text": basic_info,
            "metadata": {
                "name": data['name'],
                "status": data['status'],
                "type": "complex"
            }
        })
        
        # 청크 2: 유치업종
        if data['industries']:
            industries_info = f"""
{data['name']} 유치업종:
{chr(10).join([f'- {ind}' for ind in data['industries']])}
            """.strip()
            
            chunks.append({
                "type": "industries",
                "complex_name": data['name'],
                "display_name": f"{data['name']} 유치업종",
                "text": industries_info,
                "metadata": {
                    "name": data['name'],
                    "industries": data['industries']
                }
            })
        
        # 청크 3: 추진 일정
        if data['milestones']:
            milestones_info = f"""
{data['name']} 추진 일정:
{chr(10).join([f"- {m['event']}: {m['date']}" for m in data['milestones']])}
            """.strip()
            
            chunks.append({
                "type": "milestones",
                "complex_name": data['name'],
                "display_name": f"{data['name']} 추진일정",
                "text": milestones_info,
                "metadata": {
                    "name": data['name'],
                    "milestones": data['milestones']
                }
            })
            
    # [Case B] 일반 업무 매뉴얼
    else:
        # 섹션(## 헤더) 기반 분할 → 없으면 1000자 단위 분할
        raw = data['raw_text']
        doc_name = data['name']
        
        # 섹션 분할 시도: ## 또는 # 헤더 기준
        sections = re.split(r'\n(?=#{1,3}\s)', raw)
        
        section_chunks = []
        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue
            # 1000자 이하면 그대로, 초과면 1000자 단위로 분할
            if len(sec) <= 1200:
                section_chunks.append(sec)
            else:
                # 문단(\n\n) 기준으로 먼저 나누고, 1000자에 맞춰 합침
                paragraphs = sec.split('\n\n')
                current = ""
                for para in paragraphs:
                    if len(current) + len(para) > 1000 and current:
                        section_chunks.append(current.strip())
                        current = para
                    else:
                        current = current + "\n\n" + para if current else para
                if current.strip():
                    section_chunks.append(current.strip())
        
        # 청크가 없으면 원문 전체를 하나로
        if not section_chunks:
            section_chunks = [raw]
        
        for i, chunk_text in enumerate(section_chunks):
            # 첫 줄에서 섹션 제목 추출
            first_line = chunk_text.split('\n')[0].strip().lstrip('#').strip()
            section_label = first_line[:40] if first_line else f"섹션{i+1}"
            
            chunks.append({
                "type": "manual_section",
                "complex_name": doc_name,
                "display_name": f"{doc_name} > {section_label}",
                "text": chunk_text,
                "metadata": {
                    "name": doc_name,
                    "type": "manual",
                    "section_index": i,
                    "filename": data.get('filename', '')
                }
            })
    
    # 공통: 전체 원문 (검색 보완용)
    chunks.append({
        "type": "full_text",
        "complex_name": data['name'],
        "display_name": f"{data['name']} (원문)",
        "text": data['raw_text'],
        "metadata": {
            "name": data['name']
        }
    })
    
    return chunks
