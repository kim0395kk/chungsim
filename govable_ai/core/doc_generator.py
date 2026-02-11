# -*- coding: utf-8 -*-
"""
Govable AI - HWPX 문서 생성기

HWPX 템플릿 엔진 방식으로 공문서/보고서 자동 생성
- Linux/Cloud 서버 환경 지원 (Win32 API 불필요)
- XML(ZIP) 구조 기반 데이터 주입
- Jinja2 템플릿 렌더링

UI 의존성 없음 (streamlit import 금지)
"""
import logging
import os
import shutil
import tempfile
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape as xml_escape

logger = logging.getLogger(__name__)

# Jinja2 optional import
try:
    from jinja2 import Template, Environment
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    Template = None
    Environment = None


class HWPXGenerator:
    """
    HWPX 템플릿 기반 문서 생성기
    
    HWPX는 ZIP 압축된 XML 묶음입니다.
    템플릿 파일 내의 플레이스홀더를 AI 생성 데이터로 치환합니다.
    
    사용법:
        ```python
        generator = HWPXGenerator("template_official.hwpx")
        output_path = generator.generate({
            "DOC_TITLE": "민원 처리 회신",
            "DOC_BODY": "귀하의 민원에 대하여...",
        })
        ```
    """
    
    # HWPX 내부 본문 XML 경로 (한글 2014 이상)
    CONTENT_XML_PATHS = [
        "Contents/section0.xml",
        "Contents/Section0.xml", 
        "section0.xml",
    ]
    
    def __init__(self, template_path: str):
        """
        Args:
            template_path: HWPX 템플릿 파일 경로
        """
        self.template_path = template_path
        self._temp_dir: Optional[str] = None
    
    def _extract_template(self) -> str:
        """HWPX 압축 해제"""
        self._temp_dir = tempfile.mkdtemp(prefix="hwpx_")
        
        try:
            with zipfile.ZipFile(self.template_path, 'r') as zip_ref:
                zip_ref.extractall(self._temp_dir)
            return self._temp_dir
        except Exception as e:
            logger.error(f"HWPX 압축 해제 실패: {e}")
            raise
    
    def _find_content_xml(self) -> Optional[str]:
        """본문 XML 파일 경로 찾기"""
        for rel_path in self.CONTENT_XML_PATHS:
            full_path = os.path.join(self._temp_dir, rel_path)
            if os.path.exists(full_path):
                return full_path
        
        # Contents 폴더 내 모든 xml 파일 검색
        contents_dir = os.path.join(self._temp_dir, "Contents")
        if os.path.exists(contents_dir):
            for fname in os.listdir(contents_dir):
                if fname.endswith(".xml") and "section" in fname.lower():
                    return os.path.join(contents_dir, fname)
        
        return None
    
    def _render_content(self, context: Dict[str, Any]) -> None:
        """contents.xml에 데이터 주입"""
        content_xml_path = self._find_content_xml()
        
        if not content_xml_path:
            logger.warning("본문 XML 파일을 찾을 수 없습니다. 단순 치환 시도...")
            return
        
        with open(content_xml_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        
        # Jinja2 사용 가능하면 템플릿 렌더링
        if JINJA2_AVAILABLE and Template:
            try:
                # XML 안전 이스케이프 적용
                safe_context = {}
                for key, value in context.items():
                    if isinstance(value, str):
                        safe_context[key] = xml_escape(value)
                    elif isinstance(value, list):
                        safe_context[key] = [xml_escape(str(v)) if isinstance(v, str) else v for v in value]
                    else:
                        safe_context[key] = value
                
                template = Template(xml_content)
                rendered_xml = template.render(**safe_context)
                xml_content = rendered_xml
            except Exception as e:
                logger.warning(f"Jinja2 렌더링 실패, 단순 치환 모드: {e}")
        
        # 단순 플레이스홀더 치환 ({{VAR}} 형태)
        for key, value in context.items():
            placeholder = "{{" + key + "}}"
            if isinstance(value, str):
                xml_content = xml_content.replace(placeholder, xml_escape(value))
            elif isinstance(value, list):
                # 리스트는 줄바꿈으로 연결
                list_str = "\n".join(str(v) for v in value)
                xml_content = xml_content.replace(placeholder, xml_escape(list_str))
        
        with open(content_xml_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)
    
    def _compress_package(self, output_path: str) -> str:
        """다시 HWPX로 압축"""
        # ZIP으로 압축
        zip_path = output_path.replace('.hwpx', '.zip')
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self._temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self._temp_dir)
                    zipf.write(file_path, arcname)
        
        # .zip → .hwpx 확장자 변경
        if zip_path != output_path:
            shutil.move(zip_path, output_path)
        
        return output_path
    
    def _cleanup(self) -> None:
        """임시 디렉토리 정리"""
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
            except Exception as e:
                logger.warning(f"임시 디렉토리 정리 실패: {e}")
            self._temp_dir = None
    
    def generate(self, context: Dict[str, Any], output_path: str) -> str:
        """
        HWPX 문서 생성
        
        Args:
            context: 템플릿 치환 데이터 딕셔너리
            output_path: 출력 파일 경로
            
        Returns:
            생성된 HWPX 파일 경로
        """
        try:
            self._extract_template()
            self._render_content(context)
            result = self._compress_package(output_path)
            return result
        finally:
            self._cleanup()


class OfficialDocumentGenerator:
    """
    공문서 (Official Document) 생성기
    
    대외 발송용 공문서 양식 생성.
    형식의 엄격함이 중요: 결재 라인, 관인 생략 등.
    """
    
    def __init__(self, template_path: Optional[str] = None):
        """
        Args:
            template_path: 공문서 템플릿 경로 (None이면 기본 경로)
        """
        self.template_path = template_path or self._get_default_template_path()
    
    def _get_default_template_path(self) -> str:
        """기본 템플릿 경로"""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        return os.path.join(base_dir, "assets", "templates", "template_official.hwpx")
    
    def _format_body_paragraphs(self, paragraphs: List[str]) -> str:
        """
        본문 문단을 HWP 형식으로 변환
        
        개행 문자를 HWP 문단 태그로 변환하거나,
        템플릿에 맞는 형식으로 조정
        """
        if not paragraphs:
            return ""
        
        # 각 문단을 줄바꿈으로 연결 (템플릿에서 처리)
        formatted_lines = []
        for para in paragraphs:
            if para.strip():
                # 마크다운 bold (**text**) 유지
                formatted_lines.append(para.strip())
        
        return "\n\n".join(formatted_lines)
    
    def generate(
        self,
        doc_data: Dict[str, Any],
        meta_data: Dict[str, Any],
        output_dir: Optional[str] = None,
    ) -> str:
        """
        공문서 생성
        
        Args:
            doc_data: drafter 에이전트 출력 (title, receiver, body_paragraphs, department_head)
            meta_data: 메타데이터 (doc_num, today_str)
            output_dir: 출력 디렉토리 (None이면 임시 디렉토리)
            
        Returns:
            생성된 HWPX 파일 경로
        """
        # 출력 경로 설정
        if output_dir is None:
            output_dir = tempfile.gettempdir()
        
        doc_num = meta_data.get("doc_num", "")
        safe_doc_num = doc_num.replace("/", "-").replace("\\", "-").replace(":", "-")
        output_path = os.path.join(output_dir, f"공문_{safe_doc_num}.hwpx")
        
        # 템플릿 데이터 구성
        context = {
            "DOC_TITLE": doc_data.get("title", "민원 처리 결과 회신"),
            "DOC_RECEIVER": doc_data.get("receiver", "수신자 참조"),
            "DOC_BODY": self._format_body_paragraphs(doc_data.get("body_paragraphs", [])),
            "DOC_NUM": doc_num,
            "DOC_DATE": meta_data.get("today_str", datetime.now().strftime("%Y. %m. %d.")),
            "DEPARTMENT_HEAD": doc_data.get("department_head", "OOO과장"),
            "ATTACHMENTS": doc_data.get("attachments", ""),
        }
        
        # 템플릿 파일 존재 확인
        if not os.path.exists(self.template_path):
            logger.warning(f"템플릿 파일 없음: {self.template_path}")
            # 템플릿 없으면 빈 HWPX 생성 불가 - 에러 대신 텍스트 파일 생성
            return self._generate_fallback_txt(context, output_path.replace(".hwpx", ".txt"))
        
        generator = HWPXGenerator(self.template_path)
        return generator.generate(context, output_path)
    
    def _generate_fallback_txt(self, context: Dict[str, Any], output_path: str) -> str:
        """템플릿 없을 때 텍스트 파일로 폴백"""
        content = f"""
=====================================
{context.get('DOC_TITLE', '공문서')}
=====================================

문서번호: {context.get('DOC_NUM', '')}
시행일자: {context.get('DOC_DATE', '')}
수신: {context.get('DOC_RECEIVER', '')}

-------------------------------------

{context.get('DOC_BODY', '')}

-------------------------------------

{context.get('DEPARTMENT_HEAD', '')}

※ 이 파일은 HWPX 템플릿이 없어 텍스트로 생성되었습니다.
   assets/templates/template_official.hwpx 파일을 추가해주세요.
"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return output_path


class ReportDocumentGenerator:
    """
    보고서/계획서 (Report/Plan) 생성기
    
    내부 결재용 기안문, 결과 보고서 양식 생성.
    개조식 표현(·, □, ○)이 핵심.
    """
    
    def __init__(self, template_path: Optional[str] = None):
        """
        Args:
            template_path: 보고서 템플릿 경로 (None이면 기본 경로)
        """
        self.template_path = template_path or self._get_default_template_path()
    
    def _get_default_template_path(self) -> str:
        """기본 템플릿 경로"""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        return os.path.join(base_dir, "assets", "templates", "template_report.hwpx")
    
    def _format_bullet_list(self, items: List[str], bullet: str = "□") -> str:
        """개조식 리스트 포맷"""
        if not items:
            return ""
        return "\n".join(f"  {bullet} {item}" for item in items)
    
    def _format_timeline_steps(self, timeline: List[Dict]) -> str:
        """절차 타임라인을 개조식으로 포맷"""
        if not timeline:
            return ""
        
        lines = []
        for step in timeline:
            step_num = step.get("step", "")
            name = step.get("name", "")
            goal = step.get("goal", "")
            actions = step.get("actions", [])
            
            lines.append(f"○ {step_num}단계: {name}")
            lines.append(f"   - 목표: {goal}")
            for action in actions[:3]:  # 최대 3개 행동
                lines.append(f"   - 행동: {action}")
            lines.append("")
        
        return "\n".join(lines)
    
    def generate(
        self,
        analysis_data: Dict[str, Any],
        procedure_data: Dict[str, Any],
        strategy_text: str = "",
        legal_text: str = "",
        output_dir: Optional[str] = None,
    ) -> str:
        """
        보고서/계획서 생성
        
        Args:
            analysis_data: analyzer 에이전트 출력 (case_type, core_issue, risk_flags, etc.)
            procedure_data: planner 에이전트 출력 (timeline, checklist, templates)
            strategy_text: strategist 에이전트 출력 (처리 전략)
            legal_text: researcher 에이전트 출력 (법령 근거)
            output_dir: 출력 디렉토리 (None이면 임시 디렉토리)
            
        Returns:
            생성된 HWPX 파일 경로
        """
        # 출력 경로 설정
        if output_dir is None:
            output_dir = tempfile.gettempdir()
        
        case_type = analysis_data.get("case_type", "민원")
        safe_case_type = case_type.replace("/", "-").replace("\\", "-")
        output_path = os.path.join(output_dir, f"보고서_{safe_case_type}_{datetime.now().strftime('%Y%m%d')}.hwpx")
        
        # 템플릿 데이터 구성 (개조식 변환)
        context = {
            "REPORT_TITLE": f"{case_type} 민원 처리 계획",
            "REPORT_DATE": datetime.now().strftime("%Y. %m. %d."),
            "AUTHOR_NAME": "담당자",
            
            # 1. 현황 및 문제점
            "ANALYSIS_CASE_TYPE": case_type,
            "ANALYSIS_SUMMARY": self._format_bullet_list(
                analysis_data.get("core_issue", []), "□"
            ),
            "RISK_FLAGS": self._format_bullet_list(
                analysis_data.get("risk_flags", []), "▶"
            ),
            
            # 2. 법적 검토 결과
            "LEGAL_REVIEW": legal_text[:2000] if legal_text else "(법령 검토 결과 없음)",
            
            # 3. 처리 방안 및 계획
            "STRATEGY_SUMMARY": strategy_text[:1500] if strategy_text else "",
            "STRATEGY_STEPS": self._format_timeline_steps(
                procedure_data.get("timeline", [])
            ),
            
            # 4. 체크리스트
            "CHECKLIST": self._format_bullet_list(
                procedure_data.get("checklist", []), "☐"
            ),
            
            # 5. 필요 서식
            "TEMPLATES": self._format_bullet_list(
                procedure_data.get("templates", []), "·"
            ),
            
            # 6. 향후 계획
            "NEXT_ACTION": self._format_bullet_list(
                analysis_data.get("recommended_next_action", []), "→"
            ),
        }
        
        # 템플릿 파일 존재 확인
        if not os.path.exists(self.template_path):
            logger.warning(f"템플릿 파일 없음: {self.template_path}")
            return self._generate_fallback_txt(context, output_path.replace(".hwpx", ".txt"))
        
        generator = HWPXGenerator(self.template_path)
        return generator.generate(context, output_path)
    
    def _generate_fallback_txt(self, context: Dict[str, Any], output_path: str) -> str:
        """템플릿 없을 때 텍스트 파일로 폴백"""
        content = f"""
=====================================
{context.get('REPORT_TITLE', '보고서')}
=====================================
작성일: {context.get('REPORT_DATE', '')}
작성자: {context.get('AUTHOR_NAME', '')}

-------------------------------------
1. 현황 및 문제점
-------------------------------------
케이스 유형: {context.get('ANALYSIS_CASE_TYPE', '')}

【핵심 쟁점】
{context.get('ANALYSIS_SUMMARY', '')}

【리스크 요소】
{context.get('RISK_FLAGS', '')}

-------------------------------------
2. 법적 검토 결과
-------------------------------------
{context.get('LEGAL_REVIEW', '')}

-------------------------------------
3. 처리 방안 및 계획
-------------------------------------
{context.get('STRATEGY_SUMMARY', '')}

【단계별 조치 계획】
{context.get('STRATEGY_STEPS', '')}

-------------------------------------
4. 체크리스트
-------------------------------------
{context.get('CHECKLIST', '')}

-------------------------------------
5. 필요 서식/문서
-------------------------------------
{context.get('TEMPLATES', '')}

-------------------------------------
6. 향후 계획
-------------------------------------
{context.get('NEXT_ACTION', '')}

=====================================

※ 이 파일은 HWPX 템플릿이 없어 텍스트로 생성되었습니다.
   assets/templates/template_report.hwpx 파일을 추가해주세요.
"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return output_path


# =========================================================
# 편의 함수
# =========================================================

def generate_official_doc(
    doc_data: Dict[str, Any],
    meta_data: Dict[str, Any],
    template_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> str:
    """
    공문서 생성 (헬퍼 함수)
    
    Args:
        doc_data: drafter 에이전트 출력
        meta_data: 메타데이터
        template_path: 커스텀 템플릿 경로
        output_dir: 출력 디렉토리
        
    Returns:
        생성된 파일 경로
    """
    generator = OfficialDocumentGenerator(template_path)
    return generator.generate(doc_data, meta_data, output_dir)


def generate_report_doc(
    analysis_data: Dict[str, Any],
    procedure_data: Dict[str, Any],
    strategy_text: str = "",
    legal_text: str = "",
    template_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> str:
    """
    보고서/계획서 생성 (헬퍼 함수)
    
    Args:
        analysis_data: analyzer 에이전트 출력
        procedure_data: planner 에이전트 출력
        strategy_text: strategist 출력
        legal_text: researcher 출력
        template_path: 커스텀 템플릿 경로
        output_dir: 출력 디렉토리
        
    Returns:
        생성된 파일 경로
    """
    generator = ReportDocumentGenerator(template_path)
    return generator.generate(
        analysis_data, procedure_data, strategy_text, legal_text, output_dir
    )
