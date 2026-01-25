# HWPX 템플릿 파일 안내
# ========================

## 이 폴더에 배치할 파일:
# 1. template_official.hwpx - 공문서 템플릿
# 2. template_report.hwpx - 보고서/계획서 템플릿

## 템플릿 제작 방법:
# 1. 한글(HWP)에서 완벽한 서식(폰트, 로고, 자간)이 적용된 문서 작성
# 2. 데이터가 들어갈 위치에 플레이스홀더 삽입:
#    - 공문서: {{DOC_TITLE}}, {{DOC_BODY}}, {{DOC_NUM}}, {{DOC_DATE}}, {{DOC_RECEIVER}}, {{DEPARTMENT_HEAD}}
#    - 보고서: {{REPORT_TITLE}}, {{REPORT_DATE}}, {{ANALYSIS_SUMMARY}}, {{STRATEGY_STEPS}}, 등
# 3. .hwpx 형식으로 저장 (한글 2014 이상)

## 플레이스홀더 종류:

### 공문서 (template_official.hwpx)
# {{DOC_TITLE}}        - 문서 제목 (예: "민원 처리 결과 회신")
# {{DOC_RECEIVER}}     - 수신자
# {{DOC_BODY}}         - 본문 내용
# {{DOC_NUM}}          - 문서번호 (예: "행정-2026-001호")
# {{DOC_DATE}}         - 시행일자 (예: "2026. 01. 20.")
# {{DEPARTMENT_HEAD}}  - 발신명의 (예: "OOO과장")
# {{ATTACHMENTS}}      - 붙임 파일 목록

### 보고서 (template_report.hwpx)
# {{REPORT_TITLE}}       - 보고서 제목
# {{REPORT_DATE}}        - 작성일
# {{AUTHOR_NAME}}        - 작성자
# {{ANALYSIS_CASE_TYPE}} - 케이스 유형
# {{ANALYSIS_SUMMARY}}   - 핵심 쟁점 (개조식)
# {{RISK_FLAGS}}         - 리스크 요소
# {{LEGAL_REVIEW}}       - 법적 검토 결과
# {{STRATEGY_SUMMARY}}   - 처리 방안 요약
# {{STRATEGY_STEPS}}     - 단계별 조치 계획
# {{CHECKLIST}}          - 체크리스트
# {{TEMPLATES}}          - 필요 서식
# {{NEXT_ACTION}}        - 향후 계획

## 주의사항:
# - 폰트: 휴먼명조(본문), HHY견고딕(제목) 권장
# - 자간/장평: 표준 정부 공문서 규격 준수
# - 관인생략 이미지: 템플릿에 미리 배치
