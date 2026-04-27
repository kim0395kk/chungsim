# Streamlit 고도화/업그레이드 보고서

작성일: 2026-02-15  
대상: `streamlit_app.py` 실행 안정화 + 스킬 파서 툴 추가

## 1) 요약
- `streamlit run streamlit_app.py` 기준 앱 부팅 가능 상태를 확인했습니다.
- 설정 누락(secrets 없음) 환경에서도 최대한 안전하게 부팅되도록 폴백을 강화했습니다.
- 레포 내 에이전트/스킬 마크다운을 파싱/변환하는 별도 Streamlit 툴을 추가했습니다.

## 2) 고도화/업그레이드 내역

### A. 실행 안정화 (기동/설정 폴백)
대상 파일:
- `streamlit_app.py`
- `govable_ai/config.py`

적용 내용:
1. `streamlit_app.py`의 `get_secret()`에 환경변수(`os.environ`) 폴백 추가  
   - `st.secrets`가 없거나 키가 없을 때도 설정을 읽을 수 있게 개선
2. `govable_ai/config.py`의 Vertex 설정 폴백 강화  
   - `st.secrets["vertex_ai"]`가 없어도 `VERTEX_AI_*` 환경변수로 설정 가능
3. `streamlit_app.py` 내 로컬 LLM 클래스 이름 충돌/덮어쓰기 리스크 완화  
   - 하단 로컬 클래스를 `StreamlitLLMService`로 분리
   - 상단에서 이미 초기화된 `llm_service`가 있으면 재생성하지 않도록 보호
4. 무-secrets 환경에서 일부 초기화 시 예외 처리 보강  
   - `st.secrets` 접근 실패 시 앱 전체 부팅이 실패하지 않도록 처리

효과:
- 키/설정 일부가 비어 있어도 앱이 바로 죽지 않고 “no-keys 안전 모드”로 부팅 가능
- 설정 주입 경로(`secrets.toml` / env)가 명확해짐

### B. 운영 문서/실행 가이드 개선
대상 파일:
- `AGENTS.md`
- `README.md`
- `.env.example` (신규)

적용 내용:
1. `AGENTS.md`에 실행/검증 명령 추가
   - 설치, 앱 실행, 컴파일 체크, 테스트/린트 커맨드 정리
2. `README.md`에 `.env` 기반 로컬 안전 실행 안내 보강
3. `.env.example` 추가
   - LLM/Supabase/Naver/Vertex 관련 placeholder 제공
   - 실제 시크릿 커밋 방지 가이드 포함

### C. 신규 도구 추가 (스킬 파서)
대상 파일:
- `tools/skill_parser_app.py` (신규)

기능:
1. 자동 탐지
   - `.agents/skills/*/SKILL.md`
   - `*/agents` 폴더 내 마크다운
2. 수동 경로 입력(파일/폴더)
3. 메타 파싱
   - `source_path`, `folder_name`, `skill_name`, `description`
   - YAML front matter 없으면 heading/첫 문단 추론
4. 검색 가능한 테이블 + 통계
   - 총 개수, YAML 누락, 설명 누락
5. Import/Convert
   - `.agents/skills/<normalized_name>/SKILL.md`로 저장
   - YAML 없을 경우 자동 생성
6. Export
   - CSV, JSON 다운로드 버튼 제공

안전성:
- 네트워크 호출 없음
- 삭제/파괴적 작업 없음

## 3) 검증 결과

### 컴파일 검증
실행 명령:
```bash
python -m compileall .
```

결과:
- `exit code 0`
- `tools/skill_parser_app.py` 포함 전체 컴파일 성공

### Streamlit 실행 검증
실행 확인:
- `streamlit run streamlit_app.py` 정상 부팅 확인
- `streamlit run tools/skill_parser_app.py --server.port 8510 --server.address 0.0.0.0` 정상 부팅 확인

## 4) 실행 방법

### 메인 앱
```bash
streamlit run streamlit_app.py
```

### 스킬 파서 앱
```bash
streamlit run tools/skill_parser_app.py --server.port 8510 --server.address 0.0.0.0
```

## 5) 잔여 개선 TODO
1. `google-generativeai` deprecation 경고 대응 (`google.genai` 전환)
2. `streamlit_app.py` 대형 단일 파일 모듈 분리 (기능 단위)
3. 테스트 범위 확장 (핵심 워크플로우/폴백 경로)

