# Security Incident Response (API Key Exposure)

작성일: 2026-02-15  
상태: 즉시 조치 필요

## 1) 범위
- 현재 세션 터미널 출력/대화 기록에 API 키가 노출된 정황이 있음.
- 원격 저장소 유출 여부는 별도 확인 필요.

## 2) 즉시 대응 (Critical)
1. 키 폐기/재발급
   - `GEMINI_API_KEY`
   - `GROQ_API_KEY`
   - `SUPABASE_KEY` (필요 시 `SUPABASE_ANON_KEY`도 교체)
2. 키 교체 후 서비스 재시작
3. 노출된 기존 키 권한 차단 확인 (대시보드에서 revoked 상태 확인)

## 3) 로컬 정리
1. `.streamlit/secrets.toml`의 기존 키값 교체
2. 실행 로그/임시 파일 정리
   - `*run*.log`
   - 기타 터미널 출력 저장 파일
3. 쉘 히스토리/스크린샷/메모앱 등 2차 저장소 점검

## 4) 저장소 점검
1. 현재 워크트리 스캔 (키 패턴)
2. Git 히스토리 스캔 (키 패턴)
3. 원격 저장소/PR/이슈/CI 로그 점검
4. 노출 커밋 존재 시:
   - 즉시 키 폐기(우선)
   - 히스토리 정리(BFG/git filter-repo) + 강제 푸시
   - 조직 내 공지

## 5) 재발 방지
1. 키 마스킹 출력 원칙 적용
2. `*.log`, `secrets*` git 추적 방지 강화
3. 사전 스캔 도입 (pre-commit / CI secret scan)
4. 최소권한 키 사용, 주기적 로테이션

## 6) 완료 기준 (Definition of Done)
- [ ] 노출된 키 전부 폐기 및 신규 키 적용 완료
- [ ] 로컬/원격 스캔 결과 유출 경로 확인 완료
- [ ] 불필요 로그 삭제 완료
- [ ] 재발 방지 규칙 적용 완료

