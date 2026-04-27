"""
환각 탐지 기능 테스트 스크립트
"""
import sys
sys.path.insert(0, r'c:\Users\Mr Kim\Desktop\chungsim')

from hallucination_detection import (
    detect_hallucination,
    _detect_by_patterns,
    get_text_hash,
    detect_hallucination_cached
)

# 테스트 입력 (의도적으로 여러 환각 패턴 포함)
test_text = """
2024년 2월 31일에 민원을 제출합니다. 
주민등록법 제999조에 따라 처리해주시기 바랍니다. 
통계에 따르면 승인율은 98.7654%입니다.
"""

print("=" * 60)
print("환각 탐지 기능 테스트")
print("=" * 60)
print(f"\n테스트 입력:\n{test_text}")
print("\n" + "=" * 60)

# 1. 패턴 기반 탐지 테스트
print("\n[1] 패턴 기반 탐지 테스트")
print("-" * 60)
pattern_issues = _detect_by_patterns(test_text)
print(f"탐지된 문제: {len(pattern_issues)}건\n")

for i, issue in enumerate(pattern_issues, 1):
    print(f"{i}. {issue.get('text', 'N/A')}")
    print(f"   이유: {issue.get('reason', 'N/A')}")
    print(f"   신뢰도: {issue.get('confidence', 0):.2f}")
    print(f"   카테고리: {issue.get('category', 'N/A')}")
    print(f"   탐지 방법: {issue.get('detection_method', 'N/A')}")
    print(f"   적용 규칙: {issue.get('rule_applied', 'N/A')}")
    print()

# 2. 전체 탐지 함수 테스트 (LLM 없이)
print("\n[2] 전체 탐지 함수 테스트 (LLM 제외)")
print("-" * 60)

class MockLLMService:
    """LLM 서비스 모의 객체"""
    model_name = "mock-model"
    
    def generate_text(self, prompt, temperature=0.2):
        # LLM 호출 대신 빈 결과 반환
        return '{"issues": []}'

mock_llm = MockLLMService()
text_hash = get_text_hash(test_text)

try:
    result = detect_hallucination_cached(
        text_hash,
        test_text,
        {},
        mock_llm
    )
    
    print(f"위험도: {result.get('risk_level', 'N/A')}")
    print(f"전체 점수: {result.get('overall_score', 0):.2f}")
    print(f"총 탐지 건수: {result.get('total_issues_found', 0)}")
    print(f"캐시 사용: {result.get('cached', False)}")
    
    # verification_log 확인
    v_log = result.get('verification_log', {})
    if v_log:
        print("\n검증 로그:")
        print(f"  - 패턴 검증 건수: {v_log.get('pattern_issues_count', 0)}")
        print(f"  - LLM 상태: {v_log.get('llm_status', 'N/A')}")
        print(f"  - LLM 검증 건수: {v_log.get('llm_issues_count', 0)}")
        print(f"  - 사용 모델: {v_log.get('llm_model', 'N/A')}")
        
        steps = v_log.get('steps', [])
        if steps:
            print(f"\n  검증 단계:")
            for step in steps:
                print(f"    - {step.get('name')}: {step.get('status')} ({step.get('elapsed', 0):.2f}초)")
    
    # 의심 구간 상세 정보
    suspicious = result.get('suspicious_parts', [])
    print(f"\n의심 구간 상세 ({len(suspicious)}건):")
    for i, part in enumerate(suspicious, 1):
        print(f"\n  {i}. {part.get('text', 'N/A')}")
        print(f"     방법: {part.get('detection_method', 'N/A')}")
        print(f"     규칙: {part.get('rule_applied', 'N/A')[:50]}...")
        
    print("\n" + "=" * 60)
    print("✅ 테스트 완료!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()
