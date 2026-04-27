"""
기안문 수정 기능 테스트 스크립트
"""
import sys
sys.path.insert(0, r'c:\Users\Mr Kim\Desktop\chungsim')

from govable_ai.features.document_revision import run_revision_workflow

# Mock LLM Service
class MockLLMService:
    """LLM 서비스 모의 객체"""
    model_name = "gemini-2.5-flash"
    
    def generate_json(self, prompt, preferred_model=None):
        print(f"\n[LLM 호출] 모델: {preferred_model}")
        print(f"[프롬프트 길이]: {len(prompt)} 자")
        
        # 실제 LLM 대신 테스트용 JSON 반환
        return {
            "revised_doc": {
                "title": "2025년 시민참여 예산 설명회 개최 안내",
                "receiver": "각 부서장",
                "body_paragraphs": [
                    "시민참여 예산 설명회를 아래와 같이 개최하오니 참석하여 주시기 바랍니다.",
                    "1. 일시: 2025. 1. 28. 14:00",
                    "2. 장소: 시청 대회의실"
                ],
                "department_head": "기획예산과장"
            },
            "changelog": [
                "일시 수정: '1월 28일' → '2025. 1. 28.' (표기법 준수)",
                "표현 개선: '참석 바랍니다' → '참석하여 주시기 바랍니다' (수요자 중심)"
            ],
            "summary": "공문서 작성 표준에 맞춰 날짜 표기와 표현을 수정했습니다."
        }

# 테스트 입력
test_input = """
[원문]
제 목: 시민참여 예산 설명회 안내
수 신: 각 부서장
발 신: 기획예산과

시민참여 예산 설명회를 1월 28일 14시에 개최하니 참석 바랍니다.

[수정 요청]
날짜 표기를 표준에 맞게 수정해주세요
"""

print("=" * 60)
print("기안문 수정 기능 테스트")
print("=" * 60)
print(f"\n테스트 입력:\n{test_input}")
print("\n" + "=" * 60)

try:
    mock_llm = MockLLMService()
    result = run_revision_workflow(test_input, mock_llm)
    
    print("\n[테스트 결과]")
    print("-" * 60)
    
    if "error" in result:
        print(f"❌ 오류 발생: {result['error']}")
    else:
        print("✅ 정상 작동!")
        print(f"\n수정된 문서:")
        revised = result.get("revised_doc", {})
        print(f"  제목: {revised.get('title')}")
        print(f"  수신: {revised.get('receiver')}")
        print(f"  본문: {len(revised.get('body_paragraphs', []))}개 문단")
        
        print(f"\n변경 로그:")
        for i, log in enumerate(result.get("changelog", []), 1):
            print(f"  {i}. {log}")
        
        print(f"\n요약: {result.get('summary')}")
    
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()
