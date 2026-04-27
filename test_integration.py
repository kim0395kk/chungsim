
import sys
import os
import shutil

# App 경로 추가
sys.path.append(os.getcwd())

def test_rag_integration():
    print("Testing Civil Engineering RAG Integration...")
    
    # 1. Setup Dummy Manual
    test_docs_dir = "test_llm_docs"
    if not os.path.exists(test_docs_dir):
        os.makedirs(test_docs_dir)
        
    manual_content = """# 겨울철 콘크리트 양생 가이드
    
1. 목적
겨울철 한중콘크리트 시공 시 동결 피해를 방지하기 위함.

2. 양생 온도
- 초기 양생: 5℃ 이상 유지
- 보온 양생: 압축강도 5MPa 얻을 때까지
"""
    manual_path = os.path.join(test_docs_dir, "winter_concrete_guide.md")
    with open(manual_path, "w", encoding="utf-8") as f:
        f.write(manual_content)
        
    # Remove cached data to force re-parse
    if os.path.exists("data/test_complexes.json"):
        os.remove("data/test_complexes.json")
    
    try:
        from civil_engineering.rag_system import load_rag_system, CivilEngineeringRAG
        from civil_engineering.data_parser import parse_all_md_files
        
        # Manually parse test docs to bypass glob logic in load_rag_system if needed, 
        # or we can mock the data path.
        # Let's use the parser directly for test stability.
        
        print("(+) Parsing test documents...")
        complexes_data = parse_all_md_files([manual_path])
        
        # Initialize RAG with parsed data
        print("(+) Initializing RAG...")
        rag = CivilEngineeringRAG(complexes_data, vector_db_path="data/test_vector_db")
        
        print("(+) RAG System loaded successfully.")
        
        # Test 1: Simple Q&A
        query = "양생 온도는 몇 도?"
        print(f"Testing Query: {query}")
        
        # Mock LLM Service
        class MockLLM:
            def generate_text(self, prompt, **kwargs):
                return "Mock Answer: 5도 이상 유지해야 합니다."
        
        response = rag.answer_question(query, MockLLM())
        print(f"Response: {response['answer']}")
        print(f"Sources: {response['sources']}")
        
        # Check Source Citation
        # Expected: "겨울철 콘크리트 양생 가이드 (매뉴얼)"
        expected_source_part = "겨울철 콘크리트 양생 가이드"
        expected_type_part = "(매뉴얼)"
        
        source_found = False
        for src in response['sources']:
            if expected_source_part in src and expected_type_part in src:
                source_found = True
                break
        
        if source_found:
            print("(+) Source Citation Logic Verified: Found '매뉴얼' tag.")
        else:
            print(f"(!) Source Citation Check Failed. Got: {response['sources']}")
            return False
            
        print("(+) Integration Test Passed")
        return True
            
    except ImportError as e:
        print(f"(!) Import Error: {e}")
        return False
    except Exception as e:
        print(f"(!) Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if os.path.exists(test_docs_dir):
            shutil.rmtree(test_docs_dir)
        if os.path.exists("data/test_complexes.json"):
            os.remove("data/test_complexes.json")
        if os.path.exists("data/test_vector_db"):
            shutil.rmtree("data/test_vector_db")

if __name__ == "__main__":
    if test_rag_integration():
        print("All Integration Tests Passed!")
        sys.exit(0)
    else:
        print("Integration Tests Failed!")
        sys.exit(1)
