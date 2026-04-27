
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from civil_engineering.data_parser import parse_all_md_files, create_search_chunks  # noqa: E402

def test_parser_backtest():
    print("Testing Data Parser Back-test...")
    
    # 1. Create Dummy Manual
    dummy_content = """# 겨울철 콘크리트 양생 가이드
    
    1. 목적
    겨울철 한중콘크리트 시공 시 동결 피해를 방지하기 위함.
    
    2. 양생 온도
    - 초기 양생: 5℃ 이상 유지
    - 보온 양생: 압축강도 5MPa 얻을 때까지
    """
    
    dummy_filename = "winter_concrete_guide.md"
    with open(dummy_filename, "w", encoding="utf-8") as f:
        f.write(dummy_content)
        
    try:
        # 2. Parse
        parsed_data = parse_all_md_files([dummy_filename])
        
        if not parsed_data:
            print("(!) Parsing failed: No data returned")
            return False
            
        data = parsed_data[0]
        print(f"Parsed Name: {data['name']}")
        print(f"Parsed Type: {data.get('type')}")
        
        # 3. Check Name (Should NOT be 'Unknown' or '알 수 없음')
        if data['name'] == "알 수 없음":
            print("(!) Logic Failed: Name is still 'Unknown'")
            return False
            
        if data['name'] != "겨울철 콘크리트 양생 가이드":
            print(f"(!) Logic Failed: Expected '겨울철 콘크리트 양생 가이드', got '{data['name']}'")
            return False
            
        # 4. Check Chunking
        chunks = create_search_chunks(data)
        print(f"Generated Chunks: {len(chunks)}")
        for chunk in chunks:
            print(f"- [{chunk['type']}] {chunk['display_name']}")
            
        print("(+) Parser Back-test Passed")
        return True
        
    except Exception as e:
        print(f"(!) Error during back-test: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists(dummy_filename):
            os.remove(dummy_filename)

if __name__ == "__main__":
    if test_parser_backtest():
        sys.exit(0)
    else:
        sys.exit(1)
