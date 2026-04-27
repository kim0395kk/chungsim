
import os
from civil_engineering.data_parser import parse_markdown_file

# Test file path (one that failed in the JSON)
test_file = r"c:\Users\Mr Kim\Desktop\chungsim\llm_ready_docs\0007-7-2도시도로팀-이원국 도로구조시설기준.xls.md"

if os.path.exists(test_file):
    print(f"Testing parsing on: {test_file}")
    result = parse_markdown_file(test_file)
    print("\n--- Result ---")
    print(f"Name: {result.get('name')}")
    print(f"Type: {result.get('type')}")
    print(f"Source: {result.get('source_file')}")
else:
    print(f"File not found: {test_file}")
