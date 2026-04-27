
import os
import sys
import glob

# Add project root to path
sys.path.append(os.getcwd())

from civil_engineering.rag_system import load_rag_system
from civil_engineering.data_parser import parse_all_md_files
import json

def regenerate():
    print("Starting data regeneration from valid docs...")
    
    # 1. Define paths
    docs_dir = r"c:\Users\Mr Kim\Desktop\chungsim\llm_ready_docs"
    output_path = r"c:\Users\Mr Kim\Desktop\chungsim\data\parsed_complexes.json"
    
    # 2. Find all MD files
    md_files = glob.glob(os.path.join(docs_dir, "*.md"))
    print(f"Found {len(md_files)} markdown files.")
    
    if not md_files:
        print("Error: No MD files found!")
        return

    # 3. Parse files
    print("Parsing files...")
    data = parse_all_md_files(md_files)
    
    # 4. Save to JSON
    print(f"Saving {len(data)} documents to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print("Done! Data regenerated successfully.")

if __name__ == "__main__":
    regenerate()
