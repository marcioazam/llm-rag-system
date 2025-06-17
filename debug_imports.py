#!/usr/bin/env python3
"""
Script to debug import issues in the RAG system
"""
import os
import sys
import importlib.util

def check_file_exists(filepath):
    """Check if file exists and print its status"""
    if os.path.exists(filepath):
        print(f"✓ {filepath} exists")
        return True
    else:
        print(f"✗ {filepath} does not exist")
        return False

def check_class_in_file(filepath, class_name):
    """Check if a class exists in a Python file"""
    if not os.path.exists(filepath):
        return False
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            if f"class {class_name}" in content:
                print(f"✓ Class '{class_name}' found in {filepath}")
                return True
            else:
                print(f"✗ Class '{class_name}' not found in {filepath}")
                # Show available classes
                import re
                classes = re.findall(r'class\s+(\w+)', content)
                if classes:
                    print(f"  Available classes: {', '.join(classes)}")
                else:
                    print("  No classes found in file")
                return False
    except Exception as e:
        print(f"✗ Error reading {filepath}: {e}")
        return False

def main():
    print("=== RAG System Import Diagnostics ===\n")
    
    # Check critical files
    files_to_check = [
        "src/retrieval/retriever.py",
        "src/rag_pipeline.py",
        "src/utils/document_loader.py",
        "src/api/main.py"
    ]
    
    print("1. Checking file existence:")
    for filepath in files_to_check:
        check_file_exists(filepath)
    
    print("\n2. Checking for Retriever class:")
    check_class_in_file("src/retrieval/retriever.py", "Retriever")
    
    print("\n3. Checking imports in rag_pipeline.py:")
    if os.path.exists("src/rag_pipeline.py"):
        with open("src/rag_pipeline.py", 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines, 1):
                if "retriever" in line.lower() and ("import" in line or "from" in line):
                    print(f"  Line {i}: {line.strip()}")
    
    print("\n4. Checking Python path and virtual environment:")
    print(f"  Python executable: {sys.executable}")
    print(f"  Current working directory: {os.getcwd()}")
    print(f"  Python path includes current dir: {'.' in sys.path or os.getcwd() in sys.path}")
    
    print("\n5. Checking installed packages:")
    try:
        import bs4
        print("✓ beautifulsoup4 (bs4) is installed")
    except ImportError:
        print("✗ beautifulsoup4 (bs4) is NOT installed")
        print("  Run: pip install beautifulsoup4")
    
    try:
        import chromadb
        print("✓ chromadb is available")
    except ImportError:
        print("✗ chromadb is NOT installed")
    
    try:
        import sentence_transformers
        print("✓ sentence-transformers is available")
    except ImportError:
        print("✗ sentence-transformers is NOT installed")

if __name__ == "__main__":
    main()
