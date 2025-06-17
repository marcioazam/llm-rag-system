#!/bin/bash

echo "=== Quick Fix for RAG System Issues ==="

# 1. Install missing dependencies
echo "1. Installing missing dependencies..."
pip install beautifulsoup4 lxml html5lib

# 2. Check if retriever.py has the correct class name
echo "2. Checking retriever.py structure..."
if [ -f "src/retrieval/retriever.py" ]; then
    echo "Classes found in retriever.py:"
    grep -n "^class " src/retrieval/retriever.py
    echo ""
    echo "All imports in rag_pipeline.py related to retriever:"
    grep -n "retriever" src/rag_pipeline.py
else
    echo "retriever.py not found!"
fi

# 3. Create a simple test to verify imports
echo "3. Testing imports..."
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    print('Testing document_loader...')
    from src.utils.document_loader import DocumentLoader
    print('✓ DocumentLoader imported successfully')
except Exception as e:
    print(f'✗ DocumentLoader import failed: {e}')

try:
    print('Testing retriever module...')
    import src.retrieval.retriever as retriever_module
    print('✓ Retriever module imported successfully')
    print('Available classes/functions:')
    for attr in dir(retriever_module):
        if not attr.startswith('_') and attr[0].isupper():
            print(f'  - {attr}')
except Exception as e:
    print(f'✗ Retriever module import failed: {e}')
"

echo ""
echo "4. Suggested fixes based on common issues:"
echo "======================================="
echo "If 'Retriever' class not found, try these fixes:"
echo ""
echo "Fix 1: Check if the class is named differently"
echo "  - Look for 'HybridRetriever', 'BaseRetriever', or similar"
echo "  - Update the import in rag_pipeline.py accordingly"
echo ""
echo "Fix 2: If class doesn't exist, create a basic one:"
echo "cat > src/retrieval/retriever.py << 'EOF'
class Retriever:
    def __init__(self, config=None):
        self.config = config or {}
    
    def retrieve(self, query, top_k=5):
        # Placeholder implementation
        return []
    
    def add_documents(self, documents):
        # Placeholder implementation
        pass
EOF"
echo ""
echo "Fix 3: Check __init__.py files"
echo "  - Ensure src/__init__.py exists (can be empty)"
echo "  - Ensure src/retrieval/__init__.py exists"
echo ""
echo "After applying fixes, run: ./scripts/start_services.sh"

