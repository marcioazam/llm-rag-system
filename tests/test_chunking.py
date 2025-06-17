import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.chunking.semantic_chunker import SemanticChunker
from src.chunking.recursive_chunker import RecursiveChunker

class TestChunking(unittest.TestCase):
    
    def setUp(self):
        self.sample_text = """
        Este é o primeiro parágrafo do documento. Contém informações importantes
        sobre o tópico principal que estamos discutindo.
        
        Este é o segundo parágrafo. Ele aborda um aspecto diferente do mesmo
        tópico, mas ainda está relacionado ao contexto geral.
        
        Finalmente, temos o terceiro parágrafo que conclui a discussão e
        apresenta algumas considerações finais sobre o assunto.
        """
        
        self.metadata = {
            "document_id": "test_doc",
            "source": "test.txt"
        }
    
    def test_semantic_chunker(self):
        chunker = SemanticChunker(
            min_chunk_size=50,
            max_chunk_size=500
        )
        
        chunks = chunker.chunk(self.sample_text, self.metadata)
        
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(len(c.content) >= 50 for c in chunks))
        self.assertTrue(all(len(c.content) <= 500 for c in chunks))
    
    def test_recursive_chunker(self):
        chunker = RecursiveChunker(
            chunk_size=200,
            chunk_overlap=20
        )
        
        chunks = chunker.chunk(self.sample_text, self.metadata)
        
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(len(c.content) <= 200 for c in chunks))

if __name__ == '__main__':
    unittest.main()
