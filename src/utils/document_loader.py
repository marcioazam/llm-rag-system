import os
from typing import Dict, Any
from pathlib import Path
import uuid
import PyPDF2
import docx
import markdown
from bs4 import BeautifulSoup

class DocumentLoader:
    """Carregador universal de documentos"""
    
    def __init__(self):
        self.supported_extensions = {
            ".pdf": self._load_pdf,
            ".txt": self._load_text,
            ".md": self._load_markdown,
            ".docx": self._load_docx,
            ".html": self._load_html
        }
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """Carregar documento e extrair conteúdo"""
        
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = path.suffix.lower()
        
        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {extension}")
        
        # Carregar conteúdo
        content = self.supported_extensions[extension](file_path)
        
        # Criar metadata
        metadata = {
            "document_id": str(uuid.uuid4()),
            "source": file_path,
            "filename": path.name,
            "extension": extension,
            "size": path.stat().st_size,
            "created": path.stat().st_ctime,
            "modified": path.stat().st_mtime
        }
        
        return {
            "content": content,
            "metadata": metadata
        }
    
    def _load_pdf(self, file_path: str) -> str:
        """Carregar PDF"""
        text = ""
        
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
        
        return text.strip()
    
    def _load_text(self, file_path: str) -> str:
        """Carregar arquivo de texto"""
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    
    def _load_markdown(self, file_path: str) -> str:
        """Carregar Markdown e converter para texto"""
        with open(file_path, "r", encoding="utf-8") as file:
            md_content = file.read()
        
        # Converter para HTML e depois extrair texto
        html = markdown.markdown(md_content)
        soup = BeautifulSoup(html, "html.parser")
        
        return soup.get_text()
    
    def _load_docx(self, file_path: str) -> str:
        """Carregar DOCX"""
        doc = docx.Document(file_path)
        
        text = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text.append(paragraph.text)
        
        return "\n".join(text)
    
    def _load_html(self, file_path: str) -> str:
        """Carregar HTML"""
        with open(file_path, "r", encoding="utf-8") as file:
            html_content = file.read()
        
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remover scripts e estilos
        for script in soup(["script", "style"]):
            script.decompose()
        
        return soup.get_text()
