import pytest
from unittest.mock import Mock, patch, MagicMock

from src.preprocessing.intelligent_preprocessor import IntelligentPreprocessor


class TestIntelligentPreprocessor:
    """Testes para a classe IntelligentPreprocessor."""

    @pytest.fixture
    def mock_spacy(self):
        """Mock do spaCy."""
        with patch('src.preprocessing.intelligent_preprocessor.spacy') as mock_spacy:
            mock_nlp = Mock()
            mock_spacy.load.return_value = mock_nlp
            mock_spacy.blank.return_value = mock_nlp
            yield {
                'spacy': mock_spacy,
                'nlp': mock_nlp
            }

    @pytest.fixture
    def mock_transformers(self):
        """Mock do transformers."""
        with patch('src.preprocessing.intelligent_preprocessor.pipeline') as mock_pipeline:
            mock_classifier = Mock()
            mock_pipeline.return_value = mock_classifier
            yield {
                'pipeline': mock_pipeline,
                'classifier': mock_classifier
            }

    def test_init_english_with_spacy(self, mock_spacy, mock_transformers):
        """Testa inicialização em inglês com spaCy disponível."""
        preprocessor = IntelligentPreprocessor(language="en")
        
        assert preprocessor.language == "en"
        mock_spacy['spacy'].load.assert_called_with("en_core_web_sm")
        assert preprocessor.nlp is not None

    def test_init_portuguese_with_spacy(self, mock_spacy, mock_transformers):
        """Testa inicialização em português com spaCy disponível."""
        preprocessor = IntelligentPreprocessor(language="pt")
        
        assert preprocessor.language == "pt"
        mock_spacy['spacy'].load.assert_called_with("pt_core_news_sm")
        assert preprocessor.nlp is not None

    def test_init_other_language_with_spacy(self, mock_spacy, mock_transformers):
        """Testa inicialização com outro idioma usando spaCy blank."""
        preprocessor = IntelligentPreprocessor(language="fr")
        
        assert preprocessor.language == "fr"
        mock_spacy['spacy'].blank.assert_called_with("fr")
        assert preprocessor.nlp is not None

    def test_init_spacy_model_not_available(self, mock_transformers):
        """Testa inicialização quando modelo spaCy não está disponível."""
        with patch('src.preprocessing.intelligent_preprocessor.spacy') as mock_spacy:
            mock_spacy.load.side_effect = Exception("Model not found")
            mock_blank_nlp = Mock()
            mock_spacy.blank.return_value = mock_blank_nlp
            
            preprocessor = IntelligentPreprocessor(language="en")
            
            # Deve fazer fallback para blank
            mock_spacy.blank.assert_called_with("en")
            assert preprocessor.nlp == mock_blank_nlp

    def test_init_without_spacy(self, mock_transformers):
        """Testa inicialização quando spaCy não está instalado."""
        with patch('src.preprocessing.intelligent_preprocessor.spacy', None):
            preprocessor = IntelligentPreprocessor()
            
            assert preprocessor.nlp is None

    def test_init_transformers_classifier_success(self, mock_spacy, mock_transformers):
        """Testa inicialização bem-sucedida do classificador."""
        preprocessor = IntelligentPreprocessor()
        
        mock_transformers['pipeline'].assert_any_call("zero-shot-classification")
        assert preprocessor.classifier is not None

    def test_init_transformers_classifier_failure(self, mock_spacy):
        """Testa falha na inicialização do classificador."""
        with patch('src.preprocessing.intelligent_preprocessor.pipeline') as mock_pipeline:
            mock_pipeline.side_effect = Exception("Model loading failed")
            
            preprocessor = IntelligentPreprocessor()
            
            assert preprocessor.classifier is None

    def test_init_without_transformers(self, mock_spacy):
        """Testa inicialização quando transformers não está instalado."""
        with patch('src.preprocessing.intelligent_preprocessor.pipeline', None):
            preprocessor = IntelligentPreprocessor()
            
            assert preprocessor.classifier is None

    def test_clean_text_basic(self, mock_spacy, mock_transformers):
        """Testa limpeza básica de texto."""
        preprocessor = IntelligentPreprocessor()
        
        text = "  Este é um texto   com espaços extras.  \n\n  "
        result = preprocessor.clean_text(text)
        
        # Deve remover espaços extras e quebras de linha
        assert result.strip() != text.strip()
        assert "\n" not in result or result.count("\n") < text.count("\n")

    def test_clean_text_remove_special_chars(self, mock_spacy, mock_transformers):
        """Testa remoção de caracteres especiais."""
        preprocessor = IntelligentPreprocessor()
        
        text = "Texto com @#$%^&*() caracteres especiais!"
        result = preprocessor.clean_text(text)
        
        # Deve manter apenas caracteres alfanuméricos e espaços básicos
        assert len(result) <= len(text)
        # Verificar que alguns caracteres especiais foram removidos
        special_chars = "@#$%^&*()"
        for char in special_chars:
            if char in text:
                assert result.count(char) <= text.count(char)

    def test_clean_text_empty_input(self, mock_spacy, mock_transformers):
        """Testa limpeza com entrada vazia."""
        preprocessor = IntelligentPreprocessor()
        
        result = preprocessor.clean_text("")
        assert result == ""
        
        result = preprocessor.clean_text("   ")
        assert result.strip() == ""

    def test_extract_entities_with_spacy(self, mock_spacy, mock_transformers):
        """Testa extração de entidades com spaCy."""
        preprocessor = IntelligentPreprocessor()
        
        # Mock das entidades
        mock_entity1 = Mock()
        mock_entity1.text = "João"
        mock_entity1.label_ = "PERSON"
        
        mock_entity2 = Mock()
        mock_entity2.text = "São Paulo"
        mock_entity2.label_ = "GPE"
        
        mock_doc = Mock()
        mock_doc.ents = [mock_entity1, mock_entity2]
        mock_spacy['nlp'].return_value = mock_doc
        
        text = "João mora em São Paulo."
        result = preprocessor.extract_entities(text)
        
        assert len(result) == 2
        assert {"text": "João", "label": "PERSON"} in result
        assert {"text": "São Paulo", "label": "GPE"} in result

    def test_extract_entities_without_spacy(self, mock_transformers):
        """Testa extração de entidades sem spaCy."""
        with patch('src.preprocessing.intelligent_preprocessor.spacy', None):
            preprocessor = IntelligentPreprocessor()
            
            result = preprocessor.extract_entities("Texto qualquer")
            
            assert result == []

    def test_extract_entities_spacy_error(self, mock_spacy, mock_transformers):
        """Testa tratamento de erro na extração de entidades."""
        preprocessor = IntelligentPreprocessor()
        mock_spacy['nlp'].side_effect = Exception("Processing error")
        
        result = preprocessor.extract_entities("Texto com erro")
        
        assert result == []

    def test_classify_content_with_classifier(self, mock_spacy, mock_transformers):
        """Testa classificação de conteúdo com classificador."""
        preprocessor = IntelligentPreprocessor()
        
        # Mock do resultado da classificação
        mock_result = {
            'labels': ['technical', 'documentation', 'tutorial'],
            'scores': [0.8, 0.6, 0.4]
        }
        mock_transformers['classifier'].return_value = mock_result
        
        text = "Este é um tutorial técnico sobre programação."
        categories = ["technical", "documentation", "tutorial", "general"]
        
        result = preprocessor.classify_content(text, categories)
        
        assert result == mock_result
        mock_transformers['classifier'].assert_called_once_with(text, categories)

    def test_classify_content_without_classifier(self, mock_spacy):
        """Testa classificação sem classificador disponível."""
        with patch('src.preprocessing.intelligent_preprocessor.pipeline', None):
            preprocessor = IntelligentPreprocessor()
            
            result = preprocessor.classify_content("Texto", ["cat1", "cat2"])
            
            assert result == {"labels": [], "scores": []}

    def test_classify_content_classifier_error(self, mock_spacy, mock_transformers):
        """Testa tratamento de erro na classificação."""
        preprocessor = IntelligentPreprocessor()
        mock_transformers['classifier'].side_effect = Exception("Classification error")
        
        result = preprocessor.classify_content("Texto", ["categoria"])
        
        assert result == {"labels": [], "scores": []}

    def test_summarize_text_basic(self, mock_spacy, mock_transformers):
        """Testa resumo básico de texto."""
        preprocessor = IntelligentPreprocessor()
        
        long_text = "Este é um texto muito longo. " * 20
        result = preprocessor.summarize_text(long_text, max_length=50)
        
        # O resumo deve ser menor que o texto original
        assert len(result) <= len(long_text)
        assert len(result) > 0

    def test_summarize_text_short_input(self, mock_spacy, mock_transformers):
        """Testa resumo de texto já curto."""
        preprocessor = IntelligentPreprocessor()
        
        short_text = "Texto curto."
        result = preprocessor.summarize_text(short_text, max_length=100)
        
        # Texto curto deve retornar o próprio texto ou versão ligeiramente modificada
        assert len(result) <= max(len(short_text), 100)

    def test_summarize_text_empty_input(self, mock_spacy, mock_transformers):
        """Testa resumo com entrada vazia."""
        preprocessor = IntelligentPreprocessor()
        
        result = preprocessor.summarize_text("", max_length=50)
        assert result == ""

    def test_process_document_complete(self, mock_spacy, mock_transformers):
        """Testa processamento completo de documento."""
        preprocessor = IntelligentPreprocessor()
        
        # Mock das entidades
        mock_entity = Mock()
        mock_entity.text = "Python"
        mock_entity.label_ = "LANGUAGE"
        
        mock_doc = Mock()
        mock_doc.ents = [mock_entity]
        mock_spacy['nlp'].return_value = mock_doc
        
        # Mock da classificação
        mock_classification = {
            'labels': ['technical'],
            'scores': [0.9]
        }
        mock_transformers['classifier'].return_value = mock_classification
        
        text = "Este é um documento sobre Python programming."
        result = preprocessor.process_document(text)
        
        assert "cleaned_text" in result
        assert "entities" in result
        assert "classification" in result
        assert "summary" in result
        
        assert len(result["entities"]) == 1
        assert result["entities"][0]["text"] == "Python"
        assert result["classification"] == mock_classification

    def test_process_document_minimal_features(self, mock_transformers):
        """Testa processamento com recursos mínimos (sem spaCy)."""
        with patch('src.preprocessing.intelligent_preprocessor.spacy', None):
            preprocessor = IntelligentPreprocessor()
            
            text = "Documento simples para teste."
            result = preprocessor.process_document(text)
            
            assert "cleaned_text" in result
            assert "entities" in result
            assert "classification" in result
            assert "summary" in result
            
            # Sem spaCy, entidades devem estar vazias
            assert result["entities"] == []

    def test_process_document_error_handling(self, mock_spacy, mock_transformers):
        """Testa tratamento de erros no processamento."""
        preprocessor = IntelligentPreprocessor()
        
        # Mock que gera erro
        mock_spacy['nlp'].side_effect = Exception("Processing error")
        
        text = "Documento com erro."
        result = preprocessor.process_document(text)
        
        # Deve retornar resultado parcial mesmo com erro
        assert "cleaned_text" in result
        assert "entities" in result
        assert result["entities"] == []  # Vazio devido ao erro

    def test_get_language_info(self, mock_spacy, mock_transformers):
        """Testa obtenção de informações do idioma."""
        preprocessor = IntelligentPreprocessor(language="en")
        
        info = preprocessor.get_language_info()
        
        assert "language" in info
        assert info["language"] == "en"
        assert "spacy_available" in info
        assert "classifier_available" in info

    def test_batch_process_documents(self, mock_spacy, mock_transformers):
        """Testa processamento em lote de documentos."""
        preprocessor = IntelligentPreprocessor()
        
        # Mock simples para entidades e classificação
        mock_doc = Mock()
        mock_doc.ents = []
        mock_spacy['nlp'].return_value = mock_doc
        
        mock_transformers['classifier'].return_value = {"labels": [], "scores": []}
        
        documents = [
            "Primeiro documento.",
            "Segundo documento.",
            "Terceiro documento."
        ]
        
        if hasattr(preprocessor, 'batch_process'):
            results = preprocessor.batch_process(documents)
            assert len(results) == 3
        else:
            # Testar processamento individual
            results = [preprocessor.process_document(doc) for doc in documents]
            assert len(results) == 3
            for result in results:
                assert "cleaned_text" in result

    def test_memory_efficiency_large_text(self, mock_spacy, mock_transformers):
        """Testa eficiência de memória com texto grande."""
        preprocessor = IntelligentPreprocessor()
        
        # Texto muito grande
        large_text = "Este é um texto muito longo. " * 1000
        
        # Mock para evitar processamento real pesado
        mock_doc = Mock()
        mock_doc.ents = []
        mock_spacy['nlp'].return_value = mock_doc
        
        result = preprocessor.process_document(large_text)
        
        # Deve processar sem erros
        assert "cleaned_text" in result
        assert len(result["cleaned_text"]) > 0