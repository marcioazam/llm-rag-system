# Guia de Contribuição

Obrigado por considerar contribuir com o projeto **RAG Multimodelo**! 🚀

---

## 1. Requisitos

* Python 3.10+
* `pre-commit` instalado (lint e format automático)
* Acesso ao Neo4j (opcional) e Ollama

---

## 2. Processo de Pull Request

1. **Fork** o repositório ou crie branch `feature/<descritivo>`.
2. Execute `pre-commit install` (formatação `black`, `isort`, `flake8`).
3. Adicione/atualize **testes** em `tests/`.
4. Atualize documentação em `Docs/` quando necessário.
5. Abra PR descrevendo:
   * Motivação
   * Mudanças principais
   * Como testar
6. Aguarde revisão e feedback.

---

## 3. Convenções de Código

| Item | Regra |
|------|-------|
| Formatação | `black` (line length 100) |
| Imports | `isort` + ordem: built-in, third-party, local |
| Tipagem | Use `typing` / `pydantic` onde fizer sentido |
| Logs | Use `logging.getLogger(__name__)` |
| Strings | Prefer f-strings |
| Docstrings | Formato Google ou reStructuredText |

---

## 4. Estrutura de Testes

* Siga o padrão `tests/test_<module>.py`.
* Use `pytest` e fixtures quando necessário.
* Testes devem ser idempotentes e independentes (sem ordem).

---

## 5. Versionamento Semântico

* **MAJOR** – Mudanças incompatíveis (break API)
* **MINOR** – Novas funcionalidades retrocompatíveis
* **PATCH** – Correções e ajustes menores

---

## 6. Direitos Autorais & Licença

* Este projeto é licenciado sob MIT.
* Ao contribuir, você concorda em licenciar sua contribuição sob a mesma licença.

---

Feliz _coding_! 💙 