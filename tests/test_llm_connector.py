import pytest
from unittest.mock import MagicMock, patch
from src.llm.connector import OllamaLLMConnector

def test_initialization_defaults():
    llm = OllamaLLMConnector()
    assert llm.model_name == "llama3.1"
    assert llm.temperature == 0.0

def test_initialization_custom():
    llm = OllamaLLMConnector(model_name="mistral", temperature=0.7)
    assert llm.model_name == "mistral"
    assert llm.temperature == 0.7

def test_invoke_mocked():
    llm = OllamaLLMConnector(model_name="llama3.2:latest", temperature=0.0)
    response = llm.generate("Explain the concept of algorithmic debugging.")
    assert isinstance(response, str) and response is not None


