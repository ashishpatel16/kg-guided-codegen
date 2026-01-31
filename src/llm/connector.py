import requests
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional, Dict, Any
import os


class OllamaLLMConnector:
    """
    A custom wrapper around LangChain's ChatOllama connector.
    """

    def __init__(
        self, model_name: str = "gemma3:12b", temperature: float = 0.0, **kwargs
    ):
        """
        Initialize the CustomOllama connector.

        Args:
            model_name (str): The name of the Ollama model to use. Defaults to "llama3.1".
            temperature (float): The temperature for generation. Defaults to 0.0.
            **kwargs: Additional arguments to pass to ChatOllama.
        """
        self.model_name = model_name
        self.temperature = temperature
        self._model = ChatOllama(model=model_name, temperature=temperature, **kwargs)

    def generate(self, prompt: str) -> str:
        """
        Generate a response for the given prompt.

        Args:
            prompt (str): The input prompt.

        Returns:
            str: The generated text response.
        """
        response = self._model.invoke(prompt)
        return response.content


class GeminiLLMConnector:
    """
    A custom wrapper around LangChain's ChatGoogleGenerativeAI connector.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the Gemini connector.

        Args:
            model_name (str): The name of the Gemini model to use.
            temperature (float): The temperature for generation.
            api_key (str): The Google API key. Defaults to GOOGLE_API_KEY env var.
        """
        self.model_name = model_name
        self.temperature = temperature
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY must be provided or set as an environment variable"
            )

        self._model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key,
            **kwargs,
        )

    def generate(self, prompt: str) -> str:
        """
        Generate a response for the given prompt.

        Args:
            prompt (str): The input prompt.

        Returns:
            str: The generated text response.
        """
        response = self._model.invoke(prompt)
        return response.content


class RawOllamaConnector:
    """
    A raw HTTP client for connecting to an Ollama instances hosted on cloud directly.
    """

    def __init__(
        self,
        model_name: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
    ):
        """
        Initialize the RawOllamaConnector.

        Args:
            model_name (str): The name of the model to use.
            base_url (str): The base URL of the Ollama instance. Defaults to "http://localhost:11434".
            temperature (float): The temperature for generation. Defaults to 0.0.
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        """
        Generate a response using the /api/chat endpoint.

        Args:
            prompt (str): The input prompt.

        Returns:
            str: The content of the assistant's response.

        Raises:
            requests.exceptions.RequestException: If the HTTP request fails.
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": self.temperature},
        }

        response = requests.post(url, json=payload)
        print(response)
        response.raise_for_status()

        data = response.json()
        return data.get("message", {}).get("content", "")


if __name__ == "__main__":
    query = """
    Create a Python fastapi module that exposes a single endpoint which takes in an input number and returns the fibonacci sequence up to that number.
    Instructions:
    - Only give me the code, no other text.
    """

    # Demo usage
    print("--- LangChain Connector ---")
    try:
        llm = OllamaLLMConnector(model_name="gemma3:12b", temperature=0.0)
        response = llm.generate(query)
        print(response)
    except Exception as e:
        print(f"LangChain connector error: {e}")

    # print("\n--- Raw Connector ---")
    # raw_llm = RawOllamaConnector(model_name="gemma3:12b", temperature=0.0, base_url="http://100.84.103.5:11434")
    # # To run this, ensure you have ollama running
    # response = raw_llm.generate("hi")
    # print(response)
