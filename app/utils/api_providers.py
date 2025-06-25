"""
API Providers for different LLM services
Updated to include Anthropic Claude API support
"""
import os
import logging
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

class BaseAPIProvider(ABC):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def generate(self, messages: List[Dict], **kwargs) -> str:
        pass

class OpenAIProvider(BaseAPIProvider):
    def generate(self, messages: List[Dict], **kwargs) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            # Convert messages format
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, dict) and "content" in msg:
                    content = msg["content"]
                    if isinstance(content, list):
                        text = " ".join(item.get("text", "") for item in content if item.get("type") == "text")
                    else:
                        text = str(content)
                    formatted_messages.append({"role": "user", "content": text})
            
            response = client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 2048),
                timeout=30
                # max_retries=3
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            return f"Error: {str(e)}"

class GeminiProvider(BaseAPIProvider):
    def generate(self, messages: List[Dict], **kwargs) -> str:
        try:
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=self.api_key)
            
            # Extract prompt from messages
            prompt = ""
            for msg in messages:
                if isinstance(msg, dict) and "content" in msg:
                    content = msg["content"]
                    if isinstance(content, list):
                        prompt = " ".join(item.get("text", "") for item in content if item.get("type") == "text")
                    else:
                        prompt = str(content)
            
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                ),
            ]
            
            response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(response_mime_type="text/plain")
            )
            return response.text
        except Exception as e:
            self.logger.error(f"Gemini API error: {str(e)}")
            return f"Error: {str(e)}"

class GroqProvider(BaseAPIProvider):
    def generate(self, messages: List[Dict], **kwargs) -> str:
        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)
            
            # Convert messages format
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, dict) and "content" in msg:
                    content = msg["content"]
                    if isinstance(content, list):
                        text = " ".join(item.get("text", "") for item in content if item.get("type") == "text")
                    else:
                        text = str(content)
                    formatted_messages.append({"role": "user", "content": text})
            
            response = client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 2048),
                timeout=30
                # max_retries=3
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Groq API error: {str(e)}")
            return f"Error: {str(e)}"

class AnthropicProvider(BaseAPIProvider):
    def __init__(self, api_key: str, model: str):
        print(f"DEBUG AnthropicProvider: Received API key: {'YES (' + api_key[:10] + '...)' if api_key else 'NO - None or empty'}")
        print(f"DEBUG AnthropicProvider: API key type: {type(api_key)}")
        print(f"DEBUG AnthropicProvider: API key length: {len(api_key) if api_key else 0}")
        
        super().__init__(api_key, model)
    
    def generate(self, messages: List[Dict], **kwargs) -> str:
        print(f"DEBUG AnthropicProvider.generate: Using API key: {'YES (' + self.api_key[:10] + '...)' if self.api_key else 'NO - None or empty'}")
        
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            
            # Extract prompt from messages
            prompt = ""
            for msg in messages:
                if isinstance(msg, dict) and "content" in msg:
                    content = msg["content"]
                    if isinstance(content, list):
                        prompt = " ".join(item.get("text", "") for item in content if item.get("type") == "text")
                    else:
                        prompt = str(content)
            
            # Use correct Anthropic message format
            response = client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 2048),
                temperature=kwargs.get("temperature", 0.7),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            # Extract text from response
            return response.content[0].text
            
        except Exception as e:
            self.logger.error(f"Anthropic API error: {str(e)}")
            return f"Error: {str(e)}"

class APIProviderFactory:
    @staticmethod
    def create_provider(provider: str, api_key: str, model: str) -> BaseAPIProvider:
        providers = {
            "openai": OpenAIProvider,
            "gemini": GeminiProvider,
            "groq": GroqProvider,
            "anthropic": AnthropicProvider
        }
        
        if provider not in providers:
            raise ValueError(f"Unsupported provider: {provider}")
        
        return providers[provider](api_key, model)