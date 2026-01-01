# text2sql_mvp/core/gemini_client.py
"""
Async Gemini API client for Text-to-SQL generation.
Uses the google-genai SDK with retry logic and error handling.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import google.genai as genai
from google.genai import types
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from config.settings import get_settings, GeminiSettings


class GeminiErrorType(str, Enum):
    """Classification of Gemini API errors."""
    RATE_LIMIT = "rate_limit"
    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION = "authentication"
    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class GenerationMetrics:
    """Metrics from a generation request."""
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    temperature: float = 0.0
    finish_reason: str = ""


@dataclass
class GeminiResponse:
    """Response from Gemini API."""
    text: str
    success: bool
    metrics: GenerationMetrics = field(default_factory=GenerationMetrics)
    error: Optional[str] = None
    error_type: Optional[GeminiErrorType] = None
    raw_response: Optional[Any] = None


class GeminiClient:
    """
    Async client for Google Gemini API.
    
    Features:
    - Automatic retry with exponential backoff
    - Token counting and latency tracking
    - Error classification
    - Batch generation support
    
    Example:
        ```python
        client = GeminiClient()
        response = await client.generate("Write a SQL query...")
        if response.success:
            print(response.text)
        ```
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        settings: Optional[GeminiSettings] = None,
    ):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google API key. Falls back to settings/environment.
            model: Model name. Falls back to settings.
            settings: Optional GeminiSettings override.
        """
        self._settings = settings or self._load_settings()
        
        # Override with explicit parameters
        self._api_key = api_key or self._settings.api_key or self._get_api_key_from_env()
        self._model = model or self._settings.model
        
        # Initialize client
        self._client: Optional[genai.Client] = None
        self._initialized = False
        
    def _load_settings(self) -> GeminiSettings:
        """Load settings from configuration."""
        try:
            from config.settings import get_gemini_settings
            return get_gemini_settings()
        except Exception:
            return GeminiSettings()
    
    def _get_api_key_from_env(self) -> str:
        """Get API key from environment."""
        import os
        return os.environ.get("GOOGLE_API_KEY", "")
    
    def _ensure_initialized(self) -> None:
        """Ensure client is initialized."""
        if not self._initialized:
            if not self._api_key:
                raise ValueError(
                    "Google API key not provided. Set GOOGLE_API_KEY environment "
                    "variable or pass api_key parameter."
                )
            self._client = genai.Client(api_key=self._api_key)
            self._initialized = True
    
    @property
    def model(self) -> str:
        """Current model name."""
        return self._model
    
    @property
    def default_temperature(self) -> float:
        """Default temperature setting."""
        return self._settings.temperature
    
    def _classify_error(self, error: Exception) -> GeminiErrorType:
        """Classify an exception into error type."""
        error_str = str(error).lower()
        
        if "rate" in error_str or "quota" in error_str or "429" in error_str:
            return GeminiErrorType.RATE_LIMIT
        elif "invalid" in error_str or "400" in error_str:
            return GeminiErrorType.INVALID_REQUEST
        elif "auth" in error_str or "401" in error_str or "403" in error_str:
            return GeminiErrorType.AUTHENTICATION
        elif "500" in error_str or "503" in error_str or "server" in error_str:
            return GeminiErrorType.SERVER_ERROR
        elif "timeout" in error_str:
            return GeminiErrorType.TIMEOUT
        else:
            return GeminiErrorType.UNKNOWN
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if error is retryable."""
        error_type = self._classify_error(error)
        return error_type in {
            GeminiErrorType.RATE_LIMIT,
            GeminiErrorType.SERVER_ERROR,
            GeminiErrorType.TIMEOUT,
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    async def _generate_with_retry(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str] = None,
    ) -> types.GenerateContentResponse:
        """Generate with retry logic."""
        self._ensure_initialized()
        
        # Build config
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=self._settings.top_p,
            top_k=self._settings.top_k,
        )
        
        # Add system instruction if provided
        if system_prompt:
            config.system_instruction = system_prompt
        
        # Generate (sync call - genai SDK is synchronous)
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )
        
        return response
    
    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> GeminiResponse:
        """
        Generate text from prompt.
        
        Args:
            prompt: The input prompt
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum output tokens
            system_prompt: Optional system instruction
            
        Returns:
            GeminiResponse with generated text and metrics
        """
        start_time = time.perf_counter()
        
        # Use defaults if not specified
        temp = temperature if temperature is not None else self._settings.temperature
        tokens = max_tokens or self._settings.max_output_tokens
        
        try:
            # Run sync API call in thread pool to not block
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._generate_sync(prompt, temp, tokens, system_prompt)
            )
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Extract text
            text = ""
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    text = candidate.content.parts[0].text or ""
            
            # Build metrics
            metrics = GenerationMetrics(
                latency_ms=latency_ms,
                model=self._model,
                temperature=temp,
                finish_reason=str(response.candidates[0].finish_reason) if response.candidates else "",
            )
            
            # Try to get token counts if available
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                metrics.prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                metrics.completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
                metrics.total_tokens = getattr(response.usage_metadata, 'total_token_count', 0)
            
            return GeminiResponse(
                text=text,
                success=True,
                metrics=metrics,
                raw_response=response,
            )
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            error_type = self._classify_error(e)
            
            return GeminiResponse(
                text="",
                success=False,
                metrics=GenerationMetrics(latency_ms=latency_ms, model=self._model),
                error=str(e),
                error_type=error_type,
            )
    
    def _generate_sync(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
    ) -> types.GenerateContentResponse:
        """Synchronous generation for thread pool execution."""
        self._ensure_initialized()
        
        # Build config
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=self._settings.top_p,
            top_k=self._settings.top_k,
        )
        
        # Add system instruction if provided
        if system_prompt:
            config.system_instruction = system_prompt
        
        return self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )
    
    async def generate_batch(
        self,
        prompts: list[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        max_concurrent: int = 5,
    ) -> list[GeminiResponse]:
        """
        Generate responses for multiple prompts concurrently.
        
        Args:
            prompts: List of prompts
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            system_prompt: Optional system instruction
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of GeminiResponse objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_semaphore(prompt: str) -> GeminiResponse:
            async with semaphore:
                return await self.generate(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                )
        
        tasks = [generate_with_semaphore(p) for p in prompts]
        return await asyncio.gather(*tasks)
    
    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Token count
        """
        self._ensure_initialized()
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._client.models.count_tokens(
                    model=self._model,
                    contents=text,
                )
            )
            return result.total_tokens
        except Exception:
            # Rough estimate if API fails
            return len(text) // 4
