# -*- coding: utf-8 -*-
"""
Govable AI - LLM 서비스
Vertex AI, Gemini API, Groq 통합 및 폴백 로직

UI 의존성 없음 (streamlit import 금지)
"""
import json
import logging
import os
import tempfile
import time
from typing import Any, Dict, Optional

from govable_ai.helpers import safe_json_loads, estimate_tokens

logger = logging.getLogger(__name__)

_GEMINI_MODEL_ALIASES: Dict[str, str] = {
    # Legacy/invalid names mapped to stable model IDs
    "gemini-3.0-flash": "gemini-2.5-flash",
    "gemini-3-flash-preview": "gemini-2.5-flash",
    "models/gemini-3.0-flash": "gemini-2.5-flash",
}

_DEFAULT_GEMINI_FALLBACK_MODELS = ("gemini-2.0-flash-lite", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-1.5-pro")


class LLMService:
    """
    LLM 통합 서비스 (Vertex AI -> Gemini API -> Groq 폴백)
    
    의존성 주입으로 설정을 전달받습니다.
    UI(streamlit) 의존성이 없어 다른 환경에서도 사용 가능합니다.
    """
    
    def __init__(
        self,
        vertex_config: Optional[Dict] = None,
        gemini_key: Optional[str] = None,
        groq_key: Optional[str] = None,
        default_model: str = "gemini-2.5-flash",
    ):
        """
        Args:
            vertex_config: Vertex AI 설정 (project_id, location, model_id, credentials_json)
            gemini_key: Google Gemini API 키
            groq_key: Groq API 키
            default_model: 기본 모델명
        """
        self.vertex_config = vertex_config
        self.gemini_key = gemini_key
        self.groq_key = groq_key
        self.default_model = default_model
        
        # 상태 추적
        self.last_model_used: Optional[str] = None
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0
        self.last_latency_ms: int = 0
        
        # 초기화 상태
        self._vertex_client = None
        self._gemini_client = None
        self._groq_client = None
        self._vertex_available = False
        self._gemini_available = False
        self._groq_available = False
        
        # 클라이언트 초기화
        self._init_vertex_ai()
        self._init_gemini_api()
        self._init_groq()
    
    def _init_vertex_ai(self) -> None:
        """Vertex AI 클라이언트 초기화"""
        if not self.vertex_config:
            return
        
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
            
            project_id = self.vertex_config.get("project_id")
            location = self.vertex_config.get("location", "us-central1")
            model_id = self.vertex_config.get("model_id", "gemini-2.5-flash")
            creds_json = self.vertex_config.get("credentials_json")
            
            if not project_id:
                return
            
            # 서비스 계정 인증
            if creds_json:
                from google.oauth2 import service_account
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                    f.write(creds_json)
                    tmp_path = f.name
                try:
                    creds = service_account.Credentials.from_service_account_file(tmp_path)
                    vertexai.init(project=project_id, location=location, credentials=creds)
                finally:
                    os.unlink(tmp_path)
            else:
                vertexai.init(project=project_id, location=location)
            
            self._vertex_client = GenerativeModel(model_id)
            self._vertex_available = True
            logger.info(f"Vertex AI initialized: {model_id}")
            
        except Exception as e:
            logger.warning(f"Vertex AI 초기화 실패: {e}")
            self._vertex_available = False
    
    def _init_gemini_api(self) -> None:
        """Google Gemini API 클라이언트 초기화"""
        if not self.gemini_key:
            return
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_key)
            self._gemini_client = genai
            self._gemini_available = True
            logger.info("Gemini API initialized")
        except Exception as e:
            logger.warning(f"Gemini API 초기화 실패: {e}")
            self._gemini_available = False
    
    def _init_groq(self) -> None:
        """Groq 클라이언트 초기화"""
        if not self.groq_key:
            return
        
        try:
            from groq import Groq
            self._groq_client = Groq(api_key=self.groq_key)
            self._groq_available = True
            logger.info("Groq initialized")
        except Exception as e:
            logger.warning(f"Groq 초기화 실패: {e}")
            self._groq_available = False
    
    def is_available(self) -> bool:
        """LLM 서비스 사용 가능 여부"""
        return self._vertex_available or self._gemini_available or self._groq_available
    
    def _try_vertex_text(self, prompt: str, preferred_model: Optional[str] = None, **kwargs) -> Optional[str]:
        """Vertex AI로 텍스트 생성"""
        if not self._vertex_available or not self._vertex_client:
            return None
        
        try:
            configured_model = self.vertex_config.get("model_id", "gemini-2.5-flash")
            requested_model = _GEMINI_MODEL_ALIASES.get(preferred_model, preferred_model) if preferred_model else configured_model
            model_client = self._vertex_client
            model_used = configured_model

            # preferred_model이 있으면 해당 모델로 생성을 시도한다.
            if requested_model and requested_model != configured_model:
                try:
                    model_client = self._vertex_client.__class__(requested_model)
                    model_used = requested_model
                except Exception as model_err:
                    logger.warning(f"Vertex 모델 전환 실패 ({requested_model}): {model_err}")

            start = time.time()
            resp = model_client.generate_content(prompt)
            self.last_latency_ms = int((time.time() - start) * 1000)
            self.last_model_used = f"{model_used} (Vertex AI)"
            self.last_input_tokens = estimate_tokens(prompt)
            text = resp.text if hasattr(resp, "text") else ""
            self.last_output_tokens = estimate_tokens(text)
            return text
        except Exception as e:
            logger.warning(f"Vertex AI 생성 실패: {e}")
            return None
    
    def _try_gemini_api_text(self, prompt: str, preferred_model: Optional[str] = None, **kwargs) -> Optional[str]:
        """Gemini API로 텍스트 생성"""
        if not self._gemini_available or not self._gemini_client:
            return None

        # 지정 모델 -> 기본 모델 -> 안전 폴백 순으로 시도
        models_to_try: list[str] = []
        requested = preferred_model or self.default_model
        if requested:
            models_to_try.append(_GEMINI_MODEL_ALIASES.get(requested, requested))
        for fallback_name in _DEFAULT_GEMINI_FALLBACK_MODELS:
            if fallback_name not in models_to_try:
                models_to_try.append(fallback_name)

        self.last_input_tokens = estimate_tokens(prompt)
        for model_name in models_to_try:
            try:
                start = time.time()
                model = self._gemini_client.GenerativeModel(model_name)
                # API별 파라미터 호환성 이슈를 피하기 위해 우선 prompt만 전달
                resp = model.generate_content(prompt)
                self.last_latency_ms = int((time.time() - start) * 1000)
                self.last_model_used = f"{model_name} (Gemini API)"

                text = ""
                if hasattr(resp, "text"):
                    text = resp.text
                elif hasattr(resp, "candidates") and resp.candidates:
                    parts = resp.candidates[0].content.parts
                    text = "".join(p.text for p in parts if hasattr(p, "text"))

                self.last_output_tokens = estimate_tokens(text)
                return text
            except Exception as e:
                logger.warning(f"Gemini API 생성 실패 ({model_name}): {e}")

        return None
    
    def _try_groq_text(self, prompt: str, **kwargs) -> Optional[str]:
        """Groq로 텍스트 생성"""
        if not self._groq_available or not self._groq_client:
            return None
        
        try:
            start = time.time()
            resp = self._groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", 4096),
            )
            self.last_latency_ms = int((time.time() - start) * 1000)
            self.last_model_used = "llama-3.3-70b-versatile (Groq)"
            self.last_input_tokens = estimate_tokens(prompt)
            text = resp.choices[0].message.content if resp.choices else ""
            self.last_output_tokens = estimate_tokens(text)
            return text
        except Exception as e:
            logger.warning(f"Groq 생성 실패: {e}")
            return None
    
    def generate_text(self, prompt: str, preferred_model: Optional[str] = None, **kwargs) -> str:
        """
        텍스트 생성 (Vertex AI -> Gemini API -> Groq 순서로 폴백)
        
        Args:
            prompt: 프롬프트 문자열
            preferred_model: 우선적으로 사용할 모델 이름 (예: 'gemini-2.5-flash')
            
        Returns:
            생성된 텍스트. 모든 시도 실패 시 에러 메시지 반환.
        """
        # 1. Vertex AI 시도
        result = self._try_vertex_text(prompt, preferred_model=preferred_model, **kwargs)
        if result:
            return result
        
        # 2. Gemini API 시도
        result = self._try_gemini_api_text(prompt, preferred_model=preferred_model, **kwargs)
        if result:
            return result
        
        # 3. Groq 시도
        result = self._try_groq_text(prompt, **kwargs)
        if result:
            return result
        
        # 모두 실패
        self.last_model_used = "(failed)"
        return "[LLM 연결 실패] 모든 LLM 서비스에 연결할 수 없습니다."
    
    def generate_json(self, prompt: str, preferred_model: Optional[str] = None, **kwargs) -> Optional[Any]:
        """
        JSON 응답 생성
        
        Args:
            prompt: JSON 출력을 요청하는 프롬프트
            preferred_model: 우선적으로 사용할 모델 이름 (예: 'gemini-2.5-flash')
            
        Returns:
            파싱된 JSON 객체 또는 None
        """
        text = self.generate_text(prompt, preferred_model=preferred_model, **kwargs)
        return safe_json_loads(text)
    
    def get_last_usage(self) -> Dict[str, Any]:
        """마지막 호출의 사용량 정보 반환"""
        return {
            "model_used": self.last_model_used,
            "input_tokens": self.last_input_tokens,
            "output_tokens": self.last_output_tokens,
            "latency_ms": self.last_latency_ms,
        }

    def embed_text(self, text: str) -> list[float]:
        """
        텍스트 임베딩 생성 (Gemini API 사용)
        Args:
            text: 임베딩할 텍스트
        Returns:
            임베딩 벡터 (list of floats)
        """
        if not self._gemini_available or not self._gemini_client:
            # Fallback or empty return if no embedding service available
            # Vertex AI embedding could be added here if needed
            logger.warning("Gemini API not available for embeddings")
            return []

        try:
            # text-embedding-004 is a common model name, but verify if it works.
            # Often 'models/embedding-001' or 'models/text-embedding-004'
            result = self._gemini_client.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return []
