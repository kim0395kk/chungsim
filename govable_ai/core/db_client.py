# -*- coding: utf-8 -*-
"""
Govable AI - Supabase DB 클라이언트

UI 의존성 없음 (streamlit import 금지)
"""
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from govable_ai.config import APP_VERSION, MODEL_PRICING
from govable_ai.helpers import safe_now_utc_iso

logger = logging.getLogger(__name__)

# Optional supabase import
try:
    from supabase import create_client
except ImportError:
    create_client = None


class SupabaseClient:
    """
    Supabase DB 클라이언트
    
    의존성 주입으로 연결 정보를 전달받습니다.
    UI(streamlit) 의존성이 없어 다른 환경에서도 사용 가능합니다.
    """
    
    def __init__(self, url: Optional[str] = None, anon_key: Optional[str] = None):
        """
        Args:
            url: Supabase 프로젝트 URL
            anon_key: Supabase anon/public 키
        """
        self.url = url
        self.anon_key = anon_key
        self._client = None
        
        if url and anon_key and create_client:
            try:
                self._client = create_client(url, anon_key)
                logger.info("Supabase connected")
            except Exception as e:
                logger.error(f"Supabase connection error: {e}")
    
    def is_available(self) -> bool:
        """DB 연결 가능 여부"""
        return self._client is not None
    
    @property
    def client(self):
        """Raw Supabase client (인증 등에 필요)"""
        return self._client
    
    # =========================================================
    # Work Archive CRUD
    # =========================================================
    
    def insert_archive(
        self,
        prompt: str,
        payload: dict,
        anon_session_id: str,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
    ) -> Optional[str]:
        """
        작업 아카이브 저장
        
        Returns:
            생성된 archive_id 또는 None
        """
        if not self.is_available():
            return None
        
        archive_id = str(uuid.uuid4())
        
        row = {
            "id": archive_id,
            "prompt": prompt,
            "payload": payload,
            "anon_session_id": anon_session_id,
            "user_id": user_id,
            "user_email": user_email.strip() if user_email else None,
            "client_meta": {"app_ver": APP_VERSION},
            "app_mode": payload.get("app_mode", "신속"),
            "search_count": int(payload.get("search_count") or 0),
            "execution_time": float(payload.get("execution_time") or 0.0),
            "token_usage": int(payload.get("token_usage") or 0),
            "model_used": payload.get("model_used"),
        }
        
        try:
            self._client.table("work_archive").insert(row).execute()
            return archive_id
        except Exception as e:
            logger.error(f"Archive insert error: {e}")
            return None
    
    def fetch_history(self, limit: int = 80) -> List[dict]:
        """작업 히스토리 조회"""
        if not self.is_available():
            return []
        
        try:
            resp = (
                self._client.table("work_archive")
                .select("id,prompt,created_at,user_email,anon_session_id")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return getattr(resp, "data", None) or []
        except Exception:
            return []
    
    def fetch_payload(self, archive_id: str) -> Optional[dict]:
        """특정 아카이브 조회"""
        if not self.is_available():
            return None
        
        try:
            resp = (
                self._client.table("work_archive")
                .select("id,prompt,payload,created_at,user_email,anon_session_id")
                .eq("id", archive_id)
                .limit(1)
                .execute()
            )
            data = getattr(resp, "data", None) or []
            return data[0] if data else None
        except Exception:
            return None
    
    # =========================================================
    # Followups CRUD
    # =========================================================
    
    def fetch_followups(self, archive_id: str) -> List[dict]:
        """후속 질문 조회"""
        if not self.is_available():
            return []
        
        try:
            resp = (
                self._client.table("work_followups")
                .select("turn,role,content,created_at")
                .eq("archive_id", archive_id)
                .order("turn", desc=False)
                .execute()
            )
            return getattr(resp, "data", None) or []
        except Exception:
            return []
    
    def insert_followup(
        self,
        archive_id: str,
        turn: int,
        role: str,
        content: str,
        anon_session_id: str,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
    ) -> bool:
        """후속 질문 저장"""
        if not self.is_available():
            return False
        
        row = {
            "archive_id": archive_id,
            "turn": turn,
            "role": role,
            "content": content,
            "user_id": user_id,
            "user_email": user_email,
            "anon_session_id": anon_session_id,
        }
        
        try:
            self._client.table("work_followups").insert(row).execute()
            return True
        except Exception:
            return False
    
    # =========================================================
    # Session & Event Logging
    # =========================================================
    
    def touch_session(self, anon_session_id: str, user_id: Optional[str] = None) -> None:
        """세션 업데이트 (app_sessions 테이블)"""
        if not self.is_available():
            return
        
        try:
            row = {
                "anon_session_id": anon_session_id,
                "user_id": user_id,
                "last_seen": safe_now_utc_iso(),
            }
            self._client.table("app_sessions").upsert(
                row, 
                on_conflict="anon_session_id"
            ).execute()
        except Exception as e:
            logger.debug(f"Session touch failed: {e}")
    
    def log_event(
        self,
        event_type: str,
        anon_session_id: str,
        archive_id: Optional[str] = None,
        meta: Optional[dict] = None,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
    ) -> None:
        """이벤트 로깅"""
        if not self.is_available():
            return
        
        row = {
            "anon_session_id": anon_session_id,
            "user_id": user_id,
            "user_email": user_email,
            "event_type": event_type,
            "archive_id": archive_id,
            "meta": meta or {},
        }
        
        try:
            self._client.table("app_events").insert(row).execute()
        except Exception as e:
            logger.debug(f"Event log failed: {e}")
    
    def log_api_call(
        self,
        api_type: str,
        anon_session_id: str,
        model_name: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: int = 0,
        success: bool = True,
        error_message: Optional[str] = None,
        request_summary: Optional[str] = None,
        response_summary: Optional[str] = None,
        archive_id: Optional[str] = None,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
    ) -> None:
        """API 호출 로깅"""
        if not self.is_available():
            return
        
        # 비용 계산
        total_tokens = input_tokens + output_tokens
        price_per_m = MODEL_PRICING.get(model_name or "", MODEL_PRICING.get("(unknown)", 0.10))
        cost_usd = (total_tokens / 1_000_000) * price_per_m
        
        row = {
            "anon_session_id": anon_session_id,
            "user_id": user_id,
            "user_email": user_email,
            "api_type": api_type,
            "model_name": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "success": success,
            "error_message": error_message,
            "request_summary": request_summary[:500] if request_summary else None,
            "response_summary": response_summary[:500] if response_summary else None,
            "archive_id": archive_id,
            "cost_usd": cost_usd,
        }
        
        try:
            self._client.table("api_call_logs").insert(row).execute()
        except Exception as e:
            logger.debug(f"API log failed: {e}")
    
    # =========================================================
    # Admin Queries
    # =========================================================
    
    def admin_fetch_work_archive(self, limit: int = 2000) -> List[dict]:
        """관리자용 전체 아카이브 조회"""
        if not self.is_available():
            return []
        
        try:
            resp = (
                self._client.table("work_archive")
                .select("*")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return getattr(resp, "data", None) or []
        except Exception:
            return []
    
    def admin_fetch_sessions(self, minutes: int = 5) -> List[dict]:
        """최근 활성 세션 조회"""
        if not self.is_available():
            return []
        
        try:
            from datetime import timedelta
            cutoff = (datetime.utcnow() - timedelta(minutes=minutes)).isoformat() + "Z"
            resp = (
                self._client.table("app_sessions")
                .select("*")
                .gte("last_seen", cutoff)
                .execute()
            )
            return getattr(resp, "data", None) or []
        except Exception:
            return []
    
    def admin_fetch_events(self, limit: int = 300) -> List[dict]:
        """최근 이벤트 조회"""
        if not self.is_available():
            return []
        
        try:
            resp = (
                self._client.table("app_events")
                .select("*")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return getattr(resp, "data", None) or []
        except Exception:
            return []
    
    def admin_fetch_api_logs(self, limit: int = 500) -> List[dict]:
        """API 호출 로그 조회"""
        if not self.is_available():
            return []
        
        try:
            resp = (
                self._client.table("api_call_logs")
                .select("*")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return getattr(resp, "data", None) or []
        except Exception:
            return []
    
    def check_admin(self, email: str) -> bool:
        """관리자 여부 확인"""
        if not self.is_available() or not email:
            return False
        
        try:
            resp = (
                self._client.table("app_admins")
                .select("email")
                .eq("email", email.strip().lower())
                .limit(1)
                .execute()
            )
            data = getattr(resp, "data", None) or []
            return len(data) > 0
        except Exception:
            return False
