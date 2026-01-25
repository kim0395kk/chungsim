# -*- coding: utf-8 -*-
"""
Govable AI - 사이드바 인증 UI

이 모듈에서만 streamlit import 허용
"""
from typing import TYPE_CHECKING

import streamlit as st

from govable_ai.config import ADMIN_EMAIL
from govable_ai.helpers import shorten_one_line

if TYPE_CHECKING:
    from govable_ai.core.db_client import SupabaseClient


def is_admin_user(email: str, db_admin_flag: bool = False) -> bool:
    """관리자 여부 확인"""
    e = (email or "").strip().lower()
    if e == ADMIN_EMAIL.lower():
        return True
    return db_admin_flag


def sidebar_auth(db_client: "SupabaseClient") -> None:
    """
    사이드바 인증 UI 렌더링
    
    Args:
        db_client: Supabase DB 클라이언트
    """
    st.sidebar.markdown("## 🔐 로그인")
    
    # 세션 상태 초기화
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_email" not in st.session_state:
        st.session_state.user_email = ""
    if "admin_mode" not in st.session_state:
        st.session_state.admin_mode = False
    if "is_admin_db" not in st.session_state:
        st.session_state.is_admin_db = False
    
    sb = db_client.client if db_client else None
    
    if st.session_state.logged_in:
        email = st.session_state.user_email
        st.sidebar.success(f"✅ {email}")
        
        # 관리자 토글
        if is_admin_user(email, st.session_state.get("is_admin_db", False)):
            st.sidebar.toggle("관리자모드 켜기", key="admin_mode")
        
        if st.sidebar.button("로그아웃", use_container_width=True):
            if sb:
                try:
                    sb.auth.sign_out()
                except Exception:
                    pass
            st.session_state.logged_in = False
            st.session_state.user_email = ""
            st.session_state.admin_mode = False
            st.session_state.is_admin_db = False
            st.rerun()
        return
    
    menu = st.sidebar.radio("메뉴", ["로그인", "회원가입", "비밀번호 찾기"], horizontal=True)
    
    if "signup_stage" not in st.session_state:
        st.session_state.signup_stage = 1
    if "reset_stage" not in st.session_state:
        st.session_state.reset_stage = 1
    
    if menu == "로그인":
        email = st.sidebar.text_input("메일", placeholder="name@korea.kr", key="login_email")
        pw = st.sidebar.text_input("비밀번호", type="password", key="login_pw")
        
        if st.sidebar.button("로그인", use_container_width=True):
            if sb:
                try:
                    sb.auth.sign_in_with_password({"email": email, "password": pw})
                    st.session_state.logged_in = True
                    st.session_state.user_email = (email or "").strip()
                    
                    # DB 관리자 플래그 확인
                    if db_client:
                        st.session_state.is_admin_db = db_client.check_admin(email)
                    
                    st.rerun()
                except Exception:
                    st.sidebar.error("로그인 실패: 메일/비밀번호 확인")
            else:
                st.sidebar.error("DB 연결 없음")
    
    elif menu == "회원가입":
        if st.session_state.signup_stage == 1:
            email = st.sidebar.text_input("메일(@korea.kr)", placeholder="name@korea.kr", key="su_email")
            if st.sidebar.button("코리아 메일로 인증번호 발송", use_container_width=True):
                if not (email or "").endswith("@korea.kr"):
                    st.sidebar.error("❌ @korea.kr 메일만 가입 가능")
                elif sb:
                    try:
                        sb.auth.sign_in_with_otp({"email": email, "options": {"should_create_user": True}})
                        st.session_state.pending_email = email.strip()
                        st.session_state.signup_stage = 2
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"발송 실패: {e}")
        else:
            email = st.session_state.get("pending_email", "")
            st.sidebar.caption(f"발송 대상: {email}")
            code = st.sidebar.text_input("인증번호(OTP/토큰)", key="su_code")
            new_pw = st.sidebar.text_input("비밀번호 설정", type="password", key="su_pw")
            new_pw2 = st.sidebar.text_input("비밀번호 확인", type="password", key="su_pw2")
            
            if st.sidebar.button("인증 + 비밀번호 설정 완료", use_container_width=True):
                if not new_pw or new_pw != new_pw2:
                    st.sidebar.error("비밀번호가 일치하지 않습니다.")
                elif sb:
                    ok = False
                    for t in ["signup", "magiclink"]:
                        try:
                            sb.auth.verify_otp({"email": email, "token": code, "type": t})
                            ok = True
                            break
                        except Exception:
                            pass
                    if not ok:
                        st.sidebar.error("인증번호 검증 실패")
                    else:
                        try:
                            sb.auth.update_user({"password": new_pw})
                            st.session_state.logged_in = True
                            st.session_state.user_email = email
                            st.session_state.signup_stage = 1
                            st.rerun()
                        except Exception as e:
                            st.sidebar.error(f"비밀번호 설정 실패: {e}")
    
    elif menu == "비밀번호 찾기":
        if st.session_state.reset_stage == 1:
            email = st.sidebar.text_input("가입된 메일", key="reset_email")
            if st.sidebar.button("비밀번호 재설정 메일 발송", use_container_width=True):
                if sb:
                    try:
                        sb.auth.reset_password_email(email)
                        st.session_state.pending_email = email.strip()
                        st.session_state.reset_stage = 2
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"발송 실패: {e}")
        else:
            st.sidebar.info("메일에서 재설정 링크를 확인하세요.")
            if st.sidebar.button("처음으로", use_container_width=True):
                st.session_state.reset_stage = 1
                st.rerun()


def render_history_list(db_client: "SupabaseClient") -> None:
    """
    사이드바 히스토리 목록 렌더링
    
    Args:
        db_client: Supabase DB 클라이언트
    """
    if not db_client or not db_client.is_available():
        return
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📜 최근 기록")
    
    history = db_client.fetch_history(limit=20)
    
    if not history:
        st.sidebar.caption("기록 없음")
        return
    
    for row in history[:10]:
        prompt = row.get("prompt", "")
        archive_id = row.get("id", "")
        short_label = shorten_one_line(prompt, 25)
        
        if st.sidebar.button(f"📄 {short_label}", key=f"hist_{archive_id}", use_container_width=True):
            st.session_state.restore_archive_id = archive_id
            st.rerun()
