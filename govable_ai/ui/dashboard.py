# -*- coding: utf-8 -*-
"""
Govable AI - 관리자 대시보드 UI

이 모듈에서만 streamlit import 허용
"""
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import streamlit as st

from govable_ai.config import MODEL_PRICING, HEAVY_USER_PERCENTILE, LONG_LATENCY_THRESHOLD

if TYPE_CHECKING:
    from govable_ai.core.db_client import SupabaseClient

# Pandas optional import
try:
    import pandas as pd
except ImportError:
    pd = None


def render_master_dashboard(db_client: "SupabaseClient", llm_service=None) -> None:
    """
    관리자 마스터 대시보드 렌더링
    
    Args:
        db_client: Supabase DB 클라이언트
        llm_service: 임베딩 생성을 위한 LLM 서비스 (Optional)
    """
    if pd is None:
        st.error("pandas가 설치되지 않았습니다. `pip install pandas` 실행 필요")
        return
    
    if not db_client or not db_client.is_available():
        st.error("DB 연결 없음")
        return
    
    st.markdown("## 📊 마스터 대시보드")
    
    # [NEW] 데이터 관리 (임베딩 생성)
    with st.expander("🛠️ 데이터베이스 관리 (임베딩 생성)", expanded=False):
        st.info("당직 매뉴얼 데이터에 벡터 임베딩이 없는 경우 검색이 되지 않습니다. 아래 버튼을 눌러 임베딩을 생성하세요.")
        
        col_db1, col_db2 = st.columns(2)
        with col_db1:
            if st.button("🔄 매뉴얼 임베딩 생성(재처리)", use_container_width=True):
                if not llm_service:
                    st.error("LLM 서비스가 연결되지 않았습니다.")
                else:
                    try:
                        # 1. 임베딩 없는 데이터 조회
                        res = db_client.client.table("duty_manual_kb").select("*").is_("embedding", "null").execute()
                        rows = res.data
                        
                        if not rows:
                            st.success("모든 데이터에 임베딩이 이미 존재합니다.")
                        else:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            success_count = 0
                            
                            for idx, row in enumerate(rows):
                                content = row.get("content", "")
                                if content:
                                    emb = llm_service.embed_text(content)
                                    if emb:
                                        # 업데이트
                                        db_client.client.table("duty_manual_kb").update({"embedding": emb}).eq("id", row["id"]).execute()
                                        success_count += 1
                                
                                progress = (idx + 1) / len(rows)
                                progress_bar.progress(progress)
                                status_text.text(f"처리 중... ({idx+1}/{len(rows)})")
                            
                            st.success(f"완료! {success_count}건의 임베딩을 생성했습니다.")
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"작업 중 오류 발생: {e}")

    # 데이터 로드
    with st.spinner("데이터 로딩 중..."):
        archives = db_client.admin_fetch_work_archive(limit=2000)
        sessions = db_client.admin_fetch_sessions(minutes=5)
        events = db_client.admin_fetch_events(limit=300)
        api_logs = db_client.admin_fetch_api_logs(limit=500)
    
    # 실시간 현황
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🟢 실시간 접속", len(sessions))
    with col2:
        st.metric("📝 전체 작업", len(archives))
    with col3:
        st.metric("📊 이벤트 수", len(events))
    with col4:
        st.metric("🔌 API 호출", len(api_logs))
    
    st.markdown("---")
    
    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs(["📈 종합 통계", "👥 사용자 분석", "💰 비용 분석", "📜 상세 로그"])
    
    with tab1:
        _render_summary_stats(archives, api_logs)
    
    with tab2:
        _render_user_analysis(archives, events)
    
    with tab3:
        _render_cost_analysis(api_logs)
    
    with tab4:
        _render_detailed_logs(archives, api_logs)


def _render_summary_stats(archives: list, api_logs: list) -> None:
    """종합 통계 렌더링"""
    if not archives:
        st.info("데이터 없음")
        return
    
    df = pd.DataFrame(archives)
    
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"])
        df["date"] = df["created_at"].dt.date
        
        # 일별 작업 수 차트
        st.markdown("### 📅 일별 작업 현황")
        daily = df.groupby("date").size().reset_index(name="count")
        st.bar_chart(daily.set_index("date")["count"])
    
    # 모델별 사용량
    if "model_used" in df.columns:
        st.markdown("### 🤖 모델별 사용량")
        model_counts = df["model_used"].value_counts()
        st.bar_chart(model_counts)
    
    # 앱 모드별 사용량
    if "app_mode" in df.columns:
        st.markdown("### ⚡ 처리 모드별 분포")
        mode_counts = df["app_mode"].value_counts()
        st.bar_chart(mode_counts)


def _render_user_analysis(archives: list, events: list) -> None:
    """사용자 분석 렌더링"""
    if not archives:
        st.info("데이터 없음")
        return
    
    df = pd.DataFrame(archives)
    
    # 사용자별 작업 수
    if "user_email" in df.columns:
        st.markdown("### 👤 사용자별 작업 수 (Top 20)")
        
        # null 제외
        user_df = df[df["user_email"].notna()]
        if not user_df.empty:
            user_counts = user_df["user_email"].value_counts().head(20)
            
            # 과다 사용자 임계값 계산
            threshold = user_counts.quantile(HEAVY_USER_PERCENTILE / 100) if len(user_counts) > 5 else float("inf")
            
            for email, count in user_counts.items():
                flag = "🔴 " if count >= threshold else ""
                st.markdown(f"- {flag}**{email}**: {count}건")
        else:
            st.caption("로그인 사용자 없음")
    
    # 익명 세션 분석
    if "anon_session_id" in df.columns:
        st.markdown("### 🔒 익명 세션 분석")
        anon_counts = df["anon_session_id"].nunique()
        st.metric("고유 세션 수", anon_counts)


def _render_cost_analysis(api_logs: list) -> None:
    """비용 분석 렌더링"""
    if not api_logs:
        st.info("API 호출 로그 없음")
        return
    
    df = pd.DataFrame(api_logs)
    
    # 비용 계산
    def calc_cost(row):
        model = row.get("model_name", "")
        tokens = (row.get("input_tokens", 0) or 0) + (row.get("output_tokens", 0) or 0)
        price = MODEL_PRICING.get(model, MODEL_PRICING.get("(unknown)", 0.10))
        return (tokens / 1_000_000) * price
    
    df["cost_usd"] = df.apply(calc_cost, axis=1)
    
    # 총 비용
    total_cost = df["cost_usd"].sum()
    st.metric("💰 총 누적 비용 (추정)", f"${total_cost:.4f}")
    
    # API 유형별 비용
    if "api_type" in df.columns:
        st.markdown("### 📊 API 유형별 비용")
        api_cost = df.groupby("api_type")["cost_usd"].sum().sort_values(ascending=False)
        st.bar_chart(api_cost)
    
    # 모델별 비용
    if "model_name" in df.columns:
        st.markdown("### 🤖 모델별 비용")
        model_cost = df.groupby("model_name")["cost_usd"].sum().sort_values(ascending=False)
        for model, cost in model_cost.items():
            if model and cost > 0:
                st.markdown(f"- **{model}**: ${cost:.4f}")
    
    # 토큰 사용량
    total_tokens = (df["input_tokens"].fillna(0).sum() + df["output_tokens"].fillna(0).sum())
    st.metric("📝 총 토큰 사용량", f"{int(total_tokens):,}")


def _render_detailed_logs(archives: list, api_logs: list) -> None:
    """상세 로그 렌더링"""
    st.markdown("### 📜 최근 작업 로그")
    
    if archives:
        df = pd.DataFrame(archives)
        
        # 컬럼 선택
        display_cols = ["created_at", "user_email", "prompt", "model_used", "execution_time", "token_usage"]
        display_cols = [c for c in display_cols if c in df.columns]
        
        if display_cols:
            # 프롬프트 요약
            if "prompt" in df.columns:
                df["prompt"] = df["prompt"].apply(lambda x: (x[:50] + "...") if x and len(x) > 50 else x)
            
            # 긴 레이턴시 하이라이팅
            def highlight_rows(row):
                styles = [""] * len(row)
                if "execution_time" in row.index:
                    exec_time = row.get("execution_time", 0) or 0
                    if exec_time > LONG_LATENCY_THRESHOLD:
                        styles = ["background-color: #ffebee"] * len(row)
                return styles
            
            styled_df = df[display_cols].head(50).style.apply(highlight_rows, axis=1)
            st.dataframe(styled_df, use_container_width=True)
    
    st.markdown("### 🔌 최근 API 호출 로그")
    
    if api_logs:
        df = pd.DataFrame(api_logs)
        
        display_cols = ["created_at", "api_type", "model_name", "input_tokens", "output_tokens", "latency_ms", "success"]
        display_cols = [c for c in display_cols if c in df.columns]
        
        if display_cols:
            st.dataframe(df[display_cols].head(50), use_container_width=True)
