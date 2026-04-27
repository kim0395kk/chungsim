# civil_engineering/dashboard.py
"""
토목직 산업단지 대시보드 UI
"""

import streamlit as st
import pandas as pd
from typing import List, Dict
from datetime import datetime


def render_civil_dashboard(complexes_data: List[Dict]):
    """
    산업단지 현황 대시보드 렌더링
    
    Args:
        complexes_data: 파싱된 산업단지 데이터 리스트
    """
    
    st.markdown("## 📊 산업단지 현황 대시보드")
    
    if not complexes_data:
        st.warning("데이터가 없습니다.")
        return

    # === 1. 전체 통계 카드 ===
    render_statistics_cards(complexes_data)
    
    st.divider()
    
    # === 2. 필터 ===
    filtered_data = render_filters(complexes_data)
    
    st.divider()
    
    # === 3. 비교표 ===
    render_comparison_table(filtered_data)
    
    st.divider()
    
    # === 4. 타임라인 (간단 버전) ===
    render_timeline(filtered_data)


def render_statistics_cards(complexes_data: List[Dict]):
    """통계 카드 4개"""
    
    # 통계 계산
    total_count = len(complexes_data)
    total_area = sum(c['area_sqm'] for c in complexes_data) / 1000000  # 백만㎡
    total_budget = sum(c['budget_krw'] for c in complexes_data) / 1000000000000  # 조원
    
    status_counts = {}
    for c in complexes_data:
        status = c['status']
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # 카드 렌더링
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🏗️ 총 단지 수",
            value=f"{total_count}개"
        )
    
    with col2:
        st.metric(
            label="📐 총 면적",
            value=f"{total_area:.1f}백만㎡"
        )
    
    with col3:
        st.metric(
            label="💰 총 예산",
            value=f"{total_budget:.2f}조원"
        )
    
    with col4:
        completed = status_counts.get("조성완료", 0)
        in_progress = status_counts.get("조성중", 0)
        st.metric(
            label="✅ 완료/진행중",
            value=f"{completed}/{in_progress}개"
        )


def render_filters(complexes_data: List[Dict]) -> List[Dict]:
    """필터링 UI"""
    
    st.markdown("### 🔍 필터")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 상태 필터
        all_statuses = sorted(list(set(c['status'] for c in complexes_data)))
        selected_statuses = st.multiselect(
            "사업 상태",
            all_statuses,
            default=all_statuses,
            key="civil_status_filter"
        )
    
    with col2:
        # 개발 유형 필터
        all_types = sorted(list(set(c['development_type'] for c in complexes_data)))
        selected_types = st.multiselect(
            "개발 유형",
            all_types,
            default=all_types,
            key="civil_type_filter"
        )
    
    with col3:
        # 면적 범위 필터
        if complexes_data:
            min_area = min(c['area_sqm'] for c in complexes_data)
            max_area = max(c['area_sqm'] for c in complexes_data)
        else:
            min_area, max_area = 0, 1000000
            
        # 슬라이더 단위: 천㎡
        min_val_k = int(min_area / 1000)
        max_val_k = int(max_area / 1000)
        
        if min_val_k == max_val_k:
             max_val_k += 1 # prevent error
             
        area_range = st.slider(
            "면적 범위 (천㎡)",
            min_value=min_val_k,
            max_value=max_val_k,
            value=(min_val_k, max_val_k),
            key="civil_area_range_filter"
        )
    
    # 필터링 적용
    filtered = [
        c for c in complexes_data
        if c['status'] in selected_statuses
        and c['development_type'] in selected_types
        and area_range[0] * 1000 <= c['area_sqm'] <= area_range[1] * 1000
    ]
    
    st.caption(f"📌 {len(filtered)}개 단지 선택됨 (전체 {len(complexes_data)}개)")
    
    return filtered


def render_comparison_table(complexes_data: List[Dict]):
    """비교표"""
    st.markdown("### 📋 사업 현황 비교")
    
    if not complexes_data:
        st.info("선택된 단지가 없습니다.")
        return
        
    # DataFrame 변환
    data_for_df = []
    for c in complexes_data:
        data_for_df.append({
            "단지명": c['name'],
            "위치": c['location'],
            "상태": c['status'],
            "유형": c['development_type'],
            "면적(㎡)": f"{c['area_sqm']:,}",
            "예산(억원)": f"{c['budget_krw'] // 100000000:,}",
            "기간": f"{c['period']['start']}~{c['period']['end']}",
            "시행자": c['developer']
        })
        
    df = pd.DataFrame(data_for_df)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_timeline(complexes_data: List[Dict]):
    """간단한 타임라인 시각화"""
    st.markdown("### 📅 추진 일정 타임라인")
    
    if not complexes_data:
        st.info("선택된 단지가 없습니다.")
        return

    # 간트 차트 느낌으로 표현 (st.dataframe bar chart 활용 or text)
    # 여기서는 간단하게 텍스트 기반으로 표시하고, 추후 고도화
    
    for c in complexes_data:
        with st.expander(f"**{c['name']}** ({c['status']}) - {c['period']['start']}~{c['period']['end']}"):
            # 마일스톤 정렬
            milestones = sorted(c['milestones'], key=lambda x: x['date'])
            
            for m in milestones:
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.caption(m['date'])
                with col2:
                    st.write(f"**{m['event']}**")
            
            if c['industries']:
                st.info(f"유치업종: {', '.join(c['industries'])}")
