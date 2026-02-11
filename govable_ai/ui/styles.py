# -*- coding: utf-8 -*-
"""
Govable AI - CSS 스타일 정의

이 모듈에서만 streamlit import 허용
"""

# 페이지 설정
PAGE_CONFIG = {
    "layout": "wide",
    "page_title": "AI Bureau: The Legal Glass",
    "page_icon": "⚖️",
}

# 메인 CSS (670라인의 스타일)
MAIN_CSS = """
<style>
    /* ====================== */
    /* Design Tokens - 공무원 친화적 디자인 */
    /* ====================== */
    :root {
        /* Colors - Government Professional Palette */
        --primary-50: #f0f4f8;
        --primary-100: #d9e6f2;
        --primary-200: #b3cde0;
        --primary-500: #1e4a7a;
        --primary-600: #16375c;
        --primary-700: #0f2744;
        --primary-800: #0a1929;
        
        /* Colors - Neutral Palette (높은 가독성) */
        --neutral-50: #fafafa;
        --neutral-100: #f5f5f5;
        --neutral-200: #eeeeee;
        --neutral-300: #e0e0e0;
        --neutral-400: #bdbdbd;
        --neutral-500: #757575;
        --neutral-600: #616161;
        --neutral-700: #424242;
        --neutral-800: #2d2d2d;
        --neutral-900: #1a1a1a;
        
        /* Colors - Semantic (정부 기관 적합) */
        --success-50: #e8f5e9;
        --success-500: #2e7d32;
        --success-600: #1b5e20;
        --warning-50: #fff3e0;
        --warning-500: #ed6c02;
        --warning-600: #e65100;
        --error-50: #ffebee;
        --error-500: #d32f2f;
        --error-600: #c62828;
        --info-50: #e3f2fd;
        --info-500: #0288d1;
        --info-600: #01579b;
        
        /* Spacing (일관된 여백) */
        --space-xs: 0.25rem;
        --space-sm: 0.5rem;
        --space-md: 1rem;
        --space-lg: 1.5rem;
        --space-xl: 2rem;
        --space-2xl: 3rem;
        
        /* Border Radius (절제된 둥글기) */
        --radius-sm: 0.25rem;
        --radius-md: 0.375rem;
        --radius-lg: 0.5rem;
        --radius-xl: 0.75rem;
        
        /* Shadows (은은한 그림자) */
        --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.08);
        --shadow-md: 0 2px 8px 0 rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 4px 16px 0 rgba(0, 0, 0, 0.12);
        --shadow-xl: 0 8px 24px 0 rgba(0, 0, 0, 0.15);
        --shadow-2xl: 0 16px 48px 0 rgba(0, 0, 0, 0.2);
        
        /* Typography */
        --font-serif: 'Batang', 'Nanum Myeongjo', serif;
        --font-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans KR', 'Malgun Gothic', sans-serif;
        
        /* Accessibility */
        --focus-ring: 0 0 0 3px rgba(30, 74, 122, 0.3);
        --transition-base: all 0.2s ease;
    }
    
    /* ====================== */
    /* Base Styles - 전문적이고 깨끗한 배경 */
    /* ====================== */
    .stApp { 
        background: #f8f9fa;
        font-family: var(--font-sans);
    }
    
    /* ====================== */
    /* Document Paper Style - 공식 문서 스타일 */
    /* ====================== */
    .paper-sheet {
        background-color: white;
        width: 100%;
        max-width: 210mm;
        min-height: 297mm;
        padding: 28mm;
        margin: var(--space-xl) auto;
        box-shadow: var(--shadow-lg);
        font-family: var(--font-serif);
        color: var(--neutral-900);
        line-height: 1.9;
        position: relative;
        border-radius: var(--radius-sm);
        border: 1px solid var(--neutral-200);
    }

    .doc-header { 
        text-align: center; 
        font-size: 24pt; 
        font-weight: 900; 
        margin-bottom: var(--space-2xl); 
        letter-spacing: 2px;
        color: var(--primary-800);
        padding-bottom: var(--space-lg);
        border-bottom: 3px double var(--primary-700);
    }
    
    .doc-info { 
        display: flex; 
        justify-content: space-between; 
        font-size: 11pt; 
        background: var(--primary-50);
        padding: var(--space-lg) var(--space-xl);
        border-radius: var(--radius-md);
        margin-bottom: var(--space-xl);
        gap: var(--space-md);
        flex-wrap: wrap;
        border-left: 5px solid var(--primary-600);
        border: 1px solid var(--primary-200);
        border-left-width: 5px;
    }
    
    .doc-info span {
        font-weight: 700;
        color: var(--primary-800);
    }
    
    .doc-body { 
        font-size: 12pt; 
        text-align: justify; 
        white-space: normal;
        color: var(--neutral-800);
    }
    
    .doc-footer { 
        text-align: center; 
        font-size: 22pt; 
        font-weight: bold; 
        margin-top: 100px; 
        letter-spacing: 6px;
        color: var(--neutral-900);
    }
    
    .stamp { 
        position: absolute; 
        bottom: 85px; 
        right: 80px; 
        border: 4px solid var(--error-600); 
        color: var(--error-600); 
        padding: 12px 20px; 
        font-size: 14pt; 
        font-weight: 900; 
        transform: rotate(-12deg); 
        opacity: 0.95; 
        border-radius: var(--radius-md);
        background: rgba(255, 255, 255, 0.98);
        box-shadow: var(--shadow-lg);
    }

    /* ====================== */
    /* Lawbot Button - 전문적이고 명확한 디자인 */
    /* ====================== */
    .lawbot-btn {
        background: linear-gradient(135deg, var(--info-600) 0%, var(--info-500) 100%);
        color: white !important;
        border: 2px solid var(--info-600);
        border-radius: var(--radius-lg);
        padding: 1.2rem 2rem;
        font-weight: 700;
        font-size: 1.05rem;
        transition: var(--transition-base);
        box-shadow: var(--shadow-md);
        display: inline-block;
        width: 100%;
        text-align: center;
        text-decoration: none !important;
        position: relative;
    }
    
    .lawbot-btn:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
        background: linear-gradient(135deg, var(--info-500) 0%, #039be5 100%);
        color: white !important;
    }
    
    .lawbot-btn:focus {
        outline: none;
        box-shadow: var(--focus-ring), var(--shadow-lg);
    }
    
    .lawbot-btn:active {
        transform: translateY(0);
    }
    
    .lawbot-sub { 
        font-size: 0.9rem; 
        opacity: 0.95; 
        margin-top: var(--space-sm); 
        display: block; 
        color: rgba(255,255,255,0.95) !important; 
        font-weight: 500;
        line-height: 1.4;
    }

    /* ====================== */
    /* Sidebar Styles - 마우스 드래그로 크기 조절 */
    /* ====================== */
    
    /* 접기 버튼 완전 제거 */
    button[data-testid="stSidebarCollapseButton"],
    button[data-testid="baseButton-headerNoPadding"],
    div[data-testid="stSidebarCollapsedControl"],
    button[aria-label="Collapse sidebar"],
    button[aria-label="Expand sidebar"],
    [data-testid="collapsedControl"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
        width: 0 !important;
        height: 0 !important;
    }
    
    /* 사이드바 최상위 section */
    section[data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        transform: none !important;
        left: 0 !important;
        width: 320px;
        min-width: 60px !important;
        max-width: 500px !important;
        resize: horizontal !important;
        overflow-x: auto !important;
        overflow-y: auto !important;
    }
    
    section[data-testid="stSidebar"][aria-expanded="false"] {
        width: 320px !important;
        transform: translateX(0) !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    /* 사이드바 내부 div */
    section[data-testid="stSidebar"] > div {
        min-width: 60px !important;
        width: 100% !important;
        padding: var(--space-lg);
        overflow: visible !important;
    }
    
    div[data-testid="stSidebar"] {
        background: white;
        border-right: 2px solid var(--neutral-200);
        box-shadow: 2px 0 8px rgba(0, 0, 0, 0.05);
        position: relative !important;
        overflow: visible !important;
    }
    
    /* 오른쪽 하단에 리사이즈 힌트 */
    section[data-testid="stSidebar"]::after {
        content: '⇔ 드래그';
        position: absolute;
        right: 4px;
        bottom: 8px;
        font-size: 10px;
        color: #9ca3af;
        pointer-events: none;
    }
    
    div[data-testid="stSidebar"] button[kind="secondary"] {
        width: 100%;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 0.75rem 1rem !important;
        border-radius: var(--radius-md) !important;
        border: 2px solid var(--neutral-200) !important;
        background: white !important;
        color: var(--neutral-800) !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: var(--transition-base) !important;
        margin-bottom: var(--space-sm) !important;
        min-height: 44px !important;
    }
    
    div[data-testid="stSidebar"] button[kind="secondary"]:hover { 
        background: var(--primary-50) !important;
        border-color: var(--primary-600) !important;
        color: var(--primary-700) !important;
        transform: translateX(4px);
        box-shadow: var(--shadow-sm);
    }
    
    div[data-testid="stSidebar"] button[kind="secondary"]:focus {
        outline: none !important;
        box-shadow: var(--focus-ring) !important;
    }

    /* ====================== */
    /* Form Elements - 명확한 입력 필드 */
    /* ====================== */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: var(--radius-md) !important;
        border: 2px solid var(--neutral-300) !important;
        padding: 0.75rem 1rem !important;
        font-family: var(--font-sans) !important;
        font-size: 1rem !important;
        transition: var(--transition-base) !important;
        background: white !important;
        min-height: 44px !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary-600) !important;
        box-shadow: var(--focus-ring) !important;
        outline: none !important;
    }
    
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: var(--neutral-400) !important;
    }
    
    /* ====================== */
    /* Buttons - 명확하고 접근성 높은 버튼 */
    /* ====================== */
    .stButton > button {
        border-radius: var(--radius-lg) !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: var(--transition-base) !important;
        border: 2px solid transparent !important;
        min-height: 44px !important;
    }
    
    .stButton > button[kind="primary"] {
        background: var(--primary-600) !important;
        color: white !important;
        border-color: var(--primary-600) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: var(--primary-700) !important;
        border-color: var(--primary-700) !important;
        box-shadow: var(--shadow-md) !important;
        transform: translateY(-1px) !important;
    }
    
    .stButton > button[kind="primary"]:focus {
        outline: none !important;
        box-shadow: var(--focus-ring), var(--shadow-md) !important;
    }
    
    .stButton > button[kind="primary"]:active {
        transform: translateY(0) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    .stButton > button[kind="secondary"] {
        background: white !important;
        color: var(--primary-600) !important;
        border-color: var(--primary-600) !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: var(--primary-50) !important;
        border-color: var(--primary-700) !important;
    }

    /* ====================== */
    /* Expanders - 명확한 펼치기/접기 */
    /* ====================== */
    .streamlit-expanderHeader {
        background: white !important;
        border-radius: var(--radius-md) !important;
        padding: 1rem !important;
        font-weight: 700 !important;
        border: 2px solid var(--neutral-300) !important;
        transition: var(--transition-base) !important;
        font-size: 1rem !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--primary-50) !important;
        border-color: var(--primary-600) !important;
    }
    
    /* ====================== */
    /* Info/Warning Boxes - 명확한 알림 */
    /* ====================== */
    .stAlert {
        border-radius: var(--radius-md) !important;
        border: 2px solid !important;
        padding: 1rem 1.25rem !important;
        font-size: 1rem !important;
    }
    
    div[data-baseweb="notification"][kind="info"] {
        background: var(--info-50) !important;
        border-color: var(--info-500) !important;
        color: var(--info-600) !important;
    }
    
    div[data-baseweb="notification"][kind="success"] {
        background: var(--success-50) !important;
        border-color: var(--success-500) !important;
        color: var(--success-600) !important;
    }
    
    div[data-baseweb="notification"][kind="warning"] {
        background: var(--warning-50) !important;
        border-color: var(--warning-500) !important;
        color: var(--warning-600) !important;
    }
    
    div[data-baseweb="notification"][kind="error"] {
        background: var(--error-50) !important;
        border-color: var(--error-500) !important;
        color: var(--error-600) !important;
    }
    
    /* ====================== */
    /* Chat Messages - 대화 메시지 */
    /* ====================== */
    .stChatMessage {
        border-radius: var(--radius-md) !important;
        padding: 1rem !important;
        margin-bottom: var(--space-md) !important;
        box-shadow: var(--shadow-sm) !important;
        border: 1px solid var(--neutral-200) !important;
    }
    
    /* ====================== */
    /* Chat Input - 명확한 입력 영역 */
    /* ====================== */
    .stChatInputContainer {
        background: white !important;
        border: 3px solid var(--primary-600) !important;
        border-radius: var(--radius-lg) !important;
        padding: var(--space-md) !important;
        box-shadow: var(--shadow-md) !important;
        margin-top: var(--space-xl) !important;
        position: relative !important;
    }
    
    .stChatInputContainer::before {
        content: '💬 후속 질문 입력';
        position: absolute;
        top: -1.5rem;
        left: 0;
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--primary-800);
        background: var(--primary-50);
        padding: 0.4rem 1rem;
        border-radius: var(--radius-md);
        border: 2px solid var(--primary-600);
    }
    
    .stChatInputContainer textarea {
        border: 2px solid var(--neutral-300) !important;
        border-radius: var(--radius-md) !important;
        background: white !important;
        font-size: 1.05rem !important;
        padding: 0.75rem 1rem !important;
        transition: var(--transition-base) !important;
        line-height: 1.5 !important;
    }
    
    .stChatInputContainer textarea:focus {
        border-color: var(--primary-600) !important;
        box-shadow: var(--focus-ring) !important;
        outline: none !important;
    }
    
    .stChatInputContainer textarea::placeholder {
        color: var(--neutral-500) !important;
        font-weight: 400 !important;
    }

    /* ====================== */
    /* Headers & Text - 명확한 계층 구조 */
    /* ====================== */
    h1, h2, h3 {
        color: var(--primary-800) !important;
        font-weight: 700 !important;
        line-height: 1.3 !important;
    }
    
    h1 { 
        font-size: 2.25rem !important; 
        margin-bottom: var(--space-lg) !important;
    }
    h2 { 
        font-size: 1.5rem !important; 
        margin-top: var(--space-2xl) !important; 
        margin-bottom: var(--space-md) !important;
        padding-bottom: var(--space-sm) !important;
        border-bottom: 2px solid var(--neutral-200) !important;
    }
    h3 { 
        font-size: 1.125rem !important; 
        margin-top: var(--space-lg) !important; 
        margin-bottom: var(--space-sm) !important;
        color: var(--primary-700) !important;
    }
    
    p {
        line-height: 1.7 !important;
        color: var(--neutral-800) !important;
    }

    /* ====================== */
    /* Hide Default Elements */
    /* ====================== */
    header [data-testid="stToolbar"] { display: none !important; }
    header [data-testid="stDecoration"] { display: none !important; }
    header { height: 0px !important; }
    footer { display: none !important; }
    div[data-testid="stStatusWidget"] { display: none !important; }

    /* ====================== */
    /* Agent Logs - 명확하고 전문적인 로그 */
    /* ====================== */
    .agent-log { 
        font-family: var(--font-sans); 
        font-size: 0.95rem; 
        padding: 1rem 1.25rem; 
        border-radius: var(--radius-md); 
        margin-bottom: 0.75rem; 
        background: white;
        border: 1px solid var(--neutral-200);
        transition: var(--transition-base);
        box-shadow: var(--shadow-sm);
    }
    
    .agent-log:hover {
        transform: translateX(4px);
        box-shadow: var(--shadow-md);
    }
    
    .log-legal { 
        background: var(--primary-50); 
        color: var(--primary-800); 
        border-left: 4px solid var(--primary-600);
    }
    
    .log-legal:hover {
        background: #e3f2fd;
        border-left-color: var(--primary-700);
    }
    
    .log-search { 
        background: var(--info-50); 
        color: var(--info-600); 
        border-left: 4px solid var(--info-500);
    }
    
    .log-search:hover {
        background: #e1f5fe;
        border-left-color: var(--info-600);
    }
    
    .log-strat { 
        background: #f3e5f5; 
        color: #6a1b9a; 
        border-left: 4px solid #7b1fa2;
    }
    
    .log-strat:hover {
        background: #ede7f6;
        border-left-color: #6a1b9a;
    }
    
    .log-calc { 
        background: var(--success-50); 
        color: var(--success-600); 
        border-left: 4px solid var(--success-500);
    }
    
    .log-calc:hover {
        background: #f1f8e9;
        border-left-color: var(--success-600);
    }
    
    .log-draft { 
        background: var(--warning-50); 
        color: var(--warning-600); 
        border-left: 4px solid var(--warning-500);
    }
    
    .log-draft:hover {
        background: #fff8e1;
        border-left-color: var(--warning-600);
    }
    
    .log-sys { 
        background: var(--neutral-50); 
        color: var(--neutral-700); 
        border-left: 4px solid var(--neutral-400);
    }
    
    .log-sys:hover {
        background: var(--neutral-100);
        border-left-color: var(--neutral-500);
    }

    /* ====================== */
    /* Spinner & Active Log Animation */
    /* ====================== */
    @keyframes spin { 
        0% { transform: rotate(0deg); } 
        100% { transform: rotate(360deg); } 
    }
    
    @keyframes pulse-active { 
        0%, 100% { 
            border-left-color: var(--primary-600);
            box-shadow: 0 0 0 0 rgba(30, 74, 122, 0.3);
        } 
        50% { 
            border-left-color: var(--primary-500);
            box-shadow: 0 0 0 4px rgba(30, 74, 122, 0.15);
        } 
    }

    .spinner-icon {
        display: inline-block;
        animation: spin 1.2s linear infinite;
        margin-right: 0.5rem;
        font-size: 1.1rem;
    }

    .log-active {
        animation: pulse-active 2s infinite;
        background: white !important;
        border-width: 2px !important;
        border-left-width: 4px !important;
        font-weight: 600 !important;
    }
</style>
"""


def apply_styles() -> None:
    """Streamlit 앱에 스타일 적용"""
    import streamlit as st
    st.set_page_config(**PAGE_CONFIG)
    st.markdown(MAIN_CSS, unsafe_allow_html=True)
    
    # 사이드바 드래그 리사이즈 JavaScript
    sidebar_resize_js = """
    <script>
    (function() {
        // 이미 초기화되었으면 스킵
        if (window.sidebarResizeInit) return;
        window.sidebarResizeInit = true;
        
        function initSidebarResize() {
            const sidebar = document.querySelector('[data-testid="stSidebar"]');
            if (!sidebar) {
                setTimeout(initSidebarResize, 500);
                return;
            }
            
            // 드래그 핸들 생성
            let handle = document.getElementById('sidebar-resize-handle');
            if (!handle) {
                handle = document.createElement('div');
                handle.id = 'sidebar-resize-handle';
                handle.style.cssText = `
                    position: absolute;
                    right: 0;
                    top: 0;
                    width: 8px;
                    height: 100%;
                    cursor: ew-resize;
                    background: linear-gradient(to right, transparent 0%, #e5e7eb 50%, transparent 100%);
                    z-index: 10000;
                    transition: background 0.2s;
                `;
                handle.onmouseenter = () => { handle.style.background = 'linear-gradient(to right, transparent 0%, #3b82f6 50%, transparent 100%)'; };
                handle.onmouseleave = () => { handle.style.background = 'linear-gradient(to right, transparent 0%, #e5e7eb 50%, transparent 100%)'; };
                sidebar.appendChild(handle);
            }
            
            let isResizing = false;
            let startX = 0;
            let startWidth = 0;
            
            handle.addEventListener('mousedown', (e) => {
                isResizing = true;
                startX = e.clientX;
                startWidth = sidebar.offsetWidth;
                document.body.style.cursor = 'ew-resize';
                document.body.style.userSelect = 'none';
                e.preventDefault();
            });
            
            document.addEventListener('mousemove', (e) => {
                if (!isResizing) return;
                const diff = e.clientX - startX;
                const newWidth = Math.max(60, Math.min(600, startWidth + diff));
                sidebar.style.width = newWidth + 'px';
                sidebar.style.minWidth = newWidth + 'px';
                sidebar.style.maxWidth = newWidth + 'px';
            });
            
            document.addEventListener('mouseup', () => {
                if (isResizing) {
                    isResizing = false;
                    document.body.style.cursor = '';
                    document.body.style.userSelect = '';
                }
            });
        }
        
        // DOM 로드 후 초기화
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initSidebarResize);
        } else {
            initSidebarResize();
        }
    })();
    </script>
    """
    st.markdown(sidebar_resize_js, unsafe_allow_html=True)
