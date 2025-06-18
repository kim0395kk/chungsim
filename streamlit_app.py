import streamlit as st
import streamlit.components.v1 as components
import os

st.set_page_config(page_title="충심이 퀴즈 & 굿즈 신청", layout="wide")

st.title("충주시 굿즈/퀴즈 신청 시스템 (Streamlit UI)")

# index.html 경로 기준: 같은 폴더에 있는 경우
file_path = os.path.join(os.path.dirname(__file__), "index.html")

# 파일 존재 여부 확인
if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        html_data = f.read()
    components.html(html_data, height=1000, scrolling=True)
else:
    st.error("index.html 파일이 없습니다. GitHub에 업로드했는지 확인해주세요.")
