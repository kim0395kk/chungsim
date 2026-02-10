# Govable AI - AI 행정관 Pro

Streamlit 기반의 AI 행정 지원 시스템입니다.

## 설치 및 실행

1. **필수 요구사항**
   - Python 3.9 이상
   - Streamlit

2. **설치**
   ```bash
   pip install -r requirements.txt
   ```

3. **실행**
   ```bash
   streamlit run streamlit_app.py
   ```

## 환경 변수 설정 (Secrets)

`.streamlit/secrets.toml` 파일에 다음 API 키들을 설정해야 합니다:

```toml
[general]
GEMINI_API_KEY = "..."
GROQ_API_KEY = "..."
# ... 기타 키들

[supabase]
SUPABASE_URL = "..."
SUPABASE_KEY = "..."
```

**주의:** 보안을 위해 `secrets.toml` 파일은 GitHub에 업로드하지 마세요. Streamlit Cloud 배포 시에는 **Advanced Settings > Secrets** 메뉴에 내용을 복사하여 붙여넣으세요.
