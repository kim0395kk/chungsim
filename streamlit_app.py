# -*- coding: utf-8 -*-
"""
Govable AI — Streamlit 진입점

이 파일은 진입점 shim 입니다. 실제 로직은 ``govable_ai.main`` 에 있습니다.

Streamlit Cloud / 로컬 실행 모두 ``streamlit run streamlit_app.py`` 로 시작하므로,
이 파일은 의도적으로 얇게 유지하고 비즈니스 로직은 패키지 내부로 흡수합니다.

레거시 모놀리스(5106줄)는 ``streamlit_app_legacy.py`` 에 보존되어 있고,
revision/complaint_analyzer/hallucination_check/civil_engineering 모드는
점진적으로 ``govable_ai/ui/pages/*`` 로 이주 예정입니다.
"""
from govable_ai.main import main


if __name__ == "__main__":
    main()
else:
    # Streamlit 은 모듈을 import 하듯 실행하므로, top-level 에서 main 을 호출한다.
    main()
