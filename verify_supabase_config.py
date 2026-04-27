import os
import streamlit as st
from govable_ai.config import get_supabase_config

# Mock streamlit secrets if running outside of streamlit (though we want to test with secrets.toml)
# But streamlit secrets are only available when running via `streamlit run`.
# However, `govable_ai.config._load_streamlit_secrets` attempts to load them.
# If I run this script with `streamlit run verify_supabase_config.py`, it should work.

def main():
    print("Checking Supabase Config...")
    config = get_supabase_config()
    if config:
        print(f"Success! URL: {config.get('url')}")
        # Build masked key for display
        key = config.get('anon_key')
        if key:
            masked_key = key[:5] + "..." + key[-5:]
            print(f"Success! Key: {masked_key}")
        else:
            print("Error: Key is missing.")
    else:
        print("Error: Could not load Supabase config.")

if __name__ == "__main__":
    main()
