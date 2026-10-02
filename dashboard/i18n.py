import os
import json
import streamlit as st

# Path to the language directory
LANG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "language")

@st.cache_data
def load_translations():
    """
    Loads English and Bangla JSON translation files from the language directory.
    Returns a dictionary of languages.
    """
    translations = {}
    for lang in ["en", "bn"]:
        file_path = os.path.join(LANG_DIR, f"{lang}.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                translations[lang] = json.load(f)
        else:
            translations[lang] = {}
    return translations

def get_text(key: str) -> str:
    """
    Retrieves the translated text for a given key based on the current language 
    in Streamlit session state (st.session_state.lang). Defaults to English if key is missing.
    """
    translations = load_translations()
    lang = st.session_state.get("lang", "en")
    
    # Fallback to English if translation is missing in Bangla
    if lang not in translations or key not in translations[lang]:
        return translations["en"].get(key, key)
    
    return translations[lang].get(key, translations["en"].get(key, key))
