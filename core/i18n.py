"""
Internationalization (i18n) utilities for multi-language support.

Provides centralized text retrieval and locale management for the application.
"""

from typing import Dict, Optional
from pathlib import Path
import json


class I18n:
    """Internationalization helper for centralized text retrieval."""
    
    _instance: Optional["I18n"] = None
    _translations: Dict[str, Dict[str, str]] = {}
    _current_locale: str = "en"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_translations(self, translations_path: Path) -> None:
        """Load translations from JSON file."""
        if translations_path.exists():
            with open(translations_path, "r", encoding="utf-8") as f:
                self._translations = json.load(f)
    
    def set_locale(self, locale: str) -> None:
        """Set current locale for text retrieval."""
        self._current_locale = locale
    
    def get_locale(self) -> str:
        """Get current locale."""
        return self._current_locale
    
    def t(self, key: str, locale: Optional[str] = None) -> str:
        """Translate key to current or specified locale."""
        lang = locale or self._current_locale
        if lang not in self._translations:
            lang = "en"
        return self._translations.get(lang, {}).get(key, key)
    
    def get_available_locales(self) -> list:
        """Get list of available locale codes."""
        return list(self._translations.keys())


# Global i18n instance
i18n = I18n()


def init_i18n(translations_path: Optional[Path] = None) -> I18n:
    """Initialize i18n with translations file."""
    if translations_path is None:
        translations_path = Path(__file__).parent / "all_translations.json"
    i18n.load_translations(translations_path)
    return i18n


def _(key: str, locale: Optional[str] = None) -> str:
    """Shorthand translation function."""
    return i18n.t(key, locale)