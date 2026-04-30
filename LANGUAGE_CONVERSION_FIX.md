# Language Conversion Fix - Summary Page Headings & UI Text

## Problem Identified
When users selected any language other than English, the summary page headings and UI text (like "What Happened?", "Can You Appeal?", etc.) were not being converted to the selected language. Only the summary content itself and Tamil were working correctly.

### Root Cause
The issue was in how the UI text was being localized:

1. **app_integrated.py** (line 156) and **pages/0_Home.py** (line 42) were calling `get_localized_ui_text(language)` **WITHOUT the client parameter**
2. The `get_localized_ui_text()` function requires a client to dynamically translate missing UI text for non-hardcoded languages
3. Without the client, only hardcoded translations (Tamil) would work
4. Other Indian languages have NO hardcoded translations, so they would fall back to English text

### Why Only Tamil Worked
- Tamil has hardcoded translations in `UI_TEXT_TRANSLATIONS["Tamil"]` dictionary
- All other Indian languages (Hindi, Telugu, Kannada, etc.) rely on dynamic LLM-based translation
- Dynamic translation requires the OpenAI client to be passed to `get_localized_ui_text()`

## Solution Applied

### Files Modified
1. **app_integrated.py** - `show_judgment_analysis()` function
2. **pages/0_Home.py** - `render_page()` function

### Key Changes

#### 1. Get Client Early
```python
# Get client early (uses @st.cache_resource, so it's fast)
from app import get_client
client = get_client()
```

#### 2. Pass Client to All get_localized_ui_text() Calls
```python
# Before (❌ No translation for non-Tamil languages)
ui = core.get_localized_ui_text(language)

# After (✅ Dynamic translation works)
ui = core.get_localized_ui_text(language, client)
```

#### 3. Pass Client to get_remedies_advice()
```python
# This ensures remedies are in the selected language too
remedies = get_remedies_advice(raw_text, language, client)
```

### Changes in app_integrated.py
- Line 145-146: Get client at function start
- Line 147: Pass client to get_localized_ui_text (initial language)
- Line 156: Pass client to get_localized_ui_text (selected language)
- Line 218: Pass client to get_remedies_advice()

### Changes in pages/0_Home.py
- Line 28: Get client at function start
- Line 35: Pass client to get_localized_ui_text (initial language)
- Line 45: Pass client to get_localized_ui_text (selected language)

## How It Works Now

### For Tamil
- Uses hardcoded translations from `UI_TEXT_TRANSLATIONS["Tamil"]`
- No LLM calls needed (fast)

### For Other Languages (Hindi, Telugu, Kannada, etc.)
1. First request gets client and localized UI text
2. `get_localized_ui_text()` checks if all keys have translations
3. For missing keys, it calls `_translate_ui_text()` which:
   - Creates a JSON with untranslated keys
   - Sends to LLM with instruction to translate to target language
   - Returns translated JSON
4. Results are cached in `_LOCALIZED_UI_TEXT_CACHE` for reuse

## Benefits
✅ All Indian languages now have translated UI headings  
✅ Tamil maintains fast response (hardcoded translations)  
✅ Other languages get LLM-translated headings and labels  
✅ No additional API calls per page load (client is cached)  
✅ Consistent experience across all UI elements  

## Testing
To verify the fix:
1. Run `streamlit run app_integrated.py`
2. Select any language (Hindi, Tamil, Telugu, etc.)
3. Upload a PDF
4. Generate summary
5. Check that ALL headings and labels are in the selected language:
   - "⚖️ What Can You Do Now?" heading
   - "What Happened?" subheading
   - "Can You Appeal?" subheading
   - "Appeal Details" subheading
   - All metric labels and box titles

## Performance Impact
- **Minimal**: Client initialization is cached using `@st.cache_resource`
- **First load**: May include one LLM call to translate missing UI keys (happens once, then cached)
- **Subsequent loads**: Uses cached translations, no additional API calls
- **API Cost**: Only translation calls for new languages (not per page view)

## Future Improvements
1. Add hardcoded translations for frequently used languages (Hindi, Telugu, Kannada)
2. Pre-populate translation cache on app startup for all languages
3. Build translation dictionary for all Indian languages gradually
