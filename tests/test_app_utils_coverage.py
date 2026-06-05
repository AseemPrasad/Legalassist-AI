"""
Targeted coverage tests for core/app_utils.py uncovered lines.
Focuses on: config helpers, validate_pdf_metadata, build_judgment_result_text,
HTML rendering helpers, localize_yes_no, parse_summary_bullets, and edge cases.
"""

import io
import json
import sys
import types
import pytest
from unittest.mock import MagicMock, patch
from pypdf import PdfWriter

import core.app_utils as app_utils
from core.app_utils import (
    validate_pdf_metadata,
    output_language_mismatch_detected,
    _count_script_chars,
    build_judgment_result_text,
    _format_result_paragraph,
    _build_qa_group_html,
    _build_legal_help_group_html,
    _build_result_body_html,
    render_shareable_result_box,
    localize_yes_no,
    _normalize_yes_no,
    _plain_text_from_markdown,
    _normalize_bullet_lines,
    parse_summary_bullets,
    _parse_json_object,
    _is_untranslated_ui_value,
    get_localized_ui_text,
    extract_appeal_info,
    UI_TEXT,
    LANGUAGE_OUTPUT_RULES,
    _language_output_rule,
    get_default_model,
    DEFAULT_MODEL,
)


# ==================== VALIDATE PDF METADATA ====================

class TestValidatePdfMetadata:
    def _make_pdf(self, pages=1):
        writer = PdfWriter()
        for _ in range(pages):
            writer.add_blank_page(width=200, height=200)
        buf = io.BytesIO()
        writer.write(buf)
        buf.seek(0)
        return buf

    def test_none_file_is_valid(self):
        valid, msg, level = validate_pdf_metadata(None)
        assert valid is True
        assert msg is None

    def test_normal_pdf_is_valid(self):
        pdf = self._make_pdf(1)
        pdf.size = 1024
        valid, msg, level = validate_pdf_metadata(pdf)
        assert valid is True

    def test_large_file_warning(self):
        pdf = self._make_pdf(1)
        pdf.size = 26 * 1024 * 1024  # 26 MB
        valid, msg, level = validate_pdf_metadata(pdf)
        assert valid is True
        assert level == "warning"
        assert "large" in msg.lower()

    def test_corrupted_pdf_returns_error(self):
        bad = io.BytesIO(b"not a pdf")
        bad.size = 100
        valid, msg, level = validate_pdf_metadata(bad)
        assert valid is False
        assert level == "error"


# ==================== OUTPUT LANGUAGE MISMATCH ====================

class TestOutputLanguageMismatch:
    def test_english_language_never_mismatch(self):
        assert output_language_mismatch_detected("Some English text", "English") is False

    def test_none_text_no_mismatch(self):
        assert output_language_mismatch_detected(None, "Hindi") is False

    def test_none_language_no_mismatch(self):
        assert output_language_mismatch_detected("Some text", None) is False

    def test_wrong_script_detected(self):
        # Tamil text for a language expecting Devanagari
        tamil = "தீர்ப்பு தமிழில் உள்ளது."
        assert output_language_mismatch_detected(tamil, "Hindi") is True

    def test_correct_script_not_mismatch(self):
        devanagari = "यह निर्णय हिंदी में है।"
        assert output_language_mismatch_detected(devanagari, "Hindi") is False

    def test_count_script_chars(self):
        devanagari_text = "यह"
        count = _count_script_chars(devanagari_text, {"Devanagari"})
        assert count == 2


# ==================== LANGUAGE OUTPUT RULE ====================

class TestLanguageOutputRule:
    def test_english_returns_clear_english(self):
        rule = _language_output_rule("English")
        assert "English" in rule

    def test_none_returns_clear_english(self):
        rule = _language_output_rule(None)
        assert "English" in rule

    def test_known_language_returns_specific_rule(self):
        rule = _language_output_rule("Hindi")
        assert "Hindi" in rule
        assert "Devanagari" in rule

    def test_unknown_language_returns_generic(self):
        rule = _language_output_rule("Klingon")
        assert "Klingon" in rule


# ==================== NORMALIZE YES/NO ====================

class TestNormalizeYesNo:
    def test_yes_variants(self):
        assert _normalize_yes_no("yes") == "yes"
        assert _normalize_yes_no("Yes, they can appeal") == "yes"
        assert _normalize_yes_no("can appeal") == "yes"

    def test_no_variants(self):
        assert _normalize_yes_no("no") == "no"
        assert _normalize_yes_no("cannot appeal") == "no"
        assert _normalize_yes_no("no right to appeal") == "no"

    def test_empty_returns_empty(self):
        assert _normalize_yes_no("") == ""
        assert _normalize_yes_no(None) == ""

    def test_ambiguous_returns_empty(self):
        result = _normalize_yes_no("maybe")
        assert result == ""

    def test_localize_yes_no_yes(self):
        ui = {"yes": "हाँ", "no": "नहीं"}
        assert localize_yes_no("yes", ui) == "हाँ"

    def test_localize_yes_no_no(self):
        ui = {"yes": "हाँ", "no": "नहीं"}
        assert localize_yes_no("no", ui) == "नहीं"

    def test_localize_yes_no_passthrough(self):
        ui = {"yes": "Yes", "no": "No"}
        assert localize_yes_no("maybe", ui) == "maybe"


# ==================== BUILD JUDGMENT RESULT TEXT ====================

class TestBuildJudgmentResultText:
    def _full_remedies(self):
        return {
            "what_happened": "Plaintiff won.",
            "can_appeal": "yes",
            "appeal_days": "30",
            "appeal_court": "High Court",
            "cost": "5000-10000",
            "cost_estimate": "5000-10000",
            "first_action": "File certified copy.",
            "deadline": "30 days from judgment.",
            "_is_partial": False,
            "_warning": "",
        }

    def test_returns_tuple(self):
        result = build_judgment_result_text("- Point 1\n- Point 2", self._full_remedies(), UI_TEXT)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_plain_text_contains_summary(self):
        plain, _ = build_judgment_result_text("- Point 1", self._full_remedies(), UI_TEXT)
        assert "Point 1" in plain

    def test_structured_has_qa_pairs(self):
        _, structured = build_judgment_result_text("- Point 1", self._full_remedies(), UI_TEXT)
        assert "qa_pairs" in structured
        assert len(structured["qa_pairs"]) > 0

    def test_structured_summary_title(self):
        _, structured = build_judgment_result_text("- Point 1", self._full_remedies(), UI_TEXT)
        assert structured["summary_title"] != ""

    def test_partial_remedies_warning(self):
        remedies = {"_is_partial": True, "_warning": "Incomplete", "what_happened": ""}
        plain, structured = build_judgment_result_text("Summary", remedies, UI_TEXT)
        assert structured["partial_warning"] == "Incomplete"

    def test_no_remedies_uses_defaults(self):
        plain, structured = build_judgment_result_text("Summary", None, UI_TEXT)
        assert isinstance(plain, str)
        assert isinstance(structured, dict)

    def test_no_appeal_skips_appeal_details(self):
        remedies = {**self._full_remedies(), "can_appeal": "no"}
        _, structured = build_judgment_result_text("Summary", remedies, UI_TEXT)
        appeal_qa = [p for p in structured["qa_pairs"] if "Appeal" in p.get("question", "")]
        # When can_appeal is "no", no appeal detail lines should appear
        for qa in appeal_qa:
            assert "Days to File" not in qa["answer"]

    def test_legal_help_in_plain_text(self):
        plain, _ = build_judgment_result_text("Summary", self._full_remedies(), UI_TEXT)
        assert "nalsa" in plain.lower() or "legal" in plain.lower()


# ==================== HTML RENDERING HELPERS ====================

class TestFormatResultParagraph:
    def test_bullet_list_renders_ol(self):
        html = _format_result_paragraph("- Item one\n- Item two\n- Item three")
        assert "<ol" in html
        assert "<li>" in html

    def test_plain_paragraph_renders_p(self):
        html = _format_result_paragraph("This is a plain sentence.")
        assert "<p>" in html

    def test_empty_returns_empty(self):
        assert _format_result_paragraph("") == ""
        assert _format_result_paragraph(None) == ""

    def test_bold_key_value_rendered(self):
        html = _format_result_paragraph("Court: High Court")
        assert "<strong>" in html


class TestBuildQaGroupHtml:
    def test_renders_question_and_answer(self):
        pairs = [{"question": "What happened?", "answer": "Plaintiff won."}]
        html = _build_qa_group_html("Remedies", pairs)
        assert "What happened?" in html
        assert "Plaintiff won." in html

    def test_empty_pairs_returns_empty(self):
        assert _build_qa_group_html("", []) == ""

    def test_partial_warning_rendered(self):
        html = _build_qa_group_html("Title", [], partial_warning="Incomplete data")
        assert "Incomplete data" in html

    def test_modifier_class_applied(self):
        pairs = [{"question": "Q", "answer": "A"}]
        html = _build_qa_group_html("Title", pairs, modifier="remedies-group")
        assert "remedies-group" in html


class TestBuildLegalHelpGroupHtml:
    def test_renders_title(self):
        html = _build_legal_help_group_html("Free Legal Help", ["Intro text", "Resource 1\nDetail"])
        assert "Free Legal Help" in html

    def test_empty_title_returns_empty(self):
        assert _build_legal_help_group_html("", []) == ""

    def test_renders_resource_cards(self):
        html = _build_legal_help_group_html("Help", ["Intro", "NALSA\nPhone: 1800-180-8111"])
        assert "NALSA" in html


class TestBuildResultBodyHtml:
    def _extract_result_content(self, html):
        start = html.index('<main class="result-content">') + len('<main class="result-content">')
        end = html.index('</main>', start)
        return html[start:end]

    def test_with_structured_dict(self):
        structured = {
            "summary_title": "Simplified Judgment",
            "summary": "- Point 1\n- Point 2",
            "remedies_title": "What Can You Do?",
            "qa_pairs": [{"question": "What happened?", "answer": "Plaintiff won."}],
            "partial_warning": "",
            "free_legal_help_title": "Free Legal Help",
            "legal_help_resources": "Contact NALSA\n\nPhone: 1800-180-8111",
        }
        html = _build_result_body_html("ignored", UI_TEXT, structured=structured)
        assert "Simplified Judgment" in html
        assert "What happened?" in html

    def test_legacy_fallback_plain_text(self):
        plain = "Title\n\nSummary paragraph\n\nRemedies Title\n\nWhat happened?\n\nPlaintiff won."
        html = _build_result_body_html(plain, UI_TEXT, structured=None)
        assert "Title" in html

    def test_empty_structured_summary(self):
        structured = {
            "summary_title": "Title",
            "summary": "",
            "remedies_title": "Remedies",
            "qa_pairs": [],
            "partial_warning": "",
            "free_legal_help_title": "",
            "legal_help_resources": "",
        }
        html = _build_result_body_html("", UI_TEXT, structured=structured)
        assert "Title" in html

    def test_structured_html_escapes_malicious_content(self):
        structured = {
            "summary_title": "Simplified Judgment",
            "summary": "<script>alert('x')</script>",
            "remedies_title": "What Can You Do?",
            "qa_pairs": [
                {
                    "question": "What happened?",
                    "answer": "File <img src=x onerror=alert('x')> immediately.",
                },
                {
                    "question": "First action",
                    "answer": "<b>Contact counsel</b>",
                },
            ],
            "partial_warning": "",
            "free_legal_help_title": "Free Legal Help",
            "legal_help_resources": "Contact NALSA",
        }

        html = _build_result_body_html("ignored", UI_TEXT, structured=structured)

        assert "<script>" not in html
        assert "<img src=x onerror=alert('x')>" not in html
        assert "&lt;script&gt;" in html
        assert "&lt;img src=x onerror=alert" in html
        assert "&lt;b&gt;Contact counsel&lt;/b&gt;" in html

    def test_render_shareable_result_box_escapes_malicious_structured_content(self, monkeypatch):
        captured = {}

        fake_components = types.ModuleType("streamlit.components.v1")

        def fake_html(html, height=None, scrolling=False):
            captured["html"] = html
            captured["height"] = height
            captured["scrolling"] = scrolling

        fake_components.html = fake_html

        fake_components_pkg = types.ModuleType("streamlit.components")
        fake_streamlit = types.ModuleType("streamlit")
        fake_streamlit.components = fake_components_pkg
        fake_components_pkg.v1 = fake_components

        monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
        monkeypatch.setitem(sys.modules, "streamlit.components", fake_components_pkg)
        monkeypatch.setitem(sys.modules, "streamlit.components.v1", fake_components)

        structured = {
            "summary_title": "Simplified Judgment",
            "summary": "<script>alert('x')</script>",
            "remedies_title": "What Can You Do?",
            "qa_pairs": [
                {
                    "question": "First action",
                    "answer": "<img src=x onerror=alert('x')>",
                },
            ],
            "partial_warning": "",
            "free_legal_help_title": "Free Legal Help",
            "legal_help_resources": "Contact NALSA",
        }

        render_shareable_result_box(("Plain text", structured), UI_TEXT)

        html = captured["html"]
        start = html.index('<main class="result-content">') + len('<main class="result-content">')
        end = html.index('</main>', start)
        content = html[start:end]

        assert "<script>" not in content
        assert "<img src=x onerror=alert('x')>" not in content
        assert "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;" in content
        assert "&lt;img src=x onerror=alert(&#x27;x&#x27;)&gt;" in content


# ==================== PLAIN TEXT & BULLET HELPERS ====================

class TestPlainTextHelpers:
    def test_plain_text_from_markdown_strips_bold(self):
        result = _plain_text_from_markdown("**Bold text** here")
        assert "**" not in result
        assert "Bold text" in result

    def test_normalize_bullet_lines_handles_inline_bullets(self):
        text = "Point one * Point two * Point three"
        result = _normalize_bullet_lines(text)
        assert isinstance(result, str)

    def test_normalize_bullet_lines_empty(self):
        assert _normalize_bullet_lines("") == ""
        assert _normalize_bullet_lines(None) == ""


class TestParseSummaryBullets:
    def test_extracts_three_bullets(self):
        raw = "- First point\n- Second point\n- Third point\n- Fourth point"
        result = parse_summary_bullets(raw)
        lines = [l for l in result.split("\n") if l.strip()]
        assert len(lines) == 3

    def test_handles_numbered_bullets(self):
        raw = "1. First\n2. Second\n3. Third"
        result = parse_summary_bullets(raw)
        assert "First" in result

    def test_empty_input_returns_empty(self):
        assert parse_summary_bullets("") == ""
        assert parse_summary_bullets(None) == ""

    def test_fallback_for_unstructured_text(self):
        raw = "This is a long sentence about the case outcome and what happened next in the proceedings."
        result = parse_summary_bullets(raw)
        assert isinstance(result, str)

    def test_filters_intro_phrases(self):
        raw = "Here is the summary:\n- The plaintiff won.\n- Appeal is possible.\n- File within 30 days."
        result = parse_summary_bullets(raw)
        assert "Here is the summary" not in result


# ==================== PARSE JSON OBJECT ====================

class TestParseJsonObject:
    def test_valid_json(self):
        result = _parse_json_object('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_with_code_fence(self):
        result = _parse_json_object('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_invalid_json_returns_empty(self):
        result = _parse_json_object("not json at all")
        assert result == {}

    def test_empty_returns_empty(self):
        assert _parse_json_object("") == {}
        assert _parse_json_object(None) == {}

    def test_embedded_json_extracted(self):
        result = _parse_json_object('Some text {"key": "val"} more text')
        assert result.get("key") == "val"


# ==================== UI TEXT HELPERS ====================

class TestUiTextHelpers:
    def test_is_untranslated_returns_true_for_english_copy(self):
        key = "yes"
        assert _is_untranslated_ui_value(key, UI_TEXT[key]) is True

    def test_is_untranslated_returns_false_for_translated(self):
        assert _is_untranslated_ui_value("yes", "हाँ") is False

    def test_is_untranslated_returns_false_for_unknown_key(self):
        assert _is_untranslated_ui_value("nonexistent_key", "value") is False

    def test_get_localized_ui_text_english_returns_ui_text(self):
        result = get_localized_ui_text("English")
        assert result["yes"] == UI_TEXT["yes"]

    def test_get_localized_ui_text_none_returns_ui_text(self):
        result = get_localized_ui_text(None)
        assert result["yes"] == UI_TEXT["yes"]

    def test_get_localized_ui_text_hindi_no_client(self):
        # Without a client, should still return base UI_TEXT keys
        result = get_localized_ui_text("Hindi")
        assert "yes" in result

    def test_get_localized_ui_text_with_mock_client_caches(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({"yes": "हाँ", "no": "नहीं"})
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        # Clear cache for Marathi to force a translation call
        app_utils._LOCALIZED_UI_TEXT_CACHE.pop("Marathi", None)
        result = get_localized_ui_text("Marathi", mock_client)
        assert "yes" in result


# ==================== EXTRACT APPEAL INFO EDGE CASES ====================

class TestExtractAppealInfoEdgeCases:
    def test_none_input(self):
        info = extract_appeal_info(None)
        assert info["days"] == ""
        assert info["court"] == ""
        assert info["cost"] == ""

    def test_supreme_court_extracted(self):
        info = extract_appeal_info("File in Supreme Court within 90 days")
        assert "Supreme Court" in info["court"]
        assert info["days"] == "90"

    def test_cost_with_commas(self):
        info = extract_appeal_info("Cost: ₹10,000-₹25,000")
        assert "10" in info["cost"]
