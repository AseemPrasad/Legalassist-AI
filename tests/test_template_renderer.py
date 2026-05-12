import pytest
from core.template_renderer import validate_template, render_template, TemplateValidationError


def test_validate_known_vars():
    tmpl = "Reminder: {case_title} - {days_left} days"
    ok, unknown = validate_template(tmpl)
    assert ok
    assert unknown == []


def test_validate_unknown_vars():
    tmpl = "Hello {bad_var}"
    ok, unknown = validate_template(tmpl)
    assert not ok
    assert "bad_var" in unknown


def test_render_template_success():
    tmpl = "{case_title} due {deadline_date} ({days_left} days)"
    vals = {"case_title": "A", "deadline_date": "12 May","days_left": 5}
    out = render_template(tmpl, vals)
    assert "A" in out
    assert "12 May" in out


def test_render_template_missing_raises():
    tmpl = "{case_title} {court}"
    vals = {"case_title": "A"}
    with pytest.raises(TemplateValidationError):
        render_template(tmpl, vals, missing_as_empty=False)


def test_render_template_unknown_raises():
    tmpl = "{case_title} {@danger}"
    vals = {"case_title": "A"}
    with pytest.raises(TemplateValidationError):
        render_template(tmpl, vals)
