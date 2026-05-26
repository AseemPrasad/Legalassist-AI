from string import Formatter
from typing import Dict, Set, Tuple, List
import logging


logger = logging.getLogger(__name__)


ALLOWED_VARS: Set[str] = {
    "case_title",
    "case_number",
    "deadline_date",
    "days_left",
    "court",
    "deadline_type",
    "deadline_description",
    "link",
}


class TemplateValidationError(Exception):
    pass


class _SafeFormatter(Formatter):
    """Formatter that blocks attribute/index traversal on field values."""

    def get_field(self, field_name, args, kwargs):
        if "." in field_name or "[" in field_name:
            raise TemplateValidationError(
                f"Attribute or index access is not allowed in templates: '{field_name}'"
            )
        return super().get_field(field_name, args, kwargs)


def _has_traversal(field_name: str) -> bool:
    return "." in field_name or "[" in field_name


def extract_placeholders(template: str) -> List[str]:
    fmt = Formatter()
    fields = []
    for literal_text, field_name, format_spec, conversion in fmt.parse(template):
        if field_name is not None and field_name != "":
            fields.append(field_name)
    return fields


def validate_template(template: str, allowed: Set[str] = ALLOWED_VARS) -> Tuple[bool, List[str]]:
    """Return (is_valid, unknown_vars)"""
    fields = extract_placeholders(template)
    traversal = [f for f in fields if _has_traversal(f)]
    if traversal:
        raise TemplateValidationError(
            f"Attribute or index access is not allowed in templates: {traversal}"
        )
    unknown = [f for f in fields if f not in allowed]
    return (len(unknown) == 0, unknown)


def render_template(template: str, values: Dict[str, str], allowed: Set[str] = ALLOWED_VARS, missing_as_empty: bool = True) -> str:
    """
    Render template with provided values.
    - Validates that all placeholders are in allowed set.
    - If missing_as_empty, missing keys are replaced with empty string; else raises TemplateValidationError.
    """
    is_valid, unknown = validate_template(template, allowed)
    if not is_valid:
        raise TemplateValidationError(f"Template contains unknown variables: {unknown}")

    # Prepare mapping for format_map; ensure only allowed keys present
    safe_map = {}
    for k in allowed:
        v = values.get(k)
        if v is None:
            if missing_as_empty:
                safe_map[k] = ""
            else:
                raise TemplateValidationError(f"Missing value for variable: {k}")
        else:
            safe_map[k] = str(v)

    # Use _SafeFormatter to block attribute/index traversal on field values
    try:
        rendered = _SafeFormatter().format(template, **safe_map)
    except TemplateValidationError:
        raise
    except Exception as e:
        logger.exception("Failed to render template with values=%s", safe_map)
        raise TemplateValidationError(f"Failed to render template: {e}") from e

    return rendered
