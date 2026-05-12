from string import Formatter
from typing import Dict, Set, Tuple, List


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


def extract_placeholders(template: str) -> List[str]:
    fmt = Formatter()
    fields = []
    for literal_text, field_name, format_spec, conversion in fmt.parse(template):
        if field_name is not None and field_name != "":
            # Field name may contain indexing like a[0], strip to base name
            base = field_name.split(".")[0].split("[")[0]
            fields.append(base)
    return fields


def validate_template(template: str, allowed: Set[str] = ALLOWED_VARS) -> Tuple[bool, List[str]]:
    """Return (is_valid, unknown_vars)"""
    fields = extract_placeholders(template)
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

    # Use format_map to avoid KeyError on missing keys
    try:
        rendered = template.format_map(safe_map)
    except Exception as e:
        raise TemplateValidationError(f"Failed to render template: {e}")

    return rendered
