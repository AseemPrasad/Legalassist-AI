import os
import pytest

pytestmark = pytest.mark.contract


def _has_api_key() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"))


@pytest.mark.skipif(not _has_api_key(), reason="No OPENAI/OPENROUTER API key configured")
def test_openai_client_contract():
    """Contract test: ensure OpenAI/OpenRouter client returns expected fields.

    This test only runs when an API key is available (protected in CI by secrets).
    It verifies the live client response contains essential structural fields so
    higher-level code can rely on them.
    """
    try:
        from cli_client import get_client
        from core.app_utils import get_default_model
    except Exception as e:
        pytest.skip(f"Client modules not importable: {e}")

    client = get_client()
    assert client is not None, "LLM client initialization failed"

    model = os.getenv("DEFAULT_MODEL") or get_default_model()
    # Small, cheap prompt to validate API schema
    system = "You are a helpful assistant for contract testing."
    user = "Say hello in one short sentence."

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=16,
        temperature=0.0,
    )

    # Most OpenAI/OpenRouter responses contain 'choices' and 'usage'
    assert hasattr(resp, "choices") or (isinstance(resp, dict) and "choices" in resp), "Missing 'choices' in LLM response"
    # Validate a textual payload exists in the first choice
    first = resp.choices[0] if hasattr(resp, "choices") else resp["choices"][0]
    text = getattr(first, "message", None) or first.get("message") or getattr(first, "text", None) or first.get("text")
    assert text, "LLM response choice did not include text/message"

    # Usage information should be present
    usage = getattr(resp, "usage", None) or (resp.get("usage") if isinstance(resp, dict) else None)
    assert usage is not None, "Missing 'usage' in LLM response"