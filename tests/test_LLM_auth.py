import os
from pathlib import Path
import pytest
from dotenv import load_dotenv
import openai


def test_chatgpt_connection():
    """Attempt to call OpenAI's API and save the response or error."""
    load_dotenv()
    api_key = os.getenv("CHATGPT_API_KEY") or os.getenv("LLM_OPENAI_API_KEY")
    model = os.getenv("LLM_OPENAI_MODEL", "gpt-3.5-turbo")

    output_dir = Path(__file__).resolve().parents[1] / "output"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "chatgpt_test_output.txt"

    if not api_key:
        output_file.write_text("API key not configured")
        pytest.skip("CHATGPT_API_KEY not configured")

    client = openai.OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Tell me a joke"}],
        )
        text = response.choices[0].message.content
    except Exception as e:
        text = f"Error calling API: {e}"

    output_file.write_text(text)
    assert "Error" not in text, text
