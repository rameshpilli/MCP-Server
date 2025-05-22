import os
from pathlib import Path
from dotenv import load_dotenv
import openai

def main():
    # Load .env from the project root
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

    # Get API key and model
    api_key = os.getenv("CHATGPT_API_KEY") or os.getenv("LLM_OPENAI_API_KEY")
    model = os.getenv("LLM_OPENAI_MODEL", "gpt-3.5-turbo")

    # Prepare output path
    output_dir = Path(__file__).resolve().parents[1] / "output"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "chatgpt_test_output.txt"

    if not api_key:
        msg = "API key not configured in .env"
        print(msg)
        output_file.write_text(msg)
        return

    # Initialize OpenAI client
    client = openai.OpenAI(api_key=api_key)

    try:
        # Call ChatGPT
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Tell me a joke"}],
        )
        text = response.choices[0].message.content.strip()
        print("Response received:\n", text)
    except Exception as e:
        text = f"Error calling API: {e}"
        print(text)

    # Write output
    output_file.write_text(text)

if __name__ == "__main__":
    main()