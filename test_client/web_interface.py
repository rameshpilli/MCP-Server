from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from pathlib import Path
from test_llm import TestLLMClient
import json

app = FastAPI(title="Test LLM Web Interface")

# Create templates directory
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)

# Setup templates
templates = Jinja2Templates(directory=str(templates_dir))

# Initialize LLM client
client = TestLLMClient()

@app.on_event("startup")
async def startup_event():
    """Initialize the LLM client on startup"""
    if not client.api_key:
        client.get_api_key()
    if not client.model_id:
        client.register_model()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "messages": []}
    )

@app.post("/query")
async def query(request: Request, prompt: str = Form(...)):
    try:
        # Get context data
        context_data = client.query_data("SELECT * FROM cat_jobs LIMIT 5")
        
        # Generate response
        response = client.generate_response(prompt, context_data)
        
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response["generated_text"]},
                ],
                "tokens": response["tokens_used"]
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": str(e)
            }
        )

def create_template():
    """Create the HTML template"""
    template = """
<!DOCTYPE html>
<html>
<head>
    <title>Test LLM Interface</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .chat-container {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 5px;
        }
        .user-message {
            background-color: #e3f2fd;
            margin-left: 20%;
        }
        .assistant-message {
            background-color: #f5f5f5;
            margin-right: 20%;
        }
        .input-container {
            display: flex;
            gap: 10px;
        }
        input[type="text"] {
            flex-grow: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        button {
            padding: 10px 20px;
            background-color: #2196f3;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #1976d2;
        }
        .error {
            color: red;
            margin-bottom: 10px;
        }
        .tokens {
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <h1>Test LLM Interface</h1>
    
    <div class="chat-container">
        {% if messages %}
            {% for message in messages %}
                <div class="message {% if message.role == 'user' %}user-message{% else %}assistant-message{% endif %}">
                    <strong>{% if message.role == 'user' %}You{% else %}LLM{% endif %}:</strong>
                    <p>{{ message.content }}</p>
                </div>
            {% endfor %}
            {% if tokens %}
                <div class="tokens">Tokens used: {{ tokens }}</div>
            {% endif %}
        {% endif %}
        
        {% if error %}
            <div class="error">Error: {{ error }}</div>
        {% endif %}
    </div>

    <form action="/query" method="post">
        <div class="input-container">
            <input type="text" name="prompt" placeholder="Enter your question..." required>
            <button type="submit">Send</button>
        </div>
    </form>
</body>
</html>
"""
    
    with open(templates_dir / "index.html", "w") as f:
        f.write(template)

def main():
    # Create template file
    create_template()
    
    # Run the server
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        log_level="info"
    )

if __name__ == "__main__":
    main() 