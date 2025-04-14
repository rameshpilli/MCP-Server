"""LLM UI endpoints for model interaction and data source management."""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import json
import asyncio
from sqlalchemy import text

from app.core.database import get_db
from app.core.models import ModelRecord, APIKey, DataSource
from app.core.auth import get_current_api_key
from app.core.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Create two routers - one for UI (no auth) and one for API (with auth)
router = APIRouter(prefix="/llm", tags=["llm"])

# Create a separate router for protected API endpoints
api_router = APIRouter(
    prefix="/llm",
    tags=["llm-api"],
    dependencies=[Depends(get_current_api_key)]
)

@router.get("/ui", response_class=HTMLResponse)
async def llm_dashboard(request: Request):
    """Render the LLM dashboard."""
    return """
    <html>
        <head>
            <title>LLM Dashboard - MCP Server</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
                .dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .model-list { margin-top: 20px; }
                .model-item { padding: 10px; border-bottom: 1px solid #eee; }
                .chat-interface { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-top: 20px; }
                textarea { width: 100%; padding: 10px; margin: 10px 0; }
                button { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
                button:hover { background: #2980b9; }
                .template-selector { margin: 10px 0; }
                .examples-container { margin: 10px 0; }
                .example-pair { display: flex; gap: 10px; margin: 5px 0; }
                .example-pair input { flex: 1; }
                .streaming { color: #666; font-style: italic; }
                .system-prompt { width: 100%; margin: 10px 0; }
                .tabs { display: flex; gap: 10px; margin-bottom: 10px; }
                .tab { padding: 10px; cursor: pointer; border: 1px solid #ddd; border-radius: 4px; }
                .tab.active { background: #3498db; color: white; }
                .tab-content { display: none; }
                .tab-content.active { display: block; }
                .parameter-input { margin: 5px 0; }
                .response-container { 
                    max-height: 400px; 
                    overflow-y: auto; 
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    margin-top: 10px;
                }
                .chat-message {
                    margin: 10px 0;
                    padding: 10px;
                    border-radius: 4px;
                }
                .user-message {
                    background: #e3f2fd;
                    margin-left: 20px;
                }
                .model-message {
                    background: #f5f5f5;
                    margin-right: 20px;
                }
                .error-message {
                    color: #d32f2f;
                    background: #ffebee;
                    padding: 10px;
                    border-radius: 4px;
                    margin: 10px 0;
                }
                .modal {
                    display: none;
                    position: fixed;
                    z-index: 1;
                    left: 0;
                    top: 0;
                    width: 100%;
                    height: 100%;
                    background-color: rgba(0,0,0,0.4);
                }
                
                .modal-content {
                    background-color: #fefefe;
                    margin: 5% auto;
                    padding: 20px;
                    border: 1px solid #888;
                    width: 80%;
                    max-width: 600px;
                    border-radius: 8px;
                }
                
                .form-group {
                    margin-bottom: 15px;
                }
                
                .form-group label {
                    display: block;
                    margin-bottom: 5px;
                }
                
                .form-group input,
                .form-group select,
                .form-group textarea {
                    width: 100%;
                    padding: 8px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }
                
                .button-group {
                    display: flex;
                    gap: 10px;
                    margin-top: 20px;
                }
                
                .success-message {
                    color: #2e7d32;
                    background: #e8f5e9;
                    padding: 10px;
                    border-radius: 4px;
                    margin: 10px 0;
                }
            </style>
        </head>
        <body>
            <h1>LLM Dashboard</h1>
            
            <div class="dashboard">
                <div class="card">
                    <h2>Registered Models</h2>
                    <div class="model-list" id="modelList">
                        Loading models...
                    </div>
                    <button onclick="showRegistrationForm()" style="margin-top: 10px;">Register New Model</button>
                </div>
            </div>
            
            <div class="chat-interface">
                <h2>Model Interaction</h2>
                
                <div class="tabs">
                    <div class="tab active" onclick="switchTab('basic')">Basic Chat</div>
                    <div class="tab" onclick="switchTab('advanced')">Advanced Options</div>
                    <div class="tab" onclick="switchTab('templates')">Templates</div>
                </div>

                <div id="basicTab" class="tab-content active">
                    <select id="modelSelect">
                        <option value="">Select a model...</option>
                    </select>
                    <textarea id="prompt" rows="4" placeholder="Enter your prompt here..."></textarea>
                </div>

                <div id="advancedTab" class="tab-content">
                    <textarea id="systemPrompt" class="system-prompt" rows="3" 
                        placeholder="Enter system prompt (e.g., 'You are a helpful assistant that...')"></textarea>
                    
                    <div class="examples-container">
                        <h4>Few-Shot Examples</h4>
                        <button onclick="addExamplePair()">Add Example</button>
                        <div id="examplePairs"></div>
                    </div>

                    <div class="parameter-input">
                        <label>Temperature:</label>
                        <input type="range" id="temperature" min="0" max="2" step="0.1" value="0.7">
                        <span id="temperatureValue">0.7</span>
                    </div>

                    <div class="parameter-input">
                        <label>Max Tokens:</label>
                        <input type="number" id="maxTokens" value="1000" min="1">
                    </div>
                </div>

                <div id="templatesTab" class="tab-content">
                    <div class="template-selector">
                        <select id="templateSelect" onchange="loadTemplate()">
                            <option value="">Select a template...</option>
                            <option value="sql">SQL Query Generator</option>
                            <option value="analysis">Data Analysis</option>
                            <option value="custom">Custom Template</option>
                        </select>
                    </div>
                    <textarea id="templateEditor" rows="5" placeholder="Edit template..." style="display: none;"></textarea>
                </div>

                <div style="margin-top: 10px;">
                    <button onclick="sendPrompt()">Send</button>
                    <button onclick="clearChat()">Clear Chat</button>
                    <label><input type="checkbox" id="streamingToggle" checked> Enable streaming</label>
                </div>

                <div id="response" class="response-container"></div>
            </div>

            <!-- Model Registration Modal -->
            <div id="registrationModal" class="modal" style="display: none;">
                <div class="modal-content">
                    <h2>Register New Model</h2>
                    <form id="modelRegistrationForm">
                        <div class="form-group">
                            <label for="modelId">Model ID:</label>
                            <input type="text" id="modelId" required placeholder="e.g., gpt-4-turbo">
                        </div>
                        <div class="form-group">
                            <label for="modelName">Model Name:</label>
                            <input type="text" id="modelName" required placeholder="e.g., GPT-4 Turbo">
                        </div>
                        <div class="form-group">
                            <label for="modelDescription">Description:</label>
                            <textarea id="modelDescription" rows="3" placeholder="Brief description of the model"></textarea>
                        </div>
                        <div class="form-group">
                            <label for="modelVersion">Version:</label>
                            <input type="text" id="modelVersion" placeholder="e.g., 1.0.0">
                        </div>
                        <div class="form-group">
                            <label for="modelBackend">Backend:</label>
                            <select id="modelBackend" required>
                                <option value="openai">OpenAI</option>
                                <option value="azure_openai">Azure OpenAI</option>
                                <option value="anthropic">Anthropic</option>
                                <option value="huggingface">HuggingFace</option>
                                <option value="local">Local</option>
                                <option value="custom">Custom</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="apiBase">API Base URL:</label>
                            <input type="text" id="apiBase" placeholder="Optional: Custom API base URL">
                        </div>
                        <div class="form-group">
                            <label for="modelConfig">Additional Configuration:</label>
                            <textarea id="modelConfig" rows="4" placeholder="Optional: JSON configuration"></textarea>
                        </div>
                        <div class="button-group">
                            <button type="submit">Register Model</button>
                            <button type="button" onclick="hideRegistrationForm()">Cancel</button>
                        </div>
                    </form>
                    <div id="registrationResult" style="display: none;"></div>
                </div>
            </div>

            <script>
                let currentStream = null;

                // Tab switching
                function switchTab(tabName) {
                    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
                    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
                    
                    document.querySelector(`.tab[onclick="switchTab('${tabName}')"]`).classList.add('active');
                    document.getElementById(`${tabName}Tab`).classList.add('active');
                }

                // Template handling
                const templates = {
                    sql: {
                        system: "You are an SQL expert. Generate SQL queries based on natural language descriptions.",
                        template: "Generate an SQL query to {{description}}. Use the following schema: {{schema}}"
                    },
                    analysis: {
                        system: "You are a data analysis expert. Provide detailed analysis of data.",
                        template: "Analyze the following data and provide insights: {{data}}"
                    }
                };

                function loadTemplate() {
                    const templateName = document.getElementById('templateSelect').value;
                    const editor = document.getElementById('templateEditor');
                    
                    if (templateName && templateName !== 'custom') {
                        editor.value = templates[templateName].template;
                        document.getElementById('systemPrompt').value = templates[templateName].system;
                    }
                    
                    editor.style.display = templateName ? 'block' : 'none';
                }

                // Few-shot example handling
                function addExamplePair() {
                    const container = document.getElementById('examplePairs');
                    const pair = document.createElement('div');
                    pair.className = 'example-pair';
                    pair.innerHTML = `
                        <input type="text" placeholder="User input">
                        <input type="text" placeholder="Assistant response">
                        <button onclick="this.parentElement.remove()">Remove</button>
                    `;
                    container.appendChild(pair);
                }

                // Fetch registered models
                async function fetchModels() {
                    try {
                        const apiKey = localStorage.getItem('mcp_api_key');
                        const response = await fetch('/api/models', {
                            headers: {
                                'X-API-Key': apiKey
                            }
                        });
                        const models = await response.json();
                        const modelList = document.getElementById('modelList');
                        const modelSelect = document.getElementById('modelSelect');
                        
                        modelList.innerHTML = models.map(model => `
                            <div class="model-item">
                                <strong>${model.name}</strong> (${model.model_id})
                                <br>
                                <small>${model.description || 'No description'}</small>
                            </div>
                        `).join('');
                        
                        modelSelect.innerHTML = '<option value="">Select a model...</option>' + 
                            models.map(model => `
                                <option value="${model.model_id}">${model.name}</option>
                            `).join('');
                    } catch (error) {
                        console.error('Error fetching models:', error);
                        document.getElementById('modelList').innerHTML = '<div class="error-message">Error loading models. Please ensure you have registered a model and have a valid API key.</div>';
                    }
                }

                // Streaming response handling
                function handleStream(reader, responseDiv) {
                    const textDecoder = new TextDecoder();
                    let buffer = '';
                    
                    return reader.read().then(function processText({ done, value }) {
                        if (done) {
                            responseDiv.classList.remove('streaming');
                            return;
                        }
                        
                        buffer += textDecoder.decode(value);
                        const lines = buffer.split('\\n');
                        buffer = lines.pop();
                        
                        lines.forEach(line => {
                            if (line.trim()) {
                                try {
                                    const data = JSON.parse(line);
                                    responseDiv.textContent += data.content;
                                } catch (e) {
                                    responseDiv.textContent += line;
                                }
                            }
                        });
                        
                        return reader.read().then(processText);
                    });
                }

                // Send prompt with streaming support
                async function sendPrompt() {
                    const modelId = document.getElementById('modelSelect').value;
                    const prompt = document.getElementById('prompt').value;
                    const streaming = document.getElementById('streamingToggle').checked;
                    const response = document.getElementById('response');
                    
                    if (!modelId) {
                        alert('Please select a model');
                        return;
                    }
                    
                    // Create message containers
                    const userDiv = document.createElement('div');
                    userDiv.className = 'chat-message user-message';
                    userDiv.textContent = prompt;
                    response.appendChild(userDiv);
                    
                    const modelDiv = document.createElement('div');
                    modelDiv.className = 'chat-message model-message' + (streaming ? ' streaming' : '');
                    response.appendChild(modelDiv);
                    
                    // Scroll to bottom
                    response.scrollTop = response.scrollHeight;
                    
                    // Cancel existing stream if any
                    if (currentStream) {
                        currentStream.abort();
                    }
                    
                    try {
                        // Prepare request data
                        const requestData = {
                            model_id: modelId,
                            prompt: prompt,
                            stream: streaming,
                            parameters: {
                                temperature: parseFloat(document.getElementById('temperature').value),
                                max_tokens: parseInt(document.getElementById('maxTokens').value),
                                system_prompt: document.getElementById('systemPrompt').value,
                                examples: Array.from(document.getElementById('examplePairs').children).map(pair => ({
                                    input: pair.children[0].value,
                                    output: pair.children[1].value
                                })).filter(ex => ex.input && ex.output)
                            }
                        };
                        
                        // Create abort controller for streaming
                        currentStream = new AbortController();
                        
                        const result = await fetch('/llm/generate', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify(requestData),
                            signal: currentStream.signal
                        });
                        
                        if (streaming) {
                            const reader = result.body.getReader();
                            await handleStream(reader, modelDiv);
                        } else {
                            const data = await result.json();
                            modelDiv.textContent = data.response;
                        }
                    } catch (error) {
                        if (error.name === 'AbortError') {
                            modelDiv.textContent += '\n[Response interrupted]';
                        } else {
                            console.error('Error:', error);
                            modelDiv.innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
                        }
                    } finally {
                        currentStream = null;
                    }
                }

                function clearChat() {
                    document.getElementById('response').innerHTML = '';
                    document.getElementById('prompt').value = '';
                }

                // Model Registration Form
                function showRegistrationForm() {
                    document.getElementById('registrationModal').style.display = 'block';
                }

                function hideRegistrationForm() {
                    document.getElementById('registrationModal').style.display = 'none';
                    document.getElementById('modelRegistrationForm').reset();
                    document.getElementById('registrationResult').style.display = 'none';
                }

                document.getElementById('modelRegistrationForm').addEventListener('submit', async (e) => {
                    e.preventDefault();
                    
                    const formData = {
                        model_id: document.getElementById('modelId').value,
                        name: document.getElementById('modelName').value,
                        description: document.getElementById('modelDescription').value || null,
                        version: document.getElementById('modelVersion').value || null,
                        backend: document.getElementById('modelBackend').value,
                        api_base: document.getElementById('apiBase').value || null,
                        config: {}
                    };

                    // Parse config JSON if provided
                    const configText = document.getElementById('modelConfig').value;
                    if (configText.trim()) {
                        try {
                            formData.config = JSON.parse(configText);
                        } catch (error) {
                            showRegistrationError('Invalid JSON in configuration field');
                            return;
                        }
                    }

                    try {
                        console.log('Sending registration request...');
                        const response = await fetch('/api/models/register', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                model_id: document.getElementById('modelId').value,
                                name: document.getElementById('modelName').value,
                                description: document.getElementById('modelDescription').value || null,
                                version: document.getElementById('modelVersion').value || null,
                                backend: document.getElementById('modelBackend').value,
                                api_base: document.getElementById('apiBase').value || null,
                                config: formData.config
                            })
                        });

                        const result = await response.json();

                        if (!response.ok) {
                            throw new Error(result.detail || 'Failed to register model');
                        }

                        // Show success message with API key
                        const resultDiv = document.getElementById('registrationResult');
                        resultDiv.innerHTML = `
                            <div class="success-message">
                                <h3>Model Registered Successfully!</h3>
                                <p>Your API key: <strong>${result.api_key}</strong></p>
                                <p><em>Please save this key securely - you won't be able to see it again!</em></p>
                            </div>
                        `;
                        resultDiv.style.display = 'block';

                        // Refresh model list
                        loadModels();
                    } catch (error) {
                        showRegistrationError(error.message);
                    }
                });

                function showRegistrationError(message) {
                    const resultDiv = document.getElementById('registrationResult');
                    resultDiv.innerHTML = `
                        <div class="error-message">
                            <strong>Error:</strong> ${message}
                        </div>
                    `;
                    resultDiv.style.display = 'block';
                }

                // Initialize
                fetchModels();
                document.getElementById('temperature').addEventListener('input', function() {
                    document.getElementById('temperatureValue').textContent = this.value;
                });
            </script>
        </body>
    </html>
    """

@router.get("/docs", response_class=HTMLResponse)
async def llm_docs(request: Request):
    """Render the LLM API documentation."""
    return """
    <html>
        <head>
            <title>LLM API Documentation - MCP Server</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
                pre { background: #f5f5f5; padding: 15px; border-radius: 4px; overflow-x: auto; }
                .endpoint { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 4px; }
                .method { font-weight: bold; color: #2196F3; }
                .path { font-family: monospace; }
                .description { margin: 10px 0; }
            </style>
        </head>
        <body>
            <h1>LLM API Documentation</h1>
            
            <div class="endpoint">
                <h2><span class="method">POST</span> <span class="path">/llm/generate</span></h2>
                <div class="description">
                    Generate text using the selected LLM model.
                </div>
                <h3>Request Body:</h3>
                <pre>
{
    "model_id": "string",
    "prompt": "string",
    "system_prompt": "string (optional)",
    "temperature": float (optional, default: 0.7),
    "max_tokens": integer (optional, default: 1000)
}
                </pre>
            </div>

            <div class="endpoint">
                <h2><span class="method">POST</span> <span class="path">/llm/generate_stream</span></h2>
                <div class="description">
                    Stream text generation using the selected LLM model.
                </div>
                <h3>Request Body:</h3>
                <pre>
{
    "model_id": "string",
    "prompt": "string",
    "system_prompt": "string (optional)",
    "temperature": float (optional, default: 0.7),
    "max_tokens": integer (optional, default: 1000)
}
                </pre>
            </div>

            <div class="endpoint">
                <h2><span class="method">GET</span> <span class="path">/llm/models</span></h2>
                <div class="description">
                    List all available LLM models.
                </div>
            </div>

            <h2>Authentication</h2>
            <p>All API endpoints require an API key to be sent in the X-API-Key header.</p>
            
            <h2>Rate Limiting</h2>
            <p>API requests are rate limited. The current limits are returned in the response headers:</p>
            <ul>
                <li>X-RateLimit-Limit: Maximum requests per window</li>
                <li>X-RateLimit-Remaining: Remaining requests in current window</li>
                <li>X-RateLimit-Reset: Seconds until the rate limit resets</li>
            </ul>
        </body>
    </html>
    """

@router.get("/register", response_class=HTMLResponse)
async def register_model_ui(request: Request):
    """Render the model registration UI."""
    return """
    <html>
        <head>
            <title>Register Model - MCP Server</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .form-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .form-group { margin-bottom: 15px; }
                label { display: block; margin-bottom: 5px; font-weight: bold; }
                input, select, textarea { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
                button { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
                button:hover { background: #2980b9; }
                .error { color: #d32f2f; margin-top: 5px; }
                .success { color: #388e3c; margin-top: 5px; }
                .api-key-container { 
                    margin-top: 20px;
                    padding: 15px;
                    background: #e8f5e9;
                    border-radius: 4px;
                    display: none;
                    border: 1px solid #4caf50;
                }
                code {
                    display: block;
                    word-break: break-all;
                    white-space: pre-wrap;
                    background: #f5f5f5;
                    padding: 10px;
                    border-radius: 4px;
                    border: 1px solid #ddd;
                    font-family: monospace;
                    margin: 10px 0;
                }
                .copy-button {
                    background: #4caf50;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    cursor: pointer;
                    margin-left: 10px;
                    font-size: 14px;
                }
                .copy-button:hover {
                    background: #388e3c;
                }
                .success-message {
                    color: #388e3c;
                    margin-top: 10px;
                    font-weight: bold;
                }
                .navigation {
                    text-align: center;
                    margin-top: 20px;
                }
                .navigation a {
                    color: #3498db;
                    text-decoration: none;
                }
                .navigation a:hover {
                    text-decoration: underline;
                }
            </style>
        </head>
        <body>
            <h1>Model Registration</h1>
            
            <div class="form-container">
                <form id="registerForm" onsubmit="registerModel(event)">
                    <div class="form-group">
                        <label for="model_id">Model ID*</label>
                        <input type="text" id="model_id" required placeholder="e.g., gpt-4-turbo">
                    </div>
                    
                    <div class="form-group">
                        <label for="name">Name*</label>
                        <input type="text" id="name" required placeholder="e.g., GPT-4 Turbo">
                    </div>
                    
                    <div class="form-group">
                        <label for="description">Description</label>
                        <textarea id="description" rows="3" placeholder="Enter model description..."></textarea>
                    </div>
                    
                    <div class="form-group">
                        <label for="backend">Backend Type*</label>
                        <select id="backend" required>
                            <option value="openai">OpenAI</option>
                            <option value="azure_openai">Azure OpenAI</option>
                            <option value="anthropic">Anthropic</option>
                            <option value="huggingface">Hugging Face</option>
                            <option value="custom">Custom</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="api_base">API Base URL</label>
                        <input type="text" id="api_base" placeholder="e.g., https://api.openai.com/v1">
                    </div>
                    
                    <div class="form-group">
                        <label for="version">Version</label>
                        <input type="text" id="version" placeholder="e.g., 1.0.0">
                    </div>
                    
                    <div class="form-group">
                        <label for="config">Configuration (JSON)</label>
                        <textarea id="config" rows="4" placeholder="Enter configuration as JSON..."></textarea>
                    </div>
                    
                    <button type="submit">Register Model</button>
                </form>
                
                <div id="error" class="error"></div>
                
                <div id="apiKeyContainer" class="api-key-container">
                    <h3>Model Registered Successfully!</h3>
                    <p>Your API key has been generated. Please save it securely as it won't be shown again:</p>
                    <div style="display: flex; align-items: center;">
                        <code id="apiKey" style="flex: 1; padding: 10px; background: #f5f5f5; border-radius: 4px;"></code>
                        <button onclick="copyApiKey()" class="copy-button">Copy</button>
                    </div>
                    <p style="margin-top: 15px;">
                        Add this to your environment file:
                        <br>
                        <code id="envVar" style="display: block; margin-top: 5px; padding: 10px; background: #f5f5f5; border-radius: 4px;"></code>
                    </p>
                    <p style="margin-top: 15px; text-align: center;">
                        <a href="/llm/models" style="color: #3498db;">View all registered models</a>
                    </p>
                </div>
            </div>
            
            <div class="navigation">
                <a href="/llm/ui">Back to Dashboard</a> | 
                <a href="/llm/models">View Registered Models</a>
            </div>

            <script>
                async function registerModel(event) {
                    event.preventDefault();
                    
                    const errorDiv = document.getElementById('error');
                    const apiKeyContainer = document.getElementById('apiKeyContainer');
                    errorDiv.textContent = '';
                    apiKeyContainer.style.display = 'none';
                    
                    try {
                        // Parse config JSON if provided
                        let config = {};
                        const configText = document.getElementById('config').value;
                        if (configText.trim()) {
                            try {
                                config = JSON.parse(configText);
                            } catch (e) {
                                throw new Error('Invalid JSON in configuration field');
                            }
                        }
                        
                        const formData = {
                            model_id: document.getElementById('model_id').value,
                            name: document.getElementById('name').value,
                            description: document.getElementById('description').value,
                            backend: document.getElementById('backend').value.toLowerCase(),
                            api_base: document.getElementById('api_base').value,
                            version: document.getElementById('version').value,
                            config: config
                        };
                        
                        console.log('Sending registration request with data:', JSON.stringify(formData));
                        
                        // Show loading indicator
                        const submitButton = document.querySelector('#registerForm button[type="submit"]');
                        const originalButtonText = submitButton.textContent;
                        submitButton.textContent = 'Registering...';
                        submitButton.disabled = true;
                        
                        try {
                            const response = await fetch('/api/models/register', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json',
                                },
                                body: JSON.stringify(formData)
                            });
                            
                            console.log('Received response status:', response.status);
                            
                            // Get the response text first, then try to parse it
                            const responseText = await response.text();
                            console.log('Response text:', responseText);
                            
                            let data;
                            try {
                                data = JSON.parse(responseText);
                            } catch (err) {
                                console.error('Error parsing response JSON:', err);
                                throw new Error('Invalid response from server: ' + responseText);
                            }
                            
                            if (!response.ok) {
                                throw new Error(data.detail || 'Failed to register model');
                            }
                            
                            console.log('Parsed response data:', data);
                            
                            // Store API key in localStorage and display it
                            if (data && data.api_key) {
                                console.log('API key received:', data.api_key);
                                localStorage.setItem('mcp_api_key', data.api_key);
                                document.getElementById('apiKey').textContent = data.api_key;
                                document.getElementById('envVar').textContent = `MCP_API_KEY=${data.api_key}`;
                                apiKeyContainer.style.display = 'block';
                                
                                // Scroll to the API key container
                                apiKeyContainer.scrollIntoView({ behavior: 'smooth' });
                            } else {
                                console.error('No API key in response:', data);
                                throw new Error('No API key received from server');
                            }
                            
                            // Clear form
                            document.getElementById('registerForm').reset();
                        } finally {
                            // Restore button state
                            submitButton.textContent = originalButtonText;
                            submitButton.disabled = false;
                        }
                    } catch (error) {
                        console.error('Registration error:', error);
                        errorDiv.textContent = error.message;
                        errorDiv.style.display = 'block';
                    }
                }
                
                function copyApiKey() {
                    const apiKey = document.getElementById('apiKey').textContent;
                    if (apiKey) {
                        navigator.clipboard.writeText(apiKey).then(() => {
                            alert('API key copied to clipboard!');
                        }).catch(err => {
                            console.error('Failed to copy API key:', err);
                            // Fallback selection method
                            const range = document.createRange();
                            range.selectNode(document.getElementById('apiKey'));
                            window.getSelection().removeAllRanges();
                            window.getSelection().addRange(range);
                            document.execCommand('copy');
                            window.getSelection().removeAllRanges();
                            alert('API key copied to clipboard!');
                        });
                    }
                }
            </script>
        </body>
    </html>
    """

@router.get("/models", response_class=HTMLResponse)
async def models_list_ui(request: Request):
    """Render the models list UI."""
    return """
    <html>
        <head>
            <title>Registered Models - MCP Server</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
                .models-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .model-card {
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 15px;
                    margin-bottom: 10px;
                    background: #f8f9fa;
                }
                .model-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 10px;
                }
                .model-name {
                    font-weight: bold;
                    color: #2c3e50;
                }
                .model-id {
                    color: #7f8c8d;
                    font-size: 0.9em;
                }
                .model-stats {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 10px;
                    margin-top: 10px;
                    font-size: 0.9em;
                }
                .stat-item {
                    background: white;
                    padding: 8px;
                    border-radius: 4px;
                    text-align: center;
                }
                .stat-label {
                    color: #7f8c8d;
                    font-size: 0.8em;
                }
                .stat-value {
                    color: #2c3e50;
                    font-weight: bold;
                }
                .refresh-button {
                    background: #3498db;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 4px;
                    cursor: pointer;
                    margin-bottom: 20px;
                }
                .refresh-button:hover {
                    background: #2980b9;
                }
                .backend-badge {
                    background: #3498db;
                    color: white;
                    padding: 2px 8px;
                    border-radius: 12px;
                    font-size: 0.8em;
                }
                .error-message {
                    color: #d32f2f;
                    background: #ffebee;
                    padding: 10px;
                    border-radius: 4px;
                    margin: 10px 0;
                }
                .regenerate-key {
                    background: #27ae60;
                    color: white;
                    border: none;
                    padding: 5px 10px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 0.9em;
                    margin-left: 10px;
                }
                .regenerate-key:hover {
                    background: #219a52;
                }
                .api-key-container {
                    margin-top: 10px;
                    padding: 10px;
                    background: #e8f5e9;
                    border-radius: 4px;
                    display: none;
                }
                .api-key-value {
                    font-family: monospace;
                    background: #f5f5f5;
                    padding: 5px;
                    border-radius: 4px;
                    margin: 5px 0;
                }
                .copy-button {
                    background: #4caf50;
                    color: white;
                    border: none;
                    padding: 5px 10px;
                    border-radius: 4px;
                    cursor: pointer;
                    margin-left: 5px;
                }
                .copy-button:hover {
                    background: #388e3c;
                }
            </style>
        </head>
        <body>
            <h1>Registered Models</h1>
            
            <div class="models-container">
                <button onclick="refreshModels()" class="refresh-button">Refresh List</button>
                <div id="modelsList">Loading models...</div>
            </div>

            <script>
                async function refreshModels() {
                    try {
                        const response = await fetch('/api/models');
                        if (!response.ok) {
                            throw new Error('Failed to fetch models');
                        }
                        
                        const models = await response.json();
                        const modelsListDiv = document.getElementById('modelsList');
                        
                        if (models.length === 0) {
                            modelsListDiv.innerHTML = '<p>No models registered yet.</p>';
                            return;
                        }
                        
                        modelsListDiv.innerHTML = models.map(model => `
                            <div class="model-card">
                                <div class="model-header">
                                    <div>
                                        <div class="model-name">${model.name}</div>
                                        <div class="model-id">ID: ${model.model_id}</div>
                                    </div>
                                    <div>
                                        <span class="backend-badge">${model.backend}</span>
                                        <button onclick="regenerateKey('${model.model_id}')" class="regenerate-key">Regenerate API Key</button>
                                    </div>
                                </div>
                                ${model.description ? `<p>${model.description}</p>` : ''}
                                <div class="model-stats">
                                    <div class="stat-item">
                                        <div class="stat-label">Total Requests</div>
                                        <div class="stat-value">${model.total_requests || 0}</div>
                                    </div>
                                    <div class="stat-item">
                                        <div class="stat-label">Success Rate</div>
                                        <div class="stat-value">${model.total_requests > 0 
                                            ? Math.round((model.successful_requests / model.total_requests) * 100)
                                            : 0}%</div>
                                    </div>
                                    <div class="stat-item">
                                        <div class="stat-label">Total Tokens</div>
                                        <div class="stat-value">${model.total_tokens || 0}</div>
                                    </div>
                                    <div class="stat-item">
                                        <div class="stat-label">Avg Latency</div>
                                        <div class="stat-value">${(model.average_latency || 0).toFixed(2)}ms</div>
                                    </div>
                                </div>
                                <div style="margin-top: 10px; font-size: 0.8em; color: #7f8c8d;">
                                    Created: ${new Date(model.created_at).toLocaleString()}
                                </div>
                                <div id="apiKey-${model.model_id}" class="api-key-container">
                                    <p>New API key generated. Please save it securely:</p>
                                    <div style="display: flex; align-items: center;">
                                        <code class="api-key-value" id="apiKeyValue-${model.model_id}"></code>
                                        <button onclick="copyApiKey('${model.model_id}')" class="copy-button">Copy</button>
                                    </div>
                                </div>
                            </div>
                        `).join('');
                        
                    } catch (error) {
                        console.error('Error fetching models:', error);
                        document.getElementById('modelsList').innerHTML = 
                            '<div class="error-message">Failed to load models. Please try refreshing.</div>';
                    }
                }

                async function regenerateKey(modelId) {
                    try {
                        // Get the current API key from local storage
                        const apiKey = localStorage.getItem('mcp_api_key');
                        
                        if (!apiKey) {
                            throw new Error('No API key found. Please register a model first to get an API key.');
                        }

                        const response = await fetch(`/api/models/${modelId}/regenerate-key`, {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                                'X-API-Key': apiKey
                            }
                        });

                        if (!response.ok) {
                            const errorData = await response.json();
                            throw new Error(errorData.detail || 'Failed to regenerate API key');
                        }

                        const data = await response.json();
                        
                        // Store new API key in localStorage
                        if (data.api_key) {
                            localStorage.setItem('mcp_api_key', data.api_key);
                        }
                        
                        // Show the API key container
                        const container = document.getElementById(`apiKey-${modelId}`);
                        const valueElement = document.getElementById(`apiKeyValue-${modelId}`);
                        container.style.display = 'block';
                        valueElement.textContent = data.api_key;
                        
                    } catch (error) {
                        console.error('Error regenerating API key:', error);
                        // Create error message div if it doesn't exist
                        let errorDiv = document.getElementById(`error-${modelId}`);
                        if (!errorDiv) {
                            errorDiv = document.createElement('div');
                            errorDiv.id = `error-${modelId}`;
                            errorDiv.className = 'error-message';
                            document.getElementById(`apiKey-${modelId}`).parentElement.appendChild(errorDiv);
                        }
                        errorDiv.textContent = `Error: ${error.message}`;
                        errorDiv.style.display = 'block';
                    }
                }

                function copyApiKey(modelId) {
                    const apiKey = document.getElementById(`apiKeyValue-${modelId}`).textContent;
                    navigator.clipboard.writeText(apiKey);
                    alert('API key copied to clipboard!');
                }

                // Load models when page loads
                document.addEventListener('DOMContentLoaded', refreshModels);
            </script>
        </body>
    </html>
    """

# API Routes (Authentication Required)
@api_router.post("/generate")
async def generate_response(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Generate a response from the selected model with streaming support."""
    try:
        data = await request.json()
        model_id = data.get("model_id")
        prompt = data.get("prompt")
        stream = data.get("stream", False)
        parameters = data.get("parameters", {})
        data_source = data.get("data_source")

        if not model_id or not prompt:
            raise HTTPException(status_code=400, detail="Missing model_id or prompt")

        # Get model configuration
        model = await db.get(ModelRecord, model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        # Handle data source if specified
        if data_source:
            source_data = await get_data_source_content(db, data_source)
            # Append data source content to prompt
            prompt = f"{prompt}\n\nAvailable data:\n{source_data}"

        # Prepare generation parameters
        gen_params = {
            "temperature": float(parameters.get("temperature", 0.7)),
            "max_tokens": int(parameters.get("max_tokens", 1000)),
            "stream": stream
        }

        # Add system prompt if provided
        if system_prompt := parameters.get("system_prompt"):
            gen_params["system_prompt"] = system_prompt

        # Add few-shot examples if provided
        if examples := parameters.get("examples"):
            gen_params["examples"] = examples

        if stream:
            return StreamingResponse(
                stream_response(model, prompt, gen_params),
                media_type="text/event-stream"
            )
        else:
            response = await generate_single_response(model, prompt, gen_params)
            return {"response": response}

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def stream_response(model: ModelRecord, prompt: str, parameters: Dict[str, Any]):
    """Stream the model's response."""
    try:
        # Initialize the model client based on backend type
        client = get_model_client(model)
        
        async for chunk in client.generate_stream(prompt, parameters):
            # Yield each chunk as a JSON-encoded string
            yield json.dumps({"content": chunk}) + "\n"
            
    except Exception as e:
        logger.error(f"Error in stream_response: {str(e)}")
        yield json.dumps({"error": str(e)}) + "\n"

async def generate_single_response(model: ModelRecord, prompt: str, parameters: Dict[str, Any]) -> str:
    """Generate a single response without streaming."""
    try:
        client = get_model_client(model)
        response = await client.generate(prompt, parameters)
        return response
    except Exception as e:
        logger.error(f"Error in generate_single_response: {str(e)}")
        raise

def get_model_client(model: ModelRecord):
    """Get the appropriate model client based on the backend type."""
    if model.backend == "OPENAI":
        from app.core.backends.openai import OpenAIClient
        return OpenAIClient(model)
    elif model.backend == "AZURE_OPENAI":
        from app.core.backends.azure import AzureOpenAIClient
        return AzureOpenAIClient(model)
    elif model.backend == "ANTHROPIC":
        from app.core.backends.anthropic import AnthropicClient
        return AnthropicClient(model)
    elif model.backend == "HUGGINGFACE":
        from app.core.backends.huggingface import HuggingFaceClient
        return HuggingFaceClient(model)
    else:
        raise ValueError(f"Unsupported backend type: {model.backend}")

async def get_data_source_content(db: AsyncSession, data_source: Dict[str, Any]) -> str:
    """Get content from the specified data source."""
    source_type = data_source.get("type")
    query = data_source.get("query")

    if not source_type or not query:
        raise HTTPException(status_code=400, detail="Invalid data source configuration")

    try:
        if source_type == "postgres":
            # Execute query against PostgreSQL
            result = await db.execute(text(query))
            rows = result.fetchall()
            return "\n".join([str(row) for row in rows])
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported data source type: {source_type}")
    except Exception as e:
        logger.error(f"Error accessing data source: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error accessing data source: {str(e)}")

@api_router.post("/datasources/{source}/test")
async def test_data_source(
    source: str,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Test a data source connection and query."""
    try:
        data = await request.json()
        query = data.get("query")

        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        if source == "postgres":
            # Test PostgreSQL query
            try:
                result = await db.execute(text(query))
                rows = result.fetchall()
                return {
                    "message": f"Query executed successfully. Retrieved {len(rows)} rows.",
                    "sample": str(rows[:5]) if rows else "No data returned"
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid query: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported data source type: {source}")

    except Exception as e:
        logger.error(f"Error testing data source: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Include the API router in the main router
router.include_router(api_router) 