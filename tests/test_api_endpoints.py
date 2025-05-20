import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).parent.parent))
from app.main import app

client = TestClient(app)


def test_tool_registration_and_execution():
    resp = client.post(
        "/api/v1/register",
        json={
            "name": "echo",
            "description": "Echo text",
            "module": "tests.sample_tool",
            "function": "echo_tool",
            "namespace": "test",
            "input_schema": {"text": {"type": "string"}},
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "registered"

    exec_resp = client.post(
        "/api/v1/tool/echo",
        json={"parameters": {"text": "hi"}, "namespace": "test"},
    )
    assert exec_resp.status_code == 200
    assert exec_resp.json()["result"] == "hi"
