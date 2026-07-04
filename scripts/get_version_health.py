import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
print("=== /version ===")
print(json.dumps(client.get("/version").json(), ensure_ascii=False, indent=2))
print("\n=== /health ===")
print(json.dumps(client.get("/health").json(), ensure_ascii=False, indent=2))
