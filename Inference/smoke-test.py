#!/usr/bin/env python3
"""Basic OpenAI-compatible API check; requires only Python 3."""
import json
import os
import urllib.error
import urllib.request

base = os.environ.get("API_BASE", "http://127.0.0.1:8000").rstrip("/")
token = os.environ.get("API_KEY", "")
headers = {"Content-Type": "application/json"}
if token:
    headers["Authorization"] = "Bearer " + token
payload = {
    "model": os.environ.get("SERVED_MODEL_NAME", "qwen-flash-next"),
    "messages": [{"role": "user", "content": "请用一句话介绍你自己。"}],
    "max_tokens": 512,
    "temperature": 0.7,
    "chat_template_kwargs": {"enable_thinking": False},
}
request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(), headers=headers)
try:
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("choices") or not result["choices"][0].get("message", {}).get("content"):
        raise SystemExit("No answer content returned; inspect the response above.")
except urllib.error.HTTPError as error:
    raise SystemExit(f"HTTP {error.code}: {error.read().decode(errors='replace')}") from error
except urllib.error.URLError as error:
    raise SystemExit(f"Cannot reach {base}: {error.reason}") from error
