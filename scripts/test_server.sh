#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-../models/exported_gguf/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf}"
PORT="${2:-18088}"
EXTRA_ARGS="${3:-}"

./target/release/caelum serve "$MODEL" --port "$PORT" --ctx 2048 --kv-q8 $EXTRA_ARGS >/tmp/caelum_srv.log 2>&1 &
SRV_PID=$!
trap 'kill "$SRV_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 40); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
    break
  fi
  sleep 0.2
done

echo "=== 1. GET /health ==="
curl -s "http://127.0.0.1:${PORT}/health" && echo

echo "=== 2. GET /v1/models ==="
curl -s "http://127.0.0.1:${PORT}/v1/models" && echo

echo "=== 3. POST /v1/chat/completions (Multi-turn Turn 1, SSE stream: true) ==="
curl -sN -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"system","content":"You are a concise Rust systems expert."},{"role":"user","content":"What does Option<T> prevent in 1 sentence?"}],"max_tokens":24,"temperature":0.0,"stream":true}' | sed -n '1,10p' || true

echo "=== 4. POST /v1/chat/completions (Multi-turn Turn 2 — Prefix Cache Hit + json_object!) ==="
curl -s -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"system","content":"You are a concise Rust systems expert."},{"role":"user","content":"What does Option<T> prevent in 1 sentence?"},{"role":"assistant","content":"Option<T> prevents null pointer dereferences at compile time."},{"role":"user","content":"Return JSON with keys lang and safe."}],"max_tokens":48,"temperature":0.0,"response_format":{"type":"json_object"}}' && echo

echo "=== 5. POST /v1/completions (IDE FIM Tab Autocomplete!) ==="
curl -s -X POST "http://127.0.0.1:${PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"fn add(a: i32, b: i32) -> i32 {\n    ","suffix":"\n}","max_tokens":24,"temperature":0.0,"stop":["\n"]}' && echo

echo "=== 6. POST /v1/celestia/propose (Strict SchemaGrammar + Prefix Cache) ==="
curl -s -X POST "http://127.0.0.1:${PORT}/v1/celestia/propose" \
  -H "Content-Type: application/json" \
  -d '{"workspace_mode":"scene3d","prompt":"Spawn a blue cube at origin"}' | head -c 450 && echo

echo "=== 7. POST /v1/chat/completions (OpenAI Tool Calling — Turn 1: Cline/OpenCode read_file) ==="
curl -s -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"system","content":"You are a coding agent. Use the provided tools when asked to inspect files."},{"role":"user","content":"Please read Cargo.toml to check the crate version."}],"tools":[{"type":"function","function":{"name":"read_file","description":"Read the contents of a file at the given path","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path to read"}},"required":["path"]}}}],"tool_choice":"auto","temperature":0.0,"max_tokens":128}' && echo

echo "=== 8. POST /v1/chat/completions (OpenAI Tool Calling — SSE Streaming delta.tool_calls!) ==="
curl -sN -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"system","content":"You are a coding agent. Use the provided tools when asked to inspect files."},{"role":"user","content":"Please read Cargo.toml to check the crate version."}],"tools":[{"type":"function","function":{"name":"read_file","description":"Read the contents of a file at the given path","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path to read"}},"required":["path"]}}}],"tool_choice":"auto","temperature":0.0,"max_tokens":128,"stream":true}' && echo

echo "=== 9. POST /v1/chat/completions (OpenAI Tool Calling — Turn 2: role=tool result -> final answer!) ==="
curl -s -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"system","content":"You are a coding agent. Use the provided tools when asked to inspect files."},{"role":"user","content":"Please read Cargo.toml to check the crate version."},{"role":"assistant","content":null,"tool_calls":[{"id":"call_caelum_1_0","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"Cargo.toml\"}"}}]},{"role":"tool","tool_call_id":"call_caelum_1_0","name":"read_file","content":"[workspace.package]\nversion = \"0.7.0\"\nedition = \"2021\""}],"tools":[{"type":"function","function":{"name":"read_file","description":"Read the contents of a file at the given path","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path to read"}},"required":["path"]}}}],"tool_choice":"auto","temperature":0.0,"max_tokens":64}' && echo
