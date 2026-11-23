#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-../models/exported_gguf/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf}"
PORT="${2:-18088}"

./target/release/caelum serve "$MODEL" --port "$PORT" --ctx 2048 --kv-q8 >/tmp/caelum_srv.log 2>&1 &
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
