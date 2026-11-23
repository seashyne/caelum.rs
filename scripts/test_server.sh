#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-../models/exported_gguf/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf}"
PORT="${2:-18088}"

./target/release/caelum serve "$MODEL" --port "$PORT" --ctx 1024 --kv-q8 >/tmp/caelum_srv.log 2>&1 &
SRV_PID=$!
trap 'kill "$SRV_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
    break
  fi
  sleep 0.2
done

echo "=== 1. GET /health ==="
curl -s "http://127.0.0.1:${PORT}/health" && echo

echo "=== 2. POST /v1/chat/completions (SSE stream: true) ==="
curl -sN -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hello in 5 words."}],"max_tokens":24,"temperature":0.1,"stream":true}' | head -n 12

echo "=== 3. POST /v1/celestia/propose #1 (Cold Prefill) ==="
curl -s -X POST "http://127.0.0.1:${PORT}/v1/celestia/propose" \
  -H "Content-Type: application/json" \
  -d '{"workspace_mode":"scene3d","prompt":"Spawn a blue cube at origin and a red sphere next to it"}' | head -c 500 && echo

echo "=== 4. POST /v1/celestia/propose #2 (Prefix Cache Hit!) ==="
curl -s -X POST "http://127.0.0.1:${PORT}/v1/celestia/propose" \
  -H "Content-Type: application/json" \
  -d '{"workspace_mode":"scene3d","prompt":"Spawn a green cylinder at (2,0,0)"}' | head -c 500 && echo
