#!/usr/bin/env bash
# Reproduce LLaDA2-Uni text-to-text and text-to-image requests.
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:18002}"
MODEL="${MODEL:-LLaDA2.0-Uni}"
OUT_DIR="${OUT_DIR:-$(pwd)/llada2_uni_validation}"
TIMEOUT="${TIMEOUT:-600}"
MODE="${1:-all}"

for command in curl jq base64 file; do
  command -v "$command" >/dev/null || {
    printf 'Missing required command: %s\n' "$command" >&2
    exit 1
  }
done
mkdir -p "$OUT_DIR"

post_request() {
  local name="$1"
  local payload="$2"
  local response="$OUT_DIR/$name.json"
  local status

  status=$(curl -sS --max-time "$TIMEOUT" \
    -o "$response" \
    -w '%{http_code}' \
    "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    --data "$payload")

  printf '\n[%s] HTTP %s\n' "$name" "$status"
  if [[ "$status" != "200" ]]; then
    jq . "$response" 2>/dev/null || sed -n '1,160p' "$response"
    return 1
  fi
}

print_result() {
  local name="$1"
  local response="$OUT_DIR/$name.json"

  printf 'text: '
  jq -r '
    .choices[0].message.content as $content
    | if ($content | type) == "array" then
        ([$content[] | select(.type == "text") | .text]
          | if length == 0 then "<null>" else join("") end)
      else
        ($content // "<null>")
      end
  ' "$response"
  printf 'finish_reason: '
  jq -r '.choices[0].finish_reason // "<null>"' "$response"
  printf 'timings:\n'
  jq '.timings' "$response"
}

run_t2t() {
  local payload
  payload=$(jq -n \
    --arg model "$MODEL" \
    '{
      model: $model,
      messages: [{
        role: "user",
        content: "用一句中文解释为什么天空看起来是蓝色的。"
      }],
      modalities: ["text"],
      max_tokens: 2048
    }')

  post_request t2t "$payload"
  print_result t2t
}

run_t2i() {
  local generation_mode="$1"
  local decoder_steps="$2"
  local seed="$3"
  local name="t2i_${generation_mode}"
  local payload
  local response="$OUT_DIR/$name.json"
  local image="$OUT_DIR/$name.png"

  payload=$(jq -n \
    --arg model "$MODEL" \
    --arg mode "$generation_mode" \
    --argjson decoder_steps "$decoder_steps" \
    --argjson seed "$seed" \
    '{
      model: $model,
      messages: [{
        role: "user",
        content: "A bright red apple on a white ceramic plate, studio lighting, centered composition."
      }],
      modalities: ["text", "image"],
      image_generation: {
        mode: $mode,
        decode_mode: "decoder-turbo",
        decoder_steps: $decoder_steps,
        seed: $seed,
        cfg_scale: 4.0,
        image_h: 1024,
        image_w: 1024,
        dllm_steps: 8
      }
    }')

  post_request "$name" "$payload"
  print_result "$name"
  jq -er '
    .choices[0].message.content[]
    | select(.type == "image")
    | .image.data
  ' "$response" | base64 -d > "$image"
  printf 'image: %s\n' "$image"
  file "$image"
}

case "$MODE" in
  t2t)
    run_t2t
    ;;
  t2i-normal)
    run_t2i normal 8 42
    ;;
  t2i-thinking)
    run_t2i thinking 2 43
    ;;
  t2i)
    run_t2i normal 8 42
    run_t2i thinking 2 43
    ;;
  all)
    run_t2t
    run_t2i normal 8 42
    run_t2i thinking 2 43
    ;;
  *)
    printf 'Usage: %s [all|t2t|t2i|t2i-normal|t2i-thinking]\n' "$0" >&2
    exit 2
    ;;
esac

printf '\nArtifacts: %s\n' "$OUT_DIR"
