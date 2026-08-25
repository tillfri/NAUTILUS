#!/usr/bin/env bash
set -euo pipefail

API_URL="http://127.0.0.1:8000/v1/chat/completions"
MODEL="Qwen/Qwen3.5-9B"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MESSAGES_FILE="$DIR/qwen_messages.json"

if [[ ! -f "$MESSAGES_FILE" ]] || [[ "${1:-}" == "--reset" ]]; then
  echo "[]" > "$MESSAGES_FILE"
fi

echo "Chatting with $MODEL. Type 'exit' to quit, '/reset' to clear history, '/image <path> [caption]' to send an image."

while true; do
  printf "you> "
  IFS= read -r input || break
  [[ -z "$input" ]] && continue
  if [[ "$input" == "exit" || "$input" == "quit" ]]; then
    break
  fi
  if [[ "$input" == "/reset" ]]; then
    echo "[]" > "$MESSAGES_FILE"
    echo "(history cleared)"
    continue
  fi

  if [[ "$input" == /image* ]]; then
    rest="${input#/image}"
    rest="${rest# }"
    imgpath="${rest%% *}"
    if [[ "$rest" == *" "* ]]; then
      caption="${rest#* }"
    else
      caption="What is in this image?"
    fi

    if [[ ! -f "$imgpath" ]]; then
      echo "qwen> [error] file not found: $imgpath"
      continue
    fi

    mime=$(file --mime-type -b "$imgpath")
    b64=$(base64 -w0 "$imgpath")
    content=$(jq -n --arg text "$caption" --arg url "data:$mime;base64,$b64" \
      '[{"type":"text","text":$text},{"type":"image_url","image_url":{"url":$url}}]')

    jq --argjson content "$content" '. + [{"role":"user","content":$content}]' \
      "$MESSAGES_FILE" > "$MESSAGES_FILE.tmp" && mv "$MESSAGES_FILE.tmp" "$MESSAGES_FILE"
  else
    jq --arg content "$input" '. + [{"role":"user","content":$content}]' \
      "$MESSAGES_FILE" > "$MESSAGES_FILE.tmp" && mv "$MESSAGES_FILE.tmp" "$MESSAGES_FILE"
  fi

  payload=$(jq -n --argjson messages "$(cat "$MESSAGES_FILE")" --arg model "$MODEL" \
    '{model: $model, messages: $messages}')

  response=$(curl -s "$API_URL" -H "Content-Type: application/json" -d "$payload")

  reply=$(echo "$response" | jq -r '.choices[0].message.content // empty')

  if [[ -z "$reply" ]]; then
    echo "qwen> [error] $(echo "$response" | jq -c '.')"
    continue
  fi

  echo "qwen>"
  echo "$reply" | glow -

  jq --arg content "$reply" '. + [{"role":"assistant","content":$content}]' \
    "$MESSAGES_FILE" > "$MESSAGES_FILE.tmp" && mv "$MESSAGES_FILE.tmp" "$MESSAGES_FILE"
done

echo "Bye."
