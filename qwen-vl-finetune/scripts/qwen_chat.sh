#!/usr/bin/env bash
set -euo pipefail

API_URL="http://127.0.0.1:8000/v1/chat/completions"
MODEL="Qwen/Qwen3.5-9B"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MESSAGES_FILE="$DIR/qwen_messages.json"

if [[ ! -f "$MESSAGES_FILE" ]] || [[ "${1:-}" == "--reset" ]]; then
  echo "[]" > "$MESSAGES_FILE"
fi

echo "Chatting with $MODEL. Type 'exit' to quit, '/reset' to clear history, '/image <path> [<path> ...] [caption]' to send one or more images."

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
    read -ra tokens <<< "$rest"

    imgpaths=()
    for tok in "${tokens[@]}"; do
      [[ -f "$tok" ]] || break
      imgpaths+=("$tok")
    done

    if [[ ${#imgpaths[@]} -eq 0 ]]; then
      echo "qwen> [error] no valid image path(s) found in: $rest"
      continue
    fi

    caption_tokens=("${tokens[@]:${#imgpaths[@]}}")
    if [[ ${#caption_tokens[@]} -eq 0 ]]; then
      caption="What is in this image?"
    else
      caption="${caption_tokens[*]}"
    fi

    content_file=$(mktemp)
    jq -n --arg text "$caption" '[{"type":"text","text":$text}]' > "$content_file"
    for imgpath in "${imgpaths[@]}"; do
      mime=$(file --mime-type -b "$imgpath")
      b64_file=$(mktemp)
      base64 -w0 "$imgpath" > "$b64_file"
      jq --rawfile b64 "$b64_file" --arg mime "$mime" \
        '. + [{"type":"image_url","image_url":{"url":("data:" + $mime + ";base64," + $b64)}}]' \
        "$content_file" > "$content_file.tmp" && mv "$content_file.tmp" "$content_file"
      rm -f "$b64_file"
    done

    jq --slurpfile content "$content_file" '. + [{"role":"user","content":$content[0]}]' \
      "$MESSAGES_FILE" > "$MESSAGES_FILE.tmp" && mv "$MESSAGES_FILE.tmp" "$MESSAGES_FILE"
    rm -f "$content_file"
  else
    jq --arg content "$input" '. + [{"role":"user","content":$content}]' \
      "$MESSAGES_FILE" > "$MESSAGES_FILE.tmp" && mv "$MESSAGES_FILE.tmp" "$MESSAGES_FILE"
  fi

  payload_file=$(mktemp)
  jq -n --slurpfile messages "$MESSAGES_FILE" --arg model "$MODEL" \
    '{model: $model, messages: $messages[0]}' > "$payload_file"

  response_file=$(mktemp)
  curl -s "$API_URL" -H "Content-Type: application/json" \
    -d "@$payload_file" > "$response_file"
  rm -f "$payload_file"

  reply=$(jq -r '.choices[0].message.content // empty' "$response_file")

  if [[ -z "$reply" ]]; then
    echo "qwen> [error] $(jq -c '.' "$response_file")"
    rm -f "$response_file"
    continue
  fi
  rm -f "$response_file"

  echo "qwen>"
  printf '%s\n' "$reply" | glow -

  jq --arg content "$reply" '. + [{"role":"assistant","content":$content}]' \
    "$MESSAGES_FILE" > "$MESSAGES_FILE.tmp" && mv "$MESSAGES_FILE.tmp" "$MESSAGES_FILE"
done

echo "Bye."
