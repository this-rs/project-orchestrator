#!/bin/sh
# Logs the PreCompact input and answers according to M2C_MODE.
cat > "$M2C_LOG/input-$M2C_MODE.json"
I="End the summary with the exact line: ZEPHYR-42"
case "$M2C_MODE" in
  reason)  printf '{"continue":true,"reason":"%s"}\n' "$I" ;;
  addctx)  printf '{"hookSpecificOutput":{"hookEventName":"PreCompact","additionalContext":"%s"}}\n' "$I" ;;
  plain)   printf '%s\n' "$I" ;;
  none)    : ;;
esac
