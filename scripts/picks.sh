#!/usr/bin/env bash
# Daily buy picks to KakaoTalk. Run from cron (KST):
#   20 8  * * 1-5  .../scripts/picks.sh KR
#   50 21 * * 1-5  .../scripts/picks.sh US     # US open 22:30 KST (daylight saving)
#   50 22 * * 1-5  .../scripts/picks.sh US     # US open 23:30 KST (standard time)
#   0  9  * * 6    .../scripts/picks.sh settle
# `picks due` decides which runs actually send (trading day, send window, not yet sent).
set -euo pipefail
cd "$(dirname "$0")/.."
py=.venv/bin/python3

if [[ $1 == settle ]]; then
    exec "$py" -m coinbot.picks settle --send
fi
market=$1
"$py" -m coinbot.picks due "$market" || exit 0

# The CLI ships inside the VS Code extension, whose path changes on every update.
claude=$(command -v claude || ls -d ~/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude | sort -V | tail -1)
prompt=$("$py" -m coinbot.picks prepare "$market")
"$claude" -p "$prompt" --allowedTools "Read,Write,WebSearch,WebFetch" || echo "claude exited $?" >&2
"$py" -m coinbot.picks send "$market"
