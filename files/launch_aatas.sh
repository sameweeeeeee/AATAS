#!/bin/bash

echo "🤖 AATAS starting..."

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p logs data

# load env
export $(grep -v '^#' .env | xargs)

echo "Running from: $(pwd)"

# start server
python3 api/server.py > logs/server.log 2>&1 &
SERVER_PID=$!

sleep 2

# start bot
python3 bot/aatas_bot.py > logs/bot.log 2>&1 &
BOT_PID=$!

echo "Server PID: $SERVER_PID"
echo "Bot PID: $BOT_PID"

tail -f logs/server.log logs/bot.log

wait $BOT_PID