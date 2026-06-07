#!/usr/bin/env bash
# 启动 API 服务
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
uvicorn app.api:app --reload --port 8000
