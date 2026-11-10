#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  python_cmd="${VLA_PYTHON:-python3}"
  if [[ -z "${VLA_PYTHON:-}" ]] && command -v python3.11 >/dev/null 2>&1; then python_cmd=python3.11; fi
  if ! "$python_cmd" -c 'import sys; assert sys.version_info >= (3, 11)' 2>/dev/null; then
    printf '需要 Python 3.11 或更新版本。可设置 VLA_PYTHON 指向 Python 可执行文件。\n' >&2
    exit 1
  fi
  "$python_cmd" -m venv .venv
  .venv/bin/python -m pip install -r requirements-practice.lock
fi
if ! .venv/bin/python -c 'import fastapi,uvicorn,pydantic_settings' >/dev/null 2>&1; then
  .venv/bin/python -m pip install -r requirements-practice.lock
fi
if [[ ! -f web/out/index.html ]]; then
  (cd web && npm ci && npm run build)
fi
printf '\nVLA 面试练习室  http://127.0.0.1:8765\n按 Ctrl+C 停止，记录保存在 .local/\n\n'
exec .venv/bin/python -m uvicorn practice.main:app --host 127.0.0.1 --port 8765
