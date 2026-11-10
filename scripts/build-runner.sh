#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
args=()
if [[ -n "${VLA_BASE_IMAGE:-}" ]]; then args+=(--build-arg "BASE_IMAGE=$VLA_BASE_IMAGE"); fi
docker build --pull=false -f Dockerfile.runner "${args[@]}" -t vla-practice-runner:1 .
