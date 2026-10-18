#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 prepare.py
docker info >/dev/null
docker compose config --quiet
docker compose build po
docker compose up -d --wait --wait-timeout 300
python3 compare.py
