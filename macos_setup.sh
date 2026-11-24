#!/usr/bin/env bash

# Build and run SLEEPYLAND-PLUS from this source checkout on macOS.
# Docker Desktop must already be installed and running.

set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] Docker Desktop is not installed or is not on PATH."
    echo "Install and start Docker Desktop, then run this script again."
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "[ERROR] Docker Compose v2 is unavailable."
    echo "Update Docker Desktop so that 'docker compose' works."
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "[ERROR] Docker Desktop is not running or is not ready."
    echo "Start Docker Desktop, wait until it is ready, then run this script again."
    exit 1
fi

echo "[INFO] Building and starting SLEEPYLAND-PLUS from local source..."
docker compose -p sleepyland-plus up -d --build

echo
docker compose -p sleepyland-plus ps
echo
echo "[INFO] SLEEPYLAND-PLUS is available at http://localhost:8887"
echo "[INFO] Follow service logs with: docker compose -p sleepyland-plus logs -f"
