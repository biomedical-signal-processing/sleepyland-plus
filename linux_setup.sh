#!/usr/bin/env bash

# Build and run SLEEPYLAND-PLUS from this source checkout on Linux.
# Docker Engine and the Docker Compose v2 plugin must already be installed.

set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] Docker is not installed or is not on PATH."
    echo "Install Docker Engine and the Docker Compose v2 plugin, then run this script again."
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "[ERROR] Docker Compose v2 is unavailable."
    echo "Install the Docker Compose plugin so that 'docker compose' works."
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "[ERROR] Docker is installed but its daemon is not accessible."
    echo "Start Docker, or ensure your user has permission to access the Docker daemon."
    exit 1
fi

echo "[INFO] Building and starting SLEEPYLAND-PLUS from local source..."
docker compose -p sleepyland-plus up -d --build

echo
docker compose -p sleepyland-plus ps
echo
echo "[INFO] SLEEPYLAND-PLUS is available at http://localhost:8887"
echo "[INFO] Follow service logs with: docker compose -p sleepyland-plus logs -f"
