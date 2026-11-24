# Build and run SLEEPYLAND-PLUS from this source checkout on Windows.
# Docker Desktop must already be installed and running.

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker Desktop is not installed or Docker is not on PATH. Install and start Docker Desktop, then run this script again."
}

& docker compose version | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker Compose v2 is unavailable. Update Docker Desktop so that 'docker compose' works."
}

& docker info | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker Desktop is not running or is not ready. Start Docker Desktop, wait until it is ready, then run this script again."
}

Set-Location -LiteralPath $ProjectDir

Write-Host "[INFO] Building and starting SLEEPYLAND-PLUS from local source..."
& docker compose -p sleepyland-plus up -d --build
if ($LASTEXITCODE -ne 0) {
    throw "SLEEPYLAND-PLUS could not be started. Inspect the Docker output above for details."
}

Write-Host ""
& docker compose -p sleepyland-plus ps
Write-Host ""
Write-Host "[INFO] SLEEPYLAND-PLUS is available at http://localhost:8887"
Write-Host "[INFO] Follow service logs with: docker compose -p sleepyland-plus logs -f"
