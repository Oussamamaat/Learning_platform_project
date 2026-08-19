# Start the API backend detached, so it survives the terminal (or the agent
# session) that launched it.
#
# Why no --reload: the reloader runs a watcher parent plus a worker child.
# Editing a source file mid-request restarts the worker, and a file saved in a
# syntactically-incomplete state takes the server down entirely -- which shows
# up in the browser as "Cannot reach backend at http://127.0.0.1:8000".
# Restart this script by hand after changing backend code instead.
#
#   .\start_backend.ps1                       # default tenant
#   .\start_backend.ps1 -TenantId my_tenant   # isolated tenant
#   .\start_backend.ps1 -Stop                 # stop whatever owns port 8000

param(
    [string]$TenantId = "",
    [switch]$Stop
)

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot
$python = Join-Path $repo ".gguf_venv\Scripts\python.exe"
$logDir = Join-Path $repo "out"
$log = Join-Path $logDir "backend.log"
$errLog = Join-Path $logDir "backend.err.log"

function Get-BackendPid {
    # -ErrorAction SilentlyContinue: no listener is the normal case, not a fault.
    $conn = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
    if ($conn) { return $conn.OwningProcess | Select-Object -First 1 }
    return $null
}

$existing = Get-BackendPid
if ($existing) {
    Write-Host "Stopping existing backend (PID $existing) ..."
    Stop-Process -Id $existing -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

if ($Stop) {
    Write-Host "Backend stopped."
    exit 0
}

if (-not (Test-Path $python)) { throw "Python venv not found at $python" }
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

if ($TenantId) {
    $env:DEFAULT_TENANT_ID = $TenantId
    Write-Host "Tenant: $TenantId"
}

Start-Process -FilePath $python `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000" `
    -WorkingDirectory $repo `
    -WindowStyle Hidden `
    -RedirectStandardOutput $log `
    -RedirectStandardError $errLog

Write-Host "Starting backend ... (loading the embedding model takes ~15s)"
foreach ($i in 1..60) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) {
            Write-Host "Backend READY at http://127.0.0.1:8000  (PID $(Get-BackendPid))"
            Write-Host "Logs: $log"
            exit 0
        }
    } catch { }
}

Write-Warning "Backend did not become healthy within 60s. Check $errLog"
exit 1
