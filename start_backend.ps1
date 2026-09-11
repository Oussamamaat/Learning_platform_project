# Start the API backend detached, so it survives the terminal (or the agent
# session) that launched it.
#
# Why no --reload: the reloader runs a watcher parent plus a worker child.
# Editing a source file mid-request restarts the worker, and a file saved in a
# syntactically-incomplete state takes the server down entirely -- which shows
# up in the browser as "Cannot reach backend at http://127.0.0.1:8000".
# Restart this script by hand after changing backend code instead.
#
#   .\start_backend.ps1                       # default tenant, port 8000
#   .\start_backend.ps1 -TenantId my_tenant   # isolated tenant
#   .\start_backend.ps1 -Port 8001            # vLLM worktree: LLM_BACKEND=vllm
#                                              # smoke test (plan Phase A11) --
#                                              # 8000 stays the stable folder's
#                                              # port so the two never collide
#                                              # (see the plan's W3 rules)
#   .\start_backend.ps1 -Stop                 # stop whatever owns -Port

param(
    [string]$TenantId = "",
    [int]$Port = 8000,
    [switch]$Stop
)

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot
$python = Join-Path $repo ".gguf_venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    # Plan rev 2's W2 rule: the _vllm worktree deliberately has no venv of
    # its own -- it reuses the stable folder's .gguf_venv (no editable
    # install of `app`, so it imports whichever folder is the CWD). This
    # worktree's own name always ends in _vllm; the stable sibling sits
    # next to it with that suffix removed.
    $siblingRepo = $repo -replace '_vllm$', ''
    $sharedPython = Join-Path $siblingRepo ".gguf_venv\Scripts\python.exe"
    if (Test-Path $sharedPython) {
        $python = $sharedPython
        Write-Host "No local .gguf_venv -- using shared venv at $python"
    }
}
$logDir = Join-Path $repo "out"
# Namespaced by port so running 8000 (stable) and 8001 (this worktree, or a
# second instance for a side-by-side check) at once never clobbers one
# another's log file.
$log = Join-Path $logDir "backend.$Port.log"
$errLog = Join-Path $logDir "backend.$Port.err.log"

function Get-BackendPid {
    param([int]$ListenPort)
    # -ErrorAction SilentlyContinue: no listener is the normal case, not a fault.
    $conn = Get-NetTCPConnection -LocalPort $ListenPort -State Listen -ErrorAction SilentlyContinue
    if ($conn) { return $conn.OwningProcess | Select-Object -First 1 }
    return $null
}

$existing = Get-BackendPid -ListenPort $Port
if ($existing) {
    Write-Host "Stopping existing backend on port $Port (PID $existing) ..."
    Stop-Process -Id $existing -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

if ($Stop) {
    Write-Host "Backend on port $Port stopped."
    exit 0
}

if (-not (Test-Path $python)) { throw "Python venv not found at $python" }
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

if ($TenantId) {
    $env:DEFAULT_TENANT_ID = $TenantId
    Write-Host "Tenant: $TenantId"
}

Start-Process -FilePath $python `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$Port" `
    -WorkingDirectory $repo `
    -WindowStyle Hidden `
    -RedirectStandardOutput $log `
    -RedirectStandardError $errLog

Write-Host "Starting backend on port $Port ... (loading the embedding model takes ~15s)"
foreach ($i in 1..60) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) {
            Write-Host "Backend READY at http://127.0.0.1:$Port  (PID $(Get-BackendPid -ListenPort $Port))"
            Write-Host "Logs: $log"
            exit 0
        }
    } catch { }
}

Write-Warning "Backend did not become healthy within 60s. Check $errLog"
exit 1
