$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Dashboard = Join-Path $Root "scripts\meme_dryrun_dashboard.py"

while ($true) {
    & $Python $Dashboard | Out-Null
    Start-Sleep -Seconds 60
}
