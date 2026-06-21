$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
& $Python (Join-Path $Root "scripts\meme_dryrun_dashboard.py")
