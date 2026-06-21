$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Configs = @(
    "user_data\config-meme-limited-dca-dryrun.json",
    "user_data\config-meme-volatility-grid-dryrun.json",
    "user_data\config-meme-anti-martingale-dryrun.json",
    "user_data\config-meme-hedge-proxy-dryrun.json"
)

foreach ($Config in $Configs) {
    $ConfigPath = Join-Path $Root $Config
    $Name = [IO.Path]::GetFileNameWithoutExtension($Config)
    $LogPath = Join-Path $Root ("user_data\logs\" + $Name + ".log")
    Start-Process -FilePath $Python `
        -ArgumentList @("-m", "freqtrade", "trade", "--config", $ConfigPath, "--logfile", $LogPath) `
        -WorkingDirectory $Root `
        -WindowStyle Hidden
    Write-Host "Started $Name -> $LogPath"
}

& $Python (Join-Path $Root "scripts\meme_dryrun_dashboard.py")

Start-Process -FilePath "powershell.exe" `
    -ArgumentList @(
        "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $Root "scripts\watch_meme_dryrun_dashboard.ps1")
    ) `
    -WorkingDirectory $Root `
    -WindowStyle Hidden
Write-Host "Started dashboard refresher -> user_data\meme_dryrun_dashboard.html"
