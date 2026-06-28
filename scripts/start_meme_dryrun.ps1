$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$ConfigWriter = Join-Path $Root "scripts\write_dryrun_strategy_configs.py"
$Dashboard = Join-Path $Root "scripts\meme_dryrun_dashboard.py"

# ---- 1. 生成配置 ----
Write-Host "[1/4] Generating strategy configs ..."
& $Python $ConfigWriter | Out-Null
Write-Host "       Configs written."

# ---- 2. 启动所有 dry-run 进程 ----
$Configs = Get-ChildItem -Path (Join-Path $Root "user_data\config") -Filter "config-dryrun-*.json" |
    Sort-Object -Property Name

Write-Host "[2/4] Starting $($Configs.Count) dry-run instances ..."
$Ports = @()
foreach ($Config in $Configs) {
    $ConfigPath = $Config.FullName
    $Name = [IO.Path]::GetFileNameWithoutExtension($Config.Name)
    $LogPath = Join-Path $Root ("user_data\logs\" + $Name + ".log")

    # 提取端口号
    $Json = Get-Content $ConfigPath -Raw | ConvertFrom-Json
    $Port = $Json.api_server.listen_port
    $Ports += $Port

    Start-Process -FilePath $Python `
        -ArgumentList @("-m", "freqtrade", "trade", "--config", $ConfigPath, "--logfile", $LogPath) `
        -WorkingDirectory $Root `
        -WindowStyle Hidden
    Write-Host "       Started $Name (port $Port) -> $LogPath"
}

# ---- 3. 健康检查：等待 API 就绪 ----
Write-Host "[3/4] Waiting for API servers to become ready ..."
$Healthy = @{}
$Timeout = [DateTime]::Now.AddSeconds(60)

while ([DateTime]::Now -lt $Timeout) {
    foreach ($P in $Ports) {
        if ($Healthy.ContainsKey($P)) { continue }
        try {
            $Response = Invoke-WebRequest -Uri "http://127.0.0.1:$P/api/v1/ping" `
                -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
            if ($Response.StatusCode -eq 200) {
                $Healthy[$P] = $true
            }
        } catch {
            # 尚未就绪，继续等待
        }
    }
    if ($Healthy.Count -ge $Ports.Count) { break }
    Start-Sleep -Seconds 3
}

Write-Host "       Health check: $($Healthy.Count) / $($Ports.Count) instances responding."

$Failed = $Ports | Where-Object { -not $Healthy.ContainsKey($_) }
if ($Failed.Count -gt 0) {
    Write-Warning "       WARNING: Port(s) $($Failed -join ', ') did NOT respond after 60s!"
    Write-Warning "       Check logs in user_data\logs\ for errors."
} else {
    Write-Host "       All instances healthy."
}

# ---- 4. 生成看板 + 启动自动刷新 ----
Write-Host "[4/4] Generating dashboard and starting auto-refresh ..."
& $Python $Dashboard

Start-Process -FilePath "powershell.exe" `
    -ArgumentList @(
        "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $Root "scripts\watch_meme_dryrun_dashboard.ps1")
    ) `
    -WorkingDirectory $Root `
    -WindowStyle Hidden
Write-Host "       Dashboard: $Root\user_data\meme_dryrun_dashboard.html"
Write-Host ""
Write-Host "All done. Open the dashboard in your browser to monitor."
