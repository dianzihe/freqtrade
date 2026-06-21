param(
    [ValidateRange(1, 30)]
    [int]$Days = 5,

    [string]$ConfigPath = "user_data\config-kucoin-data-200.json",

    [string]$PairsFile = "user_data\pairs-kucoin-spot-200.json",

    [string]$Timeframe = "1m",

    [string]$TradingMode = "spot",

    [switch]$NoParallel,

    [switch]$Prepend,

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$pythonPath = Join-Path $repoRoot ".venv\Scripts\python.exe"
$resolvedConfig = Join-Path $repoRoot $ConfigPath
$resolvedPairs = Join-Path $repoRoot $PairsFile

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Python executable not found: $pythonPath"
}

if (-not (Test-Path -LiteralPath $resolvedConfig -PathType Leaf)) {
    throw "Config file not found: $resolvedConfig"
}

if (-not (Test-Path -LiteralPath $resolvedPairs -PathType Leaf)) {
    throw "Pairs file not found: $resolvedPairs"
}

$freqtradeArgs = @(
    "-m", "freqtrade",
    "download-data",
    "-c", $ConfigPath,
    "--pairs-file", $PairsFile,
    "--timeframes", $Timeframe,
    "--days", "$Days",
    "--trading-mode", $TradingMode
)

if ($NoParallel) {
    $freqtradeArgs += "--no-parallel-download"
}

if ($Prepend) {
    $freqtradeArgs += "--prepend"
}

$displayCommand = ".\.venv\Scripts\python.exe " + ($freqtradeArgs -join " ")

Write-Host "Working directory: $repoRoot"
Write-Host "Data update mode: append missing OHLCV candles; existing feather files are not edited directly."
Write-Host "Command: $displayCommand"

if ($DryRun) {
    Write-Host "Dry run only. No data download was started."
    exit 0
}

Push-Location $repoRoot
try {
    & $pythonPath @freqtradeArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
