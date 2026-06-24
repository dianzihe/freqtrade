param(
    [ValidateRange(1, 6)]
    [int]$OneMinuteDays = 6,
    [ValidateRange(7, 365)]
    [int]$FifteenMinuteDays = 60,
    [string]$DataDir = (Join-Path $PSScriptRoot '..\user_data\data\gate')
)

$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    throw "Freqtrade virtual environment not found: $python"
}

$spotPairs = @(
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT',
    'BNB/USDT', 'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT'
)
$futuresPairs = $spotPairs | ForEach-Object { "$($_):USDT" }

& $python -m freqtrade download-data `
    --exchange gate --trading-mode spot --data-dir $DataDir `
    --data-format-ohlcv feather --pairs $spotPairs `
    --timeframes 1m --days $OneMinuteDays
if ($LASTEXITCODE -ne 0) { throw 'Gate spot 1m download failed.' }

& $python -m freqtrade download-data `
    --exchange gate --trading-mode spot --data-dir $DataDir `
    --data-format-ohlcv feather --pairs $spotPairs `
    --timeframes 15m --days $FifteenMinuteDays --prepend
if ($LASTEXITCODE -ne 0) { throw 'Gate spot 15m download failed.' }

& $python -m freqtrade download-data `
    --exchange gate --trading-mode futures --data-dir $DataDir `
    --data-format-ohlcv feather --pairs $futuresPairs `
    --timeframes 1m --days $OneMinuteDays
if ($LASTEXITCODE -ne 0) { throw 'Gate futures 1m download failed.' }

& $python -m freqtrade download-data `
    --exchange gate --trading-mode futures --data-dir $DataDir `
    --data-format-ohlcv feather --pairs $futuresPairs `
    --timeframes 15m --days $FifteenMinuteDays --prepend
if ($LASTEXITCODE -ne 0) { throw 'Gate futures 15m/mark/funding download failed.' }

& $python (Join-Path $PSScriptRoot 'validate_gate_xinghe_data.py') $DataDir
exit $LASTEXITCODE
