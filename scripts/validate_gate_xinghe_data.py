import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


ASSETS = ("BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "AVAX")
TIMEFRAMES = ("1m", "15m")


def validate_feather(path: Path) -> dict:
    dataframe = pd.read_feather(path)
    required = {"date", "open", "high", "low", "close", "volume"}
    missing_columns = sorted(required.difference(dataframe.columns))
    if missing_columns:
        return {
            "path": str(path),
            "rows": len(dataframe),
            "missing_columns": missing_columns,
            "duplicate_timestamps": 0,
            "invalid_ohlc_rows": len(dataframe),
            "start": None,
            "end": None,
            "ok": False,
        }

    duplicate_timestamps = int(dataframe["date"].duplicated().sum())
    price_columns = ["open", "high", "low", "close"]
    if "funding_rate" in path.name:
        valid_values = pd.DataFrame(
            np.isfinite(dataframe[price_columns].to_numpy()), index=dataframe.index
        ).all(axis=1)
        invalid_ohlc_rows = int((~valid_values).sum())
    else:
        positive = dataframe[price_columns].gt(0).all(axis=1)
        coherent = (
            (dataframe["high"] >= dataframe[["open", "close", "low"]].max(axis=1))
            & (dataframe["low"] <= dataframe[["open", "close", "high"]].min(axis=1))
        )
        invalid_ohlc_rows = int((~(positive & coherent)).sum())
    dates = pd.to_datetime(dataframe["date"], utc=True)
    return {
        "path": str(path),
        "rows": len(dataframe),
        "missing_columns": [],
        "duplicate_timestamps": duplicate_timestamps,
        "invalid_ohlc_rows": invalid_ohlc_rows,
        "start": dates.min().isoformat() if not dates.empty else None,
        "end": dates.max().isoformat() if not dates.empty else None,
        "ok": bool(len(dataframe) and duplicate_timestamps == 0 and invalid_ohlc_rows == 0),
    }


def _required_paths(
    data_dir: Path, assets: Iterable[str], timeframes: Iterable[str]
) -> list[Path]:
    paths: list[Path] = []
    for asset in assets:
        for timeframe in timeframes:
            paths.append(data_dir / f"{asset}_USDT-{timeframe}.feather")
            paths.append(
                data_dir / "futures" / f"{asset}_USDT_USDT-{timeframe}-futures.feather"
            )
        paths.append(data_dir / "futures" / f"{asset}_USDT_USDT-1h-mark.feather")
        paths.append(data_dir / "futures" / f"{asset}_USDT_USDT-1h-funding_rate.feather")
    return paths


def validate_inventory(
    data_dir: Path,
    assets: Iterable[str] = ASSETS,
    timeframes: Iterable[str] = TIMEFRAMES,
) -> dict:
    data_dir = Path(data_dir)
    required_paths = _required_paths(data_dir, tuple(assets), tuple(timeframes))
    missing = [
        path.relative_to(data_dir).as_posix() for path in required_paths if not path.exists()
    ]
    files = [validate_feather(path) for path in required_paths if path.exists()]
    return {
        "data_dir": str(data_dir),
        "required_files": len(required_paths),
        "present_files": len(files),
        "missing": missing,
        "files": files,
        "ok": not missing and all(item["ok"] for item in files),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = validate_inventory(args.data_dir)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
