"""
Gate Meme Volatility PairList filter

Always keeps a fixed set of core pairs. From the remaining meme candidates,
selects the top-N by absolute 24h percentage change (proxy for volatility).
Refreshes every hour by default.
"""

import logging

from freqtrade.exchange.exchange_types import Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting
from freqtrade.util import FtTTLCache


logger = logging.getLogger(__name__)


class GateMemeVolatilityPairList(IPairList):
    """
    Core + Top-N volatile meme coin pairlist filter.

    Must be used AFTER a StaticPairList that provides the full candidate
    whitelist (core pairs + all meme candidates). This handler then:
      1. Always keeps the configured core_pairs.
      2. From the remaining candidates (intersection with meme_candidates),
         ranks by absolute 24h percentage change and keeps the top-N.
    """

    is_pairlist_generator = False
    supports_backtesting = SupportsBacktesting.NO

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._core_pairs: list[str] = self._pairlistconfig.get("core_pairs", [])
        self._number_assets: int = self._pairlistconfig.get("number_assets", 5)
        self._refresh_period: int = self._pairlistconfig.get("refresh_period", 3600)

        self._pair_cache: FtTTLCache = FtTTLCache(maxsize=1, ttl=self._refresh_period)

        if not self._core_pairs:
            logger.warning(
                "GateMemeVolatilityPairList: core_pairs is empty. "
                "This filter will have no effect."
            )

    @staticmethod
    def description() -> str:
        return (
            "Keeps fixed core pairs and selects top-N volatile meme coins "
            "by absolute 24h percentage change."
        )

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "core_pairs": {
                "type": "list",
                "default": [],
                "description": "Core pairs always kept",
                "help": "Fixed pairs that are always included regardless of volatility.",
            },
            "number_assets": {
                "type": "number",
                "default": 5,
                "description": "Number of volatile meme pairs to keep",
                "help": "How many top volatile coins to select from the meme candidates.",
            },
            **IPairList.refresh_period_parameter(),
        }

    @property
    def needstickers(self) -> bool:
        return True

    def short_desc(self) -> str:
        return (
            f"{self.name} - core:{len(self._core_pairs)} "
            f"+ top {self._number_assets} volatile memes"
        )

    def filter_pairlist(self, pairlist: list[str], tickers: Tickers) -> list[str]:
        """
        Filter the pairlist: keep core pairs + top-N volatile meme candidates.
        Cached for refresh_period seconds.
        """
        if not self._enabled:
            return pairlist

        if not self._core_pairs:
            return pairlist

        cached = self._pair_cache.get("pairlist")
        if cached is not None:
            return cached.copy()

        # 1. Identify core pairs that exist in the whitelist
        kept_core = [p for p in self._core_pairs if p in pairlist]

        # 2. Remaining pairs = meme candidates (excluding core)
        meme_pool = [p for p in pairlist if p not in self._core_pairs]

        # 3. Rank meme candidates by absolute 24h % change (volatility proxy)
        scored: list[tuple[str, float]] = []
        for pair in meme_pool:
            ticker = tickers.get(pair)
            if ticker is None:
                continue
            pct = ticker.get("percentage")
            if pct is None:
                continue
            scored.append((pair, abs(float(pct))))

        # Sort by absolute change descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # 4. Take top-N most volatile; fallback to remaining candidates if not enough
        top_volatile = [pair for pair, _ in scored[: self._number_assets]]

        if len(top_volatile) < self._number_assets:
            # Fallback: fill from unscored candidates (no ticker data available)
            scored_names = {pair for pair, _ in scored}
            fallback = [p for p in meme_pool if p not in scored_names]
            needed = self._number_assets - len(top_volatile)
            top_volatile.extend(fallback[:needed])

        result = kept_core + top_volatile

        self.log_once(
            f"Selected {len(result)} pairs: core={kept_core}, volatile={top_volatile}",
            logger.info,
        )

        self._pair_cache["pairlist"] = result.copy()
        return result
