from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TierConfig:
    checkpoint: str
    temperature: float = 0.0
    search_depth: int = 0
    search_weight: float = 1.0
    search_top_k: int = 6
    search_max_nodes: int = 0


TIER_CONFIGS: dict[str, TierConfig] = {
    "1000": TierConfig(checkpoint="1000", temperature=0.3),
    "1500": TierConfig(checkpoint="1500", search_depth=3, search_weight=1.0),
    "2000": TierConfig(checkpoint="1500", search_depth=6, search_weight=4.0),
}
