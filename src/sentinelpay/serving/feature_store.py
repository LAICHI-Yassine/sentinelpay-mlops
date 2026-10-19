import os
import time
from collections import defaultdict
import numpy as np
import redis


class OnlineVelocityStore:
    """
    Real-Time Online Feature Store backed by Redis (with automatic in-memory fail-safe fallback).
    Tracks per-card sliding window transaction velocity (1h and 24h) in sub-millisecond latency.
    """

    def __init__(self, redis_url: str | None = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.client: redis.Redis | None = None
        self._memory_fallback: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self.using_redis = False

        try:
            client = redis.from_url(self.redis_url, decode_responses=True, socket_connect_timeout=0.5)
            client.ping()
            self.client = client
            self.using_redis = True
        except Exception:
            # Fail-safe fallback for local testing / CI when Redis daemon isn't running
            self.using_redis = False

    def get_and_update_card_velocity(
        self, card_uid: str, tx_amount: float, current_ts: float | None = None
    ) -> dict[str, float]:
        """
        Records current transaction and computes 1h & 24h rolling features for card_uid.
        """
        now = current_ts if current_ts is not None else time.time()
        cutoff_1h = now - 3600.0
        cutoff_24h = now - 86400.0

        if self.using_redis and self.client is not None:
            key = f"sentinelpay:card:{card_uid}"
            member = f"{now:.6f}:{tx_amount:.2f}"
            pipe = self.client.pipeline()
            # Remove entries older than 24h, add current tx, fetch all 24h entries, set 24h TTL
            pipe.zremrangebyscore(key, "-inf", cutoff_24h)
            pipe.zadd(key, {member: now})
            pipe.zrange(key, 0, -1, withscores=True)
            pipe.expire(key, 86400)
            _, _, entries, _ = pipe.execute()

            records = [( float(score), float(val.split(":")[1]) ) for val, score in entries]
        else:
            history = self._memory_fallback[card_uid]
            history = [(ts, amt) for ts, amt in history if ts >= cutoff_24h]
            history.append((now, tx_amount))
            self._memory_fallback[card_uid] = history
            records = history

        amts_24h = [amt for _, amt in records]
        amts_1h = [amt for ts, amt in records if ts >= cutoff_1h]

        mean_24h = float(np.mean(amts_24h)) if amts_24h else tx_amount
        std_24h = float(np.std(amts_24h, ddof=1)) if len(amts_24h) > 1 else 0.0

        return {
            "card_tx_count_1h": float(len(amts_1h)),
            "card_amt_sum_1h": float(np.sum(amts_1h)),
            "card_tx_count_24h": float(len(amts_24h)),
            "card_amt_mean_24h": mean_24h,
            "card_amt_std_24h": std_24h,
            "amt_to_mean_24h_ratio": float(tx_amount / (mean_24h + 1.0)),
        }