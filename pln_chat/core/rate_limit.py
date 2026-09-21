"""A stdlib token bucket, so a single caller cannot monopolise the service.

Why this exists
---------------
The 2026-09-18 evaluation's §Performance finding was: "There is no
authentication, rate limit, request timeout or cap on list sizes, so any caller
can stall the service with one large ranking." Three of those four are already
fixed — patch 03 put MeTTa in a worker process with a 60 s deadline
(`core/executor.py`) and patch 03/04 capped `ontology_files`, the compound pool
and the echoed row list (`PLN_MAX_*` in config.py). What was left is the
*frequency* axis: nothing stopped one client from issuing a legitimate-looking
request a hundred times a second.

Why it is a token bucket, and not `slowapi`
-------------------------------------------
`slowapi` is not installed and is not in `pln_chat/requirements.txt`; adding a
dependency (plus, in practice, Redis) to bound a demo deployment is a worse
trade than forty lines of stdlib. A token bucket is also the right shape for
this service: a `/patients/markers` listing and a 20-compound DrugAge ranking
are both "one request", but agents legitimately arrive in bursts (discovery
endpoints first, then the real question). Capacity = the per-minute allowance,
refill = allowance/60 per second, so a burst of the full minute's budget is
allowed once and then the caller is metered.

HONESTY CONTRACT
----------------
This is a **courtesy limit, not a security control.**

* It is **per process**. Run two uvicorn workers and the effective limit is
  2x the configured number. It is deliberately not backed by Redis: this
  service is a single-process demo (hyperon's GIL behaviour is why MeTTa work
  is offloaded rather than the whole app being replicated).
* The key is `request.client.host`, which behind a reverse proxy or ngrok is
  the proxy's address, not the caller's — every client then shares one bucket.
  Nothing here reads `X-Forwarded-For`, because trusting a client-settable
  header would make the limiter trivially evadable *and* let one caller starve
  everyone else by forging addresses.
* An attacker with a botnet, or a single caller willing to rotate addresses,
  is not defended against. Put a real proxy in front for that.

What it *does* do is stop one enthusiastic agent or a runaway retry loop from
saturating a deployment, which is the failure mode the evaluation actually hit.

It is OFF by default (`PLN_RATE_LIMIT_PER_MINUTE=0`); the service behaves
exactly as it did before unless an operator turns it on.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

#: Buckets are cheap (four floats), but an unbounded dict keyed on a remote-
#: controlled value is itself a slow memory leak. Past this many tracked keys
#: the limiter drops the ones that have been idle long enough to have refilled
#: completely — their state is indistinguishable from a fresh bucket anyway.
MAX_TRACKED_KEYS = 4096


@dataclass
class _Bucket:
    tokens: float
    updated: float


@dataclass
class TokenBucketLimiter:
    """Allow `per_minute` requests per key, refilling continuously.

    `per_minute <= 0` disables the limiter entirely: `check()` then always
    returns None and no state is kept at all.
    """

    per_minute: int
    _buckets: dict[str, _Bucket] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def enabled(self) -> bool:
        return self.per_minute > 0

    @property
    def capacity(self) -> float:
        """Burst size: one full minute's allowance."""
        return float(max(0, self.per_minute))

    @property
    def refill_per_second(self) -> float:
        return self.capacity / 60.0

    def check(self, key: str, *, now: Optional[float] = None) -> Optional[int]:
        """Spend one token for `key`.

        Returns None when the request is allowed, or the number of whole
        seconds the caller should wait (>= 1, suitable for `Retry-After`) when
        it is not.
        """
        if not self.enabled:
            return None
        moment = time.monotonic() if now is None else now
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                if len(self._buckets) >= MAX_TRACKED_KEYS:
                    self._evict(moment)
                bucket = _Bucket(tokens=self.capacity, updated=moment)
                self._buckets[key] = bucket
            else:
                elapsed = max(0.0, moment - bucket.updated)
                bucket.tokens = min(
                    self.capacity, bucket.tokens + elapsed * self.refill_per_second
                )
                bucket.updated = moment

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return None

            missing = 1.0 - bucket.tokens
            wait = missing / self.refill_per_second if self.refill_per_second else 60.0
            # Retry-After is an integer number of seconds and must not round
            # DOWN to a moment when the token still is not there.
            return max(1, int(math.ceil(wait)))

    def reset(self) -> None:
        """Forget every bucket (used by tests; also handy after a config change)."""
        with self._lock:
            self._buckets.clear()

    def _evict(self, moment: float) -> None:
        """Drop keys that have refilled to full — caller must hold the lock."""
        full_after = 60.0
        stale = [
            key
            for key, bucket in self._buckets.items()
            if moment - bucket.updated >= full_after
        ]
        for key in stale:
            del self._buckets[key]
        if len(self._buckets) >= MAX_TRACKED_KEYS:
            # Everything is active. Drop the oldest half rather than grow: a
            # dropped bucket is a caller who gets a fresh allowance, which is
            # the forgiving direction for a courtesy limit.
            oldest = sorted(self._buckets.items(), key=lambda item: item[1].updated)
            for key, _ in oldest[: len(oldest) // 2]:
                del self._buckets[key]
