"""Asks the museum whether a share link is live and points at our album,
so deleted or replaced links stop working here too."""

import time
from collections.abc import Callable

import httpx

CACHE_SECONDS = 60.0


class LinkVerifier:
    def __init__(
        self,
        museum_url: str,
        collection_id: int,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.collection_id = collection_id
        self._http = httpx.Client(base_url=museum_url.rstrip("/"), transport=transport, timeout=10)
        self._clock = clock
        self._cache: dict[tuple[str, str | None], tuple[float, bool]] = {}

    def check(self, token: str, jwt: str | None = None) -> bool:
        if not token:
            return False
        key = (token, jwt)
        hit = self._cache.get(key)
        if hit and self._clock() - hit[0] < CACHE_SECONDS:
            return hit[1]
        ok = self._ask(token, jwt)
        if len(self._cache) > 10_000:
            self._cache.clear()
        self._cache[key] = (self._clock(), ok)
        return ok

    def _ask(self, token: str, jwt: str | None) -> bool:
        headers = {"X-Auth-Access-Token": token, "X-Client-Package": "io.ente.finder"}
        if jwt:
            headers["X-Auth-Access-Token-JWT"] = jwt
        try:
            res = self._http.get("/public-collection/info", headers=headers)
            return res.status_code == 200 and res.json()["collection"]["id"] == self.collection_id
        except Exception:
            return False
