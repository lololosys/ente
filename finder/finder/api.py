"""HTTP API for the shared wedding page, mounted at /finder/ by Caddy.

Every call needs the wedding link's access token. Selfies are read into
memory, never spooled to disk, and never logged."""

import re
import time
from collections import defaultdict, deque
from collections.abc import Callable

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .query import Finder, ManyFaces, NoFace

MAX_SELFIE = 8 * 1024 * 1024
MAX_QUERY = 200
LIMITS = {"find-me": (10, 60.0), "search": (60, 60.0)}


class RateLimiter:
    def __init__(self):
        self.hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)

    def allow(self, kind: str, ip: str) -> bool:
        limit, window = LIMITS[kind]
        q, now = self.hits[(kind, ip)], time.monotonic()
        while q and now - q[0] > window:
            q.popleft()
        if len(q) >= limit:
            return False
        q.append(now)
        return True


def client_ip(request: Request) -> str:
    return (
        request.headers.get("cf-connecting-ip")
        or request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        or (request.client.host if request.client else "unknown")
    )


async def read_capped(request: Request, cap: int) -> bytes | None:
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > cap + 64 * 1024:  # room for multipart framing
            return None
    return bytes(body)


def multipart_field(body: bytes, content_type: str, name: str) -> bytes | None:
    """The bytes of one field of a multipart/form-data body, in memory."""
    m = re.search(r'boundary="?([^";]+)"?', content_type)
    if not m:
        return None
    for part in body.split(b"--" + m.group(1).encode()):
        head, sep, data = part.partition(b"\r\n\r\n")
        if sep and re.search(rb'name="' + re.escape(name.encode()) + rb'"', head):
            return data[:-2] if data.endswith(b"\r\n") else data
    return None


def create_app(finder: Finder, verify: Callable[[str, str | None], bool]) -> Starlette:
    """`verify(token, jwt)` says whether a share link is live and is ours."""
    limiter = RateLimiter()

    async def authorized(request: Request) -> bool:
        token = request.headers.get("x-auth-access-token", "")
        jwt = request.headers.get("x-auth-access-token-jwt")
        return await run_in_threadpool(verify, token, jwt)

    def deny() -> JSONResponse:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    async def config(request: Request):
        if not await authorized(request):
            return deny()
        folders = [{"name": f.name, "coverFileID": f.cover_file_id, "fileIDs": f.file_ids} for f in finder.folders()]
        return JSONResponse({"folders": folders, "features": {"findMe": True, "search": True}})

    async def search(request: Request):
        if not await authorized(request):
            return deny()
        if not limiter.allow("search", client_ip(request)):
            return JSONResponse({"error": "slow_down"}, status_code=429)
        q = request.query_params.get("q", "")[:MAX_QUERY]
        return JSONResponse({"fileIDs": await run_in_threadpool(finder.search, q)})

    async def find_me(request: Request):
        if not await authorized(request):
            return deny()
        if not limiter.allow("find-me", client_ip(request)):
            return JSONResponse({"error": "slow_down"}, status_code=429)
        body = await read_capped(request, MAX_SELFIE)
        if body is None:
            return JSONResponse({"error": "too_large"}, status_code=413)
        selfie = multipart_field(body, request.headers.get("content-type", ""), "selfie")
        del body
        if selfie is None:
            return JSONResponse({"error": "no_selfie"}, status_code=400)
        if len(selfie) > MAX_SELFIE:
            return JSONResponse({"error": "too_large"}, status_code=413)
        try:
            result = await run_in_threadpool(finder.find_me, selfie)
        except NoFace:
            return JSONResponse({"error": "no_face"}, status_code=422)
        except ManyFaces:
            return JSONResponse({"error": "many_faces"}, status_code=422)
        finally:
            del selfie
        return JSONResponse({"fileIDs": result.file_ids, "faces": result.faces})

    return Starlette(
        routes=[
            Route("/config", config),
            Route("/search", search),
            Route("/find-me", find_me, methods=["POST"]),
        ]
    )


def app():
    """uvicorn --factory finder.api:app"""
    import logging

    from .config import Config
    from .ente_api import PublicAlbum
    from .links import LinkVerifier
    from .models import Models
    from .store import Store

    logging.basicConfig(level=logging.INFO)
    config = Config.from_env()
    models = Models(config.models_dir)
    # Load the sessions now so the first guest doesn't wait for them.
    models.face_detect, models.face_embed, models.clip_text, models.tokenizer
    finder = Finder(Store(config.db_path), models, config.face_threshold)
    main = PublicAlbum(config.museum_url, config.main.token, config.main.key)
    verifier = LinkVerifier(config.museum_url, main.info()["collection"]["id"])
    return create_app(finder, verifier.check)
