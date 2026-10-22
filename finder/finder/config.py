"""Settings from the environment (the root-only finder.env on the box)."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .crypto import collection_key_from_fragment


@dataclass(frozen=True)
class LinkRef:
    token: str = field(repr=False)
    key: bytes = field(repr=False)


def parse_link(url: str) -> LinkRef:
    u = urlparse(url.strip())
    token = parse_qs(u.query).get("t", [""])[0] or u.path.strip("/").split("/")[0]
    return LinkRef(token, collection_key_from_fragment(u.fragment))


@dataclass(frozen=True)
class Config:
    museum_url: str
    main: LinkRef
    parts: list[LinkRef]
    openrouter_key: str = field(repr=False)
    caption_model: str
    translate_model: str
    data_dir: Path
    face_threshold: float

    @property
    def db_path(self) -> Path:
        return self.data_dir / "finder.db"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @classmethod
    def from_env(cls) -> "Config":
        env = os.environ
        parts = [p for p in env.get("FINDER_PART_LINKS", "").split(";") if p.strip()]
        return cls(
            museum_url=env.get("FINDER_MUSEUM_URL", "http://ente-museum:8080"),
            main=parse_link(env["FINDER_MAIN_LINK"]),
            parts=[parse_link(p) for p in parts],
            openrouter_key=env.get("OPENROUTER_API_KEY", ""),
            caption_model=env.get("FINDER_CAPTION_MODEL", "anthropic/claude-haiku-5.5"),
            translate_model=env.get("FINDER_TRANSLATE_MODEL", "anthropic/claude-haiku-5.5"),
            data_dir=Path(env.get("FINDER_DATA_DIR", "/var/lib/finder")),
            face_threshold=float(env.get("FINDER_FACE_THRESHOLD", "0.6")),
        )
