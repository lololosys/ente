import builtins
import logging
import tempfile

import numpy as np
import pytest
from starlette.testclient import TestClient

from finder.api import create_app
from finder.query import Finder, NoFace
from finder.store import Store

TOKEN = "WEDTOKEN"


def unit(*v):
    v = np.array(v, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def client(tmp_path):
    s = Store(tmp_path / "f.db")
    s.set_folder("Repas", [5, 6])
    s.set_clip(5, unit(1, 0))

    def selfie(b: bytes):
        if b == b"blank":
            raise NoFace()
        return unit(1, 0)

    f = Finder(s, models=None, face_threshold=0.6, embed_selfie=selfie, embed_text=lambda q: unit(1, 0))
    return TestClient(create_app(f, TOKEN))


H = {"X-Auth-Access-Token": TOKEN}


def test_401_without_token(client):
    assert client.get("/config").status_code == 401


def test_401_wrong_token(client):
    assert client.get("/config", headers={"X-Auth-Access-Token": "other"}).status_code == 401


def test_config_shape(client):
    body = client.get("/config", headers=H).json()
    assert body == {
        "folders": [{"name": "Repas", "coverFileID": 5, "fileIDs": [5, 6]}],
        "features": {"findMe": True, "search": True},
    }


def test_search(client):
    assert client.get("/search", params={"q": "cake"}, headers=H).json() == {"fileIDs": [5]}


def test_find_me_422_no_face(client):
    r = client.post("/find-me", files={"selfie": ("s.jpg", b"blank", "image/jpeg")}, headers=H)
    assert r.status_code == 422 and r.json() == {"error": "no_face"}


def test_find_me_ok(client):
    r = client.post("/find-me", files={"selfie": ("s.jpg", b"face", "image/jpeg")}, headers=H)
    assert r.status_code == 200 and r.json() == {"fileIDs": [], "faces": 1}


def test_413_large_selfie(client):
    big = b"x" * (8 * 1024 * 1024 + 1)
    r = client.post("/find-me", files={"selfie": ("s.jpg", big, "image/jpeg")}, headers=H)
    assert r.status_code == 413


def test_rate_limit_find_me_11th_is_429(client):
    codes = [
        client.post("/find-me", files={"selfie": ("s.jpg", b"face", "image/jpeg")}, headers=H).status_code
        for _ in range(11)
    ]
    assert codes[:10] == [200] * 10 and codes[10] == 429


def test_selfie_not_written(client, monkeypatch):
    def no(*a, **k):
        raise AssertionError("selfie written to disk")

    real_open = builtins.open

    def guarded_open(file, mode="r", *a, **k):
        if any(c in mode for c in "wax+"):
            no()
        return real_open(file, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(tempfile, "SpooledTemporaryFile", no)
    monkeypatch.setattr(tempfile, "NamedTemporaryFile", no)
    r = client.post("/find-me", files={"selfie": ("s.jpg", b"face", "image/jpeg")}, headers=H)
    assert r.status_code == 200


def test_logs_have_no_selfie_or_token(client, caplog):
    caplog.set_level(logging.DEBUG)
    client.post("/find-me", files={"selfie": ("s.jpg", b"SELFIEBYTES", "image/jpeg")}, headers=H)
    client.get("/search", params={"q": "cake"}, headers=H)
    assert "SELFIEBYTES" not in caplog.text and TOKEN not in caplog.text
