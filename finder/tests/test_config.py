import base58
import pytest

from finder.config import Config, parse_link

KEY = bytes(range(32))
URL = f"https://photos-albums.example/?t=TOK1#{base58.b58encode(KEY).decode()}"


def test_parse_link():
    link = parse_link(URL)
    assert (link.token, link.key) == ("TOK1", KEY)


def test_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FINDER_MUSEUM_URL", "http://ente-museum:8080")
    monkeypatch.setenv("FINDER_MAIN_LINK", URL)
    monkeypatch.setenv("FINDER_PART_LINKS", f"{URL.replace('TOK1', 'A')} ; {URL.replace('TOK1', 'B')}")
    monkeypatch.setenv("FINDER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    c = Config.from_env()
    assert c.main.token == "TOK1"
    assert [p.token for p in c.parts] == ["A", "B"]
    assert c.db_path == tmp_path / "finder.db"
    assert c.face_threshold == 0.6


def test_repr_hides_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("FINDER_MAIN_LINK", URL)
    monkeypatch.setenv("FINDER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret")
    text = repr(Config.from_env())
    assert "sk-secret" not in text and "TOK1" not in text


def test_missing_main_link_is_an_error(monkeypatch):
    monkeypatch.delenv("FINDER_MAIN_LINK", raising=False)
    with pytest.raises(KeyError):
        Config.from_env()
