import httpx

from finder.links import LinkVerifier


def verifier(handler, clock):
    return LinkVerifier("http://museum", 42, transport=httpx.MockTransport(handler), clock=clock)


def info(cid):
    return httpx.Response(200, json={"collection": {"id": cid}, "referralCode": ""})


def test_live_link_to_the_album_is_accepted_and_cached():
    calls = []
    now = [0.0]

    def handler(req):
        calls.append(req.headers["X-Auth-Access-Token"])
        return info(42)

    v = verifier(handler, lambda: now[0])
    assert v.check("NEW") and v.check("NEW")
    assert calls == ["NEW"]
    now[0] = 61
    assert v.check("NEW") and len(calls) == 2


def test_link_to_another_album_is_refused():
    assert not verifier(lambda req: info(7), lambda: 0.0).check("REPAS")


def test_deleted_link_is_refused():
    assert not verifier(lambda req: httpx.Response(410), lambda: 0.0).check("OLD")


def test_museum_down_is_refused():
    def boom(req):
        raise httpx.ConnectError("down")

    assert not verifier(boom, lambda: 0.0).check("T")


def test_password_jwt_is_forwarded():
    seen = {}

    def handler(req):
        seen.update(req.headers)
        return info(42)

    verifier(handler, lambda: 0.0).check("T", "JWT1")
    assert seen["x-auth-access-token-jwt"] == "JWT1"


def test_empty_token_never_calls_museum():
    def handler(req):
        raise AssertionError("called")

    assert not verifier(handler, lambda: 0.0).check("")
