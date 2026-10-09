import pytest

from wq_pipeline.client import WQClient


class FakeResponse:
    status_code = 200
    text = "{}"

    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


def test_get_user_alphas_paginates_and_filters_status(monkeypatch):
    client = WQClient("email", "password")
    calls = []

    def fake_request(method, path, **kwargs):
        calls.append((method, path, kwargs["params"].copy()))
        if kwargs["params"]["offset"] == 0:
            return FakeResponse({"count": 3, "results": [{"id": "A"}, {"id": "B"}]})
        return FakeResponse({"count": 3, "results": [{"id": "C"}]})

    monkeypatch.setattr(client, "_request", fake_request)

    assert client.get_user_alphas(status="ACTIVE", page_size=2) == [
        {"id": "A"},
        {"id": "B"},
        {"id": "C"},
    ]
    assert calls == [
        ("GET", "/users/self/alphas", {"limit": 2, "offset": 0, "status": "ACTIVE"}),
        ("GET", "/users/self/alphas", {"limit": 2, "offset": 2, "status": "ACTIVE"}),
    ]


def test_get_user_alphas_rejects_invalid_page_size():
    client = WQClient("email", "password")
    with pytest.raises(ValueError, match="page_size"):
        client.get_user_alphas(page_size=0)
