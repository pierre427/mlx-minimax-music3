#!/usr/bin/env python3
"""No-model HTTP contract tests for the Music3 production API."""

from __future__ import annotations

import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

SERVER = Path(__file__).resolve().parents[1] / "scripts" / "server.py"


def load_server_module():
    spec = importlib.util.spec_from_file_location("music3_server_duration_test", SERVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def api():
    module = load_server_module()
    calls = []

    def fake_generate(caption, lyrics, seed, target_duration, num_steps, rewrite):
        calls.append({
            "caption": caption,
            "lyrics": lyrics,
            "seed": seed,
            "target_duration": target_duration,
            "num_steps": num_steps,
            "rewrite": rewrite,
        })
        return b"ID3duration-contract", {"finish_reason": "model_eos", "max_frames": 9000}

    module._generate = fake_generate
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), module.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield module, calls, f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        thread.join(timeout=1)
        httpd.server_close()


def post_json(url: str, body: dict):
    request = urllib.request.Request(
        url + "/v1/audio/music",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=2)


def test_api_accepts_target_and_fixed_9000_ceiling(api):
    _module, calls, url = api
    with post_json(url, {
        "caption": "brief dark country song",
        "target_duration": "4:30",
        "max_frames": 9000,
    }) as response:
        assert response.read() == b"ID3duration-contract"
        assert response.headers["X-Music3-finish_reason"] == "model_eos"
        assert response.headers["X-Music3-max_frames"] == "9000"
    assert calls[0]["target_duration"] == "4:30"
    assert calls[0]["rewrite"] is True


def test_api_rejects_legacy_short_frame_cap(api):
    _module, calls, url = api
    with pytest.raises(urllib.error.HTTPError) as caught:
        post_json(url, {"caption": "brief song", "max_frames": 5625})
    assert caught.value.code == 400
    error = json.loads(caught.value.read())
    assert "target_duration" in error["error"]
    assert calls == []
