#!/usr/bin/env python3
"""No-GPU integration tests for the durable Music3 MCP job contract."""

from __future__ import annotations

import importlib.util
import json
import queue
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


MCP_SERVER = Path(__file__).resolve().parents[1] / "scripts" / "mcp_server.py"


class FakeMusicAPI(BaseHTTPRequestHandler):
    should_fail = False
    last_request = None

    def log_message(self, *_args):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        type(self).last_request = json.loads(self.rfile.read(n) or b"{}")
        if self.should_fail:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"synthetic failure")
            return
        audio = b"ID3fake-music"
        self.send_response(200)
        self.send_header("Content-Type", "audio/mpeg")
        self.send_header("Content-Length", str(len(audio)))
        self.send_header("X-Music3-seconds", "0.01")
        self.end_headers()
        self.wfile.write(audio)


def load_mcp_module():
    spec = importlib.util.spec_from_file_location("music3_mcp_async_test", MCP_SERVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tool_text(module, name: str, arguments: dict) -> dict:
    response = module._handle({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    })
    return json.loads(response["result"]["content"][0]["text"])


class AsyncMusicMCPTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        FakeMusicAPI.should_fail = False
        FakeMusicAPI.last_request = None
        self.api = ThreadingHTTPServer(("127.0.0.1", 0), FakeMusicAPI)
        self.api_thread = threading.Thread(target=self.api.serve_forever, daemon=True)
        self.api_thread.start()
        self.addCleanup(self.stop_api)
        self.mcp = load_mcp_module()
        self.mcp.API = f"http://127.0.0.1:{self.api.server_port}"
        self.mcp.OUT_DIR = Path(self.tmp.name) / "output"
        self.mcp.JOB_STORE = Path(self.tmp.name) / "state" / "jobs.json"
        self.mcp._jobs.clear()
        self.mcp._job_queue = queue.Queue()
        threading.Thread(target=self.mcp._job_worker, daemon=True).start()

    def stop_api(self):
        self.api.shutdown()
        self.api_thread.join(timeout=1)
        self.api.server_close()

    def wait_for_terminal(self, job_id: str) -> dict:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status = tool_text(self.mcp, "get_music_job", {"job_id": job_id})
            if status["status"] in {"succeeded", "failed"}:
                return status
            time.sleep(0.01)
        self.fail(f"job {job_id} did not finish")

    def test_submission_is_immediate_and_result_is_durable(self):
        start = time.monotonic()
        submitted = tool_text(self.mcp, "generate_music", {
            "caption": "test caption", "duration_seconds": 270,
            "out_path": str(Path(self.tmp.name) / "song.mp3"),
        })
        self.assertLess(time.monotonic() - start, 0.25)
        self.assertEqual(submitted["status"], "queued")
        completed = self.wait_for_terminal(submitted["job_id"])
        self.assertEqual(completed["status"], "succeeded")
        self.assertEqual(completed["result"]["path"], str(Path(self.tmp.name) / "song.mp3"))
        self.assertTrue(Path(completed["result"]["path"]).is_file())
        self.assertEqual(FakeMusicAPI.last_request["max_frames"], 9000)
        self.assertEqual(FakeMusicAPI.last_request["target_seconds"], 270)
        self.assertIs(FakeMusicAPI.last_request["rewrite"], True)
        saved = json.loads(self.mcp.JOB_STORE.read_text())
        self.assertEqual(saved["jobs"][submitted["job_id"]]["status"], "succeeded")

    def test_generation_failure_is_terminal_and_queryable(self):
        FakeMusicAPI.should_fail = True
        submitted = tool_text(self.mcp, "generate_music", {"caption": "test caption"})
        completed = self.wait_for_terminal(submitted["job_id"])
        self.assertEqual(completed["status"], "failed")
        self.assertEqual(completed["error"]["code"], "generation_failed")

    def test_restart_marks_unfinished_work_as_typed_failure(self):
        # The worker is blocked on the old queue; this models a persisted submit
        # immediately before process shutdown.
        self.mcp._job_queue = queue.Queue()
        submitted = self.mcp._submit_job({"caption": "test caption"})
        self.mcp._load_jobs()
        status = tool_text(self.mcp, "get_music_job", {"job_id": submitted["job_id"]})
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["error"]["code"], "mcp_restarted_before_completion")

    def test_tool_schema_uses_duration_not_frame_cap(self):
        tool = next(item for item in self.mcp.TOOLS if item["name"] == "generate_music")
        properties = tool["inputSchema"]["properties"]
        self.assertIn("duration_seconds", properties)
        self.assertNotIn("max_frames", properties)
        self.assertEqual(properties["duration_seconds"]["maximum"], 300)

    def test_explicit_caption_duration_is_inferred_when_field_is_omitted(self):
        normalized = self.mcp._normalize_generate_args({
            "caption": "Make a 4:30 dark country duet",
        })
        self.assertEqual(normalized["duration_seconds"], 270)

    def test_legacy_short_frame_cap_is_rejected_with_migration_hint(self):
        response = self.mcp._handle({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "generate_music",
                "arguments": {"caption": "test", "max_frames": 5625},
            },
        })
        self.assertTrue(response["result"]["isError"])
        self.assertIn("duration_seconds", response["result"]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
