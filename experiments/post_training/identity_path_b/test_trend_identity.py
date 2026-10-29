# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from experiments.post_training.identity_path_b.trend_identity import Row, generate_trend_row


def test_off_native_chat_payload_preserves_original_system_and_multiturn_context_and_keeps_grades_pending():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "choices": [{"message": {"content": "I am Marin's Latest MoE."}, "finish_reason": "stop"}],
                        "usage": {"completion_tokens": 8},
                    }
                ).encode()
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        context = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "A prior question"},
            {"role": "assistant", "content": "A prior answer"},
            {"role": "user", "content": "What are you called?"},
        ]
        row = Row("fixture", "identity", "generic_system", "fixture", "dev", context, ["name"])
        result = generate_trend_row(f"http://127.0.0.1:{server.server_port}", "fixture-model", row)
        assert requests == [
            {
                "model": "fixture-model",
                "messages": context,
                "temperature": 0.0,
                "max_tokens": 1024,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ]
        assert result["coverage"] == "response_received"
        assert result["historical_prescreens"]["name"] is False
        assert result["own_profile"]["name_b"] is None
        assert result["raw_response"]["status_code"] == 200
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_transport_failure_has_one_attempt_and_never_becomes_an_incorrect_model_answer():
    attempts = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            attempts.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b'{"error":"fixture temporary failure"}')

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        row = Row(
            "fixture",
            "control",
            "third_person",
            "third-model",
            "dev",
            [{"role": "user", "content": "Who developed a different product?"}],
            [],
        )
        result = generate_trend_row(f"http://127.0.0.1:{server.server_port}", "fixture-model", row)
        assert len(attempts) == 1
        assert attempts[0]["max_tokens"] == 2048
        assert result["coverage"] == "transport_or_response_error"
        assert "answer" not in result and "historical_prescreens" not in result and "own_profile" not in result
        assert result["raw_response"]["status_code"] == 503
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
