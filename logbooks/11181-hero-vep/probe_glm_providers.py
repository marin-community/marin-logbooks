# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Bounded provider capability probe; caller supplies authenticated session."""

import time


def probe(session, record):
    for provider in ("together", "parasail/fp8", "novita/fp8"):
        base = {"model": "z-ai/glm-5.3", "provider": {"only": [provider],
                "allow_fallbacks": False, "require_parameters": True}, "temperature": 1}
        cases = [
            ("dna_echo", "completions", {"prompt": "ACGTACGTGACTGACTACGT",
             "echo": True, "logprobs": 5, "prompt_logprobs": 5, "max_tokens": 1}),
            ("output_control", "chat/completions", {"messages": [
                {"role": "user", "content": "Reply with A."}],
                "logprobs": True, "top_logprobs": 5, "max_tokens": 512}),
        ]
        for case, route, extra in cases:
            payload = {**base, **extra}
            started = time.monotonic()
            try:
                response = session.post("https://openrouter.ai/api/v1/" + route,
                                        json=payload, timeout=45, allow_redirects=False)
                result = {"status": response.status_code, "response": response.json()}
            except Exception as exc:
                result = {"error_type": type(exc).__name__}
            record({"provider_requested": provider, "case": case, "request": payload,
                    "elapsed": time.monotonic() - started, **result})
            time.sleep(1)
