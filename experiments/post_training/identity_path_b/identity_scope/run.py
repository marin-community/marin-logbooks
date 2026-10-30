# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Original identity callback on exactly66 held-out rows in both thinking modes."""

import hashlib
import importlib.util
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import requests

from experiments.post_training.identity_path_b.trend_identity import selected_rows

PAYLOAD_SHA256 = "be546b49fb4b6427dae7e4024e4c23b001c721a7abce32bfff89e38cad22482f"
HISTORICAL_SHA256 = "50e3c8730b9995b1134db0c0281e6aeab9061f1ea17d0844db24367abf8095b1"


@contextmanager
def payload():
    archive = Path(__file__).with_name("payload.zip")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != PAYLOAD_SHA256:
        raise ValueError("Committed screen payload changed")
    with TemporaryDirectory(prefix="pathb-identity-scope-") as directory:
        with zipfile.ZipFile(archive) as source:
            source.extractall(directory)
        root = Path(directory)
        historical = root / "historical.py"
        if hashlib.sha256(historical.read_bytes()).hexdigest() != HISTORICAL_SHA256:
            raise ValueError("Original9e identity callback changed")
        spec = importlib.util.spec_from_file_location("pathb_scoped_historical_identity", historical)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        yield root, module.__dict__


class RecordingSession(requests.Session):
    """Retain HTTP payloads while the unchanged historical endpoint owns transport."""

    def __init__(self):
        super().__init__()
        self.exchanges = []

    def post(self, url, data=None, json=None, **kwargs):
        exchange = {"request": json}
        self.exchanges.append(exchange)
        try:
            response = super().post(url, data=data, json=json, **kwargs)
        except requests.RequestException as error:
            exchange["error_type"] = type(error).__name__
            raise
        exchange["response"] = {"status_code": response.status_code, "body": response.text}
        return response


def jobs(root: Path, historical: dict) -> list:
    selected = {row.id for row in selected_rows(root / "prompts")}
    rows = [row for row in historical["load_rows"](root / "prompts", "dev") if row.id in selected]
    return [(row, thinking) for row in rows for thinking in (False, True)]


def generate(base_url: str, model: str, root: Path, historical: dict, write_record) -> list[dict]:
    """Drain all132 fixed requests; retain empty/length/error outcomes without repair."""
    tasks = jobs(root, historical)
    if len(tasks) != 132:
        raise ValueError("Require exactly52core+14dev third-person in both modes")

    def one(item):
        index, (row, thinking) = item
        with RecordingSession() as session:
            endpoint = historical["Endpoint"](base_url, model, session=session)
            result = {
                "row_id": row.id,
                "set": row.set,
                "category": row.category,
                "thinking": thinking,
                "condition": "no_system",
                "original_messages": row.messages,
                "own_profile": {"status": "UNGRADED_PENDING_ROLE_AWARE_READ"},
            }
            try:
                answer = historical["generate_one"](endpoint, row, historical["Condition"].NO_SYSTEM, thinking)
                assert answer is not None
                result.update(answer, coverage="response_received")
                result["historical_prescreens"] = result.pop("labels")
            except Exception as error:
                result.update(coverage="transport_or_response_error", error_type=type(error).__name__)
            result["raw_http"] = session.exchanges
            write_record(index, result)
            return result

    with ThreadPoolExecutor(max_workers=8) as executor:
        return list(executor.map(one, enumerate(tasks)))
