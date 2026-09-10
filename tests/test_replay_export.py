"""Synthetic capture tests; no private replay data or infrastructure required."""

import base64
import copy
import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "research/junie-replay/fetch_requests.py"
spec = importlib.util.spec_from_file_location("replay_export", SCRIPT)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def events(issue="one", prompt=3, timestamp=1):
    request = {
        "model": "synthetic",
        "max_tokens": 32768,
        "stream": True,
        "messages": [{"role": "user", "content": "## ISSUE DESCRIPTION\n" + issue}],
        "temperature": 1.0,
        "tools": [],
        "enable_thinking": True,
        "seed": timestamp,
    }
    response = {
        "usage": {"prompt_tokens": prompt, "completion_tokens": 1},
        "prompt_token_ids": [1] * prompt,
        "choices": [{"token_ids": [2]}],
    }
    raw = json.dumps(request).encode()
    chunks = [
        ("request_body", raw[:20]),
        ("request_body", raw[20:]),
        ("response_body", json.dumps(response).encode()),
    ]
    result = [
        {
            "seq": i,
            "kind": kind,
            "time_ns": timestamp,
            "body_b64": base64.b64encode(body).decode(),
            "request": str(timestamp),
        }
        for i, (kind, body) in enumerate(chunks)
    ]
    result.append(
        {
            "seq": 3,
            "kind": "request_end",
            "time_ns": timestamp,
            "response_complete": True,
            "request": str(timestamp),
        }
    )
    return result, request


class ReplayExportTest(unittest.TestCase):
    def test_chunk_order_and_preserved_body(self):
        rows, request = events()
        self.assertEqual(export.completed_request(rows[::-1], 32768)[3], request)
        self.assertIsNone(export.completed_request(rows, 128))

    def test_incomplete_gap_disconnected_and_malformed_excluded(self):
        for field in ("response_complete", "capture_gap", "disconnected"):
            rows, _ = events()
            rows[-1][field] = field != "response_complete"
            self.assertIsNone(export.completed_request(rows, 32768))
        rows, _ = events()
        self.assertIsNone(export.completed_request(rows[:-1], 32768))
        rows[0]["body_b64"] = "not base64!"
        self.assertIsNone(export.completed_request(rows, 32768))

    def test_token_count_mismatch_stops_export(self):
        rows, _ = events()
        response = json.loads(base64.b64decode(rows[2]["body_b64"]))
        response["usage"]["prompt_tokens"] = 999
        rows[2]["body_b64"] = base64.b64encode(json.dumps(response).encode()).decode()
        with self.assertRaises(export.ExportError):
            export.completed_request(rows, 32768)

    def test_selection_scope_ranking_and_chronology(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, model, records in [
                (
                    "session",
                    "synthetic",
                    [events("a", 7, 2)[0], events("a", 3, 1)[0], events("b", 5, 3)[0]],
                ),
                ("other", "different", [events("c", 99, 4)[0]]),
            ]:
                directory = root / "run" / name
                directory.mkdir(parents=True)
                (directory / "manifest.json").write_text(
                    json.dumps({"identity": {"model": model}})
                )
                with gzip.open(directory / "events.jsonl.gz", "wt") as target:
                    for record in records:
                        for event in record:
                            target.write(json.dumps(event) + "\n")
            result = export.collect_requests(root, "synthetic", 2, 32768, [])
            self.assertEqual([len(group) for group in result], [2, 1])
            self.assertEqual(result[0], [events("a", 3, 1)[1], events("a", 7, 2)[1]])
            self.assertEqual(
                export.collect_requests(root, "synthetic", 2, 32768, ["run/session"]),
                result,
            )
            with self.assertRaises(export.ExportError):
                export.collect_requests(root, "synthetic", 1, 32768, ["../elsewhere"])

    def test_only_request_json_private_permissions_and_no_overwrite(self):
        request = events()[1]
        before = copy.deepcopy(request)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "requests"
            export.write_requests(output, [[request]])
            directory = output / "trajectory-1"
            self.assertEqual(
                sorted(p.name for p in directory.iterdir()), ["00.json", "warm_up.json"]
            )
            self.assertEqual(
                json.loads((directory / "00.json").read_text()),
                dict(request, stream=False),
            )
            self.assertEqual(
                json.loads((directory / "warm_up.json").read_text()),
                dict(request, stream=False, max_tokens=128),
            )
            self.assertEqual((directory / "00.json").stat().st_mode & 0o777, 0o600)
            self.assertEqual(directory.stat().st_mode & 0o777, 0o700)
            with self.assertRaises(export.ExportError):
                export.write_requests(output, [[request]])
            self.assertEqual(request, before)


if __name__ == "__main__":
    unittest.main()
