"""One authorized prediction-only portal submission; no code/report metadata permitted."""

import datetime as dt
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from somerra_openadmet.constants import ACTIVITY_ENDPOINTS, IDENTIFIER_COLUMNS, TEST_FILE
from somerra_openadmet.data import load_test
from somerra_openadmet.download import sha256_file
from somerra_openadmet.validation import validate_submission

BASE = "https://openadmet-cyp-challenge.hf.space"
PARAMETERS = [
    "username",
    "user_alias",
    "anon_checkbox",
    "participant_name",
    "discord_username",
    "email",
    "affiliation",
    "model_tag",
    "paper_checkbox",
    "proprietary_data_checkbox",
    "open_code_checkbox",
    "track_select",
    "file_input",
]


def utcnow():
    return dt.datetime.now(dt.UTC)


def json_request(path, data=None, content_type="application/json"):
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": content_type} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    retry_safe = data is None or path.startswith("/gradio_api/call/partial")
    for attempt in range(3 if retry_safe else 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError):
            if not retry_safe or attempt == 2:
                raise
            time.sleep(15)


def event_result(api_name, event_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", event_id):
        raise ValueError("Unexpected event identifier")
    with urllib.request.urlopen(
        BASE + "/gradio_api/call/" + api_name + "/" + event_id, timeout=90
    ) as response:
        event = ""
        for line in response:
            text = line.decode().strip()
            if text.startswith("event:"):
                event = text.split(":", 1)[1].strip()
            if text.startswith("data:") and event in ("complete", "error"):
                return event, json.loads(text.split(":", 1)[1])
    raise RuntimeError("No terminal portal event; acceptance is unknown")


def response_message(value):
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError("Unexpected submission response schema")
    item = value[0]
    message = item.get("value") if isinstance(item, dict) else item
    if not isinstance(message, str):
        raise ValueError("Unexpected submission status type")
    return message


def payload(remote_path):
    if not isinstance(remote_path, str) or not remote_path.startswith("/") or "://" in remote_path:
        raise ValueError("Expected portal-uploaded file path")
    fields = [
        "Jenhan",
        "Somerra",
        True,
        "",
        "",
        "",
        "",
        "",
        False,
        False,
        False,
        "Regression Prediction",
        {
            "path": remote_path,
            "orig_name": "activity_submission.csv",
            "meta": {"_type": "gradio.FileData"},
        },
    ]
    check_payload(fields)
    return fields


def check_payload(fields):
    if len(fields) != 13 or fields[:3] != ["Jenhan", "Somerra", True]:
        raise ValueError("Unexpected submission identity")
    if fields[3:8] != ["", "", "", "", ""] or any(v is not False for v in fields[8:11]):
        raise ValueError("Prediction-only policy: no report/code links or publication opt-in")
    file = fields[12]
    if (
        fields[11] != "Regression Prediction"
        or not isinstance(file, dict)
        or set(file) != {"path", "orig_name", "meta"}
    ):
        raise ValueError("Prediction-only regression CSV required")
    if file["orig_name"] != "activity_submission.csv" or file["meta"] != {
        "_type": "gradio.FileData"
    }:
        raise ValueError("Unexpected file metadata")
    if (
        not isinstance(file["path"], str)
        or not file["path"].startswith("/")
        or "://" in file["path"]
    ):
        raise ValueError("Only the portal upload path may be submitted")


def live_preflight():
    api = json_request("/gradio_api/info")
    names = [
        p["parameter_name"] for p in api["named_endpoints"]["/submit_predictions"]["parameters"]
    ]
    if names != PARAMETERS:
        raise ValueError("Submission API changed; review required before uploading")
    config = json_request("/config")
    components = {c["id"]: c for c in config["components"]}
    ids = []
    for c in components.values():
        props = c.get("props", {})
        value = props.get("value", {})
        headers = props.get("headers") or (
            value.get("headers", []) if isinstance(value, dict) else []
        )
        if c["type"] == "leaderboard" and "MA-ST-RAE" in headers:
            ids.append(c["id"])
    if not ids:
        raise ValueError("Cannot locate regression leaderboard for cooldown check")
    endpoint = next(
        d["api_name"]
        for d in config["dependencies"]
        if d.get("outputs") == [ids[0]] and str(d.get("api_name", "")).startswith("partial")
    )
    spec = api["named_endpoints"]["/" + endpoint]
    if spec["parameters"] or spec["returns"][0].get("component") != "Leaderboard":
        raise ValueError("Unexpected leaderboard callback")
    event = json_request("/gradio_api/call/" + endpoint, json.dumps({"data": []}).encode())
    kind, values = event_result(endpoint, event["event_id"])
    if kind != "complete" or not isinstance(values, list) or not isinstance(values[0], dict):
        raise ValueError("Leaderboard preflight failed")
    table = values[0]
    headers = table["headers"]
    rows = [
        dict(zip(headers, r, strict=True))
        for r in table["data"]
        if r[headers.index("Username")] == "Somerra"
    ]
    return {
        "api_parameters": names,
        "leaderboard_rows": rows,
        "reference_rows": [
            dict(zip(headers, r, strict=True))
            for r in table["data"]
            if "briford" in str(r[headers.index("Username")]).lower()
        ],
        "observed_utc": utcnow().isoformat(),
    }


def wait_until(when):
    while utcnow() < when:
        time.sleep(min(30, max(0, (when - utcnow()).total_seconds())))


def submit(candidate_dir, raw_dir):
    import pandas as pd

    candidate_dir = Path(candidate_dir)
    raw_dir = Path(raw_dir).resolve()
    file = candidate_dir / "activity_submission.csv"
    manifest = json.loads((candidate_dir / "run_manifest.json").read_text())
    if manifest.get("status") != "completed":
        raise ValueError("Export incomplete")
    if manifest.get("submission_authorized") is not True:
        raise ValueError("No explicit submission authorization recorded")
    raw = file.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != manifest["submission_sha256"]:
        raise ValueError("Prediction checksum changed")
    if manifest.get("open_source_code") is not False or manifest.get("model_report_link") != "":
        raise ValueError("Export metadata conflicts with prediction-only policy")
    if sha256_file(raw_dir / TEST_FILE) != manifest["test_sha256"]:
        raise ValueError("Blinded snapshot changed")
    frame = pd.read_csv(file)
    expected_columns = list(IDENTIFIER_COLUMNS) + list(ACTIVITY_ENDPOINTS)
    if len(frame.columns) != len(expected_columns) or set(frame.columns) != set(expected_columns):
        raise ValueError("Only official identifiers and four prediction columns may be uploaded")
    validation = validate_submission(file, load_test(raw_dir), "activity")
    if not validation.valid or validation.warnings:
        raise ValueError(validation.to_dict())
    if len(raw) > 5_000_000:
        raise ValueError("Unexpected prediction CSV size")
    receipt = candidate_dir / "submission_receipt.json"
    if receipt.exists():
        raise FileExistsError("Receipt exists; reconcile it before any retry")
    with (candidate_dir / "submission.lock").open("x") as lock:
        lock.write(utcnow().isoformat())
    record = {
        "status": "preflight",
        "started_at_utc": utcnow().isoformat(),
        "space": "openadmet/cyp-challenge",
        "track": "Regression Prediction",
        "huggingface_username": "Jenhan",
        "leaderboard_alias": "Somerra",
        "anonymous": True,
        "include_in_publication": False,
        "used_proprietary_data": False,
        "open_source_code": False,
        "model_report_link": "",
        "submission_file": "activity_submission.csv",
        "submission_sha256": digest,
        "validation": validation.to_dict(),
    }

    def save():
        tmp = receipt.with_suffix(".next.json")
        tmp.write_text(json.dumps(record, indent=2))
        tmp.replace(receipt)

    save()
    try:
        preflight = live_preflight()
        (candidate_dir / "portal-before-submission.json").write_text(
            json.dumps(preflight, indent=2)
        )
        dates = []
        for row in preflight["leaderboard_rows"]:
            when = dt.datetime.strptime(row["Submitted"], "%Y-%m-%d %H:%M UTC").replace(
                tzinfo=dt.UTC
            )
            dates.append(when + dt.timedelta(hours=12, minutes=1, seconds=5))
        if dates and utcnow() < max(dates):
            record.update(status="waiting_for_cooldown", eligible_after_utc=max(dates).isoformat())
            save()
            wait_until(max(dates))
        for attempt in range(2):
            # Revalidate the current API immediately before sending the only permitted artifact.
            preflight = live_preflight()
            record.update(attempt=attempt + 1, status="uploading_predictions")
            save()
            boundary = "----cyp" + uuid.uuid4().hex
            body = (
                (
                    f"--{boundary}\r\n"
                    'Content-Disposition: form-data; name="files"; '
                    'filename="activity_submission.csv"\r\n'
                    "Content-Type: text/csv\r\n\r\n"
                ).encode()
                + raw
                + f"\r\n--{boundary}--\r\n".encode()
            )
            upload = json_request(
                "/gradio_api/upload", body, "multipart/form-data; boundary=" + boundary
            )
            if not isinstance(upload, list) or len(upload) != 1:
                raise ValueError("Unexpected upload response")
            data = payload(upload[0])
            check_payload(data)
            record.update(
                status="submit_post_started", submit_post_started_utc=utcnow().isoformat()
            )
            save()
            result = json_request(
                "/gradio_api/call/submit_predictions", json.dumps({"data": data}).encode()
            )
            record.update(event_id=result["event_id"], status="processing")
            save()
            kind, response = event_result("submit_predictions", result["event_id"])
            message = response_message(response)
            record.update(response=response, observed_at_utc=utcnow().isoformat())
            if kind == "complete" and message.startswith(
                "Submission received from 'Somerra' for the Regression Prediction track."
            ):
                record["status"] = "accepted"
                save()
                return record
            record["status"] = "rejected"
            save()
            (candidate_dir / f"submission-attempt-{attempt + 1}.json").write_text(
                json.dumps(record, indent=2)
            )
            cooldown = re.search(r"Please wait (\d+):(\d+):(\d+) before submitting again", message)
            if kind == "complete" and cooldown and attempt == 0:
                h, m, s = map(int, cooldown.groups())
                seconds = h * 3600 + m * 60 + s + 5
                if not 0 < seconds <= 12 * 3600 + 5:
                    raise ValueError("Unexpected cooldown duration")
                when = utcnow() + dt.timedelta(seconds=seconds)
                record.update(status="waiting_for_cooldown", eligible_after_utc=when.isoformat())
                save()
                wait_until(when)
                continue
            raise RuntimeError("Portal did not accept prediction-only submission: " + message)
    except BaseException as exc:
        record.update(error_type=type(exc).__name__, error=str(exc))
        if record["status"] in ("submit_post_started", "processing"):
            record["status"] = "acceptance_unknown_do_not_resubmit"
        elif record["status"] != "rejected":
            record["status"] = "failed"
        save()
        raise
