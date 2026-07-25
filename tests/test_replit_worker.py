import asyncio
import re

import pytest

from pathvla.errors import ConfigurationError
from pathvla.mujoco_sorting_demo import parse_args, run
from pathvla.replit_mac_worker import args_from_task
from pathvla.replit_worker import (
    MAX_INSTRUCTION_LENGTH,
    ReplitWorker,
    _BoundedMessageQueue,
    make_event,
    normalize_task_command,
)


def worker(queue_size=5):
    return ReplitWorker(
        control_url="wss://handmind.example/ws/v1/worker",
        worker_token="super-secret-worker-token",
        queue_size=queue_size,
    )


def test_event_envelope_is_versioned_and_uses_utc_timestamp():
    event = make_event("decision_received", "run-1", {"action": "pick"})
    assert event["protocolVersion"] == "1.0"
    assert event["runId"] == "run-1"
    assert event["eventType"] == "decision_received"
    assert event["payload"] == {"action": "pick"}
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T.*Z", event["timestamp"])


def test_worker_token_is_not_exposed_by_repr():
    client = worker()
    assert "super-secret-worker-token" not in repr(client)


def test_remote_worker_requires_secure_websocket():
    with pytest.raises(ConfigurationError, match="wss"):
        ReplitWorker(control_url="ws://public.example/ws", worker_token="secret")
    ReplitWorker(control_url="ws://localhost:8000/ws", worker_token="secret")


def test_task_command_validation_and_option_preservation():
    task = normalize_task_command(
        {
            "type": "task_command",
            "taskId": "task-1",
            "runId": "run-1",
            "instruction": "  Sort the red items.  ",
            "options": {"recordVideo": True},
        }
    )
    assert task["instruction"] == "Sort the red items."
    assert task["options"] == {"recordVideo": True}
    with pytest.raises(ValueError, match="exceeds"):
        normalize_task_command({"type": "task_command", "instruction": "x" * (MAX_INSTRUCTION_LENGTH + 1)})


def test_bounded_queue_evicts_frames_before_control_events():
    messages = _BoundedMessageQueue(maxsize=2)
    assert messages.put({"type": "telemetry_event", "id": 1}, is_frame=False)
    assert messages.put({"type": "camera_frame", "id": 2}, is_frame=True)
    assert messages.put({"type": "run_result", "id": 3}, is_frame=False)
    assert messages.get(timeout=0)["id"] == 1
    assert messages.get(timeout=0)["id"] == 3


def test_duplicate_frames_are_suppressed_per_camera():
    client = worker()
    client.send_frame(b"jpeg-one", run_id="run-1", camera_id="main")
    client.send_frame(b"jpeg-one", run_id="run-1", camera_id="main")
    client.send_frame(b"jpeg-one", run_id="run-1", camera_id="overhead")
    assert len(client._out_queue) == 2


def test_run_stop_does_not_shutdown_worker():
    client = worker()
    client.begin_run("run-1")
    client.request_run_stop()
    assert client.is_stop_requested()
    assert not client.is_shutdown_requested()
    client.end_run("run-1")
    assert not client.is_stop_requested()


def test_idle_send_loop_remains_alive():
    class WebSocket:
        async def send(self, _message):
            return None

    async def scenario():
        client = worker()
        task = asyncio.create_task(client._send_loop(WebSocket()))
        await asyncio.sleep(0.05)
        assert not task.done()
        client._shutdown_event.set()
        await asyncio.wait_for(task, timeout=1.5)

    asyncio.run(scenario())


def test_remote_task_options_are_strictly_allowlisted():
    task = {
        "instruction": "Sort all objects.",
        "options": {
            "maxActions": 20,
            "maxRejections": 2,
            "thinkingBudget": 512,
            "recordVideo": False,
            "headless": True,
            "lingerSeconds": 0,
        },
    }
    args = args_from_task(task)
    assert args.max_actions == 20
    assert args.headless is True
    assert args.record_video is False
    with pytest.raises(ValueError, match="Unsupported remote options"):
        args_from_task({"instruction": "Sort.", "options": {"shellCommand": "rm -rf /"}})


def test_received_task_and_stop_are_kept_separate_from_shutdown():
    class WebSocket:
        def __aiter__(self):
            self.messages = iter(
                [
                    '{"type":"task_command","taskId":"t1","instruction":"Sort red."}',
                    '{"type":"stop_requested","runId":"run-1"}',
                ]
            )
            return self

        async def __anext__(self):
            try:
                return next(self.messages)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    async def scenario():
        client = worker()
        client.begin_run("run-1")
        await client._recv_loop(WebSocket())
        assert client.get_pending_task()["instruction"] == "Sort red."
        assert client.is_stop_requested()
        assert not client.is_shutdown_requested()

    asyncio.run(scenario())


def test_run_persists_partial_result_when_remote_stop_is_already_pending(tmp_path):
    class StoppingWorker:
        def __init__(self):
            self.events = []
            self.results = []
            self.ended = False

        def begin_run(self, _run_id):
            return None

        def is_stop_requested(self):
            return True

        def send_telemetry_event(self, event_type, run_id, payload):
            self.events.append((event_type, run_id, payload))

        def send_run_result(self, run_id, completed, rejected, report):
            self.results.append((run_id, completed, rejected, report))

        def end_run(self, _run_id):
            self.ended = True

    client = StoppingWorker()
    args = parse_args(["--validate-only", "--output-dir", str(tmp_path / "stopped-run")])
    result = run(args, worker=client, remote_run_id="run-stop-test", task_id="task-1")
    assert result["status"] == "stopped"
    assert result["actions"] == 0
    assert (tmp_path / "stopped-run" / "result.json").is_file()
    assert any(event[0] == "run_stopped" for event in client.events)
    assert client.results[0][3]["status"] == "stopped"
    assert client.ended
