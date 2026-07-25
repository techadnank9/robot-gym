from __future__ import annotations

import os
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

from isaac_ext.pathvla_unitree import __version__
from pathvla.run_registry import create_run_id, get_output_root, read_result, write_json
from pathvla.schemas import ChecksResponseModel, HealthResponseModel, RunRequestModel, RunResultModel

APP_ROOT = Path(__file__).resolve().parents[1]
RUN_PROCESSES: dict[str, subprocess.Popen] = {}

app = FastAPI(title="PathVLA Unitree Isaac Live API", version=__version__)


def _run_script(script_name: str) -> tuple[bool, str]:
    script_path = APP_ROOT / "scripts" / script_name
    process = subprocess.run(
        ["bash", str(script_path)],
        cwd=APP_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(APP_ROOT)},
    )
    return process.returncode == 0, (process.stdout + process.stderr).strip()


def _find_run_dir(run_id: str) -> Path:
    run_dir = get_output_root() / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Unknown run_id {run_id}")
    return run_dir


@app.get("/health", response_model=HealthResponseModel)
def health():
    return HealthResponseModel(ok=True, service="pathvla-unitree-isaac-live", version=__version__)


@app.get("/checks", response_model=ChecksResponseModel)
def checks():
    gpu_ok, gpu_output = _run_script("check_gpu.sh")
    isaac_ok, isaac_output = _run_script("check_isaac.sh")
    livestream_ok, livestream_output = _run_script("check_livestream.sh")
    return ChecksResponseModel(
        gpu_ok=gpu_ok,
        isaac_ok=isaac_ok,
        livestream_ok=livestream_ok,
        details={
            "gpu": gpu_output,
            "isaac": isaac_output,
            "livestream": livestream_output,
        },
    )


@app.post("/runs", response_model=RunResultModel)
def create_run(request: RunRequestModel):
    run_id = create_run_id()
    output_dir = Path(request.output_dir).resolve() if request.output_dir else get_output_root() / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    request_payload = request.model_dump(mode="json")
    request_payload["output_dir"] = str(output_dir)
    write_json(output_dir / "request.json", request_payload)

    command = [
        "bash",
        str(APP_ROOT / "scripts" / "isaac_python.sh"),
        "-m",
        "isaac_ext.pathvla_unitree.tasks.room_nav_env",
        "--scene",
        request.scene,
        "--instruction",
        request.instruction,
        "--robot",
        "unitree_g1",
        "--live",
        request.live.value,
        "--output-dir",
        str(output_dir),
    ]
    if request.record_video:
        command.append("--record-video")
    if request.require_video:
        command.append("--require-video")
    if request.require_vla:
        command.append("--require-vla")
    else:
        command.append("--no-require-vla")
    if request.allow_proxy:
        command.append("--allow-proxy")
    if request.allow_kinematic_control:
        command.append("--allow-kinematic-control")
    if request.allow_rule_planner:
        command.append("--allow-rule-planner")

    launch_log = (output_dir / "launch.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=APP_ROOT,
        stdout=launch_log,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONPATH": str(APP_ROOT)},
    )
    RUN_PROCESSES[run_id] = process

    return RunResultModel(
        run_id=run_id,
        status="created",
        instruction=request.instruction,
        scene=request.scene,
        live=request.live,
        output_dir=str(output_dir),
        message=f"Launch started with PID {process.pid}",
    )


@app.get("/runs/{run_id}", response_model=RunResultModel)
def get_run(run_id: str):
    run_dir = _find_run_dir(run_id)
    result_path = run_dir / "result.json"
    if not result_path.exists():
        process = RUN_PROCESSES.get(run_id)
        status = "running" if process and process.poll() is None else "created"
        request_payload = json_load(run_dir / "request.json")
        return RunResultModel(
            run_id=run_id,
            status=status,
            instruction=request_payload["instruction"],
            scene=request_payload["scene"],
            live=request_payload["live"],
            output_dir=str(run_dir),
            message="Run has not emitted result.json yet.",
        )
    return RunResultModel.model_validate(read_result(run_dir))


def json_load(path: Path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/runs/{run_id}/logs")
def get_logs(run_id: str):
    run_dir = _find_run_dir(run_id)
    for candidate in ("logs.txt", "launch.log"):
        path = run_dir / candidate
        if path.exists():
            return PlainTextResponse(path.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="No logs found for this run.")


@app.get("/runs/{run_id}/result")
def get_result(run_id: str):
    run_dir = _find_run_dir(run_id)
    result_path = run_dir / "result.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="result.json not available yet.")
    return json_load(result_path)


@app.get("/runs/{run_id}/video")
def get_video(run_id: str):
    run_dir = _find_run_dir(run_id)
    video_path = run_dir / "rollout.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="rollout.mp4 not available for this run.")
    return FileResponse(video_path, media_type="video/mp4", filename=f"{run_id}.mp4")
