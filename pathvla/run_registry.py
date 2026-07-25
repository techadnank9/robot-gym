from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pathvla.schemas import RunRequestModel, RunResultModel


def get_output_root() -> Path:
    return Path(os.getenv("PATHVLA_OUTPUT_ROOT", "outputs")).resolve()


def create_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run_{timestamp}_{uuid.uuid4().hex[:8]}"


def prepare_run_directory(request: RunRequestModel) -> tuple[str, Path]:
    output_root = get_output_root()
    if request.output_dir:
        run_dir = Path(request.output_dir).resolve()
        run_id = run_dir.name if run_dir.name.startswith("run_") else create_run_id()
    else:
        run_id = create_run_id()
        run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def persist_result(result: RunResultModel) -> None:
    write_json(Path(result.output_dir) / "result.json", result.model_dump(mode="json"))


def read_result(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "result.json"
    return json.loads(path.read_text(encoding="utf-8"))
