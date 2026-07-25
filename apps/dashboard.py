from __future__ import annotations

import json
import os
from pathlib import Path

import requests
import streamlit as st

from pathvla.run_registry import get_output_root
from pathvla.schemas import RunMode

API_URL = os.getenv("PATHVLA_API_URL", f"http://127.0.0.1:{os.getenv('PATHVLA_API_PORT', '8000')}")


def list_local_runs() -> list[Path]:
    output_root = get_output_root()
    if not output_root.exists():
        return []
    return sorted([path for path in output_root.iterdir() if path.is_dir()], reverse=True)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


st.set_page_config(page_title="PathVLA Unitree Isaac Live", layout="wide")
st.title("PathVLA Unitree Isaac Live")
st.caption("This dashboard launches and monitors real Isaac Sim/Lab runs on the GPU VM.")

left, right = st.columns([1, 1])

with left:
    st.subheader("Checks")
    if st.button("Run Environment Checks", use_container_width=True):
        try:
            response = requests.get(f"{API_URL}/checks", timeout=120)
            response.raise_for_status()
            st.json(response.json())
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

with right:
    st.subheader("Live View")
    st.markdown(
        "For `webrtc`, expose the configured ports on the VM and connect from the Mac using the Isaac streaming client."
    )
    st.code(
        "make check-livestream\nmake live-demo",
        language="bash",
    )

with st.form("run_form"):
    instruction = st.text_input(
        "Instruction",
        value="Go to the red bin, avoid the chair, inspect the table, then return home.",
    )
    scene = st.selectbox("Scene", options=["room", "warehouse"], index=0)
    live_mode = st.selectbox("Live Mode", options=[mode.value for mode in RunMode], index=0)
    record_video = st.checkbox("Record Video", value=True)
    require_vla = st.checkbox("Require Real VLA Endpoint", value=True)
    allow_proxy = st.checkbox("Allow Proxy Robot", value=False)
    allow_kinematic = st.checkbox("Allow Kinematic Control", value=False)
    allow_rule_planner = st.checkbox("Allow Rule Planner (debug only)", value=False)
    submitted = st.form_submit_button("Launch Run", use_container_width=True)

if submitted:
    payload = {
        "instruction": instruction,
        "scene": scene,
        "live": live_mode,
        "record_video": record_video,
        "allow_proxy": allow_proxy,
        "allow_kinematic_control": allow_kinematic,
        "require_vla": require_vla,
        "allow_rule_planner": allow_rule_planner,
        "require_video": False,
    }
    try:
        response = requests.post(f"{API_URL}/runs", json=payload, timeout=30)
        response.raise_for_status()
        st.success("Run launched.")
        st.json(response.json())
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))

st.subheader("Recent Runs")
run_dirs = list_local_runs()
if not run_dirs:
    st.info("No runs found yet under outputs/.")
else:
    selected_run = st.selectbox("Select Run", options=[path.name for path in run_dirs], index=0)
    run_dir = get_output_root() / selected_run
    result_path = run_dir / "result.json"
    if result_path.exists():
        result = read_json(result_path)
        st.json(result)
        if (run_dir / "plan.json").exists():
            st.subheader("Plan JSON")
            st.json(read_json(run_dir / "plan.json"))
        if (run_dir / "rollout.mp4").exists():
            st.subheader("Replay")
            st.video(str(run_dir / "rollout.mp4"))
    else:
        st.warning("result.json not available yet.")
    if (run_dir / "logs.txt").exists():
        st.subheader("Logs")
        st.code((run_dir / "logs.txt").read_text(encoding="utf-8"))
    elif (run_dir / "launch.log").exists():
        st.subheader("Launch Log")
        st.code((run_dir / "launch.log").read_text(encoding="utf-8"))
