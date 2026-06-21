from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from typing import Any

import requests
import streamlit as st


DEFAULT_PROCESSOR_URL = "http://127.0.0.1:8000"


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--processor-url", default=DEFAULT_PROCESSOR_URL)
    return parser.parse_known_args(sys.argv[1:])[0]


def request_json(method: str, url: str, payload: dict[str, Any] | None = None) -> Any:
    response = requests.request(method, url, json=payload, timeout=5)
    response.raise_for_status()
    return response.json()


def default_step_from_operation(name: str, operations: dict[str, Any]) -> dict[str, Any]:
    spec = operations[name]
    params = {key: value["default"] for key, value in spec.get("parameters", {}).items()}
    return {
        "id": f"step-{len(params)}-{name}",
        "enabled": True,
        "operation": name,
        "params": params,
    }


def _number_widget(label: str, key: str, value: Any, spec: dict[str, Any]) -> Any:
    minimum = spec.get("minimum")
    maximum = spec.get("maximum")
    step = spec.get("step")
    if spec["type"] == "int":
        return int(
            st.number_input(
                label,
                value=int(value),
                min_value=int(minimum) if minimum is not None else None,
                max_value=int(maximum) if maximum is not None else None,
                step=int(step) if step is not None else 1,
                key=key,
            )
        )

    return float(
        st.number_input(
            label,
            value=float(value),
            min_value=float(minimum) if minimum is not None else None,
            max_value=float(maximum) if maximum is not None else None,
            step=float(step) if step is not None else 0.1,
            key=key,
        )
    )


def render_param_widget(step_index: int, param_name: str, spec: dict[str, Any], current_value: Any) -> Any:
    label = spec.get("label") or param_name
    widget_key = f"step-{step_index}-{param_name}"
    param_type = spec["type"]
    if param_type in {"int", "float"}:
        return _number_widget(label, widget_key, current_value, spec)
    if param_type == "bool":
        return st.checkbox(label, value=bool(current_value), key=widget_key)
    if param_type == "select":
        options = spec.get("options", [])
        current = current_value if current_value in options else spec.get("default")
        return st.selectbox(label, options=options, index=options.index(current), key=widget_key)
    if param_type == "json":
        text = current_value if isinstance(current_value, str) else json.dumps(current_value, indent=2)
        return st.text_area(label, value=text, height=120, key=widget_key)
    return st.text_input(label, value=str(current_value), key=widget_key)


def apply_step_changes(step_index: int, step: dict[str, Any], operations: dict[str, Any]) -> dict[str, Any]:
    operation = step["operation"]
    spec = operations[operation]
    updated = copy.deepcopy(step)
    updated["enabled"] = st.checkbox("Enabled", value=bool(step.get("enabled", True)), key=f"enabled-{step_index}")
    op_names = list(operations.keys())
    op_index = op_names.index(operation) if operation in op_names else 0
    updated["operation"] = st.selectbox("Operation", options=op_names, index=op_index, key=f"operation-{step_index}")
    updated.setdefault("params", {})

    if updated["operation"] != operation:
        replacement = default_step_from_operation(updated["operation"], operations)
        replacement["id"] = updated.get("id", replacement["id"])
        updated["params"] = replacement["params"]

    active_spec = operations[updated["operation"]]
    updated_params: dict[str, Any] = {}
    for param_name, param_spec in active_spec.get("parameters", {}).items():
        current_value = updated["params"].get(param_name, param_spec.get("default"))
        updated_params[param_name] = render_param_widget(step_index, param_name, param_spec, current_value)
    updated["params"] = updated_params
    return updated


def refresh_remote_data(processor_url: str) -> tuple[dict[str, Any], dict[str, Any]]:
    config = request_json("GET", f"{processor_url}/api/config")
    operations = request_json("GET", f"{processor_url}/api/operations")
    return config, operations


def render_app() -> None:
    cli_args = parse_cli_args()
    st.set_page_config(page_title="Camera Viewer", layout="wide")
    st.title("Camera Viewer")

    if "processor_url" not in st.session_state:
        st.session_state.processor_url = cli_args.processor_url
    if "config" not in st.session_state:
        st.session_state.config = None
    if "operations" not in st.session_state:
        st.session_state.operations = {}

    with st.sidebar:
        processor_url = st.text_input("Processor URL", value=st.session_state.processor_url)
        st.session_state.processor_url = processor_url.rstrip("/")

        if st.button("Load remote config", use_container_width=True):
            try:
                config, operations = refresh_remote_data(st.session_state.processor_url)
                st.session_state.config = config
                st.session_state.operations = operations
                st.success("Configuration loaded")
            except Exception as exc:
                st.error(str(exc))

        st.caption("The viewer connects to the processor over HTTP.")

    if st.session_state.config is None:
        try:
            config, operations = refresh_remote_data(st.session_state.processor_url)
            st.session_state.config = config
            st.session_state.operations = operations
        except Exception as exc:
            st.warning(f"Connect to a processor first: {exc}")
            return

    stream_left, stream_right = st.columns(2)
    with stream_left:
        st.subheader("Original")
        st.markdown(
            f'<img src="{st.session_state.processor_url}/stream/original" style="width: 100%; border: 1px solid #bbb; border-radius: 8px;" />',
            unsafe_allow_html=True,
        )
    with stream_right:
        st.subheader("Processed")
        st.markdown(
            f'<img src="{st.session_state.processor_url}/stream/processed" style="width: 100%; border: 1px solid #bbb; border-radius: 8px;" />',
            unsafe_allow_html=True,
        )

    config = copy.deepcopy(st.session_state.config)
    operations = st.session_state.operations

    st.divider()
    st.subheader("Video Settings")
    video_cols = st.columns(5)
    config["video"]["source"] = video_cols[0].text_input("Source", value=str(config["video"].get("source", "0")))
    config["video"]["width"] = int(video_cols[1].number_input("Width", value=int(config["video"].get("width") or 0), min_value=0, step=1)) or None
    config["video"]["height"] = int(video_cols[2].number_input("Height", value=int(config["video"].get("height") or 0), min_value=0, step=1)) or None
    config["video"]["fps"] = float(video_cols[3].number_input("FPS", value=float(config["video"].get("fps") or 0.0), min_value=0.0, step=1.0)) or None
    config["video"]["jpeg_quality"] = int(video_cols[4].slider("JPEG quality", min_value=10, max_value=100, value=int(config["video"].get("jpeg_quality", 85))))

    editor_tab, raw_tab, catalog_tab = st.tabs(["Editor", "Raw JSON", "Operations"])

    with editor_tab:
        steps = config.get("pipeline", {}).get("steps", [])
        updated_steps: list[dict[str, Any]] = []
        removed_index = None
        move_up_index = None
        move_down_index = None

        for index, step in enumerate(steps):
            label = f"Step {index + 1}: {step.get('operation', 'unknown')}"
            with st.expander(label, expanded=index == 0):
                updated = apply_step_changes(index, step, operations)
                button_cols = st.columns(3)
                if button_cols[0].button("Move up", key=f"up-{index}", disabled=index == 0):
                    move_up_index = index
                if button_cols[1].button("Move down", key=f"down-{index}", disabled=index == len(steps) - 1):
                    move_down_index = index
                if button_cols[2].button("Delete", key=f"delete-{index}"):
                    removed_index = index
                updated_steps.append(updated)

        config["pipeline"]["steps"] = updated_steps

        add_cols = st.columns([3, 1])
        new_operation = add_cols[0].selectbox("New step operation", options=list(operations.keys()), key="new-operation")
        if add_cols[1].button("Add step"):
            config["pipeline"]["steps"].append(default_step_from_operation(new_operation, operations))
            st.session_state.config = config
            st.rerun()

        if removed_index is not None:
            del config["pipeline"]["steps"][removed_index]
            st.session_state.config = config
            st.rerun()
        if move_up_index is not None:
            config["pipeline"]["steps"][move_up_index - 1], config["pipeline"]["steps"][move_up_index] = config["pipeline"]["steps"][move_up_index], config["pipeline"]["steps"][move_up_index - 1]
            st.session_state.config = config
            st.rerun()
        if move_down_index is not None:
            config["pipeline"]["steps"][move_down_index + 1], config["pipeline"]["steps"][move_down_index] = config["pipeline"]["steps"][move_down_index], config["pipeline"]["steps"][move_down_index + 1]
            st.session_state.config = config
            st.rerun()

        if st.button("Apply configuration", type="primary"):
            try:
                request_json("PUT", f"{st.session_state.processor_url}/api/config", payload=config)
                st.session_state.config = config
                st.success("Configuration sent to processor")
            except Exception as exc:
                st.error(str(exc))

    with raw_tab:
        raw_text = st.text_area("Config JSON", value=json.dumps(config, indent=2), height=500)
        if st.button("Send raw JSON"):
            try:
                parsed = json.loads(raw_text)
                request_json("PUT", f"{st.session_state.processor_url}/api/config", payload=parsed)
                st.session_state.config = parsed
                st.success("Raw JSON applied")
            except Exception as exc:
                st.error(str(exc))

    with catalog_tab:
        for name, spec in operations.items():
            with st.expander(f"{name}: {spec.get('label', name)}"):
                st.write(spec.get("description", ""))
                st.json(spec.get("parameters", {}))


def main() -> None:
    cli_args = parse_cli_args()
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        __file__,
        "--",
        "--processor-url",
        cli_args.processor_url,
    ]
    raise SystemExit(subprocess.call(command))


def _maybe_run_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False

    if get_script_run_ctx() is None:
        return False

    render_app()
    return True


if __name__ == "__main__":
    if not _maybe_run_streamlit():
        main()
