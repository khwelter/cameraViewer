from __future__ import annotations

import argparse
import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterator

import cv2
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from camera_viewer.models import PipelineStep, ProcessorConfig
from camera_viewer.pipeline import VisionPipeline, normalize_for_stream
from camera_viewer.pipeline_ops import OPERATION_SPECS


class ProcessorRuntime:
    def __init__(self, config_path: Path | None, config: ProcessorConfig) -> None:
        self.config_path = config_path
        self.config = config
        self.original_frame = None
        self.processed_frame = None
        self.last_error = ""
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.refresh_event = threading.Event()
        self.worker = threading.Thread(target=self._capture_loop, daemon=True)

    def start(self) -> None:
        self._load_if_present()
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.refresh_event.set()
        if self.worker.is_alive():
            self.worker.join(timeout=2.0)

    def _load_if_present(self) -> None:
        if self.config_path is None or not self.config_path.exists():
            return
        data = _load_config_file(self.config_path)
        self.config = ProcessorConfig.model_validate(data)

    def _save(self) -> None:
        if self.config_path is None:
            return
        _save_config_file(self.config_path, self.config.model_dump(mode="json"))

    def update_config(self, config: ProcessorConfig) -> None:
        self.config = config
        self._save()
        self.refresh_event.set()

    def current_config(self) -> ProcessorConfig:
        return self.config

    def frame_pair(self) -> tuple[object, object, str]:
        with self.lock:
            return self.original_frame, self.processed_frame, self.last_error

    def _capture_loop(self) -> None:
        capture = None

        while not self.stop_event.is_set():
            if capture is None or self.refresh_event.is_set():
                if capture is not None:
                    capture.release()
                capture = self._open_capture()
                self.refresh_event.clear()

            if capture is None or not capture.isOpened():
                self._set_error("Unable to open video source")
                time.sleep(1.0)
                continue

            ok, frame = capture.read()
            if not ok or frame is None:
                self._set_error("Failed to read frame")
                time.sleep(0.1)
                continue

            try:
                pipeline = VisionPipeline(self.config.pipeline.steps)
                processed = pipeline.apply(frame)
                processed = normalize_for_stream(processed)
                error = ""
            except Exception as exc:
                processed = frame.copy()
                error = str(exc)
                cv2.putText(
                    processed,
                    error[:120],
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            with self.lock:
                self.original_frame = frame.copy()
                self.processed_frame = processed
                self.last_error = error

        if capture is not None:
            capture.release()

    def _open_capture(self) -> cv2.VideoCapture | None:
        source = self._active_source()
        capture_source: int | str = int(source) if source.isdigit() else source
        capture = cv2.VideoCapture(capture_source)
        if self.config.video.fps:
            capture.set(cv2.CAP_PROP_FPS, float(self.config.video.fps))

        return capture

    def _active_source(self) -> str:
        active_name = self.config.video.active_camera
        profile = self.config.video.camera_profiles.get(active_name)
        if profile is not None:
            return profile.source
        return self.config.video.source

    def _set_error(self, message: str) -> None:
        with self.lock:
            self.last_error = message


def _encode_jpeg(frame: object, quality: int) -> bytes | None:
    if frame is None:
        return None
    ok, buffer = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), max(10, min(100, int(quality)))],
    )
    if not ok:
        return None
    return buffer.tobytes()


def _draw_crosshair(frame: object) -> object:
    if frame is None:
        return frame
    image = frame.copy()
    height, width = image.shape[:2]
    center_x = width // 2
    center_y = height // 2
    color = (0, 255, 255)
    thickness = max(1, min(width, height) // 400)
    cv2.line(image, (0, center_y), (width - 1, center_y), color, thickness)
    cv2.line(image, (center_x, 0), (center_x, height - 1), color, thickness)
    cv2.circle(image, (center_x, center_y), max(4, min(width, height) // 150), color, thickness)
    return image


def _load_config_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _save_config_file(path: Path, data: dict) -> None:
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_dump(data, sort_keys=False)
    else:
        payload = json.dumps(data, indent=2)
    path.write_text(payload + "\n", encoding="utf-8")


def _stream_frames(runtime: ProcessorRuntime, processed: bool) -> Iterator[bytes]:
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while not runtime.stop_event.is_set():
        original_frame, processed_frame, _ = runtime.frame_pair()
        frame = processed_frame if processed else original_frame
        if runtime.current_config().video.crosshair_enabled:
            frame = _draw_crosshair(frame)
        jpeg = _encode_jpeg(frame, runtime.current_config().video.jpeg_quality)
        if jpeg is None:
            time.sleep(0.05)
            continue
        yield boundary + jpeg + b"\r\n"
        time.sleep(0.03)


def build_default_config(source: str, width: int | None, height: int | None, fps: float | None) -> ProcessorConfig:
    return ProcessorConfig(
        video={
            "source": source,
            "fps": fps,
            "jpeg_quality": 85,
            "crosshair_enabled": False,
        },
        pipeline={
            "steps": [
                PipelineStep(operation="gaussian_blur", params={"ksize": 5, "sigma_x": 1.2, "sigma_y": 1.2}),
            ]
        },
    )


def create_app(runtime: ProcessorRuntime) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = runtime
        runtime.start()
        try:
            yield
        finally:
            runtime.stop()

    app = FastAPI(title="Camera Processor", lifespan=lifespan)

    @app.get("/")
    async def root() -> JSONResponse:
        return JSONResponse(
            {
                "message": "Camera processor is running",
                "config": runtime.current_config().model_dump(mode="json"),
            }
        )

    @app.get("/health")
    async def health() -> JSONResponse:
        original_frame, processed_frame, last_error = runtime.frame_pair()
        return JSONResponse(
            {
                "original_ready": original_frame is not None,
                "processed_ready": processed_frame is not None,
                "last_error": last_error,
            }
        )

    @app.get("/api/config")
    async def get_config() -> JSONResponse:
        return JSONResponse(runtime.current_config().model_dump(mode="json"))

    @app.put("/api/config")
    async def put_config(config: ProcessorConfig) -> JSONResponse:
        runtime.update_config(config)
        return JSONResponse({"status": "ok"})

    @app.get("/api/operations")
    async def get_operations() -> JSONResponse:
        payload = {name: spec.model_dump(mode="json") for name, spec in OPERATION_SPECS.items()}
        return JSONResponse(payload)

    @app.get("/stream/original")
    async def stream_original() -> StreamingResponse:
        return StreamingResponse(
            _stream_frames(runtime, processed=False),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/stream/processed")
    async def stream_processed() -> StreamingResponse:
        return StreamingResponse(
            _stream_frames(runtime, processed=True),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/frame/original.jpg")
    async def frame_original() -> StreamingResponse:
        original_frame, _, _ = runtime.frame_pair()
        jpeg = _encode_jpeg(original_frame, runtime.current_config().video.jpeg_quality)
        if jpeg is None:
            raise HTTPException(status_code=503, detail="Frame not ready")
        return StreamingResponse(iter([jpeg]), media_type="image/jpeg")

    @app.get("/frame/processed.jpg")
    async def frame_processed() -> StreamingResponse:
        _, processed_frame, _ = runtime.frame_pair()
        jpeg = _encode_jpeg(processed_frame, runtime.current_config().video.jpeg_quality)
        if jpeg is None:
            raise HTTPException(status_code=503, detail="Frame not ready")
        return StreamingResponse(iter([jpeg]), media_type="image/jpeg")

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the camera processor server")
    parser.add_argument("--source", default="0", help="OpenCV video source, camera index, URL, or file path")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface for the HTTP server")
    parser.add_argument("--port", type=int, default=8000, help="HTTP server port")
    parser.add_argument("--fps", type=float, default=None, help="Requested capture FPS")
    parser.add_argument("--config", default="config/processor.yaml", help="Path to persistent processor config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config) if args.config else None
    runtime = ProcessorRuntime(config_path, build_default_config(args.source, None, None, args.fps))
    app = create_app(runtime)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
