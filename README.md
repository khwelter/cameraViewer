# Camera Viewer

Small Python application with two parts:

1. `camera-processor` runs on the computer that has access to the camera or video stream.
2. `camera-viewer` runs on another computer and shows both the original and processed stream while also editing the remote vision pipeline.

The code is intentionally simple and centered around JSON configuration.

## Features

- camera or video file input through OpenCV
- original and processed MJPEG streams over HTTP
- configurable pipeline with multiple OpenCV operations
- remote editing through a simple viewer UI
- raw JSON editing for advanced parameters
- optional persistent configuration file

## Included operations

The processor includes common image operations such as blur, thresholding, Canny, morphology, Sobel, Laplacian, gamma, brightness and contrast, resize, rotate, flip, CLAHE, histogram equalization, `inRange`, and a generic `opencv` call for advanced single-image functions.

OpenCV is very large, so this starter project does not wrap every OpenCV function with a custom form. The pipeline is designed to be extended by adding entries in `src/camera_viewer/pipeline_ops.py`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run the processor

```bash
camera-processor --source 0 --host 0.0.0.0 --port 8000
```

Useful options:

- `--source 0` for the first local webcam
- `--source rtsp://...` for an RTSP camera
- `--source video.mp4` for a file
- `--fps` to request capture frame rate
- camera resolution stays at the native sensor or stream resolution
- `--config config/processor.yaml` to persist pipeline changes

Processor endpoints:

- `GET /stream/original`
- `GET /stream/processed`
- `GET /api/config`
- `PUT /api/config`
- `GET /api/operations`

## Run the viewer

```bash
camera-viewer --processor-url http://PROCESSOR_HOST:8000
```

Or directly:

```bash
streamlit run src/camera_viewer/viewer.py -- --processor-url http://PROCESSOR_HOST:8000
```

## Network setup

Run the processor on the machine connected to the camera. Run the viewer on any other machine that can reach the processor's HTTP port. Use the processor machine IP address in the viewer.

The processor accepts `.json`, `.yaml`, or `.yml` config files.

## Configuration format

```json
{
  "video": {
    "source": "0",
    "fps": 30,
    "jpeg_quality": 85,
    "crosshair_enabled": false
  },
  "pipeline": {
    "steps": [
      {
        "id": "step-1",
        "enabled": true,
        "operation": "gaussian_blur",
        "params": {
          "ksize": 5,
          "sigma_x": 1.2,
          "sigma_y": 1.2
        }
      }
    ]
  }
}
```
