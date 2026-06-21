from __future__ import annotations

import json
import math
from typing import Any, Callable

import cv2
import numpy as np

from camera_viewer.models import OperationSpec, ParameterSpec


OperationCallable = Callable[[np.ndarray, dict[str, Any]], np.ndarray]


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def _kernel(params: dict[str, Any]) -> np.ndarray:
    width = max(1, int(params.get("kernel_width", 3)))
    height = max(1, int(params.get("kernel_height", 3)))
    shape_name = str(params.get("shape", "rect")).lower()
    shape = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS,
    }.get(shape_name, cv2.MORPH_RECT)
    return cv2.getStructuringElement(shape, (width, height))


def _grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _display_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    scaled = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return scaled.astype(np.uint8)


def _resolve_cv2_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if hasattr(cv2, value):
        return getattr(cv2, value)
    return value


def _parse_json_value(value: Any, fallback: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return fallback
    return json.loads(text)


def op_gaussian_blur(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    ksize = _odd(params.get("ksize", 5))
    sigma_x = float(params.get("sigma_x", 1.2))
    sigma_y = float(params.get("sigma_y", sigma_x))
    return cv2.GaussianBlur(image, (ksize, ksize), sigma_x, sigmaY=sigma_y)


def op_median_blur(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    return cv2.medianBlur(image, _odd(params.get("ksize", 5)))


def op_bilateral_filter(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    return cv2.bilateralFilter(
        image,
        int(params.get("diameter", 9)),
        float(params.get("sigma_color", 75.0)),
        float(params.get("sigma_space", 75.0)),
    )


def op_grayscale(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    del params
    return _grayscale(image)


def op_threshold(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    gray = _grayscale(image)
    threshold_type = _resolve_cv2_value(params.get("threshold_type", "THRESH_BINARY"))
    _, result = cv2.threshold(
        gray,
        float(params.get("threshold", 127)),
        float(params.get("max_value", 255)),
        int(threshold_type),
    )
    return result


def op_adaptive_threshold(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    gray = _grayscale(image)
    adaptive_method = _resolve_cv2_value(params.get("adaptive_method", "ADAPTIVE_THRESH_GAUSSIAN_C"))
    threshold_type = _resolve_cv2_value(params.get("threshold_type", "THRESH_BINARY"))
    block_size = _odd(params.get("block_size", 11))
    return cv2.adaptiveThreshold(
        gray,
        int(params.get("max_value", 255)),
        int(adaptive_method),
        int(threshold_type),
        block_size,
        float(params.get("c", 2)),
    )


def op_canny(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    gray = _grayscale(image)
    return cv2.Canny(
        gray,
        float(params.get("threshold1", 100)),
        float(params.get("threshold2", 200)),
        apertureSize=_odd(params.get("aperture_size", 3)),
        L2gradient=bool(params.get("l2_gradient", False)),
    )


def op_dilate(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    return cv2.dilate(image, _kernel(params), iterations=int(params.get("iterations", 1)))


def op_erode(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    return cv2.erode(image, _kernel(params), iterations=int(params.get("iterations", 1)))


def op_morphology(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    operation = str(params.get("operation", "open")).lower()
    morph_op = {
        "open": cv2.MORPH_OPEN,
        "close": cv2.MORPH_CLOSE,
        "gradient": cv2.MORPH_GRADIENT,
        "tophat": cv2.MORPH_TOPHAT,
        "blackhat": cv2.MORPH_BLACKHAT,
    }.get(operation, cv2.MORPH_OPEN)
    return cv2.morphologyEx(image, morph_op, _kernel(params), iterations=int(params.get("iterations", 1)))


def op_resize(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    width = max(1, int(params.get("width", image.shape[1])))
    height = max(1, int(params.get("height", image.shape[0])))
    interpolation = _resolve_cv2_value(params.get("interpolation", "INTER_LINEAR"))
    return cv2.resize(image, (width, height), interpolation=int(interpolation))


def op_flip(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    mode = str(params.get("mode", "")).lower()
    if mode:
        flip_code = {
            "horizontal": 1,
            "vertical": 0,
            "both": -1,
        }.get(mode, 1)
    else:
        flip_code = int(params.get("flip_code", 1))
    return cv2.flip(image, flip_code)


def op_rotate(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    angle = float(params.get("angle_degrees", 0.0)) % 360.0
    scale = float(params.get("scale", 1.0))
    center = (image.shape[1] / 2.0, image.shape[0] / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))


def op_brightness_contrast(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    alpha = float(params.get("alpha", 1.0))
    beta = float(params.get("beta", 0.0))
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def op_gamma(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    gamma = max(0.01, float(params.get("gamma", 1.0)))
    inv_gamma = 1.0 / gamma
    table = np.array([((value / 255.0) ** inv_gamma) * 255 for value in range(256)], dtype=np.uint8)
    return cv2.LUT(_display_uint8(image), table)


def op_in_range(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    color_space = str(params.get("color_space", "bgr")).lower()
    working = image
    if color_space == "hsv":
        working = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([
        int(params.get("low_0", 0)),
        int(params.get("low_1", 0)),
        int(params.get("low_2", 0)),
    ])
    upper = np.array([
        int(params.get("high_0", 255)),
        int(params.get("high_1", 255)),
        int(params.get("high_2", 255)),
    ])
    return cv2.inRange(working, lower, upper)


def op_laplacian(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    gray = _grayscale(image)
    ksize = _odd(params.get("ksize", 3))
    result = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    return cv2.convertScaleAbs(result)


def op_sobel(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    gray = _grayscale(image)
    result = cv2.Sobel(
        gray,
        cv2.CV_64F,
        dx=int(params.get("dx", 1)),
        dy=int(params.get("dy", 0)),
        ksize=_odd(params.get("ksize", 3)),
        scale=float(params.get("scale", 1.0)),
        delta=float(params.get("delta", 0.0)),
    )
    return cv2.convertScaleAbs(result)


def op_equalize_hist(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    del params
    gray = _grayscale(image)
    return cv2.equalizeHist(gray)


def op_clahe(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    gray = _grayscale(image)
    clahe = cv2.createCLAHE(
        clipLimit=float(params.get("clip_limit", 2.0)),
        tileGridSize=(
            max(1, int(params.get("tile_width", 8))),
            max(1, int(params.get("tile_height", 8))),
        ),
    )
    return clahe.apply(gray)


def op_opencv(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    function_name = str(params.get("function", ""))
    if not function_name or not hasattr(cv2, function_name):
        raise ValueError(f"Unknown cv2 function: {function_name}")

    args = _parse_json_value(params.get("args", "[]"), [])
    kwargs = _parse_json_value(params.get("kwargs", "{}"), {})
    resolved_args = [_resolve_cv2_value(value) for value in args]
    resolved_kwargs = {key: _resolve_cv2_value(value) for key, value in kwargs.items()}

    result = getattr(cv2, function_name)(image, *resolved_args, **resolved_kwargs)
    if isinstance(result, tuple):
        arrays = [item for item in result if isinstance(item, np.ndarray)]
        if arrays:
            return arrays[-1]
        raise ValueError(f"cv2.{function_name} did not return an image")
    if not isinstance(result, np.ndarray):
        raise ValueError(f"cv2.{function_name} did not return an image")
    return result


OPERATIONS: dict[str, OperationCallable] = {
    "gaussian_blur": op_gaussian_blur,
    "median_blur": op_median_blur,
    "bilateral_filter": op_bilateral_filter,
    "grayscale": op_grayscale,
    "threshold": op_threshold,
    "adaptive_threshold": op_adaptive_threshold,
    "canny": op_canny,
    "dilate": op_dilate,
    "erode": op_erode,
    "morphology": op_morphology,
    "resize": op_resize,
    "flip": op_flip,
    "rotate": op_rotate,
    "brightness_contrast": op_brightness_contrast,
    "gamma": op_gamma,
    "in_range": op_in_range,
    "laplacian": op_laplacian,
    "sobel": op_sobel,
    "equalize_hist": op_equalize_hist,
    "clahe": op_clahe,
    "opencv": op_opencv,
}


OPERATION_SPECS: dict[str, OperationSpec] = {
    "gaussian_blur": OperationSpec(
        label="Gaussian Blur",
        description="Smooth the image with a Gaussian kernel.",
        parameters={
            "ksize": ParameterSpec(type="int", default=5, minimum=1, maximum=99, step=2, label="Kernel size"),
            "sigma_x": ParameterSpec(type="float", default=1.2, minimum=0.0, maximum=50.0, step=0.1, label="Sigma X"),
            "sigma_y": ParameterSpec(type="float", default=1.2, minimum=0.0, maximum=50.0, step=0.1, label="Sigma Y"),
        },
    ),
    "median_blur": OperationSpec(
        label="Median Blur",
        description="Replace each pixel by the median in a neighborhood.",
        parameters={
            "ksize": ParameterSpec(type="int", default=5, minimum=1, maximum=99, step=2, label="Kernel size"),
        },
    ),
    "bilateral_filter": OperationSpec(
        label="Bilateral Filter",
        description="Smooth while keeping edges.",
        parameters={
            "diameter": ParameterSpec(type="int", default=9, minimum=1, maximum=50, step=1, label="Diameter"),
            "sigma_color": ParameterSpec(type="float", default=75.0, minimum=1.0, maximum=200.0, step=1.0, label="Sigma color"),
            "sigma_space": ParameterSpec(type="float", default=75.0, minimum=1.0, maximum=200.0, step=1.0, label="Sigma space"),
        },
    ),
    "grayscale": OperationSpec(label="Grayscale", description="Convert BGR image to grayscale."),
    "threshold": OperationSpec(
        label="Threshold",
        description="Apply a fixed threshold to a grayscale image.",
        parameters={
            "threshold": ParameterSpec(type="float", default=127.0, minimum=0.0, maximum=255.0, step=1.0, label="Threshold"),
            "max_value": ParameterSpec(type="float", default=255.0, minimum=0.0, maximum=255.0, step=1.0, label="Max value"),
            "threshold_type": ParameterSpec(type="select", default="THRESH_BINARY", options=["THRESH_BINARY", "THRESH_BINARY_INV", "THRESH_TRUNC", "THRESH_TOZERO", "THRESH_TOZERO_INV"], label="Type"),
        },
    ),
    "adaptive_threshold": OperationSpec(
        label="Adaptive Threshold",
        description="Threshold using a local neighborhood.",
        parameters={
            "max_value": ParameterSpec(type="int", default=255, minimum=0, maximum=255, step=1, label="Max value"),
            "adaptive_method": ParameterSpec(type="select", default="ADAPTIVE_THRESH_GAUSSIAN_C", options=["ADAPTIVE_THRESH_GAUSSIAN_C", "ADAPTIVE_THRESH_MEAN_C"], label="Method"),
            "threshold_type": ParameterSpec(type="select", default="THRESH_BINARY", options=["THRESH_BINARY", "THRESH_BINARY_INV"], label="Type"),
            "block_size": ParameterSpec(type="int", default=11, minimum=3, maximum=99, step=2, label="Block size"),
            "c": ParameterSpec(type="float", default=2.0, minimum=-50.0, maximum=50.0, step=0.5, label="C"),
        },
    ),
    "canny": OperationSpec(
        label="Canny",
        description="Detect edges with the Canny algorithm.",
        parameters={
            "threshold1": ParameterSpec(type="float", default=100.0, minimum=0.0, maximum=500.0, step=1.0, label="Threshold 1"),
            "threshold2": ParameterSpec(type="float", default=200.0, minimum=0.0, maximum=500.0, step=1.0, label="Threshold 2"),
            "aperture_size": ParameterSpec(type="int", default=3, minimum=3, maximum=7, step=2, label="Aperture"),
            "l2_gradient": ParameterSpec(type="bool", default=False, label="L2 gradient"),
        },
    ),
    "dilate": OperationSpec(
        label="Dilate",
        description="Dilate the image with a configurable structuring element.",
        parameters={
            "kernel_width": ParameterSpec(type="int", default=3, minimum=1, maximum=99, step=1, label="Kernel width"),
            "kernel_height": ParameterSpec(type="int", default=3, minimum=1, maximum=99, step=1, label="Kernel height"),
            "shape": ParameterSpec(type="select", default="rect", options=["rect", "ellipse", "cross"], label="Shape"),
            "iterations": ParameterSpec(type="int", default=1, minimum=1, maximum=20, step=1, label="Iterations"),
        },
    ),
    "erode": OperationSpec(
        label="Erode",
        description="Erode the image with a configurable structuring element.",
        parameters={
            "kernel_width": ParameterSpec(type="int", default=3, minimum=1, maximum=99, step=1, label="Kernel width"),
            "kernel_height": ParameterSpec(type="int", default=3, minimum=1, maximum=99, step=1, label="Kernel height"),
            "shape": ParameterSpec(type="select", default="rect", options=["rect", "ellipse", "cross"], label="Shape"),
            "iterations": ParameterSpec(type="int", default=1, minimum=1, maximum=20, step=1, label="Iterations"),
        },
    ),
    "morphology": OperationSpec(
        label="Morphology",
        description="Apply an OpenCV morphologyEx operation.",
        parameters={
            "operation": ParameterSpec(type="select", default="open", options=["open", "close", "gradient", "tophat", "blackhat"], label="Operation"),
            "kernel_width": ParameterSpec(type="int", default=3, minimum=1, maximum=99, step=1, label="Kernel width"),
            "kernel_height": ParameterSpec(type="int", default=3, minimum=1, maximum=99, step=1, label="Kernel height"),
            "shape": ParameterSpec(type="select", default="rect", options=["rect", "ellipse", "cross"], label="Shape"),
            "iterations": ParameterSpec(type="int", default=1, minimum=1, maximum=20, step=1, label="Iterations"),
        },
    ),
    "resize": OperationSpec(
        label="Resize",
        description="Resize the frame to a fixed output size.",
        parameters={
            "width": ParameterSpec(type="int", default=640, minimum=1, maximum=8000, step=1, label="Width"),
            "height": ParameterSpec(type="int", default=480, minimum=1, maximum=8000, step=1, label="Height"),
            "interpolation": ParameterSpec(type="select", default="INTER_LINEAR", options=["INTER_LINEAR", "INTER_NEAREST", "INTER_AREA", "INTER_CUBIC", "INTER_LANCZOS4"], label="Interpolation"),
        },
    ),
    "flip": OperationSpec(
        label="Flip",
        description="Flip vertically, horizontally, or both.",
        parameters={
            "mode": ParameterSpec(type="select", default="horizontal", options=["horizontal", "vertical", "both"], label="Mode"),
        },
    ),
    "rotate": OperationSpec(
        label="Rotate",
        description="Rotate around the image center.",
        parameters={
            "angle_degrees": ParameterSpec(type="float", default=0.0, minimum=0.0, maximum=360.0, step=1.0, label="Angle"),
            "scale": ParameterSpec(type="float", default=1.0, minimum=0.1, maximum=5.0, step=0.1, label="Scale"),
        },
    ),
    "brightness_contrast": OperationSpec(
        label="Brightness / Contrast",
        description="Apply alpha and beta scaling.",
        parameters={
            "alpha": ParameterSpec(type="float", default=1.0, minimum=0.1, maximum=5.0, step=0.1, label="Contrast"),
            "beta": ParameterSpec(type="float", default=0.0, minimum=-255.0, maximum=255.0, step=1.0, label="Brightness"),
        },
    ),
    "gamma": OperationSpec(
        label="Gamma",
        description="Apply gamma correction with a LUT.",
        parameters={
            "gamma": ParameterSpec(type="float", default=1.0, minimum=0.1, maximum=5.0, step=0.1, label="Gamma"),
        },
    ),
    "in_range": OperationSpec(
        label="In Range",
        description="Create a binary mask by checking whether pixels lie between lower and upper bounds.",
        parameters={
            "color_space": ParameterSpec(type="select", default="bgr", options=["bgr", "hsv"], label="Color space"),
            "low_0": ParameterSpec(type="int", default=0, minimum=0, maximum=255, step=1, label="Low 0"),
            "low_1": ParameterSpec(type="int", default=0, minimum=0, maximum=255, step=1, label="Low 1"),
            "low_2": ParameterSpec(type="int", default=0, minimum=0, maximum=255, step=1, label="Low 2"),
            "high_0": ParameterSpec(type="int", default=255, minimum=0, maximum=255, step=1, label="High 0"),
            "high_1": ParameterSpec(type="int", default=255, minimum=0, maximum=255, step=1, label="High 1"),
            "high_2": ParameterSpec(type="int", default=255, minimum=0, maximum=255, step=1, label="High 2"),
        },
    ),
    "laplacian": OperationSpec(
        label="Laplacian",
        description="Second derivative edge detection.",
        parameters={
            "ksize": ParameterSpec(type="int", default=3, minimum=1, maximum=31, step=2, label="Kernel size"),
        },
    ),
    "sobel": OperationSpec(
        label="Sobel",
        description="First derivative edge detection.",
        parameters={
            "dx": ParameterSpec(type="int", default=1, minimum=0, maximum=2, step=1, label="dx"),
            "dy": ParameterSpec(type="int", default=0, minimum=0, maximum=2, step=1, label="dy"),
            "ksize": ParameterSpec(type="int", default=3, minimum=1, maximum=31, step=2, label="Kernel size"),
            "scale": ParameterSpec(type="float", default=1.0, minimum=0.1, maximum=20.0, step=0.1, label="Scale"),
            "delta": ParameterSpec(type="float", default=0.0, minimum=-255.0, maximum=255.0, step=1.0, label="Delta"),
        },
    ),
    "equalize_hist": OperationSpec(label="Equalize Histogram", description="Equalize a grayscale histogram."),
    "clahe": OperationSpec(
        label="CLAHE",
        description="Contrast Limited Adaptive Histogram Equalization.",
        parameters={
            "clip_limit": ParameterSpec(type="float", default=2.0, minimum=0.1, maximum=20.0, step=0.1, label="Clip limit"),
            "tile_width": ParameterSpec(type="int", default=8, minimum=1, maximum=64, step=1, label="Tile width"),
            "tile_height": ParameterSpec(type="int", default=8, minimum=1, maximum=64, step=1, label="Tile height"),
        },
    ),
    "opencv": OperationSpec(
        label="Generic OpenCV Call",
        description="Call a single cv2 function that accepts the current image as the first argument.",
        parameters={
            "function": ParameterSpec(type="str", default="blur", label="Function", description="OpenCV function name without the cv2 prefix."),
            "args": ParameterSpec(type="json", default="[]", label="Args JSON", description="JSON array for positional arguments after the image."),
            "kwargs": ParameterSpec(type="json", default="{}", label="Kwargs JSON", description="JSON object for keyword arguments. cv2 constant names are allowed as strings."),
        },
    ),
}
