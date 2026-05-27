"""
TensorRT inference manager — drop-in replacement for run_models.py (OpenVINO).

Dependencies
------------
    pip install tensorrt pycuda numpy torch torchvision

Building a TensorRT engine from a YOLOv8 .pt model
----------------------------------------------------
Option A — Ultralytics export (recommended):
    from ultralytics import YOLO
    YOLO("model.pt").export(format="engine", imgsz=640, batch=1)

Option B — via ONNX + trtexec (TensorRT CLI):
    # 1. Export to ONNX first
    from ultralytics import YOLO
    YOLO("model.pt").export(format="onnx", imgsz=640, batch=1)
    # 2. Convert to engine
    trtexec --onnx=model.onnx --saveEngine=model.engine --fp16

Usage (identical to the OpenVINO Models class)
----------------------------------------------
    from utils.run_models_trt import Models, MODEL_DETECTION_TYPE

    models = Models(
        model_paths=["models/bottle/bottle_best.engine"],
        class_names=[["bottle", "not_bottle"]],
        types=MODEL_DETECTION_TYPE,
    )
    result = models.predict(frame_bgr, model_index=0)

Notes
-----
- TensorRT always executes on GPU.  The ``device`` parameter is accepted for
  API compatibility with the OpenVINO version but has no effect.
- Supports TensorRT 8.x (binding-index API) and TensorRT 10.x+ (named I/O
  tensor API).  The correct code path is selected automatically at runtime.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import List, Union

import numpy as np
import torch

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401 — initialises a CUDA context on import
except ImportError as e:
    raise ImportError(
        "TensorRT and PyCUDA are required. Install with:\n"
        "    pip install tensorrt pycuda\n"
        f"Original error: {e}"
    ) from e

from .detector_utils import non_max_suppression
from .preprocess import letterbox, preprocess_image_yolov8_format
from .postprocess import inverse_letterbox

# ── Model type constants (same as run_models.py) ───────────────────────────
MODEL_DETECTION_TYPE = 0
MODEL_CLASSIFICATION_TYPE = 1
MODEL_SEGMENTATION_TYPE = 2

# ── TensorRT runtime globals ────────────────────────────────────────────────
_TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# TensorRT 10+ replaced the integer-binding API with named I/O tensors.
_TRT_MAJOR = int(trt.__version__.split(".")[0])
_USE_NAMED_IO = _TRT_MAJOR >= 10


# ── Internal engine wrapper ─────────────────────────────────────────────────

class _TRTEngine:
    """
    Loads a single serialised TensorRT engine and manages its CUDA buffers.
    Handles both the TRT 8.x (binding-index) and TRT 10.x (named-tensor) APIs.
    """

    def __init__(self, engine_path: str) -> None:
        path = Path(engine_path)
        if not path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        runtime = trt.Runtime(_TRT_LOGGER)
        with open(path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialise TensorRT engine: {engine_path}. "
                "Make sure the engine was built for this GPU and TRT version."
            )

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Allocate pinned host memory and GPU device memory for every binding.
        self.inputs: list[dict] = []
        self.outputs: list[dict] = []

        if _USE_NAMED_IO:
            self._setup_buffers_named_io()
        else:
            self._setup_buffers_binding_index()

        # The input shape is always (batch, C, H, W)
        self.input_shape: tuple = self.inputs[0]["shape"]

    # ------------------------------------------------------------------ #
    # Buffer setup — two code paths for TRT 8.x vs 10.x+                  #
    # ------------------------------------------------------------------ #

    def _setup_buffers_binding_index(self) -> None:
        """TensorRT 8.x API: integer binding indices."""
        self.bindings: list[int] = []

        for i in range(self.engine.num_bindings):
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = tuple(self.engine.get_binding_shape(i))
            size = int(np.prod(shape))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            entry = {
                "name": self.engine.get_binding_name(i),
                "dtype": dtype,
                "shape": shape,
                "host": host_mem,
                "device": device_mem,
            }
            if self.engine.binding_is_input(i):
                self.inputs.append(entry)
            else:
                self.outputs.append(entry)

    def _setup_buffers_named_io(self) -> None:
        """TensorRT 10.x+ API: named I/O tensors."""
        self.tensor_names: list[str] = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = tuple(self.engine.get_tensor_shape(name))
            size = int(np.prod(shape))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.tensor_names.append(name)

            # Register device pointer with the execution context
            self.context.set_tensor_address(name, int(device_mem))

            entry = {
                "name": name,
                "dtype": dtype,
                "shape": shape,
                "host": host_mem,
                "device": device_mem,
            }
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append(entry)
            else:
                self.outputs.append(entry)

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def infer(self, input_array: np.ndarray) -> np.ndarray:
        """
        Run synchronous inference on a single pre-processed frame.

        :param input_array: Float32 array of shape (1, C, H, W).
        :return: Raw model output as a numpy array — shape (1, classes+4, 8400)
                 for YOLOv8 detection models.
        """
        # ① Copy pre-processed input into pinned host buffer → GPU
        np.copyto(self.inputs[0]["host"], input_array.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]["device"], self.inputs[0]["host"], self.stream
        )

        # ② Execute
        if _USE_NAMED_IO:
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            self.context.execute_async_v2(
                bindings=self.bindings, stream_handle=self.stream.handle
            )

        # ③ Copy all outputs from GPU → host
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)

        self.stream.synchronize()

        # Return the first output tensor reshaped to its declared shape
        out = self.outputs[0]
        return out["host"].reshape(out["shape"])

    # ------------------------------------------------------------------ #
    # Cleanup                                                              #
    # ------------------------------------------------------------------ #

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            for binding in self.inputs + self.outputs:
                binding["device"].free()


# ── Public API ──────────────────────────────────────────────────────────────

class Models:
    """
    Manages and runs multiple TensorRT engines (detection, classification,
    segmentation).

    Drop-in replacement for the OpenVINO ``Models`` class in ``run_models.py``.
    All engines are expected to produce the YOLOv8 output format:
    ``[batch, classes + 4, 8400]``.

    Parameters
    ----------
    model_paths:
        Path or list of paths to serialised TensorRT ``.engine`` files.
    class_names:
        List of class-name lists, one per engine.
    types:
        int or list of int indicating the model type constant.  Only
        ``MODEL_DETECTION_TYPE`` is currently implemented.
    device:
        Accepted for API compatibility with the OpenVINO version; ignored —
        TensorRT always runs on GPU.
    conf_thres:
        Confidence threshold forwarded to NMS.
    iou_thres:
        IoU threshold forwarded to NMS.
    """

    def __init__(
        self,
        model_paths: Union[str, List[str]],
        class_names: List[List[str]],
        types: Union[List[int], int] = MODEL_DETECTION_TYPE,
        device: str = "GPU",
        conf_thres: float = 0.4,
        iou_thres: float = 0.5,
    ) -> None:
        if isinstance(model_paths, str):
            model_paths = [model_paths]

        if isinstance(types, int):
            types = [types] * len(model_paths)

        if len(model_paths) != len(class_names):
            raise ValueError(
                f"{len(model_paths)} engine path(s) given but class names for "
                f"{len(class_names)} model(s) provided. These must match."
            )

        self.model_paths = model_paths
        self.class_names = class_names
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Load all engines
        self.engines: List[_TRTEngine] = []
        for path in model_paths:
            try:
                self.engines.append(_TRTEngine(path))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load TensorRT engine '{path}': {e}"
                ) from e

        # Validate and cache input shapes — expected: (batch, channels, H, W)
        self.input_shapes: List[tuple] = []
        for i, eng in enumerate(self.engines):
            shape = eng.input_shape
            if len(shape) != 4:
                raise ValueError(
                    f"Engine {i}: expected 4-D input (batch, C, H, W), "
                    f"got shape {shape}"
                )
            self.input_shapes.append(shape)

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def predict(
        self, image: np.ndarray, model_index: int = 0
    ) -> Union[np.ndarray, None]:
        """
        Run inference on a single BGR image using the specified engine.

        Pre- and post-processing steps are identical to the OpenVINO version:
        letterbox → preprocess_image_yolov8_format → TensorRT infer →
        non_max_suppression → inverse_letterbox.

        Parameters
        ----------
        image:
            Input frame in BGR format (as returned by ``cv2.imread`` /
            ``cv2.VideoCapture.read``).
        model_index:
            Index of the engine to use.  Default = 0.

        Returns
        -------
        numpy.ndarray of shape ``(n, 6)`` with columns
        ``[x1, y1, x2, y2, conf, class_id]`` in *original image* coordinates,
        or ``None`` when no detection passes the confidence threshold.
        """
        if not 0 <= model_index < len(self.engines):
            raise IndexError(
                f"model_index {model_index} is out of range — "
                f"{len(self.engines)} engine(s) loaded."
            )

        img_h, img_w = image.shape[:2]
        _, _, h, w = self.input_shapes[model_index]

        # ── Pre-processing (mirrors run_models.py exactly) ──────────────
        preprocessed = letterbox(image, target_size=w, pad_value=114)
        preprocessed = preprocess_image_yolov8_format(preprocessed, in_size=(w, h))
        preprocessed = np.expand_dims(preprocessed, 0).astype(np.float32)

        # ── TensorRT inference ──────────────────────────────────────────
        raw = self.engines[model_index].infer(preprocessed)   # (1, C+4, 8400)
        result = torch.from_numpy(raw)

        # ── Post-processing (mirrors run_models.py exactly) ─────────────
        detections = non_max_suppression(
            result,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            agnostic=False,
        )[0].numpy()

        if detections.shape[0] == 0:
            return None

        detections[:4] = inverse_letterbox(detections[:4], (img_h, img_w), w)
        return detections

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def get_class_name(self, model_index: int, class_id: int) -> str:
        """Resolve a class ID to its label string for the given engine."""
        return self.class_names[model_index][class_id]
