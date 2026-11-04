"""Execute exact accepted inference definitions, with no label-reading imports.

The original modules are hashed as data. Only explicitly named functions,
model classes and literal constants are compiled; their original bodies are
unchanged. Training/evaluation entry points are never imported or executed.
"""
import ast
import hashlib
import importlib.metadata as metadata
import os
from pathlib import Path
import platform
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image
from pycocotools import mask as coco

MODEL = "src/solarspine/models/e10_01_native_roi_refiner.py"
TILES = "src/solarspine/evaluation/e10_01_a11r1w_native_roi_coverage_gate.py"
IMAGE = "src/solarspine/training/e10_01_a11r1x_roi_patch_index.py"
A4 = "src/solarspine/evaluation/e10_01_a4_fold4_native_pq.py"
A6 = "src/solarspine/submission/e10_01_a6_test_submission.py"
R2B = "src/solarspine/evaluation/e10_01_a11r2b_fold4_native_strict_pq.py"
R2C = "src/solarspine/submission/e10_01_a11r2c_test_submission.py"
REFINER_SHA = "e4b9c3bacce816cd1d4dfb863622b93774743a6a401ae844088e33540882f397"
DETECTOR_SHA = "bbb3c55175f68f80c213ac9fd6d1a1ab5bd63993696b609348d36f0472515875"
EXACT_PACKAGES = {"numpy": "2.1.3", "Pillow": "11.3.0", "pycocotools": "2.0.11",
                  "scipy": "1.16.3", "opencv-python": "4.12.0.88", "ultralytics": "8.4.177",
                  "torch": "2.11.0+cu130", "torchvision": "0.26.0+cu130"}
SELECTIONS = {
    MODEL: ("INPUT_CHANNELS", "REFINER_OUTPUT_PRIOR_LOGIT", "NEIGHBOR_FORBIDDEN_LOGIT",
            "_ensure_source_arrays", "source_roi_channels", "_ConvBlock", "NativeROIRefiner"),
    TILES: ("NATIVE", "TILE", "ANCHOR_QUANTILES", "source_tile_boxes"),
    IMAGE: ("original_halpha_pixels",),
    A4: ("NATIVE_HW", "IMGSZ", "CONF", "NMS_IOU", "MAX_DET", "compressed_rle"),
    A6: ("exclusivity_and_rle", "predict_test"),
    R2B: ("merge_source_only", "load_model"),
    R2C: ("source_exclusive_rles", "generate"),
}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_tree(raw, names):
    """No AST rewriting: select original complete top-level nodes only."""
    parsed, found, nodes = ast.parse(raw), set(), []
    for node in parsed.body:
        name = node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in names:
            need(name not in found, "Duplicate accepted definition: " + name)
            if isinstance(node, ast.Assign):
                ast.literal_eval(node.value)
            found.add(name)
            nodes.append(node)
    need(found == set(names), "Missing accepted definitions: " + str(set(names) - found))
    return ast.Module(body=nodes, type_ignores=[])


def production_functions(root, lock, torch):
    root = Path(root).resolve()
    need(set(lock["files"]) == set(SELECTIONS), "Accepted source inventory changed")
    sources = {}
    for path, spec in lock["files"].items():
        file = (root / path).resolve()
        need(file.is_relative_to(root) and file.stat().st_size == spec["bytes"]
             and sha(file) == spec["sha256"], "Accepted source bytes changed: " + path)
        sources[path] = file.read_bytes()
    shared = {"np": np, "torch": torch, "need": need, "require": need, "Image": Image,
              "Path": Path, "tempfile": tempfile, "shutil": shutil, "coco": coco, "mask_utils": coco}

    def build(path, extra):
        ns = {**shared, **extra, "__name__": path.removeprefix("src/").removesuffix(".py").replace("/", ".")}
        exec(compile(selected_tree(sources[path], SELECTIONS[path]), path, "exec"), ns)
        return ns

    model = build(MODEL, {"nn": torch.nn, "F": torch.nn.functional})
    tiles = build(TILES, {})
    image = build(IMAGE, {"NATIVE": 2048})
    a4 = build(A4, {})
    a6 = build(A6, {k: a4[k] for k in SELECTIONS[A4]})
    r2b = build(R2B, {"NATIVE": (2048, 2048), "THRESHOLD": .5,
                     "source_tile_boxes": tiles["source_tile_boxes"],
                     "source_roi_channels": model["source_roi_channels"],
                     "NativeROIRefiner": model["NativeROIRefiner"],
                     "MODEL_SHA": REFINER_SHA, "sha256_file": sha})
    r2c = build(R2C, {"original_halpha_pixels": image["original_halpha_pixels"],
                     "merge_source_only": r2b["merge_source_only"]})
    return {"detect": a6["predict_test"], "load_refiner": r2b["load_model"], "refine": r2c["generate"],
            "source_tiles": tiles["source_tile_boxes"], "source_channels": model["source_roi_channels"],
            "exclusive_masks": a6["exclusivity_and_rle"], "source_exclusive": r2c["source_exclusive_rles"]}


def kernel_flags(torch):
    return {"matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled())}


def runtime_profile():
    import torch
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        return None
    actual = {n: metadata.version(n) for n in EXACT_PACKAGES}
    need(actual == EXACT_PACKAGES, "Frozen inference package mismatch: " + str(actual))
    need(torch.version.cuda == "13.0" and torch.cuda.get_device_capability(0) == (7, 5),
         "Replay requires the recorded CUDA13/T4 compute stack")
    import cv2
    import torchvision
    # Both historical source_contract() calls import Ultralytics before
    # inference, including the R2C refiner process. Preserve that setup.
    import ultralytics
    need(ultralytics.__version__ == EXACT_PACKAGES["ultralytics"], "Ultralytics import/version mismatch")
    need(torchvision.__version__ == EXACT_PACKAGES["torchvision"], "Torchvision imported from another environment")
    props = torch.cuda.get_device_properties(0)
    return {"python": platform.python_version(), "packages": actual,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0), "compute_capability": list(torch.cuda.get_device_capability(0)),
            "gpu_total_bytes": props.total_memory, "opencv_version": cv2.__version__,
            "opencv_build_sha256": hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
            "initial_kernel_flags": kernel_flags(torch),
            "detector_precision": "FP32 half=False; exact accepted predict_test",
            "refiner_precision": "exact accepted autocast float16 on CUDA; strict threshold >0.5"}


class Backend:
    """A phase loads one model. No accepted mask/CSV paths are accepted here."""
    def __init__(self, source_root, source_lock, phase, detector_weights, refiner_directory):
        import torch
        self.torch, self.phase = torch, phase
        self.functions = production_functions(source_root, source_lock, torch)
        if phase == "detector":
            from ultralytics import YOLO
            need(Path(detector_weights).is_file() and sha(detector_weights) == DETECTOR_SHA,
                 "Accepted detector bytes changed before model loading")
            self.model, self.device = YOLO(str(detector_weights)), torch.device("cuda:0")
            need(self.model.task == "segment", "Accepted detector is not segmentation")
        elif phase == "refiner":
            self.model, self.device = self.functions["load_refiner"](
                SimpleNamespace(r2a_dir=Path(refiner_directory), device="cuda"))
            # Original R2C run() sets this after loading the refiner.
            torch.set_num_threads(min(4, os.cpu_count() or 2))
        else:
            raise ValueError("Unknown replay phase")

    def predict(self, image_path, image_id, new_detector_rles=None):
        if self.phase == "detector":
            need(new_detector_rles is None, "Detector must receive pixels, never prior masks")
            return self.functions["detect"](self.model, Path(image_path), image_id)
        need(new_detector_rles is not None, "Refiner requires the newly regenerated detector masks")
        rles, stats = self.functions["refine"](Path(image_path), {"rles": new_detector_rles}, self.model, self.device)
        return {"rles": rles, "stats": stats}

    def synchronize(self):
        self.torch.cuda.synchronize()

    def reset_peaks(self):
        self.torch.cuda.reset_peak_memory_stats()

    def memory(self):
        return {"allocated_peak_bytes": self.torch.cuda.max_memory_allocated(),
                "reserved_peak_bytes": self.torch.cuda.max_memory_reserved(),
                "kernel_flags_after_prediction": kernel_flags(self.torch)}

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()
