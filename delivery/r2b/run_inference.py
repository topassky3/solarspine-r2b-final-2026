#!/usr/bin/env python3
"""Jury-facing SolarSpine R2B inference. No personal Drive or labels required.

This wrapper calls the SHA-locked original production definitions in a
separately bundled source directory. One fresh Python process per model.
It does not select models, tune thresholds, or read accepted prediction CSVs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

NATIVE_HW = (2048, 2048)
IMAGE_ID = re.compile(r"^\d{14}[BCLMTU]h$")
STAGES = ("detector", "refiner")
EXPECTED = {"detector": "bbb3c55175f68f80c213ac9fd6d1a1ab5bd63993696b609348d36f0472515875",
            "refiner": "e4b9c3bacce816cd1d4dfb863622b93774743a6a401ae844088e33540882f397"}
SOURCE_DIR = Path(__file__).resolve().parent / "source"
LOCK_PATH = SOURCE_DIR / "accepted_sources.lock.json"
EXPECTED_BACKEND_GIT_BLOB = "2bab54d05d3115dc0ef450103dd9539c8df9a5fb"
BACKEND_DIR = SOURCE_DIR / "scripts"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha(path):
    raw = Path(path).read_bytes()
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def serial(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_once(path, contents):
    path = Path(path)
    require(not path.exists() or path.read_bytes() == contents,
            "Existing different output protected: " + str(path))
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".partial_", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(contents)
            fh.flush()
            os.fsync(fh.fileno())
        require(not path.exists(), "Concurrent writer detected: " + str(path))
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def discover_images(directory, count, smoke=False):
    directory = Path(directory).resolve(strict=True)
    require(directory.is_dir(), "--images must be an existing directory")
    images = [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")]
    require(images, "No official JPEG images discovered")
    require(smoke or len(images) == count, f"Expected {count} images for full TEST; found {len(images)}")
    result = []
    for file in images:
        image_id = file.stem
        require(IMAGE_ID.fullmatch(image_id) is not None, "Unrecognized official image ID: " + image_id)
        result.append({"image_id": image_id, "relative_path": file.relative_to(directory).as_posix(),
                       "source_sha256": digest(file)})
    result.sort(key=lambda r: r["image_id"])
    require(len({r["image_id"] for r in result}) == len(result), "Duplicate physical image IDs")
    return result


def check_sources():
    require(LOCK_PATH.is_file() and (BACKEND_DIR / "r2b_replay_backend.py").is_file(),
            "Incomplete jury distribution: missing locked production sources")
    require(git_blob_sha(BACKEND_DIR / "r2b_replay_backend.py") == EXPECTED_BACKEND_GIT_BLOB,
            "Exact accepted replay backend changed")
    lock = load(LOCK_PATH)
    require(lock.get("schema") == "R2B-accepted-production-sources-v1" and len(lock["files"]) == 7,
            "Unknown original production source lock")
    for rel, meta in lock["files"].items():
        file = (SOURCE_DIR / rel).resolve()
        require(file.is_relative_to(SOURCE_DIR.resolve()) and file.is_file(), "Missing production source: " + rel)
        require(file.stat().st_size == meta["bytes"] and digest(file) == meta["sha256"],
                "Production source mismatch: " + rel)
    return lock


def check_weights(detector, refiner):
    dp, rp = Path(detector).resolve(strict=True), (Path(refiner) / "best.pt").resolve(strict=True)
    require(dp.is_file() and rp.is_file(), "Missing one of the two approved checkpoints")
    require(digest(dp) == EXPECTED["detector"], "Detector weight SHA256 differs from accepted A3")
    require(digest(rp) == EXPECTED["refiner"], "Refiner weight SHA256 differs from accepted R2A")


def rle_validate(rles):
    """Validate actual native COCO masks and prevent within-image overlap."""
    import numpy as np
    from pycocotools import mask as coco
    require(isinstance(rles, list), "A model must return an RLE list")
    occupied = np.zeros(NATIVE_HW, dtype=bool)
    for rle in rles:
        require(set(rle) == {"size", "counts"} and rle["size"] == list(NATIVE_HW)
                and isinstance(rle["counts"], str), "Invalid native COCO compressed RLE")
        obj = {"size": list(NATIVE_HW), "counts": rle["counts"].encode("ascii")}
        mask = coco.decode(obj)
        require(mask.shape == NATIVE_HW and mask.dtype == np.uint8 and bool(mask.any()),
                "Invalid or empty predicted filament mask")
        require(not np.any(occupied & (mask != 0)), "Overlapping instance masks")
        occupied |= mask != 0
        encoded = coco.encode(np.asfortranarray(mask))["counts"].decode("ascii")
        require(encoded == rle["counts"], "Compressed COCO RLE roundtrip mismatch")
    return len(rles)


def record_path(out, stage, image_id):
    return Path(out) / stage / "records" / f"{image_id}.json"


def validated_record(out, stage, image, run_sha, parent=None):
    path = record_path(out, stage, image["image_id"])
    data = load(path)
    require(data["stage"] == stage and data["run_sha256"] == run_sha and
            data["image_id"] == image["image_id"] and
            data["source_sha256"] == image["source_sha256"] and
            data["detector_parent_sha256"] == parent, "Record lineage mismatch")
    require(data.get("GT_used") is False and data.get("new_predictions") is True,
            "Record is not independent inference")
    require(data["instances"] == rle_validate(data["rles"]), "Instance count mismatch")
    return data


def build_csv(out, stage, roster, run_sha, smoke):
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["filament_id", "segmentation_rle"])
    counts = {}
    receipts = {}
    for image in roster:
        parent = digest(record_path(out, "detector", image["image_id"])) if stage == "refiner" else None
        record = validated_record(out, stage, image, run_sha, parent)
        counts[image["image_id"]] = len(record["rles"])
        receipts[image["image_id"]] = digest(record_path(out, stage, image["image_id"]))
        for j, rle in enumerate(record["rles"], 1):
            writer.writerow([image["image_id"] + "_" + str(j), rle["counts"]])
    path = Path(out) / ("detector.csv" if stage == "detector" else "submission_r2b.csv")
    write_once(path, buf.getvalue().encode("utf-8"))
    manifest = {"stage": stage, "run_sha256": run_sha, "complete": True, "smoke": smoke,
                "images": len(roster), "predictions": sum(counts.values()),
                "empty_image_ids": [k for k, v in counts.items() if v == 0],
                "image_instance_counts": counts, "record_sha256": receipts,
                "csv_sha256": digest(path), "GT_used": False}
    write_once(Path(out) / stage / "manifest.json", serial(manifest))
    return manifest


def phase(args, roster, run_sha):
    """Run one stage; called exclusively in a fresh child interpreter."""
    check_sources()
    check_weights(args.detector_weights, args.refiner_dir)
    sys.path.insert(0, str(BACKEND_DIR))
    from r2b_replay_backend import Backend, runtime_profile
    runtime = runtime_profile()
    require(runtime is not None, "This run requires the recorded Tesla T4 GPU runtime")
    write_once(Path(args.output_dir) / args.stage / "runtime.json", serial(runtime))
    backend = Backend(SOURCE_DIR, load(LOCK_PATH), args.stage, args.detector_weights, args.refiner_dir)
    try:
        for index, image in enumerate(roster, 1):
            out = record_path(args.output_dir, args.stage, image["image_id"])
            parent_path = record_path(args.output_dir, "detector", image["image_id"])
            parent = digest(parent_path) if args.stage == "refiner" else None
            if out.exists():
                validated_record(args.output_dir, args.stage, image, run_sha, parent)
                continue
            file = Path(args.images) / image["relative_path"]
            require(digest(file) == image["source_sha256"], "Source JPEG changed during inference")
            detector_rles = None
            if args.stage == "refiner":
                detector_rles = validated_record(args.output_dir, "detector", image, run_sha)["rles"]
            backend.reset_peaks()
            backend.synchronize()
            start = time.monotonic()
            prediction = backend.predict(file, image["image_id"], detector_rles)
            backend.synchronize()
            elapsed = time.monotonic() - start
            masks = prediction["rles"]
            rle_validate(masks)
            memory = backend.memory()
            require(memory["allocated_peak_bytes"] <= 8 * 1024**3,
                    "T4 8 GiB allocation budget exceeded, preserve current results")
            entry = {"image_id": image["image_id"], "stage": args.stage,
                     "run_sha256": run_sha, "source_sha256": image["source_sha256"],
                     "detector_parent_sha256": parent, "rles": masks, "instances": len(masks),
                     "seconds": elapsed, "memory": memory,
                     "GT_used": False, "new_predictions": True}
            write_once(out, serial(entry))
            print("R2B_JURY_PREDICT", args.stage, index, "/", len(roster),
                  image["image_id"], "instances", len(masks), flush=True)
    finally:
        backend.close()
    result = build_csv(args.output_dir, args.stage, roster, run_sha, args.smoke)
    print("R2B_JURY_STAGE_COMPLETE", json.dumps({"stage": args.stage,
          "csv_sha256": result["csv_sha256"], "images": result["images"],
          "predictions": result["predictions"], "smoke": args.smoke}), flush=True)


def stage_benchmark(output, stage, roster):
    times = []
    allocated, reserved = 0, 0
    for image in roster:
        record = load(record_path(output, stage, image["image_id"]))
        times.append(float(record["seconds"]))
        memory = record["memory"]
        allocated = max(allocated, int(memory["allocated_peak_bytes"]))
        reserved = max(reserved, int(memory["reserved_peak_bytes"]))
    ordered = sorted(times)
    from math import ceil
    return {"images": len(times), "prediction_seconds_sum": sum(times),
            "prediction_seconds_mean": sum(times) / len(times),
            "prediction_seconds_p95": ordered[max(0, ceil(0.95*len(times))-1)],
            "prediction_seconds_max": max(times),
            "torch_allocated_peak_bytes": allocated, "torch_reserved_peak_bytes": reserved,
            "does_not_include_setup_or_all_cuda_driver_memory": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--detector-weights", type=Path, required=True)
    parser.add_argument("--refiner-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=180)
    parser.add_argument("--smoke", action="store_true", help="allow partial image count; never submission-ready")
    parser.add_argument("--stage", choices=STAGES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    require(args.expected_images == 180 or args.smoke, "Expected population cannot change on a full run")
    output = args.output_dir.resolve()
    images = args.images.resolve(strict=True)
    require(output != images and not output.is_relative_to(images) and
            not images.is_relative_to(output), "Outputs must be outside the image tree")
    require(not output.is_relative_to(SOURCE_DIR.resolve()), "Output must not touch accepted sources")
    require(not output.is_relative_to(args.detector_weights.resolve().parent) and
            not output.is_relative_to(args.refiner_dir.resolve()),
            "Output must be outside frozen checkpoint directories")
    roster = discover_images(images, args.expected_images, args.smoke)
    check_sources()
    check_weights(args.detector_weights, args.refiner_dir)
    contract = {"candidate": "accepted_R2B_A3_YOLO26nseg_plus_R2A_UNet", "images": roster,
                "source_lock_sha256": digest(LOCK_PATH), "backend_sha256": digest(BACKEND_DIR / "r2b_replay_backend.py"),
                "checkpoint_sha256": EXPECTED, "full_test": not args.smoke,
                "runtime_required": "T4 CUDA 13.0 pinned", "no_GT": True,
                "submission_authorized": False}
    write_once(output / "contract.json", serial(contract))
    run_sha = digest(output / "contract.json")
    if args.stage:
        phase(args, roster, run_sha)
        return 0
    for stage in STAGES:
        argv_child = [sys.executable, str(Path(__file__).resolve()), "--images", str(images),
                      "--detector-weights", str(args.detector_weights.resolve()),
                      "--refiner-dir", str(args.refiner_dir.resolve()),
                      "--output-dir", str(output), "--expected-images", str(args.expected_images), "--stage", stage]
        if args.smoke:
            argv_child.append("--smoke")
        subprocess.run(argv_child, check=True)
    results = {s: load(output / s / "manifest.json") for s in STAGES}
    write_once(output / "summary.json", serial({"decision": "R2B_JURY_INFERENCE_COMPLETED__REFERENCE_QA_PENDING",
              "stages": results, "benchmarks": {stage: stage_benchmark(output, stage, roster) for stage in STAGES},
              "complete": True, "smoke": args.smoke, "run_sha256": run_sha,
              "Kaggle_upload_executed": False, "GT_used": False,
              "comparison_to_accepted_CSV_not_performed": True}))
    print("R2B_JURY_INFERENCE_COMPLETE", str(output / "summary.json"), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print("R2B_JURY_STOP", str(exc), file=sys.stderr, flush=True)
        raise SystemExit(2)
