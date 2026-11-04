"""E10 A4: resumable SHA-locked native (2048x2048) instance PQ on E2 Fold4.

No checkpoint changes, Fold0 images, Kaggle test or training. Predictions come
only from saved A3 best.pt and permitted original H-alpha images; GT is used
only after inference, from Fold4 segmentation polygons. Append/atomic per-row
evidence lets Colab resume after disconnections without losing successful rows.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image
import torch
from pycocotools import mask as mask_utils

from solarspine.data.fold_freeze import verify_contract
from solarspine.evaluation.baseline_oracle import (
    _load_coco, annotation_rles, rle_panoptic_quality
)
from solarspine.evaluation.e10_01_a2l0_ultralytics_loader import (
    read_manifest, require, ULTRALYTICS_VERSION
)
from solarspine.evaluation.e10_01_a2l1_full_train_loader import (
    frozen_train_rows, check_manifest_alignment
)
from solarspine.training.e9_01_train_runner import (
    ANNOTATIONS_SHA, FOLDS_SHA, MANIFEST_SHA, sha256_file
)

EXPERIMENT = "E10-01-A4-FOLD4-2048-RLE-PQ-FROZEN"
EXPECTED_ALL_TRAIN_RECORDS = 923
NATIVE_HW = (2048, 2048)
VAL_FOLD = 4
IMGSZ = 1024
CONF = 0.25
NMS_IOU = 0.7
MAX_DET = 300
INFERENCE_BUDGET_SECONDS = 90 * 60
SUCCESS = "E10_A4_FOLD4_NATIVE_RLE_PQ_COMPLETE__E4_PAIRED_COMPARISON_PENDING"
PARTIAL = "E10_A4_FOLD4_NATIVE_RLE_PQ_INCOMPLETE_RESUMABLE"


def json_atomic(path: Path, obj: dict) -> None:
    """Create-only JSON: temp is cleaned and no original evidence replaced."""
    require(not path.exists(), f"Immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="._e10_a4_", suffix=".json",
                                dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        require(not path.exists(), f"Racing output created: {path}")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def csv_atomic(path: Path, columns: list[str], rows: list[dict]) -> None:
    # A notebook may disconnect after creating the first of two final CSVs.
    # If this exact immutable CSV already exists, validate all its rows and
    # continue finalization instead of overwriting or deadlocking the run.
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            actual = list(reader)
            require(reader.fieldnames == columns,
                    "Existing final CSV column schema changed")
        expected = [{key: str(item[key]) for key in columns} for item in rows]
        require(actual == expected,
                "Existing final CSV contents differ from frozen records")
        return
    fd, temp = tempfile.mkstemp(prefix="._e10_a4_", suffix=".csv",
                                dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            writer=csv.DictWriter(stream, fieldnames=columns,
                                  lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush();os.fsync(stream.fileno())
        require(not path.exists(), f"CSV created concurrently: {path}")
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def require_sources(args):
    """A3 report, actual frozen Fold4 population and single best checkpoint."""
    report_path=args.a3_dir/"pilot_manifest.json"
    best=args.a3_dir/"train"/"weights"/"best.pt"
    require(report_path.is_file() and best.is_file(),
            "A3 was not completed or best.pt missing: do NOT retrain")
    report=json.loads(report_path.read_text(encoding="utf-8"))
    require(report.get("decision")==
            "E10_A3_PILOT_COMPLETED__FOLD4_NATIVE_PQ_PENDING",
            "Existing A3 not confirmed as completed")
    require(report.get("fold0_excluded") is True
            and report.get("train_folds")==[1,2,3]
            and int(report.get("val_fold",-1))==VAL_FOLD
            and report.get("kaggle_test_unused") is True,
            "A3 train/val provenance unsafe")
    best_sha=sha256_file(best)
    require(report.get("best_checkpoint_sha256")==best_sha,
            "Frozen A3 best checkpoint differs from sealed report")
    require(report.get("A1_sha256")==sha256_file(args.a1_dir/"summary.json"),
            "A3/A1 verified segmentation source changed")
    contract=verify_contract(args.folds,args.manifest,args.lock)
    require(contract["folds_sha256"]==FOLDS_SHA
            and contract["manifest_sha256"]==MANIFEST_SHA,
            "E2 immutable split changed")
    require(sha256_file(args.annotations)==ANNOTATIONS_SHA,
            "Canonical TRAIN COCO annotations altered")
    whole=frozen_train_rows(args)
    manifest=read_manifest(args)
    check_manifest_alignment(whole,manifest)
    require(len(whole)==EXPECTED_ALL_TRAIN_RECORDS,
            "Unexpected E2 full TRAIN record count")
    frozen_train={r["image_id"] for r in whole if int(r["fold"]) in (1,2,3)}
    val=sorted((r for r in whole if int(r["fold"])==VAL_FOLD),
               key=lambda r:r["image_id"])
    require(val and frozen_train
            and set(report["val_image_ids"])=={r["image_id"] for r in val}
            and set(report["train_image_ids"])==frozen_train
            and len(report["val_image_ids"])==len(val)
            and len(report["train_image_ids"])==len(frozen_train),
            "A3 trained/validation IDs mismatch original E2 Fold4")
    for group in ("physical_image_id","temporal_group_id"):
        require(
            {r[group] for r in val}.isdisjoint(
                {r[group] for r in whole if int(r["fold"]) in (1,2,3)}),
            f"GT physical/temporal overlap across Fold4 boundary: {group}"
        )
    for row in val:
        item=manifest[row["image_id"]]
        source=Path(item["original_image_path"])
        require(source.is_file()
                and source.resolve().is_relative_to(args.images.resolve()),
                "Fold4 source must be original permitted TRAIN H-alpha")
        require((int(item["height"]),int(item["width"]))==NATIVE_HW,
                "Expected original H-alpha at exact native 2048x2048")
    return report,report_path,best,best_sha,val,manifest


def fold4_gt(args, val: list[dict]) -> dict[str,list[dict]]:
    """Only access segmentation polygons belonging to Fold4 image IDs."""
    image_ids={row["image_id"] for row in val}
    raw=_load_coco(args.annotations)
    image_shapes={}
    for img in raw["images"]:
        image_id=str(img["id"])
        if image_id in image_ids:
            require(image_id not in image_shapes, "Duplicated Fold4 image ID")
            image_shapes[image_id]=(int(img["height"]),int(img["width"]))
    require(set(image_shapes)==image_ids,"Fold4 image shapes missing from COCO")
    anns=defaultdict(list)
    for ann in raw["annotations"]:
        key=str(ann["image_id"])
        if key in image_ids:
            # GT bbox, spine, category and area are forbidden as algorithm features.
            anns[key].append({"id":str(ann["id"]),
                              "segmentation":ann["segmentation"]})
    by_id={}
    for row in val:
        id=row["image_id"]
        h,w=image_shapes[id]
        require((h,w)==NATIVE_HW,"Fold4 COCO geometry changed")
        anns[id].sort(key=lambda a:a["id"])
        rles=annotation_rles(anns[id],height=h,width=w)
        require(len(rles)>0,"Unexpected empty Fold4 GT annotations")
        by_id[id]=rles
    return by_id


def compressed_rle(binary:np.ndarray) -> dict:
    """Native compressed counts only for compact auditable internal JSON/CSV."""
    mask=np.asarray(binary)
    require(mask.ndim==2 and mask.shape==NATIVE_HW,
            "Ultralytics masks MUST be actual source-native 2048 pixels")
    mask=np.asarray(mask>0,dtype=np.uint8)
    require(bool(mask.any()),"Empty predicted instance mask")
    encoded=mask_utils.encode(np.asfortranarray(mask))
    require(list(encoded["size"])==list(NATIVE_HW),"COCO encoding shape mismatch")
    verify=mask_utils.decode(encoded)
    require(np.array_equal(verify,mask),
            "COCO-RLE prediction roundtrip changed mask pixels")
    counts=encoded["counts"]
    require(isinstance(counts,bytes),"Expected compressed COCO RLE bytes")
    return {"size":list(NATIVE_HW),"counts":counts.decode("ascii")}


def evaluate_predictions(pred_rles:list[dict], gt_rles:list[dict]) -> dict:
    """Use the already-tested competition PQ implementation in this repo."""
    pred=[{"size":list(NATIVE_HW),
           "counts":r["counts"].encode("ascii")} for r in pred_rles]
    result=rle_panoptic_quality(gt_rles,pred)
    require(0<=result.pq<=1 and np.isfinite(result.pq),
            "Invalid project canonical RLE PQ")
    return {"pq":float(result.pq),"sq":float(result.sq),"rq":float(result.rq),
            "tp":int(result.tp),"fp":int(result.fp),"fn":int(result.fn),
            "sum_iou":float(result.sum_iou)}


def predict_one(model, source:Path, image_id:str) -> tuple[list[dict],list[float]]:
    """GT is never supplied to YOLO; predict on disposable H-alpha-only copy."""
    with tempfile.TemporaryDirectory(prefix="e10_a4_image_") as dirname:
        copy=Path(dirname)/(image_id+source.suffix.lower())
        shutil.copy2(source,copy)
        with Image.open(copy) as pil:
            require(pil.size==(2048,2048),
                    "Prediction image dimensions differ from COCO 2048")
        results=model.predict(
            source=str(copy),imgsz=IMGSZ,conf=CONF,iou=NMS_IOU,
            max_det=MAX_DET,retina_masks=True,augment=False,half=False,
            device=0,save=False,verbose=False,stream=False,
        )
        require(len(results)==1,"Expected exactly one native image prediction")
        prediction=results[0]
        require(tuple(prediction.orig_shape)==NATIVE_HW,
                "YOLO original shape changed")
        boxes=prediction.boxes
        n=0 if boxes is None else len(boxes)
        require(n<=MAX_DET,"YOLO produced more than frozen max_det")
        if n==0:
            require(prediction.masks is None
                    or len(prediction.masks.data)==0,
                    "Inconsistent zero-box prediction with masks")
            return [],[]
        require(prediction.masks is not None,
                "YOLO produced boxes without instance masks")
        batch=prediction.masks.data
        require(tuple(batch.shape)==(n,*NATIVE_HW),
                f"retina_masks=True did not produce native 2048 masks: "
                f"{tuple(batch.shape)}")
        confidence=boxes.conf.detach().cpu().numpy().astype(float).tolist()
        require(len(confidence)==n
                and all(math.isfinite(c) and CONF<=c<=1 for c in confidence),
                "Predicted confidence outside frozen threshold")
        out=[]
        for index in range(n):
            mask=batch[index].detach().to("cpu").numpy()
            out.append(compressed_rle(mask))
        return out,confidence


def record_payload(image_id, row, gt_rles, predictions, confidences,
                   best_sha) -> dict:
    require(len(predictions)==len(confidences),"Predictions/confidences drift")
    metrics=evaluate_predictions(predictions,gt_rles)
    return {
        "experiment":EXPERIMENT,"image_id":image_id,
        "physical_image_id":row["physical_image_id"],
        "temporal_group_id":row["temporal_group_id"],
        "fold":VAL_FOLD,"checkpoint_sha256":best_sha,
        "gt_instances":len(gt_rles),"pred_instances":len(predictions),
        "confidence":confidences,"pred_rles":predictions,
        "native_shape":list(NATIVE_HW),
        **metrics,
    }


def verify_existing(path:Path, image_id:str, row:dict, expected_gt:int,
                    best_sha:str) -> dict:
    obj=json.loads(path.read_text(encoding="utf-8"))
    require(obj["experiment"]==EXPERIMENT
            and obj["image_id"]==image_id
            and obj["fold"]==VAL_FOLD
            and obj["checkpoint_sha256"]==best_sha
            and obj["physical_image_id"]==row["physical_image_id"]
            and obj["temporal_group_id"]==row["temporal_group_id"]
            and obj["gt_instances"]==expected_gt
            and obj["native_shape"]==list(NATIVE_HW)
            and len(obj["pred_rles"])==obj["pred_instances"]
            and len(obj["confidence"])==obj["pred_instances"],
            "Cached prediction not from identical source/record/GT")
    require(obj["pred_instances"]<=MAX_DET,"Cached image too many proposals")
    for rle in obj["pred_rles"]:
        require(rle["size"]==list(NATIVE_HW)
                and isinstance(rle["counts"],str),
                "Cached RLE shape or counts malformed")
    # Defensive numerics; no rescoring/tuning on resume.
    for key in ("pq","sq","rq","sum_iou"):
        require(math.isfinite(float(obj[key])),
                "Cached PQ numerical state nonfinite")
    return obj


def gather(rows:list[dict]) -> dict:
    require(bool(rows),"No Fold4 rows")
    tp=sum(int(r["tp"]) for r in rows)
    fp=sum(int(r["fp"]) for r in rows)
    fn=sum(int(r["fn"]) for r in rows)
    iou=sum(float(r["sum_iou"]) for r in rows)
    d=tp+0.5*(fp+fn)
    return {
        "fold":VAL_FOLD,"records":len(rows),
        "gt_instances":sum(int(r["gt_instances"]) for r in rows),
        "pred_instances":sum(int(r["pred_instances"]) for r in rows),
        "tp":tp,"fp":fp,"fn":fn,"sum_iou":iou,
        "micro_pq":iou/d if d else 0.0,
        "micro_sq":iou/tp if tp else 0.0,
        "micro_rq":tp/d if d else 0.0,
        "macro_pq":float(np.mean([r["pq"] for r in rows])),
        "zero_prediction_records":sum(r["pred_instances"]==0 for r in rows),
    }


def finalize(output_dir:Path, rows:list[dict], contract:dict,
             seconds:float) -> dict:
    require(len(rows)==contract["fold4_record_count"],
            "Cannot produce final summary from incomplete Fold4")
    overview=gather(rows)
    per_rows=[{
        key:r[key] for key in (
            "image_id","physical_image_id","fold","gt_instances",
            "pred_instances","tp","fp","fn","sum_iou","pq","sq","rq"
        )
    } for r in rows]
    internal=[]
    for r in rows:
        for i,encoded in enumerate(r["pred_rles"],1):
            internal.append({
                "image_id":r["image_id"],"pred_index":i,
                "segmentation_rle":encoded["counts"],
                "confidence":r["confidence"][i-1]
            })
    csv_atomic(output_dir/"per_record.csv",list(per_rows[0]),per_rows)
    csv_atomic(output_dir/"predictions_fold4_INTERNAL_NOT_KAGGLE.csv",
               ["image_id","pred_index","segmentation_rle","confidence"],
               internal)
    summary={
        "experiment":EXPERIMENT,"decision":SUCCESS,
        "fold4":overview,
        "official_kaggle_private_score":None,
        "note":"Development Fold4 was used by training-time validation; "
               "its PQ is not an independent final generalization score.",
        "inference_config":contract["inference"],
        "best_checkpoint_sha256":contract["best_checkpoint_sha256"],
        "a3_manifest_sha256":contract["a3_manifest_sha256"],
        "source_commit":contract["source_commit"],
        "elapsed_inference_seconds":round(seconds,2),
        "all_2048_rle_roundtrip_tested":True,
        "kaggle_submission_generated":False,
        "train_or_checkpoint_mutations":False,
        "per_record_csv_sha256":sha256_file(output_dir/"per_record.csv"),
        "internal_predictions_csv_sha256":
            sha256_file(output_dir/"predictions_fold4_INTERNAL_NOT_KAGGLE.csv"),
    }
    json_atomic(output_dir/"summary.json",summary)
    return summary


def run(args):
    import ultralytics
    from ultralytics import YOLO
    require(ultralytics.__version__==ULTRALYTICS_VERSION,
            "Ultralytics version mismatch")
    require(torch.cuda.is_available()
            and torch.cuda.get_device_name(0)=="Tesla T4",
            "Exact Tesla T4 required for fixed Fold4 inference")
    a3, report_path, best, best_sha, val, manifest=require_sources(args)
    gt=fold4_gt(args,val)
    source_commit=subprocess.check_output(
        ["git","rev-parse","HEAD"],
        cwd=Path(__file__).resolve().parents[3],text=True
    ).strip()
    frozen={
        "experiment":EXPERIMENT,"source_commit":source_commit,
        "a3_manifest_sha256":sha256_file(report_path),
        "best_checkpoint_sha256":best_sha,
        "a1_summary_sha256":sha256_file(args.a1_dir/"summary.json"),
        "e2_fold_manifest_sha256":MANIFEST_SHA,
        "annotations_sha256":ANNOTATIONS_SHA,
        "ultralytics_version":ULTRALYTICS_VERSION,
        "fold":VAL_FOLD,"fold4_record_count":len(val),
        "fold4_image_ids":[r["image_id"] for r in val],
        "native_hw":list(NATIVE_HW),
        "inference":{
            "imgsz":IMGSZ,"conf":CONF,"iou":NMS_IOU,
            "max_det":MAX_DET,"retina_masks":True,
            "augment":False,"half":False,"device":0,
            "no_postprocessing_other_than_YOLO":True,
        },
        "no_fold0_or_test_inference":True,"no_training":True,
    }
    contract_path=args.output_dir/"contract.json"
    if args.output_dir.exists():
        require(contract_path.is_file(),
                "Existing output without sealed A4 contract: no overwrite")
        existing=json.loads(contract_path.read_text(encoding="utf-8"))
        require(existing==frozen,
                "Existing A4 contract differs: do NOT merge experiment outputs")
    else:
        args.output_dir.mkdir(parents=True,exist_ok=False)
        json_atomic(contract_path,frozen)
    if (args.output_dir/"summary.json").exists():
        existing=json.loads((args.output_dir/"summary.json").read_text(
            encoding="utf-8"))
        require(existing.get("decision")==SUCCESS
                and existing.get("best_checkpoint_sha256")==best_sha,
                "Existing A4 summary invalid or drifted")
        print("E10_A4_ALREADY_COMPLETED__NO_RETRAIN_NO_OVERWRITE",flush=True)
        print("DECISION:",existing["decision"],flush=True)
        print("FOLD4:",json.dumps(existing["fold4"],sort_keys=True),flush=True)
        return existing

    completed=[]
    pending=[]
    record_dir=args.output_dir/"records"
    for row in val:
        p=record_dir/(row["image_id"]+".json")
        if p.exists():
            completed.append(verify_existing(
                p,row["image_id"],row,len(gt[row["image_id"]]),best_sha))
        else:
            pending.append(row)
    print("E10_A4_IMMUTABLE_FOLD4_CONTRACT_PASS",flush=True)
    print("E10_A4_FOLD4_TOTAL",len(val),"ALREADY_DONE",len(completed),
          "REMAINING",len(pending),flush=True)
    model=YOLO(str(best))
    require(model.task=="segment",
            "A3 best checkpoint not an instance segmentation model")
    began=time.monotonic()
    for position,row in enumerate(pending,1):
        if time.monotonic()-began>INFERENCE_BUDGET_SECONDS:
            print("E10_A4_TIME_LIMIT_INCOMPLETE_RESUMABLE",flush=True)
            break
        id=row["image_id"]
        path=Path(manifest[id]["original_image_path"])
        pred,confidence=predict_one(model,path,id)
        payload=record_payload(id,row,gt[id],pred,confidence,best_sha)
        json_atomic(record_dir/(id+".json"),payload)
        completed.append(payload)
        if position%10==0 or position==len(pending):
            print(f"E10_A4_NATIVE_FOLD4 {len(completed)}/{len(val)} "
                  f"RECENT_PQ={payload['pq']:.6f} "
                  f"GT={payload['gt_instances']} PRED={payload['pred_instances']}",
                  flush=True)
    if len(completed)!=len(val):
        print("DECISION:",PARTIAL,flush=True)
        print("COMPLETED:",len(completed),"TOTAL:",len(val),flush=True)
        print("RESUME_SAME_BRANCH_CHECKPOINT_AND_OUTPUT",flush=True)
        return {"decision":PARTIAL,"complete":len(completed),"total":len(val)}
    # Sort by original E2 fold4 record ordering, rather than order of resumes.
    indexed={x["image_id"]:x for x in completed}
    ordered=[indexed[r["image_id"]] for r in val]
    summary=finalize(args.output_dir,ordered,frozen,time.monotonic()-began)
    print("E10_A4_NATIVE_RLE_PQ_COMPLETE",flush=True)
    print("DECISION:",summary["decision"],flush=True)
    print("FOLD4:",json.dumps(summary["fold4"],sort_keys=True),flush=True)
    print("REPORT:",args.output_dir/"summary.json",flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("folds","manifest","lock","annotations","images",
                "a1-dir","a3-dir","output-dir"):
        parser.add_argument("--"+key,type=Path,required=True)
    run(parser.parse_args())


if __name__=="__main__":
    main()
