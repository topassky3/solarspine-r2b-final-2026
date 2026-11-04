"""E10-A6 frozen YOLO26n-seg TEST submission, crash-resumable and auditable.

Reads ONLY competition H-alpha test images and SHA-verified A3 best.pt.
No training, no test GT, no Fold0 or Kaggle API submissions. Official CSV has
two fields: filament_id and native 2048² compressed segmentation RLE counts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco

from solarspine.inference.competition import file_sha256
from solarspine.submission.competition import (
    COLUMNS, discover_images, verify_competition_csv,
)
from solarspine.evaluation.e10_01_a4_fold4_native_pq import (
    CONF, NMS_IOU, IMGSZ, MAX_DET, NATIVE_HW,
    ULTRALYTICS_VERSION, compressed_rle,
)

EXPERIMENT="E10-01-A6-FROZEN-2048-TEST-SUBMISSION"
REQUIRED_A3_DECISION="E10_A3_PILOT_COMPLETED__FOLD4_NATIVE_PQ_PENDING"
REQUIRED_A5_DECISION="E10_A5_FOLD4_PAIRED_MICRO_PQ_GO__SUBMISSION_ARTIFACTS_NEXT"
SUCCESS="E10_A6_FULL_TEST_CSV_VERIFIED__READY_FOR_KAGGLE_UPLOAD"
PARTIAL="E10_A6_PARTIAL_TEST_PREDICTIONS__RESUMABLE_NOT_SUBMITTABLE"
WALL_LIMIT=90*60
CSV_FILENAME="submission_e10_a6.csv"


def require(ok:bool, message:str) -> None:
    if not ok:
        raise RuntimeError(message)


def atomic_json(path:Path, item:dict):
    require(not path.exists(),f"Cannot overwrite immutable {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix="._a6_",suffix=".json",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as h:
            json.dump(item,h,ensure_ascii=False,sort_keys=True,indent=2)
            h.write("\n");h.flush();os.fsync(h.fileno())
        require(not path.exists(),f"Concurrent output: {path}")
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def source_contract(args):
    """Immutable MODEL and image-list manifest. Never read test annotations."""
    import ultralytics
    require(ultralytics.__version__==ULTRALYTICS_VERSION,
            "Ultralytics != frozen 8.4.177")
    model_report=args.a3_dir/"pilot_manifest.json"
    weights=args.a3_dir/"train"/"weights"/"best.pt"
    require(model_report.is_file() and weights.is_file(),
            "A3 completed manifest or checkpoint absent")
    a3=json.loads(model_report.read_text(encoding="utf-8"))
    require(a3.get("decision")==REQUIRED_A3_DECISION
            and a3.get("fold0_excluded") is True
            and a3.get("kaggle_test_unused") is True,
            "Original A3 training source status/holdout invalid")
    weights_sha=file_sha256(weights)
    require(a3.get("best_checkpoint_sha256")==weights_sha,
            "A3 best.pt differs from trained immutable checkpoint")
    require(args.a5_report.is_file(),"A5 paired GO report missing")
    a5=json.loads(args.a5_report.read_text(encoding="utf-8"))
    require(a5.get("decision")==REQUIRED_A5_DECISION
            and abs(float(a5["E10"]["micro_pq"])-.32498391607953403)<1e-9
            and int(a5.get("rows",0))==231,
            "E10 A5 paired Fold4 evidence did not meet preregistered GO")
    images=discover_images(args.test_root)
    require(images,"Zero official TEST images")
    roster=[]
    for image_id,source in images:
        require(source.resolve().is_relative_to(args.test_root.resolve()),
                "Source test file escaped test directory")
        with Image.open(source) as im:
            require(tuple(im.size)==(2048,2048),
                    f"Test {image_id} is not 2048 by 2048")
        roster.append({
            "image_id":image_id,
            "relative_path":str(source.relative_to(args.test_root)),
            "source_sha256":file_sha256(source),
        })
    commit=subprocess.check_output(
        ["git","rev-parse","HEAD"],
        cwd=Path(__file__).resolve().parents[3],text=True
    ).strip()
    contract={
        "experiment":EXPERIMENT,
        "source_commit":commit,
        "official_competition":"filament-segmentation-2026",
        "source_model":"A3 YOLO26n-seg best.pt ONLY",
        "model_sha256":weights_sha,
        "a3_report_sha256":file_sha256(model_report),
        "a5_report_sha256":file_sha256(args.a5_report),
        "ultralytics_version":ULTRALYTICS_VERSION,
        "native_image_hw":[2048,2048],
        "expected_test_images":len(roster),
        "roster":roster,
        "inference":{
            "imgsz":IMGSZ,"conf":CONF,"iou":NMS_IOU,
            "max_det":MAX_DET,"retina_masks":True,
            "augment":False,"half":False,
            "overlap_exclusivity":"original_YOLO_order_first_mask_owns_pixel",
            "zero_posthoc_morphology":True,
        },
        "submission_columns":list(COLUMNS),
        "test_ground_truth_never_read":True,
        "no_training_or_checkpoint_writes":True,
        "not_a_Kaggle_submission_yet":True,
    }
    return weights,images,contract


def exclusivity_and_rle(masks:np.ndarray)->tuple[list[dict],int,int]:
    """Deterministic first-instance native pixel ownership; reject empty masks.

    Every original prediction mask is processed in YOLO result index order.
    This invariant may differ from A4 unmodified Fold4 PQ and is disclosed.
    """
    masks=np.asarray(masks)
    require(masks.ndim==3 and tuple(masks.shape[1:])==NATIVE_HW,
            "Native 2048 instance tensor mismatch")
    occupied=np.zeros(NATIVE_HW,dtype=bool)
    results=[]
    removed_overlap=0
    emptied=0
    for index in range(len(masks)):
        m=np.asarray(masks[index]>0)
        removed_overlap+=int(np.count_nonzero(m & occupied))
        available=m & ~occupied
        if not available.any():
            emptied+=1
            continue
        occupied|=available
        results.append(compressed_rle(available))
    return results,removed_overlap,emptied


def predict_test(model, path:Path, image_id:str)->dict:
    """Only native H-alpha pixels from this TEST image, no train/GT metadata."""
    with tempfile.TemporaryDirectory(prefix="solarspine_a6_test_") as d:
        copy=Path(d)/(image_id+path.suffix.lower())
        shutil.copy2(path,copy)
        result=model.predict(
            source=str(copy),imgsz=IMGSZ,conf=CONF,iou=NMS_IOU,
            max_det=MAX_DET,retina_masks=True,augment=False,
            half=False,device=0,save=False,verbose=False,stream=False,
        )
        require(len(result)==1,"Expected one image inference result")
        pred=result[0]
        require(tuple(pred.orig_shape)==NATIVE_HW,
                "Test image original dimensions changed")
        box=pred.boxes
        n=0 if box is None else len(box)
        require(n<=MAX_DET,"Exceeded frozen max detection count")
        if n==0:
            require(pred.masks is None or len(pred.masks.data)==0,
                    "No-box result has non-empty masks")
            return {"rles":[],"raw_predictions":0,
                    "masked_overlap_pixels_removed":0,
                    "fully_occluded_instances_removed":0}
        require(pred.masks is not None
                and tuple(pred.masks.data.shape)==(n,*NATIVE_HW),
                "Actual retina_masks data not native 2048")
        data=pred.masks.data.detach().to("cpu").numpy()
        encoded,overlap,emptied=exclusivity_and_rle(data)
        return {
            "rles":encoded,
            "raw_predictions":n,
            "masked_overlap_pixels_removed":overlap,
            "fully_occluded_instances_removed":emptied,
        }


def verified_record(path, contract_item, checkpoint_sha):
    obj=json.loads(path.read_text(encoding="utf-8"))
    require(obj.get("experiment")==EXPERIMENT
            and obj.get("image_id")==contract_item["image_id"]
            and obj.get("source_sha256")==contract_item["source_sha256"]
            and obj.get("checkpoint_sha256")==checkpoint_sha
            and len(obj.get("rles",[]))==int(obj.get("pred_instances",-1))
            and int(obj.get("raw_predictions",-1))>=int(obj["pred_instances"]),
            "Cached TEST prediction identity/metadata conflict")
    for rle in obj["rles"]:
        require(rle["size"]==[2048,2048]
                and isinstance(rle["counts"],str)
                and rle["counts"] and "\n" not in rle["counts"],
                "Cached TEST compressed COCO RLE invalid")
    return obj


def emit_csv_and_manifest(out:Path, contract:dict, completed:dict,
                          test_root:Path)->dict:
    """Assemble only after ALL images present; safe to retry interrupted finalization."""
    require(set(completed)=={x["image_id"] for x in contract["roster"]},
            "Cannot construct Kaggle CSV from incomplete TEST directory")
    path=out/CSV_FILENAME
    manifest_path=path.with_suffix(".json")
    rows=[]
    image_counts={}
    missing=[]
    for item in contract["roster"]:
        key=item["image_id"]
        record=completed[key]
        encoded=record["rles"]
        image_counts[key]=len(encoded)
        if not encoded: missing.append(key)
        for index,rle in enumerate(encoded,1):
            rows.append({
                "filament_id":f"{key}_{index}",
                "segmentation_rle":rle["counts"],
            })
    require(rows,"Model produced zero filaments on all TEST images")
    if path.exists():
        with path.open(encoding="utf-8",newline="") as f:
            reader=csv.DictReader(f)
            actual=list(reader)
            require(reader.fieldnames==list(COLUMNS)
                    and actual==rows,
                    "Existing final CSV differs from frozen outputs")
    else:
        fd,tmp=tempfile.mkstemp(prefix="._a6_",suffix=".csv",dir=out)
        try:
            with os.fdopen(fd,"w",newline="",encoding="utf-8") as f:
                writer=csv.DictWriter(f,fieldnames=list(COLUMNS),lineterminator="\n")
                writer.writeheader();writer.writerows(rows)
                f.flush();os.fsync(f.fileno())
            require(not path.exists(),"Concurrent official CSV created")
            os.replace(tmp,path)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
    manifest={
        "experiment":EXPERIMENT,
        "is_smoke":False,
        "processed_image_ids":[x["image_id"] for x in contract["roster"]],
        "images_without_prediction":missing,
        "image_instance_counts":image_counts,
        "predicted_instances":len(rows),
        "output_sha256":file_sha256(path),
        "checkpoint_sha256":contract["model_sha256"],
        "not_test_gt_used":True,
        "format":"filament_id,segmentation_rle",
    }
    if manifest_path.exists():
        prior=json.loads(manifest_path.read_text(encoding="utf-8"))
        require(prior==manifest,
                "Existing official CSV manifest differs from frozen record data")
    else:
        atomic_json(manifest_path,manifest)
    # Mandatory original full decoded-RLE overlap and coverage validation.
    verification=verify_competition_csv(csv_path=path,test_root=test_root,
                                        manifest_path=manifest_path)
    require(verification["status"]=="PASS"
            and verification["full_coverage"] is True
            and verification["is_smoke"] is False
            and verification["overlap_pixels"]==0
            and verification["rows_checked"]==len(rows),
            "Strict original full Kaggle CSV verification did not PASS")
    return {"manifest":manifest,"verification":verification}


def run(args):
    import ultralytics
    from ultralytics import YOLO
    weights,images,contract=source_contract(args)
    path=args.output_dir
    existing_contract=path/"contract.json"
    if path.exists():
        require(existing_contract.is_file(),
                "Existing A6 output without contract: refuse to overwrite")
        prior=json.loads(existing_contract.read_text(encoding="utf-8"))
        require(prior==contract,
                "Cannot resume A6 with changed model, images, code or inference")
    else:
        path.mkdir(parents=True,exist_ok=False)
        atomic_json(existing_contract,contract)

    done={}
    pending=[]
    for item in contract["roster"]:
        record_file=path/"records"/(item["image_id"]+".json")
        if record_file.is_file():
            done[item["image_id"]]=verified_record(
                record_file,item,contract["model_sha256"])
        else:
            pending.append(item)
    print("E10_A6_TEST_ROSTER",len(images),"DONE",len(done),
          "REMAINING",len(pending),flush=True)
    model=None
    started=time.monotonic()
    if pending:
        require(torch.cuda.is_available()
                and torch.cuda.get_device_name(0)=="Tesla T4",
                "A6 frozen first official TEST inference requires Tesla T4")
        model=YOLO(str(weights))
        require(model.task=="segment","A3 checkpoint not segmentation")
    index={key:p for key,p in images}
    for ordinal,item in enumerate(pending,1):
        if time.monotonic()-started>WALL_LIMIT:
            print("DECISION:",PARTIAL,flush=True)
            print("E10_A6_TIME_CEILING__RESUME_SAME_SHA",flush=True)
            return {"decision":PARTIAL,"completed":len(done),
                    "expected":len(images)}
        id=item["image_id"]
        pred=predict_test(model,index[id],id)
        record={
            "experiment":EXPERIMENT,"image_id":id,
            "source_sha256":item["source_sha256"],
            "checkpoint_sha256":contract["model_sha256"],
            "pred_instances":len(pred["rles"]),
            **pred,
        }
        atomic_json(path/"records"/(id+".json"),record)
        done[id]=record
        if ordinal%10==0 or ordinal==len(pending):
            print(f"E10_A6_TEST_PREDICTED {len(done)}/{len(images)} "
                  f"FILAMENTS={sum(x['pred_instances'] for x in done.values())}",
                  flush=True)
    results=emit_csv_and_manifest(path,contract,done,args.test_root)
    status={
        "decision":SUCCESS,
        "test_images":len(images),
        "predicted_filaments":results["manifest"]["predicted_instances"],
        "zero_prediction_images":len(results["manifest"]["images_without_prediction"]),
        "total_overlapping_pixels_disambiguated":sum(
            x["masked_overlap_pixels_removed"] for x in done.values()),
        "total_fully_occluded_instances_removed":sum(
            x["fully_occluded_instances_removed"] for x in done.values()),
        "model_best_sha256":contract["model_sha256"],
        "csv_sha256":results["manifest"]["output_sha256"],
        "csv_path":str(path/CSV_FILENAME),
        "csv_full_original_verifier":results["verification"],
        "A5_paired_source_sha256":contract["a5_report_sha256"],
        "source_commit":contract["source_commit"],
        "contains_test_GT":False,
        "uploaded_to_kaggle":False,
        "official_private_pq":None,
        "overlap_disambiguation_differs_from_unmodified_A4_possible":True,
    }
    summary_path=path/"summary.json"
    if summary_path.exists():
        old=json.loads(summary_path.read_text(encoding="utf-8"))
        require(old==status, "Existing A6 result differs; no overwrite")
    else:
        atomic_json(summary_path,status)
    print("DECISION:",SUCCESS,flush=True)
    print("CSV:",str(path/CSV_FILENAME),flush=True)
    print("FILAMENTS:",status["predicted_filaments"],flush=True)
    print("TEST_IMAGES:",status["test_images"],flush=True)
    print("CSV_SHA256:",status["csv_sha256"],flush=True)
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("a3-dir","a5-report","test-root","output-dir"):
        p.add_argument("--"+key,type=Path,required=True)
    run(p.parse_args())


if __name__=="__main__":
    main()
