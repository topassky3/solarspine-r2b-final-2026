"""E10 A11-R2C: NEW official-format TEST CSV, source-only A6 + trained R2A U-Net.

Reads the *already accepted and immutable* A6 2048px TEST RLE records and
original TEST H-alpha images. Reuses EXACT frozen R2B ROI stitcher and model
checkpoint, without rerunning YOLO, training, reading TEST GT, or uploading.
A6 output directory, CSV and original A3/R2A checkpoints are read-only.
Resume image-by-image safely; only emit official format after ALL images.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from pycocotools import mask as coco

from solarspine.inference.competition import file_sha256
from solarspine.submission.competition import (
    COLUMNS,discover_images,verify_competition_csv
)
from solarspine.submission.e10_01_a6_test_submission import (
    verified_record,source_contract as a6_rebuild,
    SUCCESS as A6_SUCCESS,atomic_json,CSV_FILENAME as A6_FILENAME,
)
from solarspine.evaluation.e10_01_a11r2b_fold4_native_strict_pq import (
    merge_source_only,load_model,
    MODEL_SHA, DONE as R2B_DONE, BASE_PQ, PROMOTE_MIN,
)
from solarspine.training.e10_01_a11r1x_roi_patch_index import (
    original_halpha_pixels,
)
from solarspine.training.e9_01_train_runner import sha256_file

EXPERIMENT="E10-01-A11R2C-NATIVE-TEST-A6-PLUS-R2A-UNET-20261022"
DONE="E10_A11_R2C_FULL_TEST_CSV_VERIFIED__READY_FOR_MANUAL_KAGGLE_UPLOAD"
PARTIAL="E10_A11_R2C_PARTIAL_TEST_PREDICTIONS__NOT_SUBMITTABLE"
CSV_FILENAME="submission_e10_a11r2c.csv"
REQUIRED_A6_CSV_SHA="347307dc95727dbd9ea97d9d474c2125eb5bb14515e79c8bce563e51ead61a39"
MAX_SECONDS=85*60


def need(ok,msg):
    if not ok:
        raise RuntimeError(msg)


def freeze(args):
    # Original submission module verifies A3 checkpoint and A5 evidence,
    # original complete test roster + image content SHA. It never uses GT.
    weights,images,a6source=a6_rebuild(args)
    original_dir=args.a6_dir
    manifest_path=original_dir/"contract.json"
    report_path=original_dir/"summary.json"
    original_csv=original_dir/A6_FILENAME
    fold4_path=args.r2b_dir/"summary.json"
    status=args.r2a_dir/"status.json"
    model_path=args.r2a_dir/"best.pt"
    for p in (manifest_path,report_path,original_csv,fold4_path,status,model_path):
        need(p.is_file(),f"Frozen required input is missing: {p}")
    frozen=json.loads(manifest_path.read_text(encoding="utf-8"))
    a6=json.loads(report_path.read_text(encoding="utf-8"))
    r2b=json.loads(fold4_path.read_text(encoding="utf-8"))
    r2a=json.loads(status.read_text(encoding="utf-8"))
    # The original A6 was built under a DIFFERENT git commit.
    need(frozen.get("experiment")==a6source["experiment"]
         and frozen.get("roster")==a6source["roster"]
         and frozen.get("inference")==a6source["inference"]
         and frozen.get("model_sha256")==a6source["model_sha256"]
         and frozen.get("a3_report_sha256")==a6source["a3_report_sha256"]
         and frozen.get("a5_report_sha256")==a6source["a5_report_sha256"]
         and frozen.get("submission_columns")==list(COLUMNS)
         and frozen.get("test_ground_truth_never_read") is True,
         "Accepted A6 original TEST roster/inference/checkpoint drift")
    need(a6.get("decision")==A6_SUCCESS
         and a6.get("test_images")==len(images)
         and a6.get("csv_sha256")==REQUIRED_A6_CSV_SHA
         and a6.get("model_best_sha256")==frozen["model_sha256"]
         and a6.get("contains_test_GT") is False
         and a6.get("csv_full_original_verifier",{}).get("status")=="PASS"
         and a6["csv_full_original_verifier"].get("full_coverage") is True
         and sha256_file(original_csv)==REQUIRED_A6_CSV_SHA,
         "Accepted full verified A6 CSV/summary changed")
    need(r2b.get("decision")==R2B_DONE
         and r2b.get("source_model_sha256")==MODEL_SHA
         and abs(r2b.get("original_A6_fold4_strict_1to1",{}).get("micro_pq",-1)
                 -BASE_PQ)<1e-10
         and r2b.get("delta_micro_pq",0)>=.02
         and r2b.get("refined_A3_plus_R2A_fold4_strict_1to1",{})
                .get("micro_pq",0)>=PROMOTE_MIN
         and r2b.get("promotion_screen")==
             "R2B_CANDIDATE_PROMOTION_THRESHOLD_REACHED__FINAL_REVIEW"
         and r2b.get("created_Kaggle_submission") is False,
         "R2B frozen 231-image development PQ promotion evidence absent")
    need(r2a.get("best_checkpoint_sha256")==MODEL_SHA
         and sha256_file(model_path)==MODEL_SHA and
         r2a.get("no_Fold0_Fold4_Kaggle_TEST") is True,
         "R2A trained refiner model changed")
    expected={
        "experiment":EXPERIMENT,
        "parent_A6_csv_sha256":REQUIRED_A6_CSV_SHA,
        "parent_A6_contract_sha256":sha256_file(manifest_path),
        "parent_A6_summary_sha256":sha256_file(report_path),
        "parent_R2B_report_sha256":sha256_file(fold4_path),
        "R2A_refiner_sha256":MODEL_SHA,
        "source_script_sha256":sha256_file(Path(__file__)),
        "roster":frozen["roster"],
        "expected_test_images":len(images),
        "source_model":"Accepted A6 native RLE original + exact R2B source-only U-Net",
        "native_hw":[2048,2048],
        "submission_columns":list(COLUMNS),
        "no_new_YOLO_inference":True,
        "no_training":True,
        "no_TEST_GT_or_metadata":True,
        "no_Kaggle_upload":True,
        "original_A6_never_modified":True,
    }
    return images,frozen,expected


def source_exclusive_rles(obj):
    """A6 records are ALREADY first-owner exclusive compressed native RLE."""
    out=[]
    claimed=np.zeros((2048,2048),dtype=bool)
    for r in obj["rles"]:
        need(r["size"]==[2048,2048] and isinstance(r["counts"],str),
             "Original A6 TEST RLE encoding changed")
        rle={"size":[2048,2048],"counts":r["counts"].encode("ascii")}
        mask=coco.decode(rle).astype(bool)
        need(bool(mask.any()),"A6 accepted original contains empty instance")
        need(not np.any(claimed&mask),"A6 accepted original masks overlap")
        claimed|=mask
        out.append(rle)
    return out


def generate(image_path,a6_record,model,device):
    """No GT argument; exact R2B frozen source-only inference."""
    source=source_exclusive_rles(a6_record)
    if source:
        image=original_halpha_pixels(image_path)
        new,stat=merge_source_only(image,source,model,device)
    else:
        new=[]
        stat={
            "original_proposals":0,"refined_proposals":0,
            "tiles_inferred":0,
            "changed_pixels_before_final_ownership":0,
            "empty_mask_source_fallbacks":0,
            "source_instances_removed_by_new_overlap":0,
        }
    return [{
        "size":[2048,2048],
        "counts":r["counts"].decode("ascii"),
    } for r in new],stat


def verify_cached(path,item,old_sha):
    x=json.loads(path.read_text(encoding="utf-8"))
    need(x.get("experiment")==EXPERIMENT
         and x.get("image_id")==item["image_id"]
         and x.get("original_a6_record_sha256")==old_sha
         and x.get("original_test_image_sha256")==item["source_sha256"]
         and x.get("refiner_best_sha256")==MODEL_SHA
         and isinstance(x.get("pred_instances"),int)
         and len(x.get("rles",[]))==x.get("pred_instances")
         and x.get("no_TEST_GT") is True,
         "Previous per-image TEST refinement is not source-identical")
    for candidate in x["rles"]:
        need(candidate.get("size")==[2048,2048]
             and isinstance(candidate.get("counts"),str)
             and bool(candidate["counts"])
             and "\n" not in candidate["counts"]
             and "\r" not in candidate["counts"],
             "Cached native refined RLE format invalid")
    return x


def emit_csv(out,ordered,images,args,contract):
    need(len(ordered)==len(images),"Cannot emit partial Kaggle TEST CSV")
    csv_path=out/CSV_FILENAME
    manifest_path=csv_path.with_suffix(".json")
    rows=[]
    counts={}
    zero=[]
    for image_id,_path in images:
        pred=ordered[image_id]["rles"]
        counts[image_id]=len(pred)
        if not pred:zero.append(image_id)
        for i,r in enumerate(pred,1):
            rows.append({
                "filament_id":f"{image_id}_{i}",
                "segmentation_rle":r["counts"],
            })
    need(bool(rows),"Refiner generated zero instances across entire TEST set")
    if csv_path.is_file():
        with csv_path.open(newline="",encoding="utf-8") as f:
            rd=csv.DictReader(f)
            need(rd.fieldnames==list(COLUMNS) and list(rd)==rows,
                 "Existing finished R2C CSV does not match frozen records")
    else:
        fd,tmp=tempfile.mkstemp(prefix="._r2c_",suffix=".csv",dir=out)
        try:
            with os.fdopen(fd,"w",encoding="utf-8",newline="") as f:
                w=csv.DictWriter(f,fieldnames=list(COLUMNS),lineterminator="\n")
                w.writeheader();w.writerows(rows)
                f.flush();os.fsync(f.fileno())
            need(not csv_path.exists(),"R2C competing writer must not overwrite")
            os.replace(tmp,csv_path)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
    receipt={
        "experiment":EXPERIMENT,
        "format":"filament_id,segmentation_rle",
        "is_smoke":False,
        "processed_image_ids":[img for img,p in images],
        "image_instance_counts":counts,
        "images_without_prediction":zero,
        "predicted_instances":len(rows),
        "output_sha256":sha256_file(csv_path),
        "parent_A6_csv_sha256":REQUIRED_A6_CSV_SHA,
        "refiner_checkpoint_sha256":MODEL_SHA,
        "test_ground_truth_used":False,
    }
    if manifest_path.is_file():
        need(json.loads(manifest_path.read_text(encoding="utf-8"))==receipt,
             "Existing R2C manifest receipt changed")
    else:
        atomic_json(manifest_path,receipt)
    checked=verify_competition_csv(
        csv_path=csv_path,test_root=args.test_root,
        manifest_path=manifest_path)
    need(checked["status"]=="PASS"
         and checked["full_coverage"] is True
         and checked["is_smoke"] is False
         and checked["overlap_pixels"]==0
         and checked["rows_checked"]==len(rows),
         "Official TEST CSV roundtrip/coverage/zero-overlap verification failed")
    return receipt,checked


def run(args):
    images,a6_contract,contract=freeze(args)
    print("A11_R2C_SOURCE_PASS ACCEPTED_A6_INTACT FOLD4_PQ_PROMOTED",
          flush=True)
    if args.check_only:
        print("A11_R2C_PREFLIGHT_PASS__NO_TEST_REFINEMENT",flush=True)
        return
    out=args.output_dir
    contract_file=out/"contract.json"
    if out.exists():
        need(contract_file.is_file()
             and json.loads(contract_file.read_text(encoding="utf-8"))==contract,
             "R2C output directory has a different source/recipe")
    else:
        out.mkdir(parents=True,exist_ok=False)
        atomic_json(contract_file,contract)
    # Reuse exactly R2B checkpoint, implementation and tile-stitching.
    model,device=load_model(args)
    import torch
    torch.set_num_threads(min(4,os.cpu_count() or 2))
    by_id=dict(images)
    roster=a6_contract["roster"]
    done={}
    newly=0
    begin=time.monotonic()
    for index,item in enumerate(roster,1):
        key=item["image_id"]
        old_path=args.a6_dir/"records"/(key+".json")
        need(old_path.is_file(),"Accepted A6 record missing")
        original=verified_record(old_path,item,a6_contract["model_sha256"])
        source_sha=sha256_file(old_path)
        dest=out/"records"/(key+".json")
        if dest.is_file():
            saved=verify_cached(dest,item,source_sha)
        else:
            if time.monotonic()-begin>MAX_SECONDS:
                print(PARTIAL,"DONE",len(done),"TOTAL",len(roster),flush=True)
                return
            new,stats=generate(by_id[key],original,model,device)
            saved={
                "experiment":EXPERIMENT,
                "image_id":key,
                "source_test_image_sha256":item["source_sha256"],
                "original_test_image_sha256":item["source_sha256"],
                "original_a6_record_sha256":source_sha,
                "refiner_best_sha256":MODEL_SHA,
                "pred_instances":len(new),
                "rles":new,
                "stats":stats,
                "no_TEST_GT":True,
            }
            atomic_json(dest,saved)
            newly+=1
        done[key]=saved
        if index%10==0 or index==len(roster):
            print(f"A11_R2C_TEST {index}/{len(roster)} "
                  f"NEW:{newly} FILAMENTS:{sum(v['pred_instances'] for v in done.values())} "
                  f"SECONDS:{time.monotonic()-begin:.1f}",flush=True)
        if newly>=args.max_new and index<len(roster):
            print(PARTIAL,"NEW:",newly,"COMPLETE:",len(done),flush=True)
            return
    receipt,verification=emit_csv(out,done,images,args,contract)
    status={
        "experiment":EXPERIMENT,
        "decision":DONE,
        "test_images":len(images),
        "refined_instances":receipt["predicted_instances"],
        "new_submission_csv":str(out/CSV_FILENAME),
        "new_csv_sha256":receipt["output_sha256"],
        "original_accepted_A6_sha256":REQUIRED_A6_CSV_SHA,
        "refiner_checkpoint_sha256":MODEL_SHA,
        "official_csv_verifier":verification,
        "input_contract_sha256":sha256_file(contract_file),
        "model_fullFold4_exploratory_PQ_source_report_sha256":
            contract["parent_R2B_report_sha256"],
        "test_ground_truth_used":False,
        "uploaded_to_kaggle":False,
        "private_score":None,
    }
    report=out/"summary.json"
    if report.is_file():
        need(json.loads(report.read_text(encoding="utf-8"))==status,
             "Existing final R2C report does not match source")
    else:
        atomic_json(report,status)
    print("A11_R2C_FULL_TEST_CSV_VERIFIED__READY_FOR_MANUAL_KAGGLE_UPLOAD",
          flush=True)
    print("NEW_CSV:",out/CSV_FILENAME,flush=True)
    print("NEW_CSV_SHA256:",receipt["output_sha256"],flush=True)
    print("TEST_IMAGES:",len(images),"MASKS:",receipt["predicted_instances"],
          flush=True)
    print("COMPETITION_CSV_VERIFICATION: PASS NO_OVERLAP FULL_COVERAGE",
          flush=True)
    print("OLD_A6_ACCEPTED_CSV_SHA_UNCHANGED:",REQUIRED_A6_CSV_SHA,
          flush=True)
    print("NO_KAGGLE_UPLOAD_EXECUTED",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("a3-dir","a5-report","test-root","a6-dir","r2a-dir",
                 "r2b-dir","output-dir"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    p.add_argument("--max-new",type=int,default=100000)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only",action="store_true")
    mode.add_argument("--run",action="store_true")
    args=p.parse_args()
    need(args.max_new>=1,"Bad max-new budget")
    run(args)


if __name__=="__main__":
    main()
