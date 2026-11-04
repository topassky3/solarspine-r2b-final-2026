"""A11 R1X: source-locked CPU construction audit for native ROI refiner patches.

Use ONLY image, teacher OOF exclusive mask, disjoint neighbors as model inputs.
Fold3 GT is used ONLY as supervision / diagnostic, never for tile selection.
No images from Fold0/Fold4/Kaggle TEST or model optimization are permitted.
Create 3825 indexed native 256px patch coordinates without bulk PNG copies.
Read from original sources lazily at train time. This stage is a provenance
and neighbor-conflict gate, not a trained refiner or evidence of higher PQ.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import csv
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image
from pycocotools import mask as coco

from solarspine.evaluation.e10_01_a11r1u_fold3_oof_cpu import (
    verify_sources,verify_record,
)
from solarspine.evaluation.e10_01_a11r1v_fold3_oof_supervised_audit import (
    DONE as AUDIT_DONE,
)
from solarspine.evaluation.e10_01_a11r1w_native_roi_coverage_gate import (
    SUCCESS as COVER_DONE,TILE,
)
from solarspine.evaluation.e10_01_a7_one_to_one_pq_audit import (
    decode_pred,exclusive_masks,
)
from solarspine.evaluation.baseline_oracle import (
    _load_coco,build_coco_tables,annotation_rles,
)
from solarspine.evaluation.e10_01_a4_fold4_native_pq import json_atomic
from solarspine.models.e10_01_native_roi_refiner import (
    source_roi_channels,
)
from solarspine.training.e9_01_train_runner import (
    sha256_file,ANNOTATIONS_SHA,
)

EXPERIMENT="E10-01-A11R1X-GROUP-SAFE-NATIVE-ROI-PATCH-MANIFEST-20261021"
DONE="E10_A11_R1X_ROI_PATCH_INDEX_COMPLETE__REFINER_NOT_TRAINED"
PARTIAL="E10_A11_R1X_PATCH_INDEX_PARTIAL_RESUMABLE__NO_TRAINING"
EXPECTED_IMAGES=232
EXPECTED_PAIRS=1275
EXPECTED_TRAIN_PAIRS=985
EXPECTED_VAL_PAIRS=290
EXPECTED_PATCHES=3825
MAX_NEIGHBOR_CONFLICT_FRACTION=.05
NATIVE=2048
SAMPLE_SMOKE=8


def need(ok,why):
    if not ok:
        raise RuntimeError(why)


def patch_counts(coarse,neighbors,target):
    """No target (GT) can become an input feature; diagnostic only."""
    c=np.asarray(coarse).astype(bool)
    n=np.asarray(neighbors).astype(bool)
    t=np.asarray(target).astype(bool)
    need(c.shape==n.shape==t.shape==(TILE,TILE),
         "ROI masks not native 256x256")
    need(not np.any(c & n),"A6 ownership contamination in ROI features")
    overlap=int(np.count_nonzero(t & n))
    return {
        "source_positive_px":int(np.count_nonzero(c)),
        "neighbor_positive_px":int(np.count_nonzero(n)),
        "supervised_target_positive_px":int(np.count_nonzero(t)),
        "supervised_target_neighbor_conflict_px":overlap,
        "supervised_intersection_with_source_px":int(np.count_nonzero(t & c)),
    }


def original_halpha_pixels(path):
    """No image resizing; preserve original mono uint8/uint16 amplitudes."""
    with Image.open(path) as pil:
        need(pil.size==(NATIVE,NATIVE),"Native H-alpha source changed")
        image=np.asarray(pil)
    if image.ndim==3 and image.shape[2] in (3,4):
        image=(image[...,:3].astype(np.float32)*
               np.array([.299,.587,.114],dtype=np.float32)).sum(axis=2)
    need(image.shape==(NATIVE,NATIVE) and np.isfinite(image).all(),
         "H-alpha source is neither finite grayscale nor RGB")
    return image


def validated_feature_sample(image,coarse,neighbors,target,box):
    """Returns source-only float32 x and Fold3 label y separately."""
    x0,y0,x1,y1=box
    need(0<=x0<x1<=NATIVE and 0<=y0<y1<=NATIVE
         and x1-x0==y1-y0==TILE,
         "Unapproved source-only tile geometry")
    crop=np.s_[y0:y1,x0:x1]
    x=source_roi_channels(image[crop],coarse[crop],neighbors[crop])
    y=target[crop].astype(np.float32)
    need(x.shape==(3,TILE,TILE) and x.dtype==np.float32
         and y.shape==(TILE,TILE) and y.dtype==np.float32,
         "Incorrect input/target tensor schema")
    need(np.isfinite(x).all() and np.isfinite(y).all()
         and not np.any((x[1]>.5)&(x[2]>.5)),
         "Invalid source-only input or neighboring overlap")
    return x,y


def unique_tile_id(image_id,pred_index,patch_index):
    import hashlib
    blob=f"{image_id}|p{pred_index}|w{patch_index}|A11_R1W".encode()
    return hashlib.sha256(blob).hexdigest()[:24]


def audit_image(oof,review,windows,annotations,shape,source,*,smoke_left):
    """All ROI boxes from frozen R1W file; GT only diagnostic and y."""
    need(shape==(NATIVE,NATIVE),"Native Fold3 COCO geometry changed")
    weak=review["weak_pairs"]
    tiles=windows["pairs"]
    need(len(weak)==len(tiles)==review["weak_pair_count"],
         "R1V/R1W training correspondence differs")
    raw=[decode_pred(r) for r in oof["pred_rles"]]
    exclusive,removed,deleted=exclusive_masks(raw)
    need(len(exclusive)==int(review["strict"]["pred_instances"]),
         "A6 mask owner count changed from R1V")
    occupied=np.zeros((NATIVE,NATIVE),dtype=bool)
    for rle in exclusive:
        occupied|=coco.decode(rle).astype(bool)
    gt_rles=annotation_rles(annotations,height=NATIVE,width=NATIVE)
    gt_ids=[str(a["id"]) for a in annotations]
    need(len(gt_rles)==review["strict"]["gt_instances"],
         "Fold3 GT instance count differs")
    rows=[]
    checked=0
    seen_pairs=set()
    image=None
    per_pred_cache={}
    for pair,entry in zip(weak,tiles,strict=True):
        i=int(pair["gt_index"])
        j=int(pair["pred_index"])
        need(0<=i<len(gt_rles) and 0<=j<len(exclusive),
             "Supervised ROI instance index outside source")
        need(entry["pred_index"]==j and
             entry["supervised_fold3_gt_index"]==i and
             entry["gt_annotation_id"]==gt_ids[i] and
             pair["gt_annotation_id"]==gt_ids[i] and
             entry["internal_split"]==review["internal_split"] and
             entry["source_tiles_xyxy"] and
             len(entry["source_tiles_xyxy"])<=3,
             "R1W coordinates not tied to approved R1V pair")
        need(j not in seen_pairs,
             "Candidate reused by multiple GT targets")
        seen_pairs.add(j)
        if j not in per_pred_cache:
            coarse=coco.decode(exclusive[j]).astype(bool)
            neighbors=occupied & ~coarse
            need(not np.any(coarse&neighbors),
                 "A6 first-owner neighboring exclusion violated")
            per_pred_cache[j]=(coarse,neighbors)
        coarse,neighbors=per_pred_cache[j]
        target=coco.decode(gt_rles[i]).astype(bool)
        for z,box in enumerate(entry["source_tiles_xyxy"]):
            x0,y0,x1,y1=(int(v) for v in box)
            need(x1-x0==y1-y0==TILE
                 and 0<=x0<x1<=NATIVE and 0<=y0<y1<=NATIVE,
                 "R1W tile is not native")
            crop=np.s_[y0:y1,x0:x1]
            metric=patch_counts(coarse[crop],neighbors[crop],target[crop])
            one={
                "tile_id":unique_tile_id(oof["image_id"],j,z),
                "image_id":oof["image_id"],
                "physical_image_id":oof["physical_image_id"],
                "temporal_group_id":oof["temporal_group_id"],
                "fold":3,
                "internal_split":review["internal_split"],
                "teacher_best_sha256":oof["teacher_best_sha256"],
                "pred_index":j,
                "gt_index":i,
                "gt_annotation_id":gt_ids[i],
                "pair_iou_diagnostic_only":float(pair["iou"]),
                "tile_index":z,
                "x0":x0,"y0":y0,"x1":x1,"y1":y1,
                **metric,
            }
            need(math.isfinite(one["pair_iou_diagnostic_only"])
                 and 0.2<one["pair_iou_diagnostic_only"]<=1.0,
                 "Only approved R1V weak pairs are allowed")
            rows.append(one)
            if checked<smoke_left:
                if image is None:
                    image=original_halpha_pixels(source)
                features,objective=validated_feature_sample(
                    image,coarse,neighbors,target,box)
                need(features.shape[0]==3 and objective.ndim==2,
                     "Source and GT are not separated")
                checked+=1
    need(len({r["tile_id"] for r in rows})==len(rows),
         "Duplicated ROI tile index")
    return rows,checked


def source_contract(args):
    plan,teacher,best,sources=verify_sources(args)
    audit=args.audit_dir/"summary.json"
    coverage=args.coverage_dir/"summary.json"
    oof=args.oof_dir/"summary.json"
    need(all(x.is_file() for x in (audit,coverage,oof,args.annotations)),
         "Missing completed immutable R1U/R1V/R1W source")
    v=json.loads(audit.read_text(encoding="utf-8"))
    w=json.loads(coverage.read_text(encoding="utf-8"))
    need(v.get("decision")==AUDIT_DONE
         and w.get("decision")==COVER_DONE
         and w.get("weak_pairs")==EXPECTED_PAIRS
         and w.get("train_pairs")==EXPECTED_TRAIN_PAIRS
         and w.get("val_pairs")==EXPECTED_VAL_PAIRS
         and w.get("source_only_tiles_total")==EXPECTED_PATCHES
         and w.get("candidate_geometry_cover_gate")==
             "THREE_TILES_SOURCE_ONLY_SUFFICIENT_FOR_BOUNDED_PILOT"
         and w.get("r1v_summary_sha256")==sha256_file(audit)
         and v.get("teacher_best_sha256")==teacher["best_checkpoint_sha256"]
         and sha256_file(args.annotations)==ANNOTATIONS_SHA,
         "Frozen complete R1V/R1W study or COCO source drift")
    return sources,teacher,v,w,audit,coverage,oof


def run(args):
    sources,teacher,v,w,audit,coverage,oof=source_contract(args)
    print("A11_R1X_SOURCE_CONTRACT_PASS "
          "PAIRS=1275 TILES=3825 NO_GPU",flush=True)
    if args.check_only:
        print("A11_R1X_PREFLIGHT_PASS__NO_PATCHES",flush=True)
        return
    shapes,by_ann=build_coco_tables(_load_coco(args.annotations))
    root=args.output_dir
    contract={
        "experiment":EXPERIMENT,
        "r1v_summary_sha256":sha256_file(audit),
        "r1w_summary_sha256":sha256_file(coverage),
        "r1u_summary_sha256":sha256_file(oof),
        "teacher_checkpoint_sha256":teacher["best_checkpoint_sha256"],
        "annotations_sha256":sha256_file(args.annotations),
        "source_code_sha256":sha256_file(Path(__file__)),
        "weak_pairs":EXPECTED_PAIRS,
        "expected_tiles":EXPECTED_PATCHES,
        "patch_side":TILE,
        "max_fraction_fold3_gt_foreground_in_forbidden_neighbors":
            MAX_NEIGHBOR_CONFLICT_FRACTION,
        "source_only_features":["native_Halpha","teacher_exclusive_mask",
                                "neighbor_union"],
        "supervision_only":["Fold3_gt_instance_mask"],
        "no_fold0_fold4_test_gt":True,"no_model_training":True,
    }
    if root.exists():
        path=root/"contract.json"
        need(path.is_file()
             and json.loads(path.read_text(encoding="utf-8"))==contract,
             "R1X frozen source/recipe changed; do not overwrite")
    else:
        root.mkdir(parents=True,exist_ok=False)
        json_atomic(root/"contract.json",contract)
    completed=[]
    begin=time.monotonic()
    smoke_remaining=SAMPLE_SMOKE
    for position,(e2,a1) in enumerate(sources,1):
        key=e2["image_id"]
        source=Path(a1["original_image_path"])
        file_oof=args.oof_dir/"records"/(key+".json")
        file_review=args.audit_dir/"records"/(key+".json")
        file_window=args.coverage_dir/"records"/(key+".json")
        dest=root/"records"/(key+".json")
        need(all(p.is_file() for p in (file_oof,file_review,file_window)),
             "Missing original per-image OOF/supervision/source-only tiles")
        original=verify_record(file_oof,e2,a1,teacher["best_checkpoint_sha256"])
        review=json.loads(file_review.read_text(encoding="utf-8"))
        windows=json.loads(file_window.read_text(encoding="utf-8"))
        need(review["image_id"]==windows["image_id"]==key
             and windows["r1v_record_sha256"]==sha256_file(file_review)
             and windows["r1u_record_sha256"]==sha256_file(file_oof)
             and review["oof_prediction_record_sha256"]==sha256_file(file_oof)
             and review["internal_split"]==windows["internal_split"],
             "R1X source records not provably the frozen OOF model")
        need(shapes.get(key)==(NATIVE,NATIVE),
             "Fold3 annotation geometry changed")
        if dest.is_file():
            record=json.loads(dest.read_text(encoding="utf-8"))
            need(record.get("experiment")==EXPERIMENT
                 and record.get("image_id")==key
                 and record.get("r1w_record_sha256")==sha256_file(file_window)
                 and record.get("r1u_record_sha256")==sha256_file(file_oof)
                 and record.get("r1v_record_sha256")==sha256_file(file_review),
                 "Existing patch source record lacks correct immutable origin")
        else:
            rows,checked=audit_image(
                original,review,windows,by_ann.get(key,[]),
                shapes[key],source,smoke_left=smoke_remaining)
            smoke_remaining-=checked
            record={
                "experiment":EXPERIMENT,"image_id":key,
                "physical_image_id":e2["physical_image_id"],
                "temporal_group_id":e2["temporal_group_id"],
                "fold":3,"internal_split":review["internal_split"],
                "r1u_record_sha256":sha256_file(file_oof),
                "r1v_record_sha256":sha256_file(file_review),
                "r1w_record_sha256":sha256_file(file_window),
                "teacher_best_sha256":teacher["best_checkpoint_sha256"],
                "tiles":rows,"no_trained_model":True,
            }
            json_atomic(dest,record)
        completed.append(record)
        if position%20==0 or position==EXPECTED_IMAGES:
            count=sum(len(x["tiles"]) for x in completed)
            print(f"A11_R1X_PATCH_INDEX {position}/{EXPECTED_IMAGES} "
                  f"TILES:{count} ELAPSED:{time.monotonic()-begin:.1f}",
                  flush=True)
        if position>=args.max_records and position<EXPECTED_IMAGES:
            print(PARTIAL,flush=True)
            return
    need(len(completed)==EXPECTED_IMAGES,
         "Cannot approve index with incomplete Fold3 sources")
    rows=[r for obj in completed for r in obj["tiles"]]
    need(len(rows)==EXPECTED_PATCHES and
         len({r["tile_id"] for r in rows})==EXPECTED_PATCHES,
         "ROI dataset did not index all preregistered 3825 tiles")
    count=Counter(r["internal_split"] for r in rows)
    need(set(count)=={"train","val"} and count["train"]>0
         and count["val"]>0,
         "Invalid or empty train/validation patch split")
    pair_counts=Counter()
    for item in completed:
        unique={(t["pred_index"],t["gt_annotation_id"]) for t in item["tiles"]}
        pair_counts[item["internal_split"]]+=len(unique)
    need(pair_counts==Counter({"train":EXPECTED_TRAIN_PAIRS,
                              "val":EXPECTED_VAL_PAIRS}),
         "Fold3 physical/temporal grouped pair count drift")
    for group in ("physical_image_id","temporal_group_id"):
        t={r[group] for r in rows if r["internal_split"]=="train"}
        vset={r[group] for r in rows if r["internal_split"]=="val"}
        need(t.isdisjoint(vset),
             "Physical/temporal leakage in source ROI patches")
    headers=list(rows[0].keys())
    csv_out=root/"patch_index.csv"
    if csv_out.exists():
        with csv_out.open(newline="",encoding="utf-8") as f:
            reader=csv.DictReader(f)
            need(reader.fieldnames==headers and
                 list(reader)==[{k:str(r[k]) for k in headers} for r in rows],
                 "Existing patch index drift: cannot overwrite")
    else:
        fd,tmp=tempfile.mkstemp(prefix="._a11r1x_",suffix=".csv",
                               dir=root)
        try:
            with os.fdopen(fd,"w",newline="",encoding="utf-8") as f:
                writer=csv.DictWriter(f,fieldnames=headers,
                    lineterminator="\n")
                writer.writeheader();writer.writerows(rows)
                f.flush();os.fsync(f.fileno())
            os.replace(tmp,csv_out)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
    gt_pos=sum(r["supervised_target_positive_px"] for r in rows)
    conflicts=sum(r["supervised_target_neighbor_conflict_px"] for r in rows)
    # Overlapping tiles may count the same native GT pixels multiple times;
    # this is a patch-sampled conflict metric, not unique-image GT area.
    fraction=conflicts/gt_pos if gt_pos else 1.0
    summary={
        "experiment":EXPERIMENT,"decision":DONE,
        "source_images":EXPECTED_IMAGES,"source_pairs":EXPECTED_PAIRS,
        "patches":len(rows),
        "train_tiles":count["train"],"val_tiles":count["val"],
        "patch_target_positive_pixels_with_overlap":gt_pos,
        "patch_gt_neighbor_conflict_pixels_with_overlap":conflicts,
        "patch_sampled_target_neighbor_conflict_fraction":fraction,
        "geometry_gate":(
            "R1X_NEIGHBOR_CONFLICT_ACCEPTABLE_FOR_FIRST_REFINER_SMOKE"
            if fraction<=MAX_NEIGHBOR_CONFLICT_FRACTION else
            "R1X_NEIGHBOR_CONFLICT_HIGH__REVISIT_HARD_MASK_BEFORE_TRAINING"),
        "indexed_patch_csv_sha256":sha256_file(csv_out),
        "source_r1w_summary_sha256":sha256_file(coverage),
        "source_code_sha256":sha256_file(Path(__file__)),
        "no_model_training":True,"fold0_fold4_test_gt_not_used":True,
        "input_channels":["Halpha","teacher_mask","neighbor_union"],
        "GT_only_supervised_y":True,
        "notes":"Counted target/neighbor conflicting pixels in native 256px tiles "
                "with tile overlaps; not a measured model PQ improvement.",
    }
    out=root/"summary.json"
    if out.exists():
        need(json.loads(out.read_text(encoding="utf-8"))==summary,
             "Existing index summary differs from recomputed sources")
    else:json_atomic(out,summary)
    print("A11_R1X_ROI_PATCH_INDEX_COMPLETE",flush=True)
    print("PATCHES:",len(rows),"TRAIN_TILES:",count["train"],
          "VAL_TILES:",count["val"],flush=True)
    print("NEIGHBOR_CONFLICT_FRACTION:",fraction,flush=True)
    print("GEOMETRY_GATE:",summary["geometry_gate"],flush=True)
    print("MODEL_TRAINING: False",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("plan-dir","teacher-dir","fold-manifest","a1-dir","images",
              "oof-dir","audit-dir","coverage-dir",
              "annotations","output-dir"):
        p.add_argument("--"+k,required=True,type=Path)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only",action="store_true")
    mode.add_argument("--run",action="store_true")
    p.add_argument("--max-records",type=int,default=EXPECTED_IMAGES)
    a=p.parse_args()
    need(1<=a.max_records<=EXPECTED_IMAGES,"Invalid bounded record count")
    run(a)


if __name__=="__main__":
    main()
