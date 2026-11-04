"""A11-R1W: preregistered source-only ROI tile coverage feasibility on Fold3.

NOT training or materializing learning data. Test a fixed three-tile strategy
defined ONLY from predicted native masks; Fold3 GT is used after tile selection
solely to MEASURE how much of each matched target would be visible for a future
refiner. Never consult GT, IoU, or outcomes to place tiles or change the
strategy. One immutable resumable image record. Fold0/4/Test GT excluded.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pycocotools import mask as coco

from solarspine.evaluation.baseline_oracle import (
    _load_coco,build_coco_tables,annotation_rles,
)
from solarspine.evaluation.e10_01_a11r1u_fold3_oof_cpu import (
    verify_sources,verify_record
)
from solarspine.evaluation.e10_01_a7_one_to_one_pq_audit import (
    decode_pred, exclusive_masks,
)
from solarspine.evaluation.e10_01_a4_fold4_native_pq import json_atomic
from solarspine.training.e9_01_train_runner import (
    sha256_file, ANNOTATIONS_SHA,
)

EXPERIMENT="E10-01-A11R1W-FOLD3-SOURCE-ONLY-ROI-TILE-COVERAGE-20261021"
SUCCESS="E10_A11_R1W_SOURCE_ONLY_TILING_FEASIBILITY_COMPLETE__NO_TRAINING"
PARTIAL="E10_A11_R1W_PARTIAL_RESUMABLE__NO_TRAINING"
PRIOR="E10_A11_R1V_FOLD3_OOF_SUPERVISED_AUDIT_COMPLETE__NO_TRAINING"
TILE=256
NATIVE=2048
ANCHOR_QUANTILES=(.15,.50,.85)
EXPECTED=232
EXPECTED_WEAK=1275
MIN_COVERAGE=.50
MIN_PASS_RATE=.70


def need(ok,why):
    if not ok:
        raise RuntimeError(why)


def source_tile_boxes(pred_mask):
    """Place at most 3 native 256x256 tiles on source PRED alone.

    Each anchor is the foreground pixel whose coordinate along the longest
    predicted bbox axis is nearest one preregistered .15/.50/.85 quantile.
    GT, loss, confidence, Fold3 pair IoU and annotation metadata are forbidden
    as inputs. Reuse exact boxes on inference; no GT-driven crop selection.
    """
    mask=np.asarray(pred_mask,dtype=np.bool_)
    need(mask.shape==(NATIVE,NATIVE) and bool(mask.any()),
         "Input mask is not positive native 2048 geometry")
    ys,xs=np.nonzero(mask)
    long_axis=ys if np.ptp(ys)>=np.ptp(xs) else xs
    out=[]
    for q in ANCHOR_QUANTILES:
        target=float(np.quantile(long_axis,q))
        # np.argmin is deterministic, selects lowest image-row-major index
        index=int(np.argmin(np.abs(long_axis-target)))
        y=int(np.clip(int(ys[index])-TILE//2,0,NATIVE-TILE))
        x=int(np.clip(int(xs[index])-TILE//2,0,NATIVE-TILE))
        box=[x,y,x+TILE,y+TILE]
        if box not in out:
            out.append(box)
    need(1<=len(out)<=3 and all(
        0<=x0<x1<=NATIVE and 0<=y0<y1<=NATIVE
        and (x1-x0)==TILE and (y1-y0)==TILE
        for x0,y0,x1,y1 in out),
        "Predicted source tiles escaped native image geometry")
    return out


def coverage(mask,boxes):
    """Exact union area fraction without double-counting overlapping tiles."""
    value=np.asarray(mask,dtype=np.bool_)
    need(value.shape==(NATIVE,NATIVE) and len(boxes)>0,
         "Invalid mask/tile geometry in coverage audit")
    ys,xs=np.nonzero(value)
    if not len(ys):
        return 0.0
    present=np.zeros(len(ys),dtype=bool)
    for x0,y0,x1,y1 in boxes:
        present |= (xs>=x0)&(xs<x1)&(ys>=y0)&(ys<y1)
    return float(np.mean(present))


def audit_one(oof,review,gt_rles,gt_ids):
    """Pred-only tiles, then TRAIN Fold3 GT-based coverage measurement."""
    raw=[decode_pred(r) for r in oof["pred_rles"]]
    pred,removed,deleted=exclusive_masks(raw)
    need(len(pred)==review["strict"]["pred_instances"],
         "R1V and A6 owner-exclusive prediction counts mismatch")
    need(len(gt_rles)==review["strict"]["gt_instances"]
         and len(gt_ids)==len(gt_rles),
         "R1V target instance population changed")
    records=[]
    for p in review["weak_pairs"]:
        j=int(p["pred_index"])
        i=int(p["gt_index"])
        need(0<=j<len(pred) and 0<=i<len(gt_rles)
             and p["gt_annotation_id"]==gt_ids[i],
             "R1V paired GT annotation mismatch")
        # CRITICAL: no GT is passed into source_tile_boxes().
        mask=coco.decode(pred[j]).astype(bool)
        boxes=source_tile_boxes(mask)
        pred_coverage=coverage(mask,boxes)
        gt_coverage=coverage(coco.decode(gt_rles[i]),boxes)
        # Coordinates are computed before consulting either GT or GT coverage.
        records.append({
            "pred_index":j,
            "gt_annotation_id":gt_ids[i],
            "supervised_fold3_gt_index":i,
            "source_tiles_xyxy":boxes,
            "source_pred_coverage":pred_coverage,
            "supervised_gt_coverage_diagnostic_only":gt_coverage,
            "source_oof_pred_gt_iou_diagnostic_only":p["iou"],
            "internal_split":review["internal_split"],
            "strict_tp":p["strict_tp"],
        })
    need(len(records)==review["weak_pair_count"],
         "Missing weak pair in source-only tiling audit")
    return records


def run(args):
    plan,teacher,best,source_rows=verify_sources(args)
    prior=args.audit_dir/"summary.json"
    oof_summary=args.oof_dir/"summary.json"
    for p in (prior,oof_summary,args.annotations):
        need(p.is_file(),f"Missing immutable source file: {p}")
    r1v=json.loads(prior.read_text(encoding="utf-8"))
    need(r1v.get("decision")==PRIOR and
         r1v.get("records")==EXPECTED and
         r1v.get("weak_pair_count")==EXPECTED_WEAK and
         r1v.get("grouped_train_weak_pairs")==985 and
         r1v.get("grouped_val_weak_pairs")==290 and
         r1v.get("teacher_best_sha256")==teacher["best_checkpoint_sha256"],
         "Real 1275 OOF weak-pair source not verified")
    need(sha256_file(args.annotations)==ANNOTATIONS_SHA,
         "Frozen TRAIN COCO SHA drifted")
    shapes,anns=build_coco_tables(_load_coco(args.annotations))
    source_ids={e2["image_id"] for e2,a1 in source_rows}
    need(len(source_ids)==EXPECTED,"Source must be exactly blind Fold3")
    report=args.output_dir
    contract={
        "experiment":EXPERIMENT,
        "r1v_summary_sha256":sha256_file(prior),
        "r1u_summary_sha256":sha256_file(oof_summary),
        "annotations_sha256":sha256_file(args.annotations),
        "teacher_sha256":teacher["best_checkpoint_sha256"],
        "source_code_sha256":sha256_file(Path(__file__)),
        "frozen_fold3_ids":sorted(source_ids),
        "patch_side":TILE,"native_side":NATIVE,
        "max_source_only_tiles_per_weak_pair":len(ANCHOR_QUANTILES),
        "anchor_quantiles":list(ANCHOR_QUANTILES),
        "min_desirable_gt_coverage":MIN_COVERAGE,
        "min_fraction_of_pairs_reaching_desired_coverage":MIN_PASS_RATE,
        "strictly_source_only_tile_coordinates":True,
        "gt_used_only_for_fold3_training_coverage_diagnostic":True,
        "fold0_fold4_and_test_gt_never_decoded":True,
        "no_training_no_new_detector_inference":True,
    }
    if report.exists():
        p=report/"contract.json"
        need(p.exists() and json.loads(p.read_text(encoding="utf-8"))==contract,
             "Cannot change immutable R1W tile selection after start")
    else:
        report.mkdir(parents=True,exist_ok=False)
        json_atomic(report/"contract.json",contract)
    print("A11_R1W_SOURCE_CONTRACT_PASS "
          "RECORDS=232 PAIRS=1275 PATCH=256 ANCHORS=3",flush=True)
    if args.check_only:
        print("A11_R1W_PREFLIGHT_PASS__NO_TILING",flush=True)
        return

    completed=[]
    start=time.monotonic()
    for position,(e2,a1) in enumerate(source_rows,1):
        key=e2["image_id"]
        oof_file=args.oof_dir/"records"/(key+".json")
        review_file=args.audit_dir/"records"/(key+".json")
        dest=report/"records"/(key+".json")
        need(oof_file.is_file() and review_file.is_file(),
             "Missing frozen A11R1U/V per-image evidence")
        oof=verify_record(oof_file,e2,a1,
                          teacher["best_checkpoint_sha256"])
        review=json.loads(review_file.read_text(encoding="utf-8"))
        need(review.get("image_id")==key and
             review.get("teacher_best_sha256")==teacher["best_checkpoint_sha256"]
             and review.get("oof_prediction_record_sha256")==sha256_file(oof_file)
             and review.get("internal_split") in ("train","val"),
             "R1V source/origin mismatch")
        if dest.is_file():
            obj=json.loads(dest.read_text(encoding="utf-8"))
            need(obj.get("experiment")==EXPERIMENT and
                 obj.get("image_id")==key and
                 obj.get("r1v_record_sha256")==sha256_file(review_file)
                 and obj.get("r1u_record_sha256")==sha256_file(oof_file)
                 and len(obj.get("pairs",[]))==review["weak_pair_count"],
                 "Previously saved source tile record cannot be trusted")
        else:
            need(shapes.get(key)==(NATIVE,NATIVE),
                 "Fold3 source annotation geometry invalid")
            ground=anns.get(key,[])
            gt_ids=[str(a["id"]) for a in ground]
            gt_rles=annotation_rles(ground,height=NATIVE,width=NATIVE)
            pairs=audit_one(oof,review,gt_rles,gt_ids)
            obj={
                "experiment":EXPERIMENT,
                "image_id":key,
                "physical_image_id":e2["physical_image_id"],
                "temporal_group_id":e2["temporal_group_id"],
                "fold":3,
                "internal_split":review["internal_split"],
                "r1v_record_sha256":sha256_file(review_file),
                "r1u_record_sha256":sha256_file(oof_file),
                "pairs":pairs,
                "no_training":True,
                "all_boxes_source_pred_only":True,
            }
            json_atomic(dest,obj)
        completed.append(obj)
        if position%20==0 or position==EXPECTED:
            print(f"A11_R1W_TILE_COVERAGE {position}/{EXPECTED} "
                  f"PAIRS:{sum(len(x['pairs']) for x in completed)} "
                  f"ELAPSED:{time.monotonic()-start:.1f}",flush=True)
        if position>=args.max_records and position<EXPECTED:
            print(PARTIAL,flush=True)
            return
    need(len(completed)==EXPECTED and
         sum(len(x["pairs"]) for x in completed)==EXPECTED_WEAK,
         "Source ROI tiling lost OOF weak pairs")
    all_pairs=[p for item in completed for p in item["pairs"]]
    train=[p for p in all_pairs if p["internal_split"]=="train"]
    val=[p for p in all_pairs if p["internal_split"]=="val"]
    need((len(train),len(val))==(985,290),
         "Physical/temporal group validation split drift")
    coverage_all=[p["supervised_gt_coverage_diagnostic_only"]
                  for p in all_pairs]
    good=sum(p>=MIN_COVERAGE for p in coverage_all)
    near=[p for p in all_pairs if
          .35<=p["source_oof_pred_gt_iou_diagnostic_only"]<=.50]
    summary={
        "experiment":EXPERIMENT,
        "decision":SUCCESS,
        "source_records":EXPECTED,
        "weak_pairs":EXPECTED_WEAK,
        "train_pairs":len(train),
        "val_pairs":len(val),
        "max_tiles_per_pair":3,
        "source_only_tiles_total":sum(len(p["source_tiles_xyxy"])
                                     for p in all_pairs),
        "gt_coverage_mean":float(np.mean(coverage_all)),
        "gt_coverage_median":float(np.median(coverage_all)),
        "gt_coverage_at_least_050":good,
        "gt_coverage_rate_at_least_050":good/len(all_pairs),
        "near_miss_035_to_050_supervised_pairs":len(near),
        "near_miss_gt_coverage_mean":(
            float(np.mean([
                p["supervised_gt_coverage_diagnostic_only"] for p in near]))
            if near else None),
        "candidate_geometry_cover_gate":(
            "THREE_TILES_SOURCE_ONLY_SUFFICIENT_FOR_BOUNDED_PILOT"
            if good/len(all_pairs)>=MIN_PASS_RATE else
            "THREE_TILES_LOW_COVERAGE__REPLAN_GEOMETRY_BEFORE_TRAINING"
        ),
        "no_training_performed":True,
        "no_gt_guided_tiles":True,
        "fold0_fold4_test_gt_not_used":True,
        "r1v_summary_sha256":sha256_file(prior),
        "r1u_summary_sha256":sha256_file(oof_summary),
        "source_code_sha256":sha256_file(Path(__file__)),
    }
    result=report/"summary.json"
    if result.exists():
        need(json.loads(result.read_text(encoding="utf-8"))==summary,
             "Existing immutable R1W summary contradicts completed audit")
    else:
        json_atomic(result,summary)
    print("A11_R1W_COMPLETE",flush=True)
    print("TRAIN_PAIRS:",len(train),"VAL_PAIRS:",len(val),flush=True)
    print("GT_COVERAGE_MEAN:",summary["gt_coverage_mean"],flush=True)
    print("GT_COVERAGE_MEDIAN:",summary["gt_coverage_median"],flush=True)
    print("GT_COVERAGE_GE_050:",good,flush=True)
    print("NEAR_MISS_SOURCE_PAIRS:",len(near),flush=True)
    print("COVERAGE_GATE:",summary["candidate_geometry_cover_gate"],flush=True)
    print("TRAINING_PERFORMED: False",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("plan-dir","teacher-dir","fold-manifest","a1-dir",
                 "images","oof-dir","audit-dir","annotations","output-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only",action="store_true")
    mode.add_argument("--run",action="store_true")
    parser.add_argument("--max-records",type=int,default=232)
    args=parser.parse_args()
    need(1<=args.max_records<=EXPECTED,"Invalid source record limit")
    run(args)


if __name__=="__main__":
    main()
