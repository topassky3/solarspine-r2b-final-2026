"""E10 A11-R2B: frozen-source native Fold4 strict PQ for A3+R2A U-Net.

READ ONLY existing A3/A4/A7 and R2A best model. No new YOLO inference,
no model training, no Kaggle TEST, no Fold0 images, no leaderboard submission.
For each saved Fold4 image use A4 native RLE proposals, A6 pixel first-owner,
predicted-mask-only 256x256 R1W tiling, learned R2A residual U-Net corrections,
uncovered pixels unchanged, and global first-owner conflict resolution.
GT Fold4 segmentation polygons are loaded strictly AFTER source-only 2048
refinement, and used only to compute the original/refined A7 1:1 Hungarian
IoU>0.5 strict scores. Fold4 is a development set, NOT a blind test.
All source thresholds/geometry are frozen; refuse post-hoc tuning.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import numpy as np
from pycocotools import mask as coco
import torch

from solarspine.evaluation.e10_01_a4_fold4_native_pq import (
    require_sources,fold4_gt,verify_existing,json_atomic,
    SUCCESS as A4_SUCCESS,compressed_rle
)
from solarspine.evaluation.e10_01_a7_one_to_one_pq_audit import (
    decode_pred,exclusive_masks,score,
    SUCCESS as A7_SUCCESS
)
from solarspine.evaluation.e10_01_a11r1w_native_roi_coverage_gate import (
    source_tile_boxes
)
from solarspine.models.e10_01_native_roi_refiner import (
    NativeROIRefiner,source_roi_channels
)
from solarspine.training.e10_01_a11r2a_t4_refiner_train import (
    COMPLETE as TRAIN_COMPLETE
)
from solarspine.training.e10_01_a11r1x_roi_patch_index import (
    original_halpha_pixels
)
from solarspine.training.e9_01_train_runner import sha256_file

EXPERIMENT="E10-01-A11R2B-FROZEN-FOLD4-NATIVE-A3-UNET-STRICT-1TO1-20261022"
DONE="E10_A11_R2B_FOLD4_NATIVE_REFINER_STRICT_PQ_COMPLETE"
PARTIAL="E10_A11_R2B_PARTIAL_RESUMABLE__NO_KAGGLE_SUBMISSION"
MODEL_SHA="e4b9c3bacce816cd1d4dfb863622b93774743a6a401ae844088e33540882f397"
EXPECTED=231
EXPECTED_GT=1638
BASE_TP=791
BASE_FP=997
BASE_FN=847
BASE_PQ=.2935407215569436
PROMOTE_MIN=BASE_PQ+.02
NATIVE=(2048,2048)
ROI=256
BATCH=8
THRESHOLD=.5
MAX_SECONDS=80*60


def need(ok,msg):
    if not ok:
        raise RuntimeError(msg)


def freeze_sources(args):
    source,a3_manifest,best_a3,sha_a3,rows,manifest=require_sources(args)
    need(len(rows)==EXPECTED and
         source["best_checkpoint_sha256"]==sha_a3,
         "Frozen A3 Fold4 source identity changed")
    a4_summary=args.a4_dir/"summary.json"
    a7_summary=args.a7_dir/"summary.json"
    model_status=args.r2a_dir/"status.json"
    model_best=args.r2a_dir/"best.pt"
    for p in (a4_summary,a7_summary,model_status,model_best):
        need(p.is_file(),f"Missing immutable required evidence {p}")
    a4=json.loads(a4_summary.read_text(encoding="utf-8"))
    a7=json.loads(a7_summary.read_text(encoding="utf-8"))
    r2a=json.loads(model_status.read_text(encoding="utf-8"))
    need(a4.get("decision")==A4_SUCCESS and
         a4.get("best_checkpoint_sha256")==sha_a3 and
         a4.get("fold4",{}).get("records")==EXPECTED and
         a7.get("decision")==A7_SUCCESS and
         a7.get("source_a4_sha256")==sha256_file(a4_summary),
         "Frozen A4/A7 reports not valid or lineage broken")
    ref=a7["one_to_one_after_A6_exclusivity_on_Fold4"]
    need(ref["records"]==EXPECTED and
         (ref["tp"],ref["fp"],ref["fn"],ref["gt_instances"])==
             (BASE_TP,BASE_FP,BASE_FN,EXPECTED_GT) and
         abs(ref["micro_pq"]-BASE_PQ)<1e-10,
         "Prior A7 exact one-to-one A6 baseline changed")
    need(r2a.get("decision")==TRAIN_COMPLETE and
         int(r2a.get("completed_epochs",0))==8 and
         int(r2a.get("best_epoch",0))==8 and
         r2a.get("best_checkpoint_sha256")==MODEL_SHA and
         sha256_file(model_best)==MODEL_SHA and
         r2a.get("no_Fold0_Fold4_Kaggle_TEST") is True,
         "R2A completed refiner source, epoch or checkpoint changed")
    contract={
        "experiment":EXPERIMENT,
        "a3_best_sha256":sha_a3,
        "a4_summary_sha256":sha256_file(a4_summary),
        "a7_summary_sha256":sha256_file(a7_summary),
        "a11_r2a_status_sha256":sha256_file(model_status),
        "a11_r2a_best_sha256":MODEL_SHA,
        "runner_source_sha256":sha256_file(Path(__file__)),
        "fold4_image_ids":[r["image_id"] for r in rows],
        "native_hw":list(NATIVE),"patch_hw":[ROI,ROI],
        "source_only_three_channels":["Halpha","A6_exclusive_coarse",
                                      "original_other_instances_union"],
        "tile_placement":"R1W original predicted foreground .15,.50,.85",
        "tile_forward_batch":BATCH,
        "threshold_refined_probability":THRESHOLD,
        "combine_overlapping_roi":"mean source-only U-Net probabilities",
        "pixels_uncovered_by_any_tile":"KEEP_ORIGINAL",
        "hard_exclusion":"original-other-instance occupied source pixels",
        "pred_empty_after_refinement":"KEEP_ORIGINAL",
        "output_ownership":"first A6 prediction owns, no overlapping RLE",
        "no_GT_based_inference_selection":True,
        "score":"A7 exact one-to-one max cardinality then max sum IoU",
        "strict_iou_cutoff":.5,
        "minimum_accepted_local_pq":PROMOTE_MIN,
        "no_train_no_fold0_no_kaggle_test":True,
        "development_fold4_not_blind":True,
    }
    return rows,manifest,sha_a3,contract


def merge_source_only(image,source,model,device):
    """Refine complete A6-exclusive source masks; GT is inaccessible.

    The R1W 3 x 256 windows cover part of each source candidate. Only votes
    within those selected GT-blind windows are changed. Uncovered pixels
    retain the original candidate segmentation. Hard neighbor pixels remain
    forbidden. Sequential ownership resolves newly contested background.
    """
    need(image.shape==NATIVE and len(source)<=300,
         "Source-only Fold4 native frame geometry changed")
    previous=[coco.decode(r).astype(bool) for r in source]
    occupied=np.zeros(NATIVE,dtype=bool)
    for mask in previous:
        need(bool(mask.any()),"Existing A6 exclusive source instance empty")
        need(not np.any(occupied&mask),"Original A6 ownership not exclusive")
        occupied|=mask
    refined=[]
    output_claims=np.zeros(NATIVE,dtype=bool)
    roi_count=changed=empty_fallback=dropped=0
    with torch.inference_mode():
        for coarse in previous:
            neighbors=occupied&~coarse
            boxes=source_tile_boxes(coarse)
            patches=[]
            for x0,y0,x1,y1 in boxes:
                patch=np.s_[y0:y1,x0:x1]
                patches.append(source_roi_channels(
                    image[patch],coarse[patch],neighbors[patch]))
            x=torch.from_numpy(np.stack(patches)).to(device)
            with torch.autocast("cuda",dtype=torch.float16,
                                enabled=device.type=="cuda"):
                logits=model(x)
            probs=logits.sigmoid().float().cpu().numpy()[:,0]
            vote=np.zeros(NATIVE,dtype=np.float32)
            hits=np.zeros(NATIVE,dtype=np.uint8)
            for n,(x0,y0,x1,y1) in enumerate(boxes):
                clip=np.s_[y0:y1,x0:x1]
                vote[clip]+=probs[n]
                hits[clip]+=1
            final=coarse.copy()
            covered=hits>0
            final[covered]=(vote[covered]/hits[covered])>THRESHOLD
            final &= ~neighbors
            if not final.any():
                final=coarse.copy()
                empty_fallback+=1
            changed+=int(np.count_nonzero(final!=coarse))
            # First-owner output rule: even competing new background pixels
            # can only be owned by the earlier original prediction.
            final &= ~output_claims
            if not final.any():
                dropped+=1
                continue
            output_claims|=final
            refined.append(coco.encode(np.asfortranarray(final.astype(np.uint8))))
            roi_count+=len(boxes)
    need(all(int(coco.area(r))>0 for r in refined),
         "Refiner emitted empty native instance after ownership")
    return refined,{
        "original_proposals":len(source),
        "refined_proposals":len(refined),
        "tiles_inferred":roi_count,
        "changed_pixels_before_final_ownership":changed,
        "empty_mask_source_fallbacks":empty_fallback,
        "source_instances_removed_by_new_overlap":dropped,
    }


def load_model(args):
    best=args.r2a_dir/"best.pt"
    need(sha256_file(best)==MODEL_SHA,
         "Refiner model checkpoint SHA drift")
    # Load only a trusted checkpoint created by this local pipeline.
    trained=torch.load(best,map_location="cpu",weights_only=False)
    need(trained.get("epoch")==8 and trained.get("best_epoch")==8
         and trained.get("recipe",{}).get("architecture")=="NativeROIRefiner"
         and trained.get("source_contract_sha256")==
             sha256_file(args.r2a_dir/"training_contract.json"),
         "Refiner checkpoint state/recipe invalid")
    device=torch.device("cuda:0" if args.device=="cuda" else "cpu")
    need(args.device!="cuda" or torch.cuda.is_available(),
         "Requested CUDA not available")
    model=NativeROIRefiner().to(device)
    model.load_state_dict(trained["model"],strict=True)
    model.eval()
    return model,device


def evaluation(gt,pred):
    return score(gt,pred)


def aggregate(items,key):
    tp=sum(x[key]["tp"] for x in items)
    fp=sum(x[key]["fp"] for x in items)
    fn=sum(x[key]["fn"] for x in items)
    ng=sum(x[key]["gt_instances"] for x in items)
    npred=sum(x[key]["pred_instances"] for x in items)
    si=sum(x[key]["sum_iou"] for x in items)
    need(tp+fn==ng and tp+fp==npred,
         "One-to-one original or refined matches not reconciled")
    den=tp+.5*(fp+fn)
    return {
        "records":len(items),"gt":ng,"pred":npred,
        "tp":tp,"fp":fp,"fn":fn,"sum_iou":si,
        "micro_pq":si/den if den else 0.0,
        "sq":si/tp if tp else 0.0,
        "rq":tp/den if den else 0.0,
    }


def run(args):
    rows,manifest,sha_a3,contract=freeze_sources(args)
    print("A11_R2B_FOLD4_SOURCE_PREFLIGHT_PASS 231_A4 8_EPOCH_T4_MODEL",
          flush=True)
    if args.check_only:
        print("A11_R2B_CHECK_ONLY_NO_GT_INFERENCE",flush=True)
        return
    out=args.output_dir
    contract_file=out/"contract.json"
    if out.exists():
        need(contract_file.is_file()
             and json.loads(contract_file.read_text(encoding="utf-8"))==contract,
             "Cannot reuse Fold4 score folder with a different evaluation contract")
    else:
        out.mkdir(parents=True,exist_ok=False)
        json_atomic(contract_file,contract)
    checkpoint=load_model(args)
    model,device=checkpoint
    torch.set_num_threads(min(4,os.cpu_count() or 2))
    started=time.monotonic()
    done=[]
    new_records=0
    for position,row in enumerate(rows,1):
        key=row["image_id"]
        source_path=args.a4_dir/"records"/(key+".json")
        dest=out/"records"/(key+".json")
        need(source_path.is_file(),"Frozen A4 image prediction absent")
        a4=json.loads(source_path.read_text(encoding="utf-8"))
        need(a4.get("image_id")==key and
             a4.get("checkpoint_sha256")==sha_a3 and
             a4.get("physical_image_id")==row["physical_image_id"]
             and a4.get("fold")==4 and
             len(a4.get("pred_rles",[]))==a4.get("pred_instances"),
             "Original A4 source record does not match trusted A3 Fold4")
        existing_source_sha=sha256_file(source_path)
        if dest.is_file():
            item=json.loads(dest.read_text(encoding="utf-8"))
            need(item.get("image_id")==key and
                 item.get("a4_record_sha256")==existing_source_sha and
                 item.get("r2a_checkpoint_sha256")==MODEL_SHA and
                 item.get("original",{}).get("gt_instances")==a4["gt_instances"]
                 and len(item.get("refined_rles",[]))==
                     item.get("refined",{}).get("pred_instances"),
                 "Existing Fold4 evaluated result lacks exact provenance")
        else:
            if time.monotonic()-started>MAX_SECONDS:
                print("A11_R2B_TIME_BUDGET_REACHED_PARTIAL",flush=True)
                break
            original,_removed,_deleted=exclusive_masks(
                [decode_pred(x) for x in a4["pred_rles"]])
            if original:
                image=original_halpha_pixels(
                    Path(manifest[key]["original_image_path"]))
                refined,stats=merge_source_only(
                    image,original,model,device)
            else:
                refined=[]
                stats={"original_proposals":0,"refined_proposals":0,
                       "tiles_inferred":0,
                       "changed_pixels_before_final_ownership":0,
                       "empty_mask_source_fallbacks":0,
                       "source_instances_removed_by_new_overlap":0}
            # GT decoded ONLY after the entire source-only prediction, and
            # is never passed to the U-Net or tile-location function.
            gt=fold4_gt_one(args,key,set(contract['fold4_image_ids']))
            baseline=evaluation(gt,original)
            trial=evaluation(gt,refined)
            need(baseline["gt_instances"]==len(gt)
                 and baseline["gt_instances"]==a4["gt_instances"]
                 and baseline["pred_instances"]==len(original)
                 and trial["pred_instances"]==len(refined),
                 "Fold4 canonical one-to-one source population mismatch")
            item={
                "experiment":EXPERIMENT,"image_id":key,
                "physical_image_id":row["physical_image_id"],
                "fold":4,
                "a4_record_sha256":existing_source_sha,
                "r2a_checkpoint_sha256":MODEL_SHA,
                "original":baseline,"refined":trial,
                "refined_rles":[{
                    "size":[2048,2048],
                    "counts":r["counts"].decode("ascii"),
                } for r in refined],
                "inference":stats,
                "no_GT_selected_mask_or_tile":True,
            }
            json_atomic(dest,item)
            new_records+=1
        done.append(item)
        if position%10==0 or position==EXPECTED:
            print(f"A11_R2B_FOLD4 {position}/{EXPECTED} "
                  f"OLD_TP:{sum(r['original']['tp'] for r in done)} "
                  f"NEW_TP:{sum(r['refined']['tp'] for r in done)} "
                  f"ELAPSED:{time.monotonic()-started:.1f}",flush=True)
        if new_records>=args.max_new and position<EXPECTED:
            print(PARTIAL,"NEW_RECORDS:",new_records,
                  "TOTAL_COMPLETE:",len(done),flush=True)
            return
    if len(done)<EXPECTED:
        print(PARTIAL,"COMPLETE:",len(done),"TOTAL:",EXPECTED,flush=True)
        return
    old=aggregate(done,"original")
    new=aggregate(done,"refined")
    need(old["gt"]==EXPECTED_GT and old["tp"]==BASE_TP
         and old["fp"]==BASE_FP and old["fn"]==BASE_FN
         and abs(old["micro_pq"]-BASE_PQ)<1e-10,
         "Full frozen A6 baseline did NOT replay exactly; STOP")
    delta=new["micro_pq"]-old["micro_pq"]
    lost=sum(
        max(0,i["original"]["tp"]-i["refined"]["tp"]) for i in done)
    gained=sum(
        max(0,i["refined"]["tp"]-i["original"]["tp"]) for i in done)
    verdict=("R2B_CANDIDATE_PROMOTION_THRESHOLD_REACHED__FINAL_REVIEW"
             if delta>=.02 and gained>=lost
             else "R2B_NO_PROMOTION__KEEP_ACCEPTED_A6")
    summary={
        "experiment":EXPERIMENT,"decision":DONE,
        "source_contract_sha256":sha256_file(contract_file),
        "source_a4_summary_sha256":contract["a4_summary_sha256"],
        "source_a7_summary_sha256":contract["a7_summary_sha256"],
        "source_model_sha256":MODEL_SHA,
        "original_A6_fold4_strict_1to1":old,
        "refined_A3_plus_R2A_fold4_strict_1to1":new,
        "delta_micro_pq":delta,
        "image_level_tp_recoveries":gained,
        "image_level_tp_regressions":lost,
        "total_gt_blind_tiles":sum(x["inference"]["tiles_inferred"] for x in done),
        "total_changed_pixels_before_owner":sum(
            x["inference"]["changed_pixels_before_final_ownership"] for x in done),
        "promotion_screen":verdict,
        "development_fold4_not_blind":True,
        "Kaggle_TEST_scored":False,
        "retrained_detector":False,
        "created_Kaggle_submission":False,
        "warning":"The same Fold4 previously served as A3 development validation "
                  "and A11 near-miss feasibility. This is not independent "
                  "blind generalization or the private leaderboard score.",
    }
    final=out/"summary.json"
    if final.is_file():
        need(json.loads(final.read_text(encoding="utf-8"))==summary,
             "Existing immutable A11R2B Fold4 report changed")
    else:
        json_atomic(final,summary)
    print("A11_R2B_FOLD4_COMPLETE",flush=True)
    print("OLD_STRICT_PQ:",old["micro_pq"],flush=True)
    print("NEW_STRICT_PQ:",new["micro_pq"],flush=True)
    print("DELTA_PQ:",delta,flush=True)
    print("ORIGINAL_TP_FP_FN:",old["tp"],old["fp"],old["fn"],flush=True)
    print("REFINED_TP_FP_FN:",new["tp"],new["fp"],new["fn"],flush=True)
    print("RECOVERED_IMAGE_LEVEL_TP:",gained,"LOST:",lost,flush=True)
    print("PROMOTION_SCREEN:",verdict,flush=True)
    print("ACCEPTED_A6_NOT_TOUCHED",flush=True)


def fold4_gt_one(args,image_id,fold4_ids):
    """Decode permitted Fold4 TRAIN GT only after predictions are fixed.

    The evaluator's official Fold4 annotations require same canonical COCO
    polygon decoder, but loading all 231 per-image GT each call would be
    wasteful. Keep the exact A4 segmentation rasterization and sort by ann id.
    """
    from solarspine.evaluation.baseline_oracle import (
        _load_coco,annotation_rles
    )
    need(len(fold4_ids)==EXPECTED and image_id in fold4_ids,
         "Only the frozen A3 Fold4 validation source has scoring GT")
    if not hasattr(fold4_gt_one,"_cache"):
        payload=_load_coco(args.annotations)
        by_img={}
        shapes={}
        for im in payload["images"]:
            key=str(im["id"])
            if key in fold4_ids:
                shapes[key]=(int(im["height"]),int(im["width"]))
        for ann in payload["annotations"]:
            key=str(ann["image_id"])
            if key in fold4_ids:
                by_img.setdefault(key,[]).append(ann)
        fold4_gt_one._cache=(shapes,by_img)
    shapes,by_img=fold4_gt_one._cache
    need(shapes.get(image_id)==NATIVE,
         "Fold4 TRAIN GT native dimensions mismatch")
    anns=sorted(by_img.get(image_id,[]),key=lambda a:str(a["id"]))
    need(anns,"Frozen Fold4 source has no GT polygons")
    return annotation_rles(anns,height=2048,width=2048)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for opt in ("folds","manifest","lock","annotations","images",
                "a1-dir","a3-dir","a4-dir","a7-dir","r2a-dir","output-dir"):
        p.add_argument("--"+opt,type=Path,required=True)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--max-new",type=int,default=EXPECTED)
    p.add_argument("--check-only",action="store_true")
    p.add_argument("--run",action="store_true")
    args=p.parse_args()
    need(args.check_only!=args.run,"Choose exactly one of --check-only/--run")
    need(1<=args.max_new<=EXPECTED,"Invalid max-new budget")
    run(args)


if __name__=="__main__":
    main()
