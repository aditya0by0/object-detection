import argparse
import json
import os
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from transformers import DetrForObjectDetection, DetrImageProcessor

from constants import COCO_DIR, IMAGE_DIR
from train import ID2LABEL

SCORE_THRESHOLD = 0.5
IOU_MATCH_THRESHOLD = 0.5


def run_inference(model, processor, coco: COCO, images_dir: Path, device):
    """Runs the model on every image in the COCO given set, returns COCO-format results."""
    results = []
    per_image_preds = {}

    for image_id in coco.imgs:
        info = coco.imgs[image_id]
        image = Image.open(images_dir / info["file_name"]).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[info["height"], info["width"]]])
        processed = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.0
        )[0]

        preds = []
        for score, label, box in zip(
            processed["scores"], processed["labels"], processed["boxes"]
        ):
            score = float(score)
            label = int(label) + 1  # HF model labels are 0-indexed internally
            xmin, ymin, xmax, ymax = [float(v) for v in box]
            w, h = xmax - xmin, ymax - ymin
            results.append(
                {
                    "image_id": image_id,
                    "category_id": label,
                    "bbox": [xmin, ymin, w, h],
                    "score": score,
                }
            )
            preds.append((score, label, [xmin, ymin, xmax, ymax]))

        per_image_preds[image_id] = preds

    return results, per_image_preds


def compute_coco_map(coco_gt: COCO, results: list, out_dir: Path):
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = {
        "mAP@[.5:.95]": coco_eval.stats[0],
        "mAP@0.5": coco_eval.stats[1],
        "mAP@0.75": coco_eval.stats[2],
    }
    (out_dir / "coco_map.json").write_text(json.dumps(metrics, indent=2))
    return coco_eval


def plot_precision_recall(coco_eval: COCOeval, out_dir: Path):
    # precision has shape [T, R, K, A, M] -> T=IoU thresholds, R=recall thresholds,
    # K=categories, A=area ranges, M=max dets. We take IoU=0.5 (index 0), all
    # areas, max dets=100, averaged across the 3 classes.
    precision = coco_eval.eval["precision"]
    recall_thresholds = coco_eval.params.recThrs

    iou_idx = 0  # IoU = 0.5
    area_idx = 0  # 'all'
    maxdet_idx = -1  # 100 dets

    plt.figure(figsize=(6, 5))
    for k, cls_name in ID2LABEL.items():
        cat_idx = k - 1
        pr = precision[iou_idx, :, cat_idx, area_idx, maxdet_idx]
        valid = pr > -1
        plt.plot(recall_thresholds[valid], pr[valid], label=cls_name)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (IoU=0.5)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "precision_recall_curve.png", dpi=150)
    plt.close()


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def build_confusion_matrix(coco_gt: COCO, per_image_preds: dict, out_dir: Path):
    """Greedy matches predictions (score > threshold) to ground truth by IoU
    to build a class-level confusion matrix (includes a 'background' row/col
    for missed detections / false positives)."""
    y_true, y_pred = [], []
    labels = list(ID2LABEL.keys())
    label_names = list(ID2LABEL.values()) + ["background"]

    for image_id in coco_gt.imgs:
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_id))
        gts = [
            (
                a["category_id"],
                [
                    a["bbox"][0],
                    a["bbox"][1],
                    a["bbox"][0] + a["bbox"][2],
                    a["bbox"][1] + a["bbox"][3],
                ],
            )
            for a in gt_anns
        ]
        preds = [p for p in per_image_preds[image_id] if p[0] >= SCORE_THRESHOLD]

        matched_gt = set()
        for score, pred_label, pred_box in preds:
            best_iou, best_j = 0.0, -1
            for j, (gt_label, gt_box) in enumerate(gts):
                if j in matched_gt:
                    continue
                cur_iou = iou(pred_box, gt_box)
                if cur_iou > best_iou:
                    best_iou, best_j = cur_iou, j
            if best_iou >= IOU_MATCH_THRESHOLD:
                matched_gt.add(best_j)
                y_true.append(gts[best_j][0])
                y_pred.append(pred_label)
            else:
                y_true.append(len(labels) + 1)  # background (false positive)
                y_pred.append(pred_label)

        for j, (gt_label, _) in enumerate(gts):
            if j not in matched_gt:
                y_true.append(gt_label)
                y_pred.append(len(labels) + 1)  # missed detection

    all_labels = labels + [len(labels) + 1]
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix (BCCD classes + background)")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()


def visualize_predictions(
    coco_gt: COCO, per_image_preds: dict, images_dir: Path, out_dir: Path, n=10
):
    image_ids = list(coco_gt.imgs.keys())[:n]
    colors = {1: "red", 2: "lime", 3: "yellow"}

    for image_id in image_ids:
        info = coco_gt.imgs[image_id]
        image = Image.open(images_dir / info["file_name"]).convert("RGB")

        fig, ax = plt.subplots(1)
        ax.imshow(image)
        for score, label, box in per_image_preds[image_id]:
            if score < SCORE_THRESHOLD:
                continue
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor=colors.get(label, "white"),
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                x1,
                max(0, y1 - 5),
                f"{ID2LABEL[label]} {score:.2f}",
                color=colors.get(label, "white"),
                fontsize=8,
            )
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"pred_{info['file_name']}", dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val-ann",
        type=Path,
        default=Path(os.path.join(COCO_DIR, "train.json")),
        help="Path to COCO-format validation annotations JSON",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = os.path.dirname(args.model_dir) / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = DetrImageProcessor.from_pretrained(args.model_dir)
    model = DetrForObjectDetection.from_pretrained(args.model_dir).to(device).eval()

    coco_gt = COCO(str(args.val_ann))
    results, per_image_preds = run_inference(
        model, processor, coco_gt, Path(IMAGE_DIR), device
    )

    coco_eval = compute_coco_map(coco_gt, results, output_dir)
    plot_precision_recall(coco_eval, output_dir)
    build_confusion_matrix(coco_gt, per_image_preds, output_dir)
    visualize_predictions(coco_gt, per_image_preds, Path(IMAGE_DIR), output_dir, n=10)

    print(f"All evaluation artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
