import argparse
import json
import os
from pathlib import Path

import fiftyone as fo
import torch
from PIL import Image
from pycocotools.coco import COCO
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import DetrForObjectDetection, DetrImageProcessor

from constants import COCO_DIR, IMAGE_DIR
from train import ID2LABEL

SCORE_THRESHOLD = 0.5


def run_inference(model, processor, coco, images_dir, device):
    """
    Runs DETR inference and returns TorchMetrics-compatible predictions/targets
    plus FiftyOne samples.
    """
    predictions = []
    targets = []
    fiftyone_samples = []

    for image_id in coco.imgs:
        info = coco.imgs[image_id]

        image_path = images_dir / info["file_name"]
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[info["height"], info["width"]]], device=device)
        result = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=SCORE_THRESHOLD
        )[0]

        # TorchMetrics predictions (Shift +1 to match 1-indexed ground-truth category_ids)
        boxes, scores, labels = [], [], []

        for score, label, box in zip(
            result["scores"], result["labels"], result["boxes"]
        ):
            boxes.append(box.cpu())
            scores.append(score.cpu())
            labels.append(label.cpu())

        predictions.append(
            {
                "boxes": torch.stack(boxes) if boxes else torch.empty((0, 4)),
                "scores": torch.stack(scores) if scores else torch.empty((0,)),
                "labels": torch.stack(labels)
                if labels
                else torch.empty((0,), dtype=torch.long),
            }
        )

        # Ground truth annotations
        gt_boxes = []
        gt_labels = []
        anns = coco.loadAnns(coco.getAnnIds(imgIds=image_id))

        for ann in anns:
            x, y, w, h = ann["bbox"]
            gt_boxes.append([x, y, x + w, y + h])
            gt_labels.append(ann["category_id"])

        targets.append(
            {
                "boxes": torch.tensor(gt_boxes, dtype=torch.float32),
                "labels": torch.tensor(gt_labels, dtype=torch.long),
            }
        )

        # FiftyOne visualization
        detections = []
        for score, label, box in zip(
            result["scores"], result["labels"], result["boxes"]
        ):
            x1, y1, x2, y2 = box.tolist()
            detections.append(
                fo.Detection(
                    label=ID2LABEL[int(label)],
                    bounding_box=[
                        x1 / info["width"],
                        y1 / info["height"],
                        (x2 - x1) / info["width"],
                        (y2 - y1) / info["height"],
                    ],
                    confidence=float(score),
                )
            )

        sample = fo.Sample(filepath=str(image_path))
        sample["predictions"] = fo.Detections(detections=detections)
        fiftyone_samples.append(sample)

    return predictions, targets, fiftyone_samples


def evaluate(predictions, targets, output_dir: Path):
    """Calculates overall and per-class Mean Average Precision metrics using TorchMetrics."""
    # class_metrics=True enables per-class mAP and mAR breakdown
    metric = MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox", class_metrics=True
    )
    metric.update(predictions, targets)
    raw_results = metric.compute()

    results = {}

    # Standard scalar metrics
    metric_keys = [
        ("map", "Map"),
        ("map_50", "Map 50"),
        ("map_75", "Map 75"),
        ("map_small", "Map Small"),
        ("map_medium", "Map Medium"),
        ("map_large", "Map Large"),
        ("mar_1", "Mar 1"),
        ("mar_10", "Mar 10"),
        ("mar_100", "Mar 100"),
        ("mar_small", "Mar Small"),
        ("mar_medium", "Mar Medium"),
        ("mar_large", "Mar Large"),
    ]

    for raw_key, print_label in metric_keys:
        val = raw_results.get(raw_key)
        if val is not None and torch.is_tensor(val) and val.numel() == 1:
            results[print_label] = float(val)

    # Per-class metrics
    # torchmetrics returns a tensor of shape (num_classes,) or a list of class IDs in 'classes'
    classes_tensor = raw_results.get("classes")
    map_per_class = raw_results.get("map_per_class")
    mar_100_per_class = raw_results.get("mar_100_per_class")

    if classes_tensor is not None and map_per_class is not None:
        class_ids = classes_tensor.tolist()
        map_list = map_per_class.tolist()
        mar_list = mar_100_per_class.tolist() if mar_100_per_class is not None else []

        for cid, map_val in zip(class_ids, map_list):
            class_name = ID2LABEL.get(int(cid), f"Class_{cid}")
            formatted_name = class_name.capitalize()
            results[f"Map {formatted_name}"] = float(map_val)

            if mar_list:
                mar_val = mar_list[class_ids.index(cid)]
                results[f"Mar 100 {formatted_name}"] = float(mar_val)

    print("\nDetection metrics")
    print("-----------------")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2))


def launch_fiftyone(samples):
    """Launches the FiftyOne visualization app."""
    dataset = fo.Dataset("detr-evaluation", overwrite=True)
    dataset.add_samples(samples)
    session = fo.launch_app(dataset)
    session.wait()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DETR model and visualize results."
    )
    parser.add_argument(
        "--model-dir", type=Path, required=True, help="Path to trained model directory"
    )
    parser.add_argument(
        "--val-ann",
        type=Path,
        default=Path(os.path.join(COCO_DIR, "val.json")),
        help="Path to validation COCO annotation file",
    )
    args = parser.parse_args()

    output_dir = args.model_dir / "eval"
    output_dir.mkdir(exist_ok=True, parents=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print("Loading model and processor...")
    processor = DetrImageProcessor.from_pretrained(args.model_dir)
    model = DetrForObjectDetection.from_pretrained(args.model_dir).to(device).eval()

    print(f"Loading annotations from: {args.val_ann}")
    coco = COCO(str(args.val_ann))

    print("Running inference...")
    predictions, targets, samples = run_inference(
        model, processor, coco, Path(IMAGE_DIR), device
    )

    print("Computing metrics...")
    evaluate(predictions, targets, output_dir)

    print("\nLaunching FiftyOne App...")
    launch_fiftyone(samples)


if __name__ == "__main__":
    main()
