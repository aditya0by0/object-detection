import argparse
import json
import os
from pathlib import Path

import fiftyone as fo
import torch
from PIL import Image
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

        # TorchMetrics format
        boxes, scores, labels = [], [], []

        for score, label, box in zip(
            result["scores"], result["labels"], result["boxes"]
        ):
            boxes.append(box.cpu())
            scores.append(score.cpu())
            labels.append(label.cpu() + 1)

        predictions.append(
            {
                "boxes": torch.stack(boxes) if boxes else torch.empty((0, 4)),
                "scores": torch.stack(scores) if scores else torch.empty((0,)),
                "labels": torch.stack(labels)
                if labels
                else torch.empty((0,), dtype=torch.long),
            }
        )

        # Ground truth
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
                    label=ID2LABEL[int(label) + 1],
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
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    metric.update(predictions, targets)
    results = metric.compute()
    results = {
        k: float(v) for k, v in results.items() if torch.is_tensor(v) and v.numel() == 1
    }

    print("\nDetection metrics")
    print("-----------------")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2))


def launch_fiftyone(samples):
    dataset = fo.Dataset("detr-evaluation")
    dataset.add_samples(samples)
    session = fo.launch_app(dataset)
    session.wait()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--val-ann", type=Path, default=Path(os.path.join(COCO_DIR, "train.json"))
    )
    args = parser.parse_args()

    output_dir = args.model_dir / "eval"
    output_dir.mkdir(exist_ok=True, parents=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model...")
    processor = DetrImageProcessor.from_pretrained(args.model_dir)
    model = DetrForObjectDetection.from_pretrained(args.model_dir).to(device).eval()
    from pycocotools.coco import COCO

    coco = COCO(str(args.val_ann))
    predictions, targets, samples = run_inference(
        model, processor, coco, Path(IMAGE_DIR), device
    )
    evaluate(predictions, targets, output_dir)
    print("\nLaunching FiftyOne...")
    launch_fiftyone(samples)


if __name__ == "__main__":
    main()
