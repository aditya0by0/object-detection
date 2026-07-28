"""
Generate validation sample images with ground truth and predicted bounding boxes.

Usage:
    uv run python src/visualize_predictions.py \
        --model-dir .output/detr-resnet-50_100ep_8bs_1e-05lr \
        --image-names BloodImage_00000 BloodImage_00002 BloodImage_00014 \
        --output-dir results/val_samples
"""

import argparse
import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from transformers import DetrForObjectDetection, DetrImageProcessor

from constants import COCO_DIR, IMAGE_DIR
from train import CLASSES, ID2LABEL

SCORE_THRESHOLD = 0.5

# Colors for each class (R, G, B)
CLASS_COLORS = {
    "RBC": (0, 255, 0),  # Green
    "WBC": (255, 0, 0),  # Red
    "Platelets": (0, 0, 255),  # Blue
}


def draw_boxes(image, boxes, labels, scores=None, color=(0, 255, 0), line_width=2):
    """Draw bounding boxes on an image. boxes are in xyxy format."""
    draw = ImageDraw.Draw(image)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        label = labels[i] if i < len(labels) else ""
        label_color = CLASS_COLORS.get(label, color)

        draw.rectangle([x1, y1, x2, y2], outline=label_color, width=line_width)

        # Draw label background
        if scores is not None:
            text = f"{label} {scores[i]:.2f}"
        else:
            text = label

        # Get text size
        bbox = draw.textbbox((0, 0), text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.rectangle(
            [x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=label_color + (200,)
        )
        draw.text((x1 + 2, y1 - text_h - 2), text, fill=(255, 255, 255))


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ground truth and predicted bounding boxes on validation images."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--image-names",
        type=str,
        nargs="+",
        required=True,
        help="List of image names (without extension), e.g. BloodImage_00000 BloodImage_00002",
    )
    parser.add_argument(
        "--val-ann",
        type=Path,
        default=Path(os.path.join(COCO_DIR, "val.json")),
        help="Path to validation COCO annotation file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/val_samples"),
        help="Output directory for annotated images",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print("Loading model and processor...")
    processor = DetrImageProcessor.from_pretrained(args.model_dir)
    model = DetrForObjectDetection.from_pretrained(args.model_dir).to(device).eval()

    print(f"Loading annotations from: {args.val_ann}")
    coco = COCO(str(args.val_ann))

    # Build a mapping from file_name to image_id
    file_to_id = {info["file_name"]: img_id for img_id, info in coco.imgs.items()}

    for img_name in args.image_names:
        file_name = f"{img_name}.jpg"
        if file_name not in file_to_id:
            print(
                f"  WARNING: {file_name} not found in validation annotations, skipping"
            )
            continue

        image_id = file_to_id[file_name]
        image_path = Path(IMAGE_DIR) / file_name
        image = Image.open(image_path).convert("RGB")
        info = coco.imgs[image_id]

        # --- Ground Truth ---
        gt_image = image.copy()
        anns = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
        gt_boxes = []
        gt_labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            gt_boxes.append([x, y, x + w, y + h])
            gt_labels.append(ID2LABEL[ann["category_id"]])

        draw_boxes(gt_image, gt_boxes, gt_labels, color=(0, 255, 0), line_width=2)

        # --- Predictions ---
        pred_image = image.copy()
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[info["height"], info["width"]]], device=device)
        result = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=SCORE_THRESHOLD
        )[0]

        pred_boxes = []
        pred_labels = []
        pred_scores = []
        for score, label, box in zip(
            result["scores"], result["labels"], result["boxes"]
        ):
            pred_boxes.append(box.tolist())
            pred_labels.append(ID2LABEL[int(label)])
            pred_scores.append(float(score))

        draw_boxes(
            pred_image,
            pred_boxes,
            pred_labels,
            scores=pred_scores,
            color=(255, 0, 0),
            line_width=2,
        )

        # --- Side-by-side comparison ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].imshow(gt_image)
        axes[0].set_title("Ground Truth", fontsize=14, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(pred_image)
        axes[1].set_title("Predictions", fontsize=14, fontweight="bold")
        axes[1].axis("off")

        # Add a shared legend
        legend_patches = [
            mpatches.Patch(color=(r / 255, g / 255, b / 255), label=cls)
            for cls, (r, g, b) in CLASS_COLORS.items()
        ]
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=len(CLASSES),
            fontsize=12,
            frameon=True,
        )

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        save_path = output_dir / f"{img_name}_annotated.png"
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"  Saved: {save_path}")

    print("\nDone! Annotated images saved to:", output_dir)


if __name__ == "__main__":
    main()
