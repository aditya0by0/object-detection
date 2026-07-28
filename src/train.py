# srun --partition=gpu --constraint="A100|H100.80gb" --ntasks=1 --cpus-per-task=8 --threads-per-core=1 --mem=64G --time=02:00:00 --gres=gpu:1 --pty bash
import argparse
import os
from functools import partial

from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Trainer,
    TrainingArguments,
)

from constants import COCO_DIR, IMAGE_DIR
from dataset import BCCDDataset

CLASSES = ["RBC", "WBC", "Platelets"]
ID2LABEL = {
    0: "RBC",
    1: "WBC",
    2: "Platelets",
}

LABEL2ID = {
    "RBC": 0,
    "WBC": 1,
    "Platelets": 2,
}

# References:
# https://huggingface.co/docs/transformers/tasks/object_detection
# https://huggingface.co/docs/transformers/v5.14.0/en/model_doc/detr?usage=Pipeline#detr
# https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb


def train(
    model_name: str,
    image_dir: str,
    train_coco_fp: str,
    val_coco_fp: str,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-5,
):
    model_id = model_name.split("/")[-1]

    output_dir = os.path.join(
        ".output",
        f"{model_id}_{epochs}ep_{batch_size}bs_{lr}lr",
    )

    print(f"Loading {model_name}")

    image_processor = AutoImageProcessor.from_pretrained(model_name)

    train_ds = BCCDDataset(image_dir, train_coco_fp, image_processor)
    val_ds = BCCDDataset(image_dir, val_coco_fp, image_processor)

    model = AutoModelForObjectDetection.from_pretrained(
        model_name,
        num_labels=len(CLASSES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=1e-4,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        logging_strategy="epoch",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        bf16=True,
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=partial(collate_fn, image_processor=image_processor),
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
    trainer.save_model(output_dir)
    image_processor.save_pretrained(output_dir)
    print(f"Model + processor saved to {output_dir}")


def collate_fn(batch, image_processor):
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]

    encoding = image_processor.pad(pixel_values, return_tensors="pt")

    return {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DETR model and visualize results."
    )
    # PekingU/rtdetr_r50vd, microsoft/conditional-detr-resnet-50, PekingU/rtdetr_r18vd
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Object detection model name (e.g., facebook/detr-resnet-50)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training and evaluation (default: 8)",
    )

    args = parser.parse_args()

    train(
        model_name=args.model_name,
        image_dir=IMAGE_DIR,
        train_coco_fp=os.path.join(COCO_DIR, "train.json"),
        val_coco_fp=os.path.join(COCO_DIR, "val.json"),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
