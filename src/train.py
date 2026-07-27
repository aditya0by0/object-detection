# srun --partition=gpu --constraint="A100|H100.80gb" --ntasks=1 --cpus-per-task=8 --threads-per-core=1 --mem=64G --time=02:00:00 --gres=gpu:1 --pty bash
import os
from functools import partial

import torch
import torch.nn.functional as F
from transformers import (
    DetrForObjectDetection,
    DetrImageProcessor,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from dataset import BCCDDataset

CLASSES = ["RBC", "WBC", "Platelets"]
ID2LABEL = {i + 1: name for i, name in enumerate(CLASSES)}
LABEL2ID = {name: i + 1 for i, name in enumerate(CLASSES)}


# References:
# https://huggingface.co/docs/transformers/tasks/object_detection
# https://huggingface.co/docs/transformers/v5.14.0/en/model_doc/detr?usage=Pipeline#detr
# https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb


def main(image_dir, train_coco_fp, val_coco_fp, epochs=50, batch_size=4, lr=1e-5):
    output_dir = os.path.join(".output", f"detr_{epochs}ep_{batch_size}bs_{lr}lr")
    image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    train_ds = BCCDDataset(image_dir, train_coco_fp, image_processor)
    val_ds = BCCDDataset(image_dir, val_coco_fp, image_processor)

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
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
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_strategy="epoch",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        fp16=True,
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=partial(collate_fn),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
    trainer.save_model(output_dir)
    image_processor.save_pretrained(output_dir)
    print(f"Model + processor saved to {output_dir}")


def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]

    max_h = max(pv.shape[-2] for pv in pixel_values)
    max_w = max(pv.shape[-1] for pv in pixel_values)

    padded_pixel_values, pixel_masks = [], []
    for pv in pixel_values:
        _, h, w = pv.shape
        # pad only bottom/right, matching DETR's own padding convention
        padded = F.pad(pv, (0, max_w - w, 0, max_h - h), value=0.0)
        mask = torch.zeros((max_h, max_w), dtype=torch.long)
        mask[:h, :w] = 1
        padded_pixel_values.append(padded)
        pixel_masks.append(mask)

    return {
        "pixel_values": torch.stack(padded_pixel_values),
        "pixel_mask": torch.stack(pixel_masks),
        "labels": labels,
    }


if __name__ == "__main__":
    import os

    from constants import COCO_DIR, IMAGE_DIR

    main(
        IMAGE_DIR,
        os.path.join(COCO_DIR, "train.json"),
        os.path.join(COCO_DIR, "val.json"),
    )
