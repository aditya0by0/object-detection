import os

from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class BCCDDataset(Dataset):
    """Reads a COCO-style json + image folder and returns HF-DETR-ready samples."""

    def __init__(self, images_dir: str, ann_file: str, image_processor):
        self.images_dir = images_dir
        self.coco = COCO(ann_file)
        self.image_ids = sorted(self.coco.imgs.keys())
        self.image_processor = image_processor

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.coco.imgs[image_id]
        image_path = os.path.join(self.images_dir, img_info["file_name"])
        image = Image.open(image_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        target = {"image_id": image_id, "annotations": anns}

        encoding = self.image_processor(
            images=image, annotations=target, return_tensors="pt"
        )
        pixel_values = encoding["pixel_values"][0]
        labels = encoding["labels"][0]
        return {"pixel_values": pixel_values, "labels": labels}


if __name__ == "__main__":
    from transformers import DetrImageProcessor

    from constants import COCO_DIR, IMAGE_DIR

    image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    dataset = BCCDDataset(
        IMAGE_DIR, os.path.join(COCO_DIR, "train.json"), image_processor
    )
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Pixel values shape: {sample['pixel_values'].shape}")
    print(f"Labels: {sample['labels']}")
