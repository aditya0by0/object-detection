import json
import os
import xml.etree.ElementTree as ET

from tqdm import tqdm

from constants import ANNOTATION_DIR, DATASET_DIR, IMAGESETS_DIR
from train import CLASSES, LABEL2ID


def convert(split):
    images, annotations = [], []
    annotation_id = 1
    txt_file = os.path.join(IMAGESETS_DIR, split + ".txt")

    with open(txt_file) as f:
        image_names = [x.strip() for x in f.readlines()]

    for name in tqdm(image_names):
        xml_path = os.path.join(ANNOTATION_DIR, name + ".xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        filename = name + ".jpg"

        # image size
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        image_id = int(name.split("_")[-1])  # Extract image ID from filename
        images.append(
            {"id": image_id, "file_name": filename, "width": width, "height": height}
        )

        for obj in root.findall("object"):
            category = obj.find("name").text
            if category not in CLASSES:
                raise ValueError(f"Unknown category: {category} for image {filename}")

            bbox = obj.find("bndbox")
            # https://github.com/soumenpramanik/Convert-Pascal-VOC-to-COCO/blob/master/convertVOC2COCO.py#L78-L79
            # Subtract 1 to shift from VOC's 1-indexed to COCO's 0-indexed format
            xmin = int(bbox.find("xmin").text) - 1
            ymin = int(bbox.find("ymin").text) - 1
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            w = xmax - xmin
            h = ymax - ymin
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": LABEL2ID[category],
                    "bbox": [xmin, ymin, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "RBC"},
            {"id": 2, "name": "WBC"},
            {"id": 3, "name": "Platelets"},
        ],
    }

    output = f"{split}.json"
    path = os.path.join(DATASET_DIR, "coco", output)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(coco, f, indent=4)
    print(f"{split}.json created")


if __name__ == "__main__":
    convert("train")
    convert("trainval")
    convert("val")
    convert("test")
