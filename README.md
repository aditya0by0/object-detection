# Object Detection — Blood Cell Detection (BCCD)

Fine-tune [DETR](https://huggingface.co/docs/transformers/en/model_doc/detr) (DEtection TRansformer) and other Hugging Face object detection models on the **BCCD (Blood Cell Count and Detection)** dataset. The task is to detect and classify three types of blood cells: **RBC** (Red Blood Cells), **WBC** (White Blood Cells), and **Platelets**.

## Dataset

The [BCCD Dataset](https://github.com/Shenggan/BCCD_Dataset) contains microscopic blood cell images with Pascal VOC‑format XML annotations. A conversion script (`src/convert_to_coco.py`) transforms the annotations into COCO JSON format, which is then consumed by the PyTorch `Dataset` and Hugging Face `Trainer`.

| Split       | Images |
|-------------|--------|
| train       | 162    |
| trainval    | 205    |
| val         | 43     |
| test        | 72     |

Class mapping (0‑indexed):

| ID | Label     |
|----|-----------|
| 0  | RBC       |
| 1  | WBC       |
| 2  | Platelets |

## Project Structure

```
.
├── main.py                    # Placeholder entry point
├── pyproject.toml             # Dependencies (uv/pip)
├── .pre-commit-config.yaml    # Linting & formatting hooks
├── README.md
├── data/
│   └── BCCD/
│       ├── JPEGImages/        # Blood cell images (.jpg)
│       ├── Annotations/       # Pascal VOC XML annotations
│       ├── ImageSets/Main/    # Train/val/test split files (.txt)
│       └── coco/              # Converted COCO JSON annotations
└── src/
    ├── __init__.py
    ├── constants.py           # Dataset directory paths
    ├── convert_to_coco.py     # VOC → COCO conversion
    ├── dataset.py             # PyTorch Dataset wrapping COCO
    ├── train.py               # Training script (HF Trainer)
    └── evaluate.py            # Evaluation + FiftyOne visualization
```

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for package management. Alternatively, `pip` can be used directly.

### Using uv

```bash
uv sync
```

### Using pip

```bash
pip install -e .
```

### Pre-commit hooks (optional but recommended)

```bash
uv run pre-commit install
```

## Usage

### 1. Convert Annotations

Convert the Pascal VOC XML files in `data/BCCD/Annotations/` to COCO JSON format:

```bash
uv run python src/convert_to_coco.py
```

> **Note:** This step has already been performed and the COCO JSON files are already committed to the repository.

This creates `train.json`, `val.json`, `trainval.json`, and `test.json` inside `data/BCCD/coco/`.

### 2. Train

Train a DETR model (or any compatible Hugging Face object detection model) on the BCCD dataset:

```bash
uv run python src/train.py \
    --model-name facebook/detr-resnet-50 \
    --epochs 50 \
    --batch-size 8
```

**Arguments:**

| Argument        | Default | Description                             |
|-----------------|---------|-----------------------------------------|
| `--model-name`  | —       | HF model ID (e.g. `facebook/detr-resnet-50`, `PekingU/rtdetr_r50vd`, `microsoft/conditional-detr-resnet-50`) |
| `--epochs`      | 50      | Number of training epochs               |
| `--batch-size`  | 8       | Per‑device batch size                   |

Checkpoints and logs are saved to `.output/{model_id}_{epochs}ep_{batch_size}bs_{lr}lr/`.

### 3. Evaluate

Run inference on the validation set and compute metrics:

```bash
uv run python src/evaluate.py --model-dir .output/detr-resnet-50_50ep_8bs_1e-05lr
```

This script will:

- Run inference with a score threshold of 0.5
- Compute **mean Average Precision (mAP)** metrics using `torchmetrics.detection.MeanAveragePrecision` (overall and per‑class)
- Save numeric metrics to `{model_dir}/eval/metrics.json`
- Launch a [**FiftyOne**](https://voxel51.com/fiftyone) visualization app to explore predictions interactively
- Generate an **interactive confusion matrix** and **precision‑recall curves** (saved as HTML to `{model_dir}/eval/`)

**Arguments:**

| Argument       | Default                      | Description                     |
|----------------|------------------------------|---------------------------------|
| `--model-dir`  | —                            | Path to trained model directory |
| `--val-ann`    | `data/BCCD/coco/val.json`    | Validation COCO annotation file |

> **Tip:** The HTML plots require a web browser to view; they are interactive (Plotly). To save as static PNG instead, install `kaleido` (`uv add kaleido`) and change the file extension to `.png` in the source code.