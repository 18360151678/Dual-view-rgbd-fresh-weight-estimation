"""Evaluate the released YOLO12 segmentation checkpoint on the test set."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights" / "yolo12_best.pt")
    parser.add_argument("--test-root", type=Path, default=ROOT / "test_set")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    return parser.parse_args()


def polygon_labels_to_mask(label_path: Path, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if not label_path.exists():
        return mask
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        coordinates = np.asarray(parts[1:], dtype=np.float32).reshape(-1, 2)
        coordinates[:, 0] *= width
        coordinates[:, 1] *= height
        cv2.fillPoly(mask, [coordinates.astype(np.int32)], 1)
    return mask


def overlap_metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    total = prediction.sum() + target.sum()
    iou = 1.0 if union == 0 else float(intersection / union)
    dice = 1.0 if total == 0 else float(2.0 * intersection / total)
    return iou, dice


def write_runtime_yaml(test_root: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_yaml = output_dir / "dataset.yaml"
    payload = {
        "path": test_root.resolve().as_posix(),
        "train": "images",
        "val": "images",
        "test": "images",
        "nc": 1,
        "names": ["non-heading Chinese cabbage"],
    }
    runtime_yaml.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return runtime_yaml


def main() -> None:
    args = parse_args()
    image_dir = args.test_root / "images"
    label_dir = args.test_root / "labels"
    if not args.weights.exists():
        raise FileNotFoundError(args.weights)
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(
            "Extract segmentation_test_set.zip so test_set/images and "
            "test_set/labels are available."
        )

    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No test images found in {image_dir}")

    runtime_yaml = write_runtime_yaml(args.test_root, args.output_dir)
    model = YOLO(str(args.weights))
    params_m = sum(parameter.numel() for parameter in model.model.parameters()) / 1e6
    flops_g = get_flops(model.model, imgsz=args.imgsz) / 1e9

    official = model.val(
        data=str(runtime_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        project=str(args.output_dir),
        name="official_metrics",
        exist_ok=True,
        verbose=False,
    )

    rows: list[dict] = []
    elapsed = 0.0
    for image_path in tqdm(image_paths, desc="YOLO12 test"):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        height, width = image.shape[:2]

        started = time.perf_counter()
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
            retina_masks=True,
        )[0]
        elapsed += time.perf_counter() - started

        predicted_mask = np.zeros((height, width), dtype=bool)
        if result.masks is not None:
            for mask in result.masks.data:
                resized = cv2.resize(
                    mask.cpu().numpy(),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                predicted_mask |= resized >= args.mask_threshold

        label_path = label_dir / f"{image_path.stem}.txt"
        target_mask = polygon_labels_to_mask(label_path, height, width)
        iou, dice = overlap_metrics(predicted_mask, target_mask)
        rows.append({"image": image_path.name, "iou": iou, "dice": dice})

    per_image = pd.DataFrame(rows)
    summary = {
        "model": "YOLO12 segmentation",
        "test_images": len(per_image),
        "parameters_m": float(params_m),
        "flops_g": float(flops_g),
        "latency_ms_per_image": float(elapsed / len(per_image) * 1000.0),
        "fps": float(len(per_image) / elapsed),
        "mask_map50": float(official.seg.map50),
        "mask_map50_95": float(official.seg.map),
        "mean_iou": float(per_image["iou"].mean()),
        "mean_dice": float(per_image["dice"].mean()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(args.output_dir / "per_image_metrics.csv", index=False)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
