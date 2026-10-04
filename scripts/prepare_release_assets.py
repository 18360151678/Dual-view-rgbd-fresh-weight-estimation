"""Build release assets and reproducibility tables from completed local runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd


TOP_K = 10
SPLIT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--fresh-output", type=Path, required=True)
    parser.add_argument("--fresh-data", type=Path, required=True)
    parser.add_argument("--fresh-labels", type=Path, required=True)
    parser.add_argument(
        "--published-validation-manifest", type=Path, required=True
    )
    parser.add_argument("--segmentation-weight", type=Path, required=True)
    parser.add_argument("--segmentation-images", type=Path, required=True)
    parser.add_argument("--segmentation-labels", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def zip_directory(source: Path, destination: Path) -> None:
    if destination.exists():
        destination.unlink()
    with zipfile.ZipFile(
        destination, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source.parent))


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    fresh_root = repo_root / "fresh_weight"
    sys.path.insert(0, str(fresh_root))
    from rgbd_experiment import MultiViewRGBDDataset, create_split_manifest

    results_dir = fresh_root / "results"
    tensorboard_dir = fresh_root / "output" / "tensorboard"
    release_dir = repo_root / "release_assets"
    results_dir.mkdir(parents=True, exist_ok=True)
    reset_directory(tensorboard_dir)
    reset_directory(release_dir)

    report = args.fresh_output / "validation_results.xlsx"
    summary = pd.read_excel(report, sheet_name="mean_std").sort_values(
        "r2_mean", ascending=False
    )
    top10 = summary.head(TOP_K).copy().reset_index(drop=True)
    top10.insert(0, "rank", range(1, TOP_K + 1))
    per_seed = pd.read_excel(report, sheet_name="per_seed_metrics")
    selected = (
        per_seed[per_seed["model"].isin(top10["model"])]
        .merge(top10[["rank", "model", "r2_mean"]], on="model", validate="many_to_one")
        .sort_values(["rank", "r2"], ascending=[True, False])
        .groupby("model", sort=False, as_index=False)
        .head(1)
        .sort_values("rank")
        .reset_index(drop=True)
    )
    top10.to_csv(results_dir / "top10_metrics_mean_sd.csv", index=False)

    weights_dir = release_dir / "fresh_weight_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_rows: list[dict] = []
    for row in selected.itertuples(index=False):
        run_dir = args.fresh_output / "runs" / row.model / f"seed_{int(row.seed)}"
        source_checkpoint = run_dir / "best.pt"
        filename = f"rank_{int(row.rank):02d}_{row.model}_seed_{int(row.seed)}_best.pt"
        destination = weights_dir / filename
        shutil.copy2(source_checkpoint, destination)

        log_destination = tensorboard_dir / row.model / f"seed_{int(row.seed)}"
        log_destination.mkdir(parents=True, exist_ok=True)
        for event_file in sorted(run_dir.rglob("events.out.tfevents.*")):
            shutil.copy2(event_file, log_destination / event_file.name)
        if (run_dir / "history.csv").exists():
            shutil.copy2(run_dir / "history.csv", log_destination / "history.csv")
        metadata = {
            "rank": int(row.rank),
            "model": row.model,
            "selected_seed": int(row.seed),
            "selection_rule": "highest validation R2 among seeds 42, 2024, and 3407",
            "validation_r2": float(row.r2),
            "validation_mae": float(row.mae),
            "validation_rmse": float(row.rmse),
        }
        (log_destination / "selection.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        checkpoint_rows.append(
            {
                **metadata,
                "filename": filename,
                "size_bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
    pd.DataFrame(checkpoint_rows).to_csv(
        results_dir / "selected_checkpoints.csv", index=False
    )

    dataset = MultiViewRGBDDataset(args.fresh_data, args.fresh_labels)
    manifest = create_split_manifest(dataset, split_seed=SPLIT_SEED)
    data_metadata_dir = fresh_root / "data"
    data_metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(data_metadata_dir / "split_manifest.csv", index=False)

    published_manifest = pd.read_csv(args.published_validation_manifest)
    required_columns = {"sample_id", "target", "split"}
    if not required_columns.issubset(published_manifest.columns):
        raise ValueError(
            "Published validation manifest must contain sample_id, target, and split."
        )
    validation = published_manifest.loc[
        published_manifest["split"] == "val", ["sample_id", "target"]
    ].copy()
    if len(validation) != 34:
        raise ValueError(
            f"Expected 34 samples in the published validation set, found {len(validation)}."
        )
    validation.columns = ["sample_id", "fresh_weight_g"]
    validation.to_csv(
        data_metadata_dir / "published_validation_labels.csv", index=False
    )

    validation_asset = release_dir / "fresh_weight_validation_set"
    validation_arrays = validation_asset / "RGB_D"
    validation_arrays.mkdir(parents=True, exist_ok=True)
    for sample_id in validation["sample_id"]:
        for view in ("top", "side"):
            filename = f"{sample_id}_rgbd_{view}.npy"
            shutil.copy2(args.fresh_data / filename, validation_arrays / filename)
    shutil.copy2(
        data_metadata_dir / "published_validation_labels.csv",
        validation_asset / "validation_labels.csv",
    )
    zip_directory(validation_asset, release_dir / "fresh_weight_validation_set.zip")

    segmentation_weight = release_dir / "yolo12_best.pt"
    shutil.copy2(args.segmentation_weight, segmentation_weight)
    segmentation_asset = release_dir / "segmentation_test_set"
    image_destination = segmentation_asset / "images"
    label_destination = segmentation_asset / "labels"
    image_destination.mkdir(parents=True, exist_ok=True)
    label_destination.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(args.segmentation_images.iterdir()):
        if image_path.is_file():
            shutil.copy2(image_path, image_destination / image_path.name)
    for label_path in sorted(args.segmentation_labels.glob("*.txt")):
        shutil.copy2(label_path, label_destination / label_path.name)
    zip_directory(segmentation_asset, release_dir / "segmentation_test_set.zip")

    packaged_files = [
        release_dir / "yolo12_best.pt",
        release_dir / "segmentation_test_set.zip",
        release_dir / "fresh_weight_validation_set.zip",
        *sorted(weights_dir.glob("*.pt")),
    ]
    checksum_lines = [f"{sha256(path)}  {path.name}" for path in packaged_files]
    (release_dir / "checksums_sha256.txt").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    print(f"Selected checkpoints: {len(checkpoint_rows)}")
    print(f"Validation samples: {len(validation)}")
    print(f"Segmentation images: {len(list(image_destination.iterdir()))}")
    print(f"Release assets: {release_dir}")


if __name__ == "__main__":
    main()
