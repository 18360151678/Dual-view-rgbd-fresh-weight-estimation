from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

# Required by CUDA/cuBLAS when deterministic PyTorch algorithms are requested.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from rgbd_experiment import (
    DEFAULT_SEEDS,
    DualBranchModel,
    MultiViewRGBDDataset,
    config_to_dict,
    load_or_create_manifest,
    parse_model_names,
    regression_metrics,
    save_run_metadata,
    seed_worker,
    set_seed,
    split_indices,
)


ROOT = Path(__file__).resolve().parent

PUBLISHED_TOP10_MODELS = (
    "resnet18_shufflenet_mul",
    "shufflenet_shufflenet_mul",
    "resnet18_squeezenet_mul",
    "shufflenet_vgg11_attn",
    "resnet18_vgg11_attn",
    "squeezenet_squeezenet_mul",
    "squeezenet_resnet18_mul",
    "squeezenet_shufflenet_mul",
    "resnet18_vgg11_mul",
    "squeezenet_vgg11_mul",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "RGB_D")
    parser.add_argument("--labels", type=Path, default=ROOT / "data" / "data.xlsx")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help=(
            "Existing split manifest to reuse exactly. When omitted, the manifest "
            "inside output-dir is loaded or created from split-seed."
        ),
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max-depth", type=float, default=3.0)
    parser.add_argument("--label-scale", type=float, default=300.0)
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(PUBLISHED_TOP10_MODELS),
        help="Comma-separated top_side_fusion model names. Default: published top 10.",
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Train all 75 top/side backbone and fusion combinations.",
    )
    parser.add_argument(
        "--exclude-vgg",
        action="store_true",
        help="Exclude configurations containing VGG; useful for safe multi-process training.",
    )
    parser.add_argument(
        "--only-vgg-pairs",
        action="store_true",
        help="Train only configurations where the top or side backbone is VGG.",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Optional unique summary workbook path used by the task queue.",
    )
    return parser.parse_args()


def make_loader(
    dataset,
    indices: list[int],
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    label_scale: float,
    optimizer: Adam | None,
) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    predictions: list[float] = []
    targets: list[float] = []

    for top, side, target, _ in loader:
        top = top.to(device, non_blocking=True)
        side = side.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(top, side)
            loss = criterion(output, target)
            if training:
                loss.backward()
                optimizer.step()
        batch_count = target.shape[0]
        total_loss += float(loss.item()) * batch_count
        total_count += batch_count
        predictions.extend((output.detach().cpu().numpy().reshape(-1) * label_scale).tolist())
        targets.extend((target.detach().cpu().numpy().reshape(-1) * label_scale).tolist())

    return total_loss / total_count, regression_metrics(targets, predictions)


def train_one_run(
    config,
    seed: int,
    dataset: MultiViewRGBDDataset,
    train_idx: list[int],
    val_idx: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    set_seed(seed)
    run_dir = args.output_dir / "runs" / config.name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best.pt"
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"), purge_step=0)
    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        dataset, train_idx, args.batch_size, True, seed, args.num_workers, pin_memory
    )
    val_loader = make_loader(
        dataset, val_idx, args.batch_size, False, seed, args.num_workers, pin_memory
    )
    model = DualBranchModel(
        **config_to_dict(config), pretrained=not args.no_pretrained
    ).to(device)
    optimizer = Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.MSELoss()
    best_val_r2 = -np.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict] = []
    started = time.time()

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_metrics = run_epoch(
                model, train_loader, criterion, device, args.label_scale, optimizer
            )
            with torch.no_grad():
                val_loss, val_metrics = run_epoch(
                    model, val_loader, criterion, device, args.label_scale, None
                )
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            history.append(row)
            for phase in ("train", "val"):
                writer.add_scalar(f"loss/{phase}", row[f"{phase}_loss"], epoch)
                for metric in ("r2", "mae", "rmse"):
                    writer.add_scalar(
                        f"metrics/{phase}_{metric}", row[f"{phase}_{metric}"], epoch
                    )

            improved = np.isfinite(val_metrics["r2"]) and val_metrics["r2"] > best_val_r2
            if improved:
                best_val_r2 = val_metrics["r2"]
                best_epoch = epoch
                stale_epochs = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_config": config_to_dict(config),
                        "model_name": config.name,
                        "training_seed": seed,
                        "split_seed": args.split_seed,
                        "best_epoch": best_epoch,
                        "best_val_r2": best_val_r2,
                        "best_val_mae": val_metrics["mae"],
                        "best_val_rmse": val_metrics["rmse"],
                        "selection_rule": "highest validation R2",
                        "pretrained": not args.no_pretrained,
                        "max_depth": args.max_depth,
                        "label_scale": args.label_scale,
                    },
                    checkpoint_path,
                )
            else:
                stale_epochs += 1

            print(
                f"{config.name} seed={seed} epoch={epoch:03d} "
                f"train_R2={train_metrics['r2']:.4f} val_R2={val_metrics['r2']:.4f}"
            )
            if args.patience > 0 and stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break
    finally:
        writer.close()

    if not history:
        raise RuntimeError("No training epochs were completed.")
    history_path = run_dir / "history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")
    summary = {
        "model": config.name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_r2": best_val_r2,
        "pretrained": not args.no_pretrained,
        "elapsed_minutes": (time.time() - started) / 60.0,
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
    }
    save_run_metadata(run_dir / "completed.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = MultiViewRGBDDataset(
        args.data_dir,
        args.labels,
        max_depth=args.max_depth,
        label_scale=args.label_scale,
    )
    output_manifest_path = args.output_dir / "split_manifest.csv"
    source_manifest_path = args.split_manifest or output_manifest_path
    if args.split_manifest is not None and not source_manifest_path.exists():
        raise FileNotFoundError(
            f"The requested fixed split manifest does not exist: {source_manifest_path}"
        )
    manifest = load_or_create_manifest(dataset, source_manifest_path, args.split_seed)
    if source_manifest_path.resolve() != output_manifest_path.resolve():
        if output_manifest_path.exists():
            existing_manifest = pd.read_csv(output_manifest_path)
            columns = ["row_index", "sample_id", "target", "split", "split_seed"]
            if existing_manifest[columns].astype(str).values.tolist() != (
                manifest[columns].astype(str).values.tolist()
            ):
                raise ValueError(
                    "The output split manifest differs from the requested fixed manifest: "
                    f"{output_manifest_path}"
                )
        else:
            manifest.to_csv(output_manifest_path, index=False, encoding="utf-8-sig")
    train_idx = split_indices(manifest, "train")
    val_idx = split_indices(manifest, "val")
    test_idx = split_indices(manifest, "test")
    print(
        f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, "
        f"test={len(test_idx)}"
    )
    configs = parse_model_names(args.models, all_pairs=args.all_pairs)
    if args.exclude_vgg and args.only_vgg_pairs:
        raise ValueError("--exclude-vgg and --only-vgg-pairs cannot be used together.")
    if args.exclude_vgg:
        configs = [
            config
            for config in configs
            if config.top_name != "vgg11" and config.side_name != "vgg11"
        ]
    if args.only_vgg_pairs:
        configs = [
            config
            for config in configs
            if config.top_name == "vgg11" or config.side_name == "vgg11"
        ]
    if not configs:
        raise ValueError("No model configurations remain after filtering.")
    device = torch.device(args.device)
    print(f"Device: {device}; models={len(configs)}; seeds={args.seeds}")
    summary_name = (
        f"training_summary_seed_{args.seeds[0]}.xlsx"
        if len(args.seeds) == 1
        else "training_summary.xlsx"
    )
    summary_path = args.summary_file or (args.output_dir / summary_name)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    for config in configs:
        for seed in args.seeds:
            checkpoint = args.output_dir / "runs" / config.name / f"seed_{seed}" / "best.pt"
            completion_marker = checkpoint.parent / "completed.json"
            if checkpoint.exists() and completion_marker.exists():
                print(f"Skipping existing checkpoint: {checkpoint}")
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                summaries.append(
                    {
                        "model": config.name,
                        "seed": seed,
                        "best_epoch": saved["best_epoch"],
                        "best_val_r2": saved["best_val_r2"],
                        "pretrained": saved.get("pretrained", True),
                        "elapsed_minutes": np.nan,
                        "checkpoint": str(checkpoint),
                        "history": str(checkpoint.parent / "history.csv"),
                    }
                )
                continue
            try:
                summaries.append(
                    train_one_run(
                        config, seed, dataset, train_idx, val_idx, args, device
                    )
                )
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            pd.DataFrame(summaries).to_excel(
                summary_path, index=False
            )

    pd.DataFrame(summaries).to_excel(
        summary_path, index=False
    )
    print("Training and validation are complete. The independent test set was untouched.")
    print(f"Next: python evaluate.py --output-dir \"{args.output_dir}\"")


if __name__ == "__main__":
    main()
