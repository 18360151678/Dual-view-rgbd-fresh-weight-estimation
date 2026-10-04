"""Shared data, model, split, and statistical utilities."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import models


BACKBONES = ("resnet18", "mobilenet", "shufflenet", "squeezenet", "vgg11")
FUSIONS = ("cat", "mul", "attn")
DEFAULT_SEEDS = (42, 2024, 3407)
SPLIT_RATIOS = {"train": 0.64, "val": 0.16, "test": 0.20}


@dataclass(frozen=True)
class ModelConfig:
    top_name: str
    side_name: str
    fusion: str

    @property
    def name(self) -> str:
        return f"{self.top_name}_{self.side_name}_{self.fusion}"


@dataclass(frozen=True)
class SingleViewConfig:
    backbone_name: str
    view: str

    @property
    def name(self) -> str:
        return f"{self.backbone_name}_{self.view}"


def default_single_view_configs() -> list[SingleViewConfig]:
    return [
        SingleViewConfig(backbone, view)
        for backbone in BACKBONES
        for view in ("top", "side")
    ]


def parse_single_view_names(names: str | None) -> list[SingleViewConfig]:
    if not names:
        return default_single_view_configs()
    configs: list[SingleViewConfig] = []
    valid = {config.name: config for config in default_single_view_configs()}
    for raw_name in names.split(","):
        name = raw_name.strip().lower()
        if name not in valid:
            raise ValueError(
                f"Invalid single-view model '{name}'. Expected names such as "
                "resnet18_top or mobilenet_side."
            )
        configs.append(valid[name])
    return configs


def default_model_configs(all_pairs: bool = False) -> list[ModelConfig]:
    """Return 15 same-backbone models, or all 75 top/side combinations."""
    if all_pairs:
        return [
            ModelConfig(top, side, fusion)
            for top in BACKBONES
            for side in BACKBONES
            for fusion in FUSIONS
        ]
    return [
        ModelConfig(backbone, backbone, fusion)
        for backbone in BACKBONES
        for fusion in FUSIONS
    ]


def parse_model_names(names: str | None, all_pairs: bool = False) -> list[ModelConfig]:
    if not names:
        return default_model_configs(all_pairs=all_pairs)

    configs: list[ModelConfig] = []
    for raw_name in names.split(","):
        name = raw_name.strip()
        parts = name.rsplit("_", 1)
        if len(parts) != 2 or parts[1] not in FUSIONS:
            raise ValueError(
                f"Invalid model name '{name}'. Expected top_side_fusion, for example "
                "resnet18_resnet18_attn."
            )
        pair, fusion = parts
        matches = [
            (top, side)
            for top in BACKBONES
            for side in BACKBONES
            if pair == f"{top}_{side}"
        ]
        if len(matches) != 1:
            raise ValueError(f"Cannot parse model name '{name}'.")
        top, side = matches[0]
        configs.append(ModelConfig(top, side, fusion))
    return configs


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class MultiViewRGBDDataset(Dataset):
    """One plant is one sample; both views always remain in the same split."""

    def __init__(
        self,
        data_dir: str | Path,
        excel_path: str | Path,
        max_depth: float = 3.0,
        label_scale: float = 300.0,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.excel_path = Path(excel_path)
        if self.excel_path.suffix.lower() == ".csv":
            frame = pd.read_csv(self.excel_path)
        else:
            frame = pd.read_excel(self.excel_path)
        if frame.shape[1] < 2:
            raise ValueError("The label workbook must contain ID and target columns.")
        self.ids = frame.iloc[:, 0].tolist()
        self.weights = frame.iloc[:, 1].astype(float).tolist()
        if len(set(map(str, self.ids))) != len(self.ids):
            raise ValueError("Sample IDs must be unique.")
        self.max_depth = float(max_depth)
        self.label_scale = float(label_scale)

    def __len__(self) -> int:
        return len(self.ids)

    def _load_view(self, sample_id: object, view: str) -> torch.Tensor:
        path = self.data_dir / f"{sample_id}_rgbd_{view}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        image = np.array(np.load(path, mmap_mode="r"), dtype=np.float32, copy=True)
        if image.ndim != 3 or image.shape[2] != 4:
            raise ValueError(f"Expected HxWx4 RGB-D array, got {image.shape} in {path}")
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image[:, :, :3] = np.clip(image[:, :, :3], 0.0, 255.0) / 255.0
        image[:, :, 3] = np.clip(image[:, :, 3], 0.0, self.max_depth) / self.max_depth
        return torch.from_numpy(image).permute(2, 0, 1).contiguous()

    def __getitem__(self, index: int):
        sample_id = self.ids[index]
        top = self._load_view(sample_id, "top")
        side = self._load_view(sample_id, "side")
        target = torch.tensor(
            [self.weights[index] / self.label_scale], dtype=torch.float32
        )
        return top, side, target, sample_id


class SingleViewRGBDDataset(MultiViewRGBDDataset):
    """Load only one RGB-D view while retaining the same plant-level manifest."""

    def __init__(
        self,
        data_dir: str | Path,
        excel_path: str | Path,
        view: str,
        max_depth: float = 3.0,
        label_scale: float = 300.0,
    ) -> None:
        super().__init__(data_dir, excel_path, max_depth=max_depth, label_scale=label_scale)
        view = view.lower()
        if view not in {"top", "side"}:
            raise ValueError("view must be 'top' or 'side'.")
        self.view = view

    def __getitem__(self, index: int):
        sample_id = self.ids[index]
        image = self._load_view(sample_id, self.view)
        target = torch.tensor(
            [self.weights[index] / self.label_scale], dtype=torch.float32
        )
        return image, target, sample_id


def create_split_manifest(
    dataset: MultiViewRGBDDataset,
    split_seed: int = 42,
) -> pd.DataFrame:
    """Hold out 20% for testing, then split the remaining 80% into train/val."""
    indices = np.arange(len(dataset))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        shuffle=True,
        random_state=split_seed,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.20,
        shuffle=True,
        random_state=split_seed,
    )

    split_by_index = {int(i): "train" for i in train_idx}
    split_by_index.update({int(i): "val" for i in val_idx})
    split_by_index.update({int(i): "test" for i in test_idx})
    manifest = pd.DataFrame(
        {
            "row_index": indices,
            "sample_id": dataset.ids,
            "target": dataset.weights,
            "split": [split_by_index[int(i)] for i in indices],
            "split_seed": split_seed,
        }
    )
    expected = SPLIT_RATIOS
    actual = manifest["split"].value_counts(normalize=True).to_dict()
    tolerance = 1.0 / max(1, len(dataset)) + 1e-12
    for split, ratio in expected.items():
        if abs(actual.get(split, 0.0) - ratio) > tolerance:
            raise RuntimeError(f"Unexpected {split} split ratio: {actual.get(split, 0.0):.4f}")
    return manifest


def load_or_create_manifest(
    dataset: MultiViewRGBDDataset,
    manifest_path: str | Path,
    split_seed: int = 42,
) -> pd.DataFrame:
    """Persist the split so every model and training seed uses identical plants."""
    path = Path(manifest_path)
    if path.exists():
        manifest = pd.read_csv(path)
        required = {"row_index", "sample_id", "target", "split", "split_seed"}
        if not required.issubset(manifest.columns):
            raise ValueError(f"Invalid split manifest: {path}")
        if len(manifest) != len(dataset):
            raise ValueError("Existing split manifest does not match dataset length.")
        if manifest["sample_id"].astype(str).tolist() != list(map(str, dataset.ids)):
            raise ValueError("Existing split manifest IDs do not match the label workbook.")
        stored_seed = int(manifest["split_seed"].iloc[0])
        if stored_seed != split_seed:
            raise ValueError(
                f"Manifest was created with split seed {stored_seed}, not {split_seed}. "
                "Use the original seed or a new output directory."
            )
        actual = manifest["split"].value_counts(normalize=True).to_dict()
        tolerance = 1.0 / max(1, len(dataset)) + 1e-12
        incompatible = {
            split: actual.get(split, 0.0)
            for split, expected_ratio in SPLIT_RATIOS.items()
            if abs(actual.get(split, 0.0) - expected_ratio) > tolerance
        }
        if incompatible:
            raise ValueError(
                "Existing split manifest does not match the required two-stage split. "
                "Use a new output directory so old checkpoints are not mixed with the new experiment."
            )
        return manifest

    manifest = create_split_manifest(dataset, split_seed=split_seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False, encoding="utf-8-sig")
    return manifest


def split_indices(manifest: pd.DataFrame, split: str) -> list[int]:
    return manifest.loc[manifest["split"] == split, "row_index"].astype(int).tolist()


def _replace_first_conv(conv: nn.Conv2d) -> nn.Conv2d:
    replacement = nn.Conv2d(
        4,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        if conv.weight.shape[1] == 3:
            replacement.weight[:, :3].copy_(conv.weight)
            replacement.weight[:, 3:4].copy_(conv.weight.mean(dim=1, keepdim=True))
            if conv.bias is not None:
                replacement.bias.copy_(conv.bias)
    return replacement


def create_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.conv1 = _replace_first_conv(model.conv1)
        model.fc = nn.Identity()
        return model, 512
    if name == "mobilenet":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.features[0][0] = _replace_first_conv(model.features[0][0])
        model.classifier = nn.Identity()
        return model, 1280
    if name == "shufflenet":
        weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        model = models.shufflenet_v2_x1_0(weights=weights)
        model.conv1[0] = _replace_first_conv(model.conv1[0])
        model.fc = nn.Identity()
        return model, 1024
    if name == "squeezenet":
        weights = models.SqueezeNet1_0_Weights.DEFAULT if pretrained else None
        model = models.squeezenet1_0(weights=weights)
        model.features[0] = _replace_first_conv(model.features[0])
        model.classifier = nn.Identity()
        return model, 512
    if name == "vgg11":
        weights = models.VGG11_Weights.DEFAULT if pretrained else None
        model = models.vgg11(weights=weights)
        model.features[0] = _replace_first_conv(model.features[0])
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        return model, 4096
    raise ValueError(f"Unknown backbone: {name}")


class FusionHead(nn.Module):
    def __init__(self, feature_dim: int, fusion: str, dropout: float = 0.5) -> None:
        super().__init__()
        self.fusion = fusion
        if fusion == "cat":
            in_dim = feature_dim * 2
        elif fusion == "mul":
            in_dim = feature_dim
        elif fusion == "attn":
            in_dim = feature_dim
            self.attn = nn.Sequential(
                nn.Linear(feature_dim * 2, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2),
                nn.Softmax(dim=1),
            )
        else:
            raise ValueError(f"Unknown fusion: {fusion}")
        self.regressor = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, 1))

    def forward(self, top_feature: torch.Tensor, side_feature: torch.Tensor):
        if self.fusion == "cat":
            feature = torch.cat([top_feature, side_feature], dim=1)
        elif self.fusion == "mul":
            feature = top_feature * side_feature
        else:
            weights = self.attn(torch.cat([top_feature, side_feature], dim=1))
            feature = top_feature * weights[:, 0:1] + side_feature * weights[:, 1:2]
        return self.regressor(feature)


class DualBranchModel(nn.Module):
    def __init__(
        self,
        top_name: str,
        side_name: str,
        fusion: str,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.top_name = top_name
        self.side_name = side_name
        self.top, top_dim = create_backbone(top_name, pretrained=pretrained)
        self.side, side_dim = create_backbone(side_name, pretrained=pretrained)
        common_dim = min(1024, (top_dim + side_dim) // 2)
        self.top_proj = nn.Linear(top_dim, common_dim)
        self.side_proj = nn.Linear(side_dim, common_dim)
        self.fusion = FusionHead(common_dim, fusion=fusion, dropout=0.5)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def _extract(self, backbone: nn.Module, image: torch.Tensor) -> torch.Tensor:
        if isinstance(backbone, models.SqueezeNet):
            feature = backbone.features(image)
            return self.pool(feature).flatten(1)
        return backbone(image)

    def forward(self, top: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        top_feature = self.top_proj(self._extract(self.top, top))
        side_feature = self.side_proj(self._extract(self.side, side))
        return self.fusion(top_feature, side_feature)


class SingleViewModel(nn.Module):
    """One RGB-D backbone followed by a dropout-linear regression head."""

    def __init__(self, backbone_name: str, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone, feature_dim = create_backbone(
            backbone_name, pretrained=pretrained
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(nn.Dropout(0.5), nn.Linear(feature_dim, 1))

    def _extract(self, image: torch.Tensor) -> torch.Tensor:
        if isinstance(self.backbone, models.SqueezeNet):
            feature = self.backbone.features(image)
            return self.pool(feature).flatten(1)
        return self.backbone(image)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.regressor(self._extract(image))


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    return {
        "r2": float(r2_score(true, pred)),
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(math.sqrt(mean_squared_error(true, pred))),
    }


def _bootstrap_metric_arrays(
    true: np.ndarray,
    pred: np.ndarray,
    sample_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    sampled_true = true[sample_indices]
    sampled_pred = pred[sample_indices]
    residual = sampled_true - sampled_pred
    mae = np.mean(np.abs(residual), axis=1)
    squared_error = np.sum(residual**2, axis=1)
    rmse = np.sqrt(squared_error / sampled_true.shape[1])
    centered = sampled_true - np.mean(sampled_true, axis=1, keepdims=True)
    denominator = np.sum(centered**2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - squared_error / denominator
    return {"r2": r2, "mae": mae, "rmse": rmse}


def bootstrap_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    n_bootstrap: int = 10000,
    seed: int = 12345,
) -> tuple[dict[str, tuple[float, float]], dict[str, np.ndarray]]:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(true), size=(n_bootstrap, len(true)))
    arrays = _bootstrap_metric_arrays(true, pred, sample_indices)
    intervals = {
        key: (float(np.nanpercentile(value, 2.5)), float(np.nanpercentile(value, 97.5)))
        for key, value in arrays.items()
    }
    return intervals, arrays


def paired_bootstrap_difference(
    y_true: Sequence[float],
    candidate_pred: Sequence[float],
    reference_pred: Sequence[float],
    n_bootstrap: int = 10000,
    seed: int = 12345,
) -> list[dict[str, float | str]]:
    true = np.asarray(y_true, dtype=np.float64)
    candidate = np.asarray(candidate_pred, dtype=np.float64)
    reference = np.asarray(reference_pred, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(true), size=(n_bootstrap, len(true)))
    candidate_metrics = _bootstrap_metric_arrays(true, candidate, sample_indices)
    reference_metrics = _bootstrap_metric_arrays(true, reference, sample_indices)
    differences = {
        metric: candidate_metrics[metric] - reference_metrics[metric]
        for metric in ("r2", "mae", "rmse")
    }

    rows: list[dict[str, float | str]] = []
    for metric, values in differences.items():
        finite_values = values[np.isfinite(values)]
        p_value = min(
            1.0,
            2.0 * min(np.mean(finite_values <= 0), np.mean(finite_values >= 0)),
        )
        rows.append(
            {
                "metric": metric,
                "difference_definition": "candidate - reference",
                "mean_difference": float(np.mean(finite_values)),
                "ci95_low": float(np.percentile(finite_values, 2.5)),
                "ci95_high": float(np.percentile(finite_values, 97.5)),
                "p_value_two_sided": float(p_value),
            }
        )
    return rows


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    p = np.asarray(list(p_values), dtype=np.float64)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=np.float64)
    running_max = 0.0
    for rank, index in enumerate(order):
        current = min(1.0, (len(p) - rank) * p[index])
        running_max = max(running_max, current)
        adjusted[index] = running_max
    return adjusted.tolist()


def save_run_metadata(path: str | Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def config_to_dict(config: ModelConfig) -> dict[str, str]:
    return asdict(config)
