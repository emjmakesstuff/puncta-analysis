from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import json
import math


@dataclass
class PunctaConfig:
    """Configuration parameters for puncta analysis (port of PunctaConfig.m)."""

    # GMM parameters
    num_gmm_components: int = 2
    gmm_replicates: int = 20
    gmm_max_iter: int = 1000
    gmm_regularization: float = 1e-5
    gmm_downsample_factor: float = math.sqrt(1 / 50)  # ~0.1414

    # DoG (Difference of Gaussians) parameters  -- tune after qualitative check
    puncta_sigma1: float = 1.5
    puncta_sigma2: float = 3.0          # typically 2x sigma1
    dog_sensitivity: float = 2.0

    # Filtering
    min_puncta_area: int = 1            # minimum puncta size in pixels

    # Visualization
    roi_transparency: float = 0.25
    puncta_marker_size: int = 10
    puncta_marker_color: str = "r"
    manual_marker_color: str = "g"
    roi_boundary_color: str = "c"
    roi_boundary_width: float = 1.5

    # Physical calibration (optional). If None, areas are reported in pixels.
    # Set to your microscope's µm/pixel to get areas in µm².
    pixel_size_um: float | None = None

    # ---- (de)serialization -------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "PunctaConfig":
        """Build a config from a dict, ignoring unknown keys (with a warning)."""
        valid = {f.name for f in fields(cls)}
        clean = {}
        for k, v in (d or {}).items():
            if k in valid:
                clean[k] = v
            else:
                print(f"Warning: Property '{k}' does not exist")
        return cls(**clean)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PunctaConfig":
        import yaml
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        # allow either a flat dict or nested 'detection'/'gmm' sections
        flat = dict(data)
        for section in ("detection", "gmm"):
            if isinstance(data.get(section), dict):
                flat.update(data[section])
        return cls.from_dict(flat)

    def save(self, path: str | Path = "puncta_config.json") -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))
        print(f"Configuration saved to: {path}")

    @classmethod
    def load(cls, path: str | Path = "puncta_config.json") -> "PunctaConfig":
        data = json.loads(Path(path).read_text())
        print(f"Configuration loaded from: {path}")
        return cls.from_dict(data)

    def __str__(self) -> str:
        return (
            "\nPuncta Analysis Configuration\n\n"
            "GMM Parameters:\n"
            f"  Components: {self.num_gmm_components}\n"
            f"  Replicates: {self.gmm_replicates}\n"
            "\nDoG Parameters:\n"
            f"  Sigma1: {self.puncta_sigma1:.2f}\n"
            f"  Sigma2: {self.puncta_sigma2:.2f}\n"
            f"  Sensitivity: {self.dog_sensitivity:.2f}\n"
            "\nFiltering:\n"
            f"  Min puncta area: {self.min_puncta_area} pixels\n"
        )