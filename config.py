from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic.dataclasses import dataclass
from pyfedappwrap.learning.run_runfig import AppConfig, AppInputConfig, AppOutputConfig


@dataclass
class SpectralAppConfig(AppConfig):
    n_clusters: int = 3
    affinity: str = "rbf"  # rbf|nearest_neighbors|precomputed
    gamma: float = 1.0  # only for rbf
    n_neighbors: int = 10  # only for nearest_neighbors
    assign_labels: str = "kmeans"  # kmeans|discretize
    n_init: int = 10  # for kmeans
    eigen_solver: str = "arpack"  # arpack|lobpcg|amg
    random_state: int = 42
    standardize: bool = 1
    output_embedding: bool = 0


@dataclass
class SpectralAppInputConfig(AppInputConfig):
    features: Any = None


@dataclass
class SpectralAppOutputConfig(AppOutputConfig):
    labels: Any = None
    report: str | None = None
