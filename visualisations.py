from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.manifold import spectral_embedding


def build_echarts_visualisations(
        *,
        ids: Optional[pd.Series],
        labels: np.ndarray,
        affinity: str,
        X_raw: np.ndarray,
        model: SpectralClustering,
        random_state: int,
):
    try:
        coords = _compute_2d_embedding(
            affinity=affinity,
            X_raw=X_raw,
            model=model,
            random_state=random_state,
        )
        option = _echarts_scatter_option(coords=coords, labels=labels, ids=ids)

        return [
            {
                "name": "Clusters (2D embedding)",
                "description": "2D spectral embedding scatter plot (points colored by cluster label).",
                "visualisation": option,
            }
        ]
    except Exception:
        return None


def _compute_2d_embedding(
        *,
        affinity: str,
        X_raw: np.ndarray,
        model: SpectralClustering,
        random_state: int,
) -> np.ndarray:
    affinity_mat = getattr(model, "affinity_matrix_", None)

    if affinity_mat is None:
        if str(affinity) != "precomputed":
            raise ValueError(
                "Cannot compute embedding: model has no affinity_matrix_ and affinity != 'precomputed'."
            )
        affinity_mat = X_raw

    affinity_mat = np.asarray(affinity_mat, dtype=float)
    if affinity_mat.ndim != 2 or affinity_mat.shape[0] != affinity_mat.shape[1]:
        raise ValueError("Affinity matrix must be square to compute a spectral embedding.")

    coords = spectral_embedding(
        affinity_mat,
        n_components=2,
        random_state=random_state,
        drop_first=False,
    )

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("Expected a 2D embedding with shape (n_samples, 2).")

    return coords


def _echarts_scatter_option(
        *,
        coords: np.ndarray,
        labels: np.ndarray,
        ids: Optional[pd.Series],
) -> dict:
    labels = np.asarray(labels)
    unique_labels = np.unique(labels).tolist()

    series = []
    for k in unique_labels:
        mask = labels == k
        points = coords[mask]

        if ids is None:
            point_payload = [
                {
                    "value": [float(x), float(y)],
                    "meta": {"label": int(k)},
                }
                for (x, y) in points
            ]
        else:
            id_vals = np.asarray(ids.values)[mask]
            point_payload = [
                {
                    "value": [float(x), float(y)],
                    "meta": {"id": str(i), "label": int(k)},
                }
                for (i, (x, y)) in zip(id_vals, points)
            ]

        series.append(
            {
                "name": f"Cluster {int(k)}",
                "type": "scatter",
                "symbolSize": 8,
                "data": point_payload,
                "emphasis": {"focus": "series"},
            }
        )

    return {
        "title": {"text": "Spectral Clustering (2D embedding)"},
        "tooltip": {"trigger": "item"},
        "legend": {"type": "scroll"},
        "grid": {"containLabel": True},
        "xAxis": {"type": "value", "name": "emb-1"},
        "yAxis": {"type": "value", "name": "emb-2"},
        "series": series,
    }
