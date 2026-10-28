from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from pyfedappwrap.learning.app_types import BaseSelfLearnedApp
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from typing_extensions import override

from config import SpectralAppConfig, SpectralAppInputConfig, SpectralAppOutputConfig
from report import html_report
from visualisations import build_echarts_visualisations


class SpectralClusteringAPP(
    BaseSelfLearnedApp[SpectralAppConfig, SpectralAppInputConfig, SpectralAppOutputConfig]):

    @override
    def get_test_data(self) -> Optional[dict[str, Path]]:
        return {"features": Path("data/iris_encoded.csv")}

    def run_algorithm(self, data: SpectralAppInputConfig) -> SpectralAppOutputConfig:
        df: pd.DataFrame = data.features
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'features' must be a pandas DataFrame.")

        ids, feats = _split_id_and_features(df)
        X_raw = _ensure_numeric_matrix(feats)

        affinity = str(self.config.affinity)
        if affinity != "precomputed":
            X = _maybe_standardize(X_raw, self.config.standardize)
        else:
            X = X_raw

        model = SpectralClustering(
            n_clusters=int(self.config.n_clusters),
            affinity=affinity,
            gamma=float(self.config.gamma),
            n_neighbors=int(self.config.n_neighbors),
            assign_labels=str(self.config.assign_labels),
            n_init=int(self.config.n_init),
            eigen_solver=str(self.config.eigen_solver),
            random_state=int(self.config.random_state),
        )

        self.logger.info(
            f"Running SpectralClustering on {X.shape[0]} samples "
            f"(affinity={affinity}, n_clusters={self.config.n_clusters})"
        )

        labels = model.fit_predict(X)

        label_df = _get_labels_df(ids, labels)

        visualisations = build_echarts_visualisations(
            ids=ids,
            labels=labels,
            affinity=affinity,
            X_raw=X_raw,
            model=model,
            random_state=int(self.config.random_state),
        )

        unique, counts = np.unique(labels, return_counts=True)
        for k, c in zip(unique.tolist(), counts.tolist()):
            self.send_metric("cluster_size", float(c), float(k))

        report = html_report(
            "SpectralClustering Report",
            params={
                "n_clusters": self.config.n_clusters,
                "affinity": affinity,
                "gamma": self.config.gamma,
                "n_neighbors": self.config.n_neighbors,
                "assign_labels": self.config.assign_labels,
                "n_init": self.config.n_init,
                "eigen_solver": self.config.eigen_solver,
                "random_state": self.config.random_state,
                "standardize": bool(self.config.standardize),
                "output_embedding": bool(self.config.output_embedding),
            },
            extra={
                "n_samples": X.shape[0],
                "n_features": X.shape[
                    1] if affinity != "precomputed" else "precomputed_affinity_matrix",
                "n_output_clusters": int(len(np.unique(labels))),
            },
        )

        return SpectralAppOutputConfig(
            visualisations=visualisations,
            labels=label_df,
            report=report,
        )


def _split_id_and_features(df: pd.DataFrame) -> Tuple[Optional[pd.Series], pd.DataFrame]:
    if df.shape[1] == 0:
        raise ValueError("Input dataframe has no columns.")

    first_col = str(df.columns[0]).strip().lower()
    if first_col == "id":
        ids = df.iloc[:, 0]
        feats = df.iloc[:, 1:]
        return ids, feats
    return None, df


def _ensure_numeric_matrix(feats: pd.DataFrame) -> np.ndarray:
    X = feats.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("Feature matrix must be 2D with at least one feature column.")
    return X


def _maybe_standardize(X: np.ndarray, standardize: bool) -> np.ndarray:
    if bool(standardize):
        return StandardScaler().fit_transform(X)
    return X


def _get_labels_df(ids: Optional[pd.Series], labels: np.ndarray) -> pd.DataFrame:
    if ids is None:
        return pd.DataFrame({"index": np.arange(len(labels)), "label": labels})
    else:
        return pd.DataFrame({"id": ids.values, "label": labels})
