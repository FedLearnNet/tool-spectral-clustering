# spectral-clustering

Runs spectral clustering on numeric tables and exports sample labels and an interactive cluster visualization.

## Runtime inputs

Supply input files according to the schemas in `app.yml`.

## Inputs and outputs

### Inputs

| Name | Type | Description |
| --- | --- | --- |
| `features` | CSV | Tabular feature matrix (or affinity matrix if affinity='precomputed') |

### Outputs

| Name | Type | Description |
| --- | --- | --- |
| `labels` | CSV | Cluster assignments per sample |
| `report` | HTML | HTML report with parameters, affinity settings, eigen solver and cluster-size stats. |

## Configuration

| Name | Default | Description |
| --- | --- | --- |
| `n_clusters` | `3` | Number of clusters. |
| `affinity` | `rbf` | Affinity strategy. 'precomputed' expects an affinity matrix as input. |
| `gamma` | `1.0` | Gamma for RBF kernel (only used if affinity='rbf'). |
| `n_neighbors` | `10` | Number of neighbors for the k-NN graph (only used if affinity='nearest_neighbors'). |
| `assign_labels` | `kmeans` | Label assignment strategy in embedding space. |
| `n_init` | `10` | Number of k-means initializations (used when assign_labels='kmeans'). |
| `eigen_solver` | `arpack` | Eigen solver. 'amg' requires pyamg to be installed. |
| `random_state` | `42` | Random seed for reproducibility. |
| `standardize` | `1` | Standardize features to zero mean / unit variance before clustering (recommended). Ignored if affinity='precomputed'. |

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
# Set APP_ID and platform URLs before connecting.
python main.py
```

The Docker base image provides the FL-Net API. `requirements.txt` lists additional
runtime dependencies; `requirements-dev.txt` includes the API for local development.
Runtime datasets and trained models must be supplied separately; see `data/README.md`.

Build with `docker build -t tool-spectral-clustering .`.
