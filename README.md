# simpleGraphEmbedding

An end-to-end study of graph embeddings on a custom-built social-graph
dataset: three classical embedding algorithms, two downstream tasks
(link prediction and clustering), and a Django web app that visualises
everything interactively.

> The motivating question was personal: *what does my social network
> within video games actually look like, and is there hidden structure
> in it?* I collected my friends' tastes across 36 video games and
> built two graphs from them — then trained graph embeddings to find
> latent connections and predict game preferences.

---

## Methods

Three classical graph-embedding algorithms, implemented and compared
on the same dataset:

| Algorithm | Idea | Key parameters |
|---|---|---|
| **DeepWalk** | Uniform random walks + Word2Vec (skip-gram) | walk length, num walks, window, dim |
| **Node2Vec** | Biased 2nd-order random walks + Word2Vec | `p` (return), `q` (in-out), walk length, dim |
| **Spectral (Laplacian Eigenmaps)** | Top-k eigenvectors of the graph Laplacian | `n_components`, affinity |

## Downstream tasks

1. **Link prediction** — held-out edge prediction via cosine similarity
   of node embeddings, evaluated with **ROC AUC** and precision.
2. **Node clustering** — `KMeans` on the embeddings, projected to 2D
   with PCA / t-SNE for visualisation.

## Dataset

Two hand-crafted graphs built from a 36-game taste survey:

* `data/userNode.csv` — users (graph nodes)
* `data/gameNode.csv` — games (graph nodes)
* Edges are derived by rules described in `graphEmbWeb/graphEmbWeb/`.

## Screenshots

The Django dashboard and the analysis pages:

![Web UI](pic/web.png)
![Analysis page](pic/analysis.png)

Embedding / link-prediction / clustering result figures:

![Embedding vectors](pic/vectors.png)
![Link prediction](pic/Link%20prediction.png)
![Clustering (Spectral)](pic/cluster-sp.png)

## Tech stack

Python · NumPy · pandas · NetworkX · gensim (Word2Vec for DeepWalk /
Node2Vec) · scikit-learn (KMeans, PCA, metrics, Laplacian Eigenmaps) ·
Matplotlib · Django (visualisation web app) · D3.js + ECharts (in-page
interactivity).

## Project structure

```
.
├── Deepwalk.py       # Standalone DeepWalk training + PCA scatter plot
├── cluster.py        # Embed → KMeans → 2D projection → plot
├── link_pre.py       # Embed → cosine-sim → ROC-AUC / precision comparison
├── Display.py        # NetworkX spring-layout visualisation of the raw graph
├── data/             # The two custom CSV node files
├── pic/              # Result figures + screenshots used in this README
└── graphEmbWeb/      # Django app: interactive D3.js + ECharts dashboard
    ├── manage.py
    ├── graphEmbWeb/  # Django project package (settings, urls, views, algos)
    ├── templates/
    └── static/
```

## Run it

```bash
pip install -r requirements.txt

# Reproduce the algorithm comparison plots
python Deepwalk.py
python cluster.py
python link_pre.py

# Or launch the Django dashboard
cd graphEmbWeb
python manage.py migrate         # (first time only — recreates the local SQLite DB)
python manage.py runserver
```

Then open <http://127.0.0.1:8000/>.
