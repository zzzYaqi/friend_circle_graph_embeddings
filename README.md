# Friend-Circle Graph Embeddings

> Started from a personal question: *what does my friend circle actually
> look like as a graph if the only signal is "who plays which games"?*
> I surveyed 30+ friends across 36 video games, built two graphs from
> the raw responses, and trained three classical graph-embedding
> algorithms — and the most interesting finding came out of the
> embedding **without me asking for it**.
>
>

---

 ![Django + D3 playground — graph on the left, hyper-parameter controls in the middle, live embedding scatter on the right with genre legend](pic/analysis.png)


## Headline finding — the embedding rediscovered genre

![Game embeddings projected to 2D, colour-coded by genre — Action / Role-Playing / Strategy / Adventure / Simulation / Sports & Racing](pic/embedding-result.png)

Trained only on the **co-play graph** (two games share an edge if ≥3
of the same friends like both) — **no genre labels, no metadata, no
descriptions** — the learned 2D projection separates the six gaming
genres into distinguishable clusters anyway. The random-walk structure
of "who plays what together" implicitly encodes the same taste
grouping the industry uses to categorise games.

This is the textbook payoff of unsupervised representation learning:
the model finds structure no one told it to look for.

---

## The dataset (custom-collected)

I asked **30+ friends** to mark which of **36 video games** they
enjoyed, then constructed two graphs from the raw survey:

| Graph | Nodes | Edge rule |
|---|---|---|
| **Game graph** (`data/gameNode.csv`) | 36 games | two games share an edge if **≥3 of the same friends** like both |
| **User graph** (`data/userNode.csv`) | 30+ friends | two users share an edge if they overlap enough on game taste |

Genre labels were captured per game but **held out from training** —
they're used only at evaluation time, to test whether the unsupervised
embedding recovers them (which it does, see above).

---

## Methods — three classical graph-embedding algorithms

All three implemented from scratch / from their canonical libraries
and benchmarked on the same dataset and the same splits, so the
comparison is apples-to-apples:

| Algorithm | Family | Key hyper-parameters |
|---|---|---|
| **DeepWalk** | Uniform random walks → skip-gram Word2Vec | `walks=30, length=5, dim=128, window=15, epochs=10` |
| **Node2Vec** | Biased 2nd-order random walks → Word2Vec | `walks=41, length=15, dim=32, p=0.25, q=0.5, epochs=50` |
| **Spectral (Laplacian Eigenmaps)** | Top-k eigenvectors of the graph Laplacian | `n_components=3, affinity=nearest_neighbors` |

The `p` (return) and `q` (in-out) parameters for Node2Vec were chosen
to bias toward DFS-like exploration of the sparse friend graph;
DeepWalk's longer embedding dimension reflects that uniform walks need
more capacity to capture the same neighbourhood signal.

---

## Downstream tasks

### 1. Link prediction

Held-out edges are scored by **cosine similarity** of the two
endpoints' embeddings; evaluated with **ROC AUC** and precision. The
figure below shows a successfully predicted edge between user 8 and
user 0 (red) — that edge was not in the training graph but the
embedding inferred it from the surrounding structure:

![Held-out edge prediction on the user graph — predicted link in red](pic/Link%20prediction.png)

### 2. Node clustering

**KMeans** (k=6) on the embeddings, projected to 2D with PCA for
visualisation. Clusters correspond loosely to the implicit play-
together communities in the friend group:

![KMeans clustering of nodes — six colour-coded communities recovered](pic/cluster-sp.png)

---

## Interactive playground (Django + D3 + ECharts)

The `graphEmbWeb/` Django app turns the whole pipeline into a live
demo. You can:

* pick an algorithm (DeepWalk / Node2Vec / Spectral)
* live-tune each algorithm's hyper-parameters (walk length, num walks,
  dimensions, window size, epochs) and **re-embed in real time**
* watch random-walk paths animate across the graph
* see the resulting embedding redraw, colour-coded by genre

![Django + D3 playground — graph on the left, hyper-parameter controls in the middle, live embedding scatter on the right with genre legend](pic/analysis.png)

---

## Tech stack

Python · NumPy · pandas · NetworkX · gensim (Word2Vec for DeepWalk /
Node2Vec) · scikit-learn (SpectralEmbedding, KMeans, PCA, metrics) ·
Matplotlib · Django (interactive demo) · D3.js + ECharts (in-page
animation + interactive charts).

---

## Project structure

```
.
├── Deepwalk.py       # Standalone DeepWalk training + PCA scatter plot
├── cluster.py        # Embed → KMeans → 2D projection → plot
├── link_pre.py       # Embed → cosine-sim → ROC-AUC / precision comparison
├── Display.py        # NetworkX spring-layout visualisation of the raw graph
├── data/             # The two custom CSV node files (raw survey output)
├── pic/              # Result figures used in this README
└── graphEmbWeb/      # Django app — interactive D3 + ECharts playground
    ├── manage.py
    ├── graphEmbWeb/  # Project package: settings, urls, views, algorithm modules
    ├── templates/
    └── static/
```

---

## Run it

```bash
pip install -r requirements.txt

# Reproduce the algorithm comparison plots
python Deepwalk.py
python cluster.py
python link_pre.py

# Or launch the Django playground
cd graphEmbWeb
python manage.py migrate         # first time only — recreates the local SQLite DB
python manage.py runserver
```

Then open <http://127.0.0.1:8000/>.
