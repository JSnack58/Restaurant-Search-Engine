"""
Embedding Cluster Visualiser
=============================
Encodes a sample of queries and business texts, reduces to 2D with t-SNE
(or PCA for a faster preview), and writes an interactive HTML scatter plot.

Hover over any point to see the full text.  Queries and businesses are
plotted with distinct colours; businesses are further split by query_tier
so you can see whether templates from different tiers cluster differently.

Usage
-----
    # Fine-tuned model (default)
    python scripts/plot_embeddings.py

    # Compare against the unfine-tuned base model
    python scripts/plot_embeddings.py --model_dir bert-base-uncased --out plots/base_model.html

    # Faster preview using PCA instead of t-SNE
    python scripts/plot_embeddings.py --method pca
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.Finetuning.config import ESConfig
from scripts.Finetuning.es_dataset import ElasticSearchDataset


def _stream_sample(es_cfg: ESConfig, n: int) -> list[dict]:
    dataset = ElasticSearchDataset(
        es_host=es_cfg.host,
        index_name=es_cfg.training_pairs_index,
        split="test",
        es_batch_size=256,
        shuffle_buffer=n,
        max_samples=n,
    )
    return list(dataset)


def _reduce(embeddings: np.ndarray, method: str, n_components: int = 2) -> np.ndarray:
    if method == "pca":
        return PCA(n_components=n_components).fit_transform(embeddings)
    return TSNE(
        n_components=n_components,
        perplexity=30,
        random_state=42,
        init="pca",
        learning_rate="auto",
    ).fit_transform(embeddings)


def run(model_dir: str, es_cfg: ESConfig, n_samples: int, method: str, out: Path) -> None:
    print(f"Loading model: {model_dir}")
    model = SentenceTransformer(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print(f"Streaming {n_samples} docs from ES…")
    docs = _stream_sample(es_cfg, n_samples)

    queries       = [d["query"]         for d in docs]
    biz_texts     = [d["business_text"] for d in docs]
    tiers         = [d.get("query_tier", "unknown") for d in docs]

    all_texts = queries + biz_texts
    labels    = ["query"] * len(queries) + tiers  # colour by type + tier

    print(f"Encoding {len(all_texts)} texts…")
    with torch.no_grad():
        emb = model.encode(all_texts, convert_to_tensor=True, device=device, show_progress_bar=True)
        emb = F.normalize(emb, dim=-1).cpu().numpy()

    print(f"Reducing to 2D with {method.upper()}…")
    coords = _reduce(emb, method)

    # ── Build plotly figure ───────────────────────────────────────────────
    colour_map = {
        "query":            "#636EFA",
        "vague_templates":  "#EF553B",
        "atmospheric_tuning": "#00CC96",
        "price_tuning":     "#AB63FA",
        "unknown":          "#FFA15A",
    }

    fig = go.Figure()
    for label in sorted(set(labels)):
        mask = [i for i, l in enumerate(labels) if l == label]
        fig.add_trace(go.Scatter(
            x=coords[mask, 0],
            y=coords[mask, 1],
            mode="markers",
            name=label,
            marker=dict(
                size=5 if label != "query" else 8,
                color=colour_map.get(label, "#19D3F3"),
                opacity=0.7,
                symbol="circle" if label != "query" else "diamond",
            ),
            text=[all_texts[i] for i in mask],
            hovertemplate="<b>%{text}</b><extra></extra>",
        ))

    fig.update_layout(
        title=f"Embedding clusters — {Path(model_dir).name} ({method.upper()}, n={n_samples})",
        xaxis_title="dim 1",
        yaxis_title="dim 2",
        legend_title="Type / Tier",
        hovermode="closest",
        width=1200,
        height=800,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out))
    print(f"Plot saved → {out}")
    print("Open in your browser to explore interactively.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_dir", default="models/checkpoints/best_model")
    p.add_argument("--n_samples", type=int, default=2_000)
    p.add_argument("--method",    choices=["tsne", "pca"], default="tsne")
    p.add_argument("--out",       type=Path, default=Path("plots/embeddings.html"))
    args = p.parse_args()

    run(args.model_dir, ESConfig(), args.n_samples, args.method, args.out)
