"""Agrupamento de entregas usando K-means."""

from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans

from .models import Delivery


def cluster_deliveries(
    deliveries: Sequence[Delivery],
    n_clusters: int,
    *,
    random_state: int = 42,
) -> Tuple[Dict[int, List[Delivery]], np.ndarray]:
    """Executa K-means e devolve clusters e centróides."""

    coords = np.array([(d.lat, d.lon) for d in deliveries])
    model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    labels = model.fit_predict(coords)
    clusters: Dict[int, List[Delivery]] = {idx: [] for idx in range(n_clusters)}
    for label, delivery in zip(labels, deliveries):
        clusters[int(label)].append(delivery)
    return clusters, model.cluster_centers_
