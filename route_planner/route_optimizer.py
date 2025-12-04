"""Rotinas relacionadas à otimização de rotas (baseline e 2-opt)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import networkx as nx
import numpy as np
import osmnx as ox
from python_tsp.heuristics import solve_tsp_local_search

from .models import Delivery, Truck


@dataclass(frozen=True)
class RouteResult:
    nodes: List[int]
    distance_m: float


def build_route(
    graph: nx.MultiDiGraph, truck: Truck, deliveries: Sequence[Delivery]
) -> RouteResult:
    """Resolve a ordem ótima de visitas usando 2-opt e retorna nós e etiquetas."""

    nodes = [nearest_node(graph, truck.lat, truck.lon)]

    for delivery in deliveries:
        nodes.append(nearest_node(graph, delivery.lat, delivery.lon))

    if len(nodes) <= 2:
        nodes = nodes + nodes[:1]
        distance = compute_path_length(graph, nodes)
        return RouteResult(nodes, distance)

    distance_matrix = compute_distance_matrix(graph, nodes)
    permutation, _ = solve_tsp_local_search(distance_matrix)
    ordered_indices = reorder_cycle(permutation, start_index=0)

    route_nodes = [nodes[idx] for idx in ordered_indices]
    distance = compute_path_length(graph, route_nodes)
    return RouteResult(route_nodes, distance)


def build_baseline_route(
    graph: nx.MultiDiGraph, truck: Truck, deliveries: Sequence[Delivery]
) -> RouteResult:
    """Gera rota simples (ordem sequencial) para comparação com 2-opt."""

    nodes = [nearest_node(graph, truck.lat, truck.lon)]

    for delivery in deliveries:
        nodes.append(nearest_node(graph, delivery.lat, delivery.lon))

    nodes.append(nodes[0])
    distance = compute_path_length(graph, nodes)
    return RouteResult(nodes, distance)


def nearest_node(graph: nx.MultiDiGraph, lat: float, lon: float) -> int:
    return ox.distance.nearest_nodes(graph, lon, lat)  # type: ignore[arg-type]


def compute_distance_matrix(graph: nx.MultiDiGraph, nodes: Sequence[int]) -> np.ndarray:
    n_nodes = len(nodes)
    matrix = np.zeros((n_nodes, n_nodes), dtype=float)
    for i, source in enumerate(nodes):
        lengths = nx.single_source_dijkstra_path_length(graph, source, weight="length")
        for j in range(i + 1, n_nodes):
            target = nodes[j]
            length = lengths.get(target)
            if length is None:
                length = haversine(
                    (graph.nodes[source]["y"], graph.nodes[source]["x"]),
                    (graph.nodes[target]["y"], graph.nodes[target]["x"]),
                )
            matrix[i, j] = matrix[j, i] = float(length)
    return matrix


def compute_path_length(graph: nx.MultiDiGraph, route_nodes: Sequence[int]) -> float:
    total = 0.0
    for start, end in zip(route_nodes, route_nodes[1:]):
        try:
            total += nx.shortest_path_length(graph, start, end, weight="length")
        except nx.NetworkXNoPath:
            total += haversine(
                (graph.nodes[start]["y"], graph.nodes[start]["x"]),
                (graph.nodes[end]["y"], graph.nodes[end]["x"]),
            )
    return float(total)


def reorder_cycle(permutation: Sequence[int], start_index: int) -> List[int]:
    start_pos = permutation.index(start_index)
    ordered = list(permutation[start_pos:]) + list(permutation[:start_pos])
    ordered.append(start_index)
    return ordered


def haversine(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = map(np.radians, a)
    lat2, lon2 = map(np.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(h))
