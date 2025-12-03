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
    labels: List[str]
    distance_m: float


def build_route(
    graph: nx.MultiDiGraph, truck: Truck, deliveries: Sequence[Delivery]
) -> RouteResult:
    """Resolve a ordem ótima de visitas usando 2-opt e retorna nós e etiquetas."""

    nodes = [nearest_node(graph, truck.lat, truck.lon)]
    ordered_codes = [f"Saída {truck.name}"]

    for delivery in deliveries:
        nodes.append(nearest_node(graph, delivery.lat, delivery.lon))
        ordered_codes.append(delivery.code)

    distance_matrix = compute_distance_matrix(graph, nodes)
    permutation, _ = solve_tsp_local_search(distance_matrix)
    ordered_indices = reorder_cycle(permutation, start_index=0)

    route_nodes = [nodes[idx] for idx in ordered_indices]
    ordered_codes = [ordered_codes[idx] for idx in ordered_indices]
    distance = compute_path_length(graph, route_nodes)
    return RouteResult(route_nodes, ordered_codes, distance)


def build_baseline_route(
    graph: nx.MultiDiGraph, truck: Truck, deliveries: Sequence[Delivery]
) -> RouteResult:
    """Gera rota simples (ordem sequencial) para comparação com 2-opt."""

    nodes = [nearest_node(graph, truck.lat, truck.lon)]
    ordered_codes = [f"Saída {truck.name}"]

    for delivery in deliveries:
        nodes.append(nearest_node(graph, delivery.lat, delivery.lon))
        ordered_codes.append(delivery.code)

    nodes.append(nodes[0])
    ordered_codes.append(ordered_codes[0])
    distance = compute_path_length(graph, nodes)
    return RouteResult(nodes, ordered_codes, distance)


def nearest_node(graph: nx.MultiDiGraph, lat: float, lon: float) -> int:
    return ox.distance.nearest_nodes(graph, lon, lat)  # type: ignore[arg-type]


def compute_distance_matrix(graph: nx.MultiDiGraph, nodes: Sequence[int]) -> np.ndarray:
    n_nodes = len(nodes)
    matrix = np.zeros((n_nodes, n_nodes), dtype=float)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            try:
                length = nx.shortest_path_length(
                    graph, nodes[i], nodes[j], weight="length"
                )
            except nx.NetworkXNoPath:
                length = haversine(
                    (graph.nodes[nodes[i]]["y"], graph.nodes[nodes[i]]["x"]),
                    (graph.nodes[nodes[j]]["y"], graph.nodes[nodes[j]]["x"]),
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
