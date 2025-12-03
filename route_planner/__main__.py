"""Planejamento de rotas para k caminhões usando OSMnx, K-Means e 2-opt."""

import argparse
import secrets
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import folium
import networkx as nx
import numpy as np
import osmnx as ox

from .clustering import cluster_deliveries
from .models import Delivery, Truck
from .route_optimizer import build_baseline_route, build_route, haversine

OUTPUT_DIR = Path("output")
CITY_CENTER = (-23.550520, -46.633308)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    trucks = generate_trucks(args.num_trucks, rng, CITY_CENTER, args.radius_m)
    deliveries = generate_deliveries(
        args.num_deliveries, rng, CITY_CENTER, args.radius_m, trucks
    )

    graph = build_graph(trucks + deliveries, dist=args.radius_m)

    baseline_clusters = initial_cluster_assignment(deliveries, len(trucks))
    baseline_centroids = compute_centroids(baseline_clusters)
    baseline_assignments = match_clusters_to_trucks(trucks, baseline_centroids)

    clusters, centroids = cluster_deliveries(deliveries, len(trucks))
    assignments = match_clusters_to_trucks(trucks, centroids)

    folium_map = build_base_map(trucks, deliveries)
    points_layer = folium.FeatureGroup(name="Pontos (sem rotas)", show=True)
    baseline_cluster_layer = folium.FeatureGroup(
        name="Clusters - distribuição inicial", show=False
    )
    kmeans_cluster_layer = folium.FeatureGroup(name="Clusters - K-Means", show=False)
    baseline_routes_layer = folium.FeatureGroup(
        name="Rotas - ordem sequencial", show=False
    )
    routes_layer = folium.FeatureGroup(name="Rotas - 2-opt", show=True)

    for layer in (
        points_layer,
        baseline_cluster_layer,
        kmeans_cluster_layer,
        baseline_routes_layer,
        routes_layer,
    ):
        layer.add_to(folium_map)

    palette = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "darkred",
        "cadetblue",
        "darkgreen",
    ]
    color_lookup = {
        truck.name: palette[idx % len(palette)] for idx, truck in enumerate(trucks)
    }

    cluster_stats: List[Tuple[str, float, float]] = []
    route_stats: List[Tuple[str, float, float, Sequence[str]]] = []

    for truck in trucks:
        color = color_lookup[truck.name]
        baseline_cluster_id = baseline_assignments.get(truck.name)
        baseline_deliveries = (
            baseline_clusters.get(baseline_cluster_id, [])
            if baseline_cluster_id is not None
            else []
        )
        current_cluster_id = assignments.get(truck.name)
        selected_deliveries = (
            clusters.get(current_cluster_id, [])
            if current_cluster_id is not None
            else []
        )

        plot_markers(points_layer, truck, selected_deliveries, "gray")
        plot_markers(routes_layer, truck, selected_deliveries, color)
        plot_cluster_connections(
            baseline_cluster_layer,
            truck,
            baseline_deliveries,
            color,
            dash_array="6,6",
            label_prefix="Baseline",
        )
        plot_cluster_connections(
            kmeans_cluster_layer,
            truck,
            selected_deliveries,
            color,
            label_prefix="K-Means",
        )

        baseline_cost = assignment_cost(truck, baseline_deliveries)
        optimized_cost = assignment_cost(truck, selected_deliveries)
        cluster_stats.append((truck.name, baseline_cost, optimized_cost))

        baseline_route = build_baseline_route(graph, truck, selected_deliveries)
        optimized_route = build_route(graph, truck, selected_deliveries)
        route_stats.append(
            (
                truck.name,
                baseline_route.distance_m,
                optimized_route.distance_m,
                optimized_route.labels,
            )
        )

        plot_route(
            baseline_routes_layer,
            graph,
            baseline_route.nodes,
            color,
            truck.name,
            dash_array="6,3",
        )
        plot_route(routes_layer, graph, optimized_route.nodes, color, truck.name)
        print_summary(truck.name, optimized_route.labels)

    print_cluster_stats(cluster_stats)
    print_route_stats(route_stats)

    folium.LayerControl(collapsed=False).add_to(folium_map)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "map.html"
    folium_map.save(str(output_file))
    print(f"Mapa salvo em {output_file}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Planejamento de rotas em SP")
    parser.add_argument(
        "--num-trucks",
        type=int,
        default=3,
        help="Número de caminhões/pontos de partida",
    )
    parser.add_argument(
        "--num-deliveries",
        type=int,
        default=12,
        help="Número de pontos de entrega",
    )
    parser.add_argument(
        "--radius-m",
        type=int,
        default=12000,
        help="Raio (em metros) para baixar o grafo e gerar pontos",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semente para geração pseudo-aleatória (aleatória se não fornecida)",
    )
    args = parser.parse_args()
    if args.num_trucks <= 0:
        parser.error("--num-trucks deve ser maior que zero")
    if args.num_deliveries < args.num_trucks:
        parser.error("--num-deliveries deve ser >= --num-trucks")
    if args.radius_m <= 0:
        parser.error("--radius-m deve ser positivo")
    if args.seed is None:
        args.seed = secrets.randbits(32)
        print(f"Semente aleatória escolhida: {args.seed}")
    return args


def generate_trucks(
    num_trucks: int,
    rng: np.random.Generator,
    center: Tuple[float, float],
    radius_m: int,
) -> List[Truck]:
    lat_center, lon_center = center
    lat_offset, lon_offset = _degree_offsets(radius_m, lat_center)
    trucks = []
    for idx in range(num_trucks):
        lat = lat_center + rng.uniform(-lat_offset, lat_offset)
        lon = lon_center + rng.uniform(-lon_offset, lon_offset)
        trucks.append(Truck(name=f"Caminhão {idx + 1}", lat=lat, lon=lon))
    return trucks


def generate_deliveries(
    num_deliveries: int,
    rng: np.random.Generator,
    center: Tuple[float, float],
    radius_m: int,
    trucks: Sequence[Truck],
) -> List[Delivery]:
    lat_center, lon_center = center
    lat_offset, lon_offset = _degree_offsets(radius_m, lat_center)
    deliveries: List[Delivery] = []
    for idx in range(num_deliveries):
        base_lat = lat_center + rng.uniform(-lat_offset, lat_offset)
        base_lon = lon_center + rng.uniform(-lon_offset, lon_offset)
        reference = rng.choice(trucks)
        lat = (base_lat + reference.lat) / 2
        lon = (base_lon + reference.lon) / 2
        deliveries.append(Delivery(code=f"ET{idx + 1:02d}", lat=lat, lon=lon))
    return deliveries


def _degree_offsets(radius_m: int, lat_center: float) -> Tuple[float, float]:
    lat_offset = radius_m / 111_320
    lon_offset = radius_m / (111_320 * np.cos(np.radians(lat_center)))
    return lat_offset, lon_offset


def build_graph(points: Sequence[object], *, dist: int = 12000) -> nx.MultiDiGraph:
    coords = np.array([(p.lat, p.lon) for p in points])
    center_lat, center_lon = coords.mean(axis=0)
    graph = ox.graph_from_point(
        (center_lat, center_lon), dist=dist, network_type="drive"
    )
    return graph


def match_clusters_to_trucks(
    trucks: Sequence[Truck], centroids: np.ndarray
) -> Dict[str, int]:
    pairs: List[Tuple[float, int, int]] = []
    for cluster_idx, (lat, lon) in enumerate(centroids):
        if np.isnan(lat) or np.isnan(lon):
            continue
        for truck_idx, truck in enumerate(trucks):
            distance = haversine((lat, lon), (truck.lat, truck.lon))
            pairs.append((distance, cluster_idx, truck_idx))

    assignments: Dict[str, int] = {}
    used_clusters: set[int] = set()
    used_trucks: set[int] = set()
    for _, cluster_idx, truck_idx in sorted(pairs, key=lambda item: item[0]):
        if cluster_idx in used_clusters or truck_idx in used_trucks:
            continue
        assignments[trucks[truck_idx].name] = cluster_idx
        used_clusters.add(cluster_idx)
        used_trucks.add(truck_idx)
    return assignments


def build_base_map(
    trucks: Sequence[Truck], deliveries: Sequence[Delivery]
) -> folium.Map:
    coords = np.array(
        [(d.lat, d.lon) for d in deliveries] + [(t.lat, t.lon) for t in trucks]
    )
    center_lat, center_lon = coords.mean(axis=0)
    folium_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    return folium_map


def plot_markers(
    layer: folium.FeatureGroup,
    truck: Truck,
    deliveries: Sequence[Delivery],
    color: str,
) -> None:
    folium.Marker(
        location=[truck.lat, truck.lon],
        popup=f"Depósito {truck.name}",
        icon=folium.Icon(color=color, icon="truck", prefix="fa"),
    ).add_to(layer)

    for delivery in deliveries:
        folium.CircleMarker(
            location=[delivery.lat, delivery.lon],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            popup=f"Entrega {delivery.code}",
        ).add_to(layer)


def plot_route(
    layer: folium.FeatureGroup,
    graph: nx.MultiDiGraph,
    route_nodes: Sequence[int],
    color: str,
    truck_name: str,
    dash_array: str | None = None,
) -> None:
    path_coords: List[Tuple[float, float]] = []
    for start, end in zip(route_nodes, route_nodes[1:]):
        try:
            segment = ox.shortest_path(graph, start, end, weight="length")
        except nx.NetworkXNoPath:
            segment = None

        if segment is None:
            start_coord = (graph.nodes[start]["y"], graph.nodes[start]["x"])
            end_coord = (graph.nodes[end]["y"], graph.nodes[end]["x"])
            if not path_coords or path_coords[-1] != start_coord:
                path_coords.append(start_coord)
            path_coords.append(end_coord)
            continue

        for idx, node in enumerate(segment):
            lat = graph.nodes[node]["y"]
            lon = graph.nodes[node]["x"]
            if idx and path_coords and path_coords[-1] == (lat, lon):
                continue
            path_coords.append((lat, lon))

    folium.PolyLine(
        locations=path_coords,
        color=color,
        tooltip=f"Rota {truck_name}",
        weight=4,
        dash_array=dash_array,
    ).add_to(layer)


def print_summary(truck_name: str, ordered_points: Sequence[str]) -> None:
    readable = " -> ".join(ordered_points)
    print(f"Rota otimizada ({truck_name}): {readable}")


def plot_cluster_connections(
    layer: folium.FeatureGroup,
    truck: Truck,
    deliveries: Sequence[Delivery],
    color: str,
    *,
    dash_array: str | None = None,
    label_prefix: str,
) -> None:
    if not deliveries:
        return
    for delivery in deliveries:
        folium.PolyLine(
            locations=[(truck.lat, truck.lon), (delivery.lat, delivery.lon)],
            color=color,
            weight=2,
            opacity=0.8,
            dash_array=dash_array,
            tooltip=f"{label_prefix}: {truck.name} -> {delivery.code}",
        ).add_to(layer)


def initial_cluster_assignment(
    deliveries: Sequence[Delivery], n_clusters: int
) -> Dict[int, List[Delivery]]:
    clusters: Dict[int, List[Delivery]] = {idx: [] for idx in range(n_clusters)}
    for idx, delivery in enumerate(deliveries):
        clusters[idx % n_clusters].append(delivery)
    return clusters


def compute_centroids(clusters: Dict[int, Sequence[Delivery]]) -> np.ndarray:
    centroids = np.zeros((len(clusters), 2), dtype=float)
    for cluster_idx, cluster_deliveries in clusters.items():
        if cluster_deliveries:
            coords = np.array([(d.lat, d.lon) for d in cluster_deliveries])
            centroids[cluster_idx] = coords.mean(axis=0)
        else:
            centroids[cluster_idx] = np.array([np.nan, np.nan])
    return centroids


def assignment_cost(truck: Truck, deliveries: Sequence[Delivery]) -> float:
    return float(
        sum(haversine((truck.lat, truck.lon), (d.lat, d.lon)) for d in deliveries)
    )


def print_cluster_stats(stats: Sequence[Tuple[str, float, float]]) -> None:
    if not stats:
        return
    print("\n== Agrupamento: baseline vs K-Means ==")
    total_baseline = sum(b for _, b, _ in stats)
    total_optimized = sum(o for _, _, o in stats)
    for truck_name, baseline_cost, optimized_cost in stats:
        improvement = baseline_cost - optimized_cost
        pct = (improvement / baseline_cost * 100) if baseline_cost else 0.0
        print(
            f"{truck_name}: {baseline_cost / 1000:.2f} km -> {optimized_cost / 1000:.2f} km "
            f"(Δ {improvement / 1000:.2f} km, {pct:.1f}%)"
        )
    print(
        f"Total: {total_baseline / 1000:.2f} km -> {total_optimized / 1000:.2f} km "
        f"(Δ {(total_baseline - total_optimized) / 1000:.2f} km)"
    )


def print_route_stats(stats: Sequence[Tuple[str, float, float, Sequence[str]]]) -> None:
    if not stats:
        return
    print("\n== Rotas: antes vs 2-opt ==")
    total_baseline = sum(b for _, b, _, _ in stats)
    total_optimized = sum(o for _, _, o, _ in stats)
    for truck_name, baseline_len, optimized_len, ordered_points in stats:
        improvement = baseline_len - optimized_len
        pct = (improvement / baseline_len * 100) if baseline_len else 0.0
        readable = " -> ".join(ordered_points)
        print(
            f"{truck_name}: {baseline_len / 1000:.2f} km -> {optimized_len / 1000:.2f} km "
            f"(Δ {improvement / 1000:.2f} km, {pct:.1f}%) | Sequência: {readable}"
        )
    print(
        f"Total: {total_baseline / 1000:.2f} km -> {total_optimized / 1000:.2f} km "
        f"(Δ {(total_baseline - total_optimized) / 1000:.2f} km)"
    )


if __name__ == "__main__":
    main()
