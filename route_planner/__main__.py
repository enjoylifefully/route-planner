"""Planejamento de rotas para k caminhões usando OSMnx, K-means e 2-opt."""

import argparse
import os
import secrets
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import folium
import networkx as nx
import numpy as np
import osmnx as ox
from scipy.optimize import linear_sum_assignment

from .clustering import cluster_deliveries
from .models import Delivery, Truck
from .route_optimizer import RouteResult, build_baseline_route, build_route, haversine

OUTPUT_DIR = Path("output")
CENTER = (-23.550520, -46.633308)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    ox.settings.use_cache = True

    print("[1/4] Construindo grafo")

    trucks = generate_trucks(args.num_trucks, rng, CENTER, args.radius_m)
    deliveries = generate_deliveries(
        args.num_deliveries, rng, CENTER, args.radius_m, trucks
    )

    graph = build_graph(center=CENTER, dist=args.radius_m)

    baseline_clusters = initial_cluster_assignment(deliveries, len(trucks))
    baseline_centroids = compute_centroids(baseline_clusters)
    baseline_assignments = match_clusters_to_trucks(trucks, baseline_centroids)

    print("[2/4] K-means")
    clusters, centroids = cluster_deliveries(
        deliveries, len(trucks), random_state=args.seed
    )
    assignments = match_clusters_to_trucks(trucks, centroids)

    folium_map = build_base_map(trucks, deliveries)
    points_layer = folium.FeatureGroup(name="Pontos (sem rotas)", show=True)
    baseline_cluster_layer = folium.FeatureGroup(
        name="Clusters - distribuição inicial", show=False
    )
    kmeans_cluster_layer = folium.FeatureGroup(name="Clusters - K-means", show=False)
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

    color_lookup = build_color_lookup(trucks)

    cluster_stats: List[Tuple[int, float, float]] = []
    route_stats: List[Tuple[int, float, float, Sequence[str]]] = []
    deliveries_lookup: Dict[int, List[Delivery]] = {}

    for truck in trucks:
        color = color_lookup[truck.code]
        baseline_cluster_id = baseline_assignments.get(truck.code)
        baseline_deliveries = (
            baseline_clusters.get(baseline_cluster_id, [])
            if baseline_cluster_id is not None
            else []
        )
        current_cluster_id = assignments.get(truck.code)
        selected_deliveries = (
            clusters.get(current_cluster_id, [])
            if current_cluster_id is not None
            else []
        )

        deliveries_lookup[truck.code] = selected_deliveries
        plot_markers(points_layer, truck, selected_deliveries, "gray")
        plot_cluster_connections(
            baseline_cluster_layer,
            truck,
            baseline_deliveries,
            color,
            dash_array="6,6",
            label_prefix="Random",
        )
        plot_cluster_connections(
            kmeans_cluster_layer,
            truck,
            selected_deliveries,
            color,
            label_prefix="K-means",
        )

        baseline_cost = assignment_cost(truck, baseline_deliveries)
        optimized_cost = assignment_cost(truck, selected_deliveries)
        cluster_stats.append((truck.code, baseline_cost, optimized_cost))
    truck_by_code = {t.code: t for t in trucks}

    def compute_routes(
        job: Tuple[int, List[Delivery]],
    ) -> Tuple[int, RouteResult, RouteResult]:
        code, assigned_deliveries = job
        truck = truck_by_code[code]
        baseline_route = build_baseline_route(graph, truck, assigned_deliveries)
        optimized_route = build_route(graph, truck, assigned_deliveries)
        return code, baseline_route, optimized_route

    jobs = [(truck.code, deliveries_lookup.get(truck.code, [])) for truck in trucks]
    total = len(jobs)

    sys.stdout.write(f"\r[3/4] 2-opt em paralelo (0/{total})")
    sys.stdout.flush()

    max_workers = min(len(trucks), (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(compute_routes, job): job[0] for job in jobs}
        completed = 0
        for future in as_completed(future_map):
            code, baseline_route, optimized_route = future.result()
            truck = truck_by_code[code]
            color = color_lookup[code]
            selected_deliveries = deliveries_lookup.get(code, [])

            plot_markers(routes_layer, truck, selected_deliveries, color)

            route_stats.append(
                (
                    code,
                    baseline_route.distance_m,
                    optimized_route.distance_m,
                )
            )

            plot_route(
                baseline_routes_layer,
                graph,
                baseline_route.nodes,
                color,
                str(code),
                dash_array="6,3",
            )
            plot_route(routes_layer, graph, optimized_route.nodes, color, str(code))

            completed += 1
            sys.stdout.write(f"\r[3/4] 2-opt em paralelo ({completed}/{total})")
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()

    print("[4/4] Gerando mapa")
    print_cluster_stats(cluster_stats)
    print_route_stats(route_stats)

    folium.LayerControl(collapsed=False).add_to(folium_map)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = next_available_map(OUTPUT_DIR)
    folium_map.save(str(output_file))
    print(f"\nMapa salvo em {output_file}.")


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
        default=8000,
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
        trucks.append(Truck(code=idx + 1, lat=lat, lon=lon))
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
        deliveries.append(Delivery(code=idx + 1, lat=lat, lon=lon))
    return deliveries


def _degree_offsets(radius_m: int, lat_center: float) -> Tuple[float, float]:
    lat_offset = radius_m / 111_320
    lon_offset = radius_m / (111_320 * np.cos(np.radians(lat_center)))
    return lat_offset, lon_offset


def build_graph(
    center: Tuple[float, float] = CENTER,
    dist: int = 12000,
) -> nx.MultiDiGraph:
    cache_dir = OUTPUT_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_key = f"point_{dist}_{center[0]}_{center[1]}"
    cache_file = cache_dir / f"graph_{cache_key}.gpickle"

    if cache_file.exists():
        try:
            return nx.read_gpickle(cache_file)
        except Exception:
            cache_file.unlink(missing_ok=True)

    graph = ox.graph_from_point(center, dist=dist, network_type="drive")

    try:
        nx.write_gpickle(graph, cache_file)
    except Exception:
        cache_file.unlink(missing_ok=True)

    return graph


def match_clusters_to_trucks(
    trucks: Sequence[Truck], centroids: np.ndarray
) -> Dict[str, int]:
    if centroids.size == 0 or not trucks:
        return {}

    valid_mask = ~np.isnan(centroids).any(axis=1)
    if not valid_mask.any():
        return {}

    valid_centroids = centroids[valid_mask]
    truck_coords = np.array([(t.lat, t.lon) for t in trucks])
    cost_matrix = haversine_matrix(valid_centroids, truck_coords)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assignments: Dict[int, int] = {}
    valid_indices = np.nonzero(valid_mask)[0]
    for c_idx, t_idx in zip(row_ind, col_ind):
        assignments[trucks[t_idx].code] = int(valid_indices[c_idx])
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


def build_color_lookup(trucks: Sequence[Truck]) -> Dict[int, str]:
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
    return {truck.code: palette[idx % len(palette)] for idx, truck in enumerate(trucks)}


def haversine_matrix(centroids: np.ndarray, trucks: np.ndarray) -> np.ndarray:
    cent_lat = np.radians(centroids[:, 0])[:, None]
    cent_lon = np.radians(centroids[:, 1])[:, None]
    truck_lat = np.radians(trucks[:, 0])[None, :]
    truck_lon = np.radians(trucks[:, 1])[None, :]

    dlat = truck_lat - cent_lat
    dlon = truck_lon - cent_lon
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(cent_lat) * np.cos(truck_lat) * np.sin(dlon / 2) ** 2
    )
    return 2 * 6371000 * np.arcsin(np.sqrt(a))


def plot_markers(
    layer: folium.FeatureGroup,
    truck: Truck,
    deliveries: Sequence[Delivery],
    color: str,
) -> None:
    folium.Marker(
        location=[truck.lat, truck.lon],
        popup=f"Depósito {truck.code}",
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
            tooltip=f"{label_prefix}: {truck.code} -> {delivery.code}",
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


def print_cluster_stats(stats: Sequence[Tuple[int, float, float]]) -> None:
    if not stats:
        return
    print("\nAleatório vs K-means (total):")
    total_baseline = sum(b for _, b, _ in stats)
    total_optimized = sum(o for _, _, o in stats)
    delta = total_baseline - total_optimized
    pct = (delta / total_baseline * 100) if total_baseline else 0.0
    print(
        f"{total_baseline / 1000:.2f} km -> {total_optimized / 1000:.2f} km "
        f"(Δ {delta / 1000:.2f} km, {pct:.1f}%)"
    )


def print_route_stats(stats: Sequence[Tuple[int, float, float]]) -> None:
    if not stats:
        return
    print("\nSequencial vs 2-opt (total):")
    total_baseline = sum(b for _, b, _ in stats)
    total_optimized = sum(o for _, _, o in stats)
    delta = total_baseline - total_optimized
    pct = (delta / total_baseline * 100) if total_baseline else 0.0
    print(
        f"{total_baseline / 1000:.2f} km -> {total_optimized / 1000:.2f} km "
        f"(Δ {delta / 1000:.2f} km, {pct:.1f}%)"
    )


def next_available_map(output_dir: Path) -> Path:
    base = output_dir / "map.html"
    if not base.exists():
        return base

    idx = 1
    while True:
        candidate = output_dir / f"map_{idx}.html"
        if not candidate.exists():
            return candidate
        idx += 1


if __name__ == "__main__":
    main()
