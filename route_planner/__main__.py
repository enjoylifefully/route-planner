"""Planejamento de rotas para k caminhões usando OSMnx, K-Means e 2-opt."""

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import folium
import networkx as nx
import numpy as np
import osmnx as ox

from .clustering import cluster_deliveries
from .models import Delivery, Truck
from .route_optimizer import build_route, haversine

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
    clusters, centroids = cluster_deliveries(deliveries, len(trucks))
    assignments = match_clusters_to_trucks(trucks, centroids)

    folium_map = build_base_map(trucks, deliveries)
    points_layer = folium.FeatureGroup(name="Pontos (sem rotas)", show=True)
    routes_layer = folium.FeatureGroup(name="Rotas e pontos", show=False)
    points_layer.add_to(folium_map)
    routes_layer.add_to(folium_map)

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
        name: palette[idx % len(palette)] for idx, name in enumerate(assignments.keys())
    }

    for truck_name, cluster_id in assignments.items():
        truck = next(t for t in trucks if t.name == truck_name)
        selected_deliveries = clusters[cluster_id]
        route_nodes, ordered_points = build_route(graph, truck, selected_deliveries)
        color = color_lookup[truck_name]
        plot_markers(points_layer, truck, selected_deliveries, "gray")
        plot_markers(routes_layer, truck, selected_deliveries, color)
        plot_route(routes_layer, graph, route_nodes, color, truck_name)
        print_summary(truck_name, ordered_points)

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
        default=42,
        help="Semente para geração pseudo-aleatória",
    )
    args = parser.parse_args()
    if args.num_trucks <= 0:
        parser.error("--num-trucks deve ser maior que zero")
    if args.num_deliveries < args.num_trucks:
        parser.error("--num-deliveries deve ser >= --num-trucks")
    if args.radius_m <= 0:
        parser.error("--radius-m deve ser positivo")
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
) -> None:
    path_coords: List[Tuple[float, float]] = []
    for start, end in zip(route_nodes, route_nodes[1:]):
        segment = ox.shortest_path(graph, start, end, weight="length")
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
    ).add_to(layer)


def print_summary(truck_name: str, ordered_points: Sequence[str]) -> None:
    readable = " -> ".join(ordered_points)
    print(f"Rota otimizada ({truck_name}): {readable}")


if __name__ == "__main__":
    main()
