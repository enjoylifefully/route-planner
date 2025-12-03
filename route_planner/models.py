"""Entidades centrais do planejamento de rotas."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Truck:
    name: str
    lat: float
    lon: float


@dataclass(frozen=True)
class Delivery:
    code: str
    lat: float
    lon: float
