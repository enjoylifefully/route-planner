"""Entidades centrais do planejamento de rotas."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Truck:
    code: int
    lat: float
    lon: float


@dataclass(frozen=True)
class Delivery:
    code: int
    lat: float
    lon: float
