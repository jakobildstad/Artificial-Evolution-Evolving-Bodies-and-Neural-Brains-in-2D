"""A connected pond with solid islands and shared terrain queries."""

import pymunk

WORLD_WIDTH, WORLD_HEIGHT = 1600, 1100
TERRAIN = 1
CREATURE = 2
TERRAIN_FILTER = pymunk.ShapeFilter(mask=TERRAIN)


def pond_islands(width: int, height: int) -> list[list[tuple[float, float]]]:
    """Staggered islands leave several channels, including routes for large bodies."""
    outlines = [
        [(0.30, 0), (0.38, 0), (0.39, 0.15), (0.36, 0.28), (0.32, 0.25)],
        [(0.33, 0.41), (0.38, 0.39), (0.40, 0.56), (0.37, 0.76), (0.31, 0.71)],
        [(0.32, 0.89), (0.37, 0.88), (0.39, 1), (0.30, 1)],
        [(0.64, 0), (0.72, 0), (0.70, 0.14), (0.66, 0.16)],
        [(0.66, 0.30), (0.71, 0.28), (0.73, 0.46), (0.69, 0.61), (0.64, 0.56)],
        [(0.67, 0.76), (0.72, 0.78), (0.74, 1), (0.64, 1)],
    ]
    return [[(x * width, y * height) for x, y in polygon] for polygon in outlines]


def add_islands(space: pymunk.Space, islands: list) -> None:
    for vertices in islands:
        shape = pymunk.Poly(space.static_body, vertices)
        shape.friction = 0.6
        shape.filter = pymunk.ShapeFilter(categories=TERRAIN)
        space.add(shape)


def open_water(space: pymunk.Space, position: tuple, radius: float) -> bool:
    return space.point_query_nearest(position, radius, TERRAIN_FILTER) is None


def visible(space: pymunk.Space, start: tuple, end: tuple) -> bool:
    return space.segment_query_first(start, end, 0, TERRAIN_FILTER) is None
