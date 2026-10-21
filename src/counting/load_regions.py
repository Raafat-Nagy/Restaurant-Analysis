import json
from pathlib import Path


def load_zones(path: Path) -> list[dict]:
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Regions config not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    zones = data.get("zones")

    if not isinstance(zones, list) or not zones:
        raise ValueError(f"'zones' must be a non-empty list in {path}")

    names = set()

    for zone in zones:
        if not isinstance(zone, dict):
            raise ValueError("Each zone must be an object.")

        name = zone.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each zone must have a non-empty name.")

        if name in names:
            raise ValueError(f"Duplicate zone name: {name}")
        names.add(name)

        polygon_keys = ["counting_polygon"]
        if "display_polygon" in zone:
            polygon_keys.append("display_polygon")

        for key in polygon_keys:
            polygon = zone.get(key)

            if not isinstance(polygon, list) or len(polygon) < 3:
                raise ValueError(
                    f"Zone '{name}': '{key}' must contain at least 3 points."
                )

            for point in polygon:
                if (
                    not isinstance(point, list)
                    or len(point) != 2
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        for value in point
                    )
                ):
                    raise ValueError(
                        f"Zone '{name}': invalid point in '{key}': {point}"
                    )

        maximum = zone.get("max")
        if (
            isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or maximum <= 0
        ):
            raise ValueError(
                f"Zone '{name}': 'max' must be a positive integer."
            )

    return zones
