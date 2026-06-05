#!/usr/bin/env python3
"""Download Norwegian and Swedish price-zone polygons from Electricity Maps.

Electricity Maps exposes zone keys through the API, while the offline
zone-finder repository contains the zone geometries. This script reads the
official generated geometry file and writes a filtered GeoJSON file for the
Norwegian and Swedish bidding zones used in this project.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml


LOGGER = logging.getLogger(__name__)

ZONE_GEOMETRY_URL = "https://raw.githubusercontent.com/electricitymaps/zone-finder/main/geo.generated.json"
ZONES_API_URL = "https://api.electricitymaps.com/v3/zones"
DEFAULT_ZONES = (
    "NO-NO1",
    "NO-NO2",
    "NO-NO3",
    "NO-NO4",
    "NO-NO5",
    "SE-SE1",
    "SE-SE2",
    "SE-SE3",
    "SE-SE4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Electricity Maps polygons for Norwegian and Swedish price zones."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/electricitymaps_no_se_price_zones.geojson"),
        help="GeoJSON output path.",
    )
    parser.add_argument(
        "--source-url",
        default=ZONE_GEOMETRY_URL,
        help="Electricity Maps zone-finder generated geometry URL.",
    )
    parser.add_argument(
        "--zones",
        nargs="+",
        default=list(DEFAULT_ZONES),
        help="Electricity Maps zone keys to export. Comma-separated values are also accepted.",
    )
    parser.add_argument(
        "--validate-api",
        action="store_true",
        help="Also check selected zones against the Electricity Maps /v3/zones endpoint.",
    )
    parser.add_argument(
        "--api-config",
        type=Path,
        default=Path("config/electricity_map.yaml"),
        help="YAML file containing API_KEY for --validate-api.",
    )
    parser.add_argument(
        "--api-token-env",
        default="ELECTRICITYMAPS_API_TOKEN",
        help="Fallback environment variable containing an Electricity Maps API token for --validate-api.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Write indented GeoJSON. By default the output is compact.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser.parse_args()


def fetch_json(url: str, headers: dict[str, str] | None = None) -> dict:
    request_headers = {"User-Agent": "emps-electricitymaps-zone-downloader"}
    request_headers.update(headers or {})
    request = Request(url, headers=request_headers)
    try:
        with urlopen(request, timeout=60) as response:
            return json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} while fetching {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not fetch {url}: {exc.reason}") from exc


def normalize_zones(raw_zones: list[str]) -> list[str]:
    zones: list[str] = []
    for raw_zone in raw_zones:
        zones.extend(part.strip().upper() for part in raw_zone.split(",") if part.strip())
    return list(dict.fromkeys(zones))


def read_api_key(config_path: Path) -> str | None:
    if not config_path.exists():
        return None

    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    api_key = config.get("API_KEY")
    if api_key in (None, "", "<KEY>", "<value>"):
        return None
    return str(api_key)


def validate_zones_against_api(zones: list[str], config_path: Path, token_env: str) -> None:
    headers = {}
    token = read_api_key(config_path) or os.getenv(token_env)
    if token:
        headers["auth-token"] = token

    try:
        api_zones = fetch_json(ZONES_API_URL, headers=headers)
    except RuntimeError as exc:
        if not token or "HTTP 403" not in str(exc):
            raise

        LOGGER.warning(
            "Electricity Maps /v3/zones rejected the configured API token; retrying public zone lookup without it."
        )
        api_zones = fetch_json(ZONES_API_URL)

    missing = sorted(set(zones) - set(api_zones))
    if missing and token:
        LOGGER.warning(
            "Electricity Maps /v3/zones did not expose selected zones with the configured API token; "
            "retrying public zone lookup without it."
        )
        api_zones = fetch_json(ZONES_API_URL)
        missing = sorted(set(zones) - set(api_zones))

    if missing:
        raise RuntimeError(f"Selected zones are not present in {ZONES_API_URL}: {', '.join(missing)}")

    LOGGER.info("Validated %d zone keys against %s", len(zones), ZONES_API_URL)


def build_feature_collection(source_data: dict, zones: list[str], source_url: str) -> dict:
    zone_geometries = source_data.get("zoneToGeometryFeatures")
    if not isinstance(zone_geometries, dict):
        raise RuntimeError("Geometry source does not contain 'zoneToGeometryFeatures'.")

    features: list[dict] = []
    missing_zones: list[str] = []

    for zone in zones:
        zone_features = zone_geometries.get(zone)
        if not zone_features:
            missing_zones.append(zone)
            continue

        country_code, price_area = zone.split("-", maxsplit=1)
        for feature_index, feature in enumerate(zone_features):
            if feature.get("type") != "Feature":
                raise RuntimeError(f"Unexpected geometry entry for {zone}: {feature.get('type')}")

            output_feature = copy.deepcopy(feature)
            properties = output_feature.setdefault("properties", {})
            properties.update(
                {
                    "electricitymaps_zone": zone,
                    "country_code": country_code,
                    "price_area": price_area,
                    "feature_index": feature_index,
                    "source": source_url,
                }
            )
            features.append(output_feature)

    if missing_zones:
        raise RuntimeError(f"No geometry found for zones: {', '.join(missing_zones)}")

    return {
        "type": "FeatureCollection",
        "name": "electricitymaps_no_se_price_zones",
        "properties": {
            "source": source_url,
            "zones": zones,
        },
        "features": features,
    }


def write_geojson(feature_collection: dict, output_path: Path, pretty: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            feature_collection,
            f,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
        )
        f.write("\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s")

    zones = normalize_zones(args.zones)
    if args.validate_api:
        validate_zones_against_api(zones, args.api_config, args.api_token_env)

    LOGGER.info("Downloading Electricity Maps zone geometry from %s", args.source_url)
    source_data = fetch_json(args.source_url)
    feature_collection = build_feature_collection(source_data, zones, args.source_url)
    write_geojson(feature_collection, args.output, args.pretty)

    exported_zones = sorted({feature["properties"]["electricitymaps_zone"] for feature in feature_collection["features"]})
    LOGGER.info(
        "Wrote %d polygon feature(s) for %d zone(s) to %s",
        len(feature_collection["features"]),
        len(exported_zones),
        args.output,
    )


if __name__ == "__main__":
    main()
