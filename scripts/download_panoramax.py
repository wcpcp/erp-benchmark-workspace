#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen


USER_AGENT = "erp-data-pipeline/0.1 panoramax downloader"


def eprint(message: str) -> None:
    print(message, file=sys.stderr)


def request_json(url: str, timeout: float, retries: int) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            sleep_seconds = min(2 ** attempt, 10)
            eprint(f"[retry] {url} failed with {exc}; retrying in {sleep_seconds}s")
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Failed to fetch JSON from {url}") from last_error


def download_file(url: str, destination: Path, timeout: float, retries: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return

    tmp_path = destination.with_suffix(destination.suffix + ".part")
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response, tmp_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            tmp_path.replace(destination)
            return
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt >= retries:
                break
            sleep_seconds = min(2 ** attempt, 10)
            eprint(f"[retry] {url} failed with {exc}; retrying in {sleep_seconds}s")
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Failed to download {url}") from last_error


def merge_query(url: str, extra_params: Dict[str, Any]) -> str:
    parsed = urlparse(url)
    params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    for key, value in extra_params.items():
        if value is None:
            continue
        params[key] = str(value)
    new_query = urlencode(params, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def normalize_api_url(api_url: str) -> str:
    return api_url.rstrip("/")


def asset_urls(feature: Dict[str, Any]) -> Dict[str, str]:
    assets = feature.get("assets", {}) or {}
    result: Dict[str, str] = {}
    for quality in ("hd", "sd", "thumb"):
        href = (assets.get(quality) or {}).get("href")
        if href:
            result[quality] = href
    return result


def flatten_semantics(records: Iterable[Dict[str, Any]]) -> List[str]:
    flattened: List[str] = []
    for record in records or []:
        key = str(record.get("key", "")).strip()
        value = str(record.get("value", "")).strip()
        if key and value:
            flattened.append(f"{key}={value}")
        elif key:
            flattened.append(key)
        elif value:
            flattened.append(value)
    return flattened


def extract_annotation_semantics(annotations: Iterable[Dict[str, Any]]) -> List[str]:
    flattened: List[str] = []
    for annotation in annotations or []:
        flattened.extend(flatten_semantics(annotation.get("semantics", []) or []))
    return flattened


def normalize_collection(collection: Dict[str, Any], api_url: str) -> Dict[str, Any]:
    temporal_intervals = (((collection.get("extent") or {}).get("temporal") or {}).get("interval") or [[]])
    spatial_boxes = (((collection.get("extent") or {}).get("spatial") or {}).get("bbox") or [[]])
    interval = temporal_intervals[0] if temporal_intervals else []
    bbox = spatial_boxes[0] if spatial_boxes else []

    return {
        "instance_api_url": api_url,
        "collection_id": collection.get("id"),
        "collection_title": collection.get("title"),
        "collection_description": collection.get("description"),
        "collection_license": collection.get("license"),
        "collection_length_km": collection.get("geovisio:length_km"),
        "collection_bbox": bbox,
        "collection_start_datetime": interval[0] if len(interval) > 0 else None,
        "collection_end_datetime": interval[1] if len(interval) > 1 else None,
        "collection_keywords": collection.get("keywords") or [],
        "collection_provider_names": [provider.get("name") for provider in collection.get("providers", []) if provider.get("name")],
        "collection_raw": collection,
    }


def normalize_item(
    feature: Dict[str, Any],
    collection_summary: Dict[str, Any],
) -> Dict[str, Any]:
    properties = feature.get("properties", {}) or {}
    exif = properties.get("exif", {}) or {}
    interior_orientation = properties.get("pers:interior_orientation", {}) or {}
    sensor_dims = interior_orientation.get("sensor_array_dimensions") or []
    geometry = feature.get("geometry", {}) or {}
    coordinates = geometry.get("coordinates") or [None, None]
    providers = [provider.get("name") for provider in feature.get("providers", []) if provider.get("name")]
    annotation_semantics = extract_annotation_semantics(properties.get("annotations", []) or [])

    return {
        "instance_api_url": collection_summary["instance_api_url"],
        "collection_id": collection_summary["collection_id"],
        "collection_title": collection_summary["collection_title"],
        "collection_description": collection_summary["collection_description"],
        "collection_license": collection_summary["collection_license"],
        "collection_length_km": collection_summary["collection_length_km"],
        "collection_bbox": collection_summary["collection_bbox"],
        "collection_start_datetime": collection_summary["collection_start_datetime"],
        "collection_end_datetime": collection_summary["collection_end_datetime"],
        "collection_keywords": collection_summary["collection_keywords"],
        "collection_provider_names": collection_summary["collection_provider_names"],
        "item_id": feature.get("id"),
        "item_type": feature.get("type"),
        "item_datetime": properties.get("datetime"),
        "item_created": properties.get("created"),
        "item_updated": properties.get("updated"),
        "license": properties.get("license") or collection_summary["collection_license"],
        "projection_type": exif.get("Xmp.GPano.ProjectionType"),
        "field_of_view": interior_orientation.get("field_of_view"),
        "sensor_width": sensor_dims[0] if len(sensor_dims) > 0 else None,
        "sensor_height": sensor_dims[1] if len(sensor_dims) > 1 else None,
        "view_azimuth": properties.get("view:azimuth"),
        "pitch": properties.get("pers:pitch"),
        "roll": properties.get("pers:roll"),
        "rank_in_collection": properties.get("geovisio:rank_in_collection"),
        "horizontal_accuracy": properties.get("quality:horizontal_accuracy"),
        "original_file_name": properties.get("original_file:name"),
        "original_file_size": properties.get("original_file:size"),
        "geometry_type": geometry.get("type"),
        "lon": coordinates[0] if len(coordinates) > 0 else None,
        "lat": coordinates[1] if len(coordinates) > 1 else None,
        "bbox": feature.get("bbox"),
        "provider_names": providers,
        "item_semantics": flatten_semantics(properties.get("semantics", []) or []),
        "collection_semantics": flatten_semantics(((properties.get("collection") or {}).get("semantics") or [])),
        "annotation_semantics": annotation_semantics,
        "assets": asset_urls(feature),
        "has_tiled_asset_template": bool((feature.get("asset_templates") or {}).get("tiles")),
        "feature_url": next((link.get("href") for link in feature.get("links", []) if link.get("rel") == "self"), None),
    }


def write_jsonl_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def collection_items_url(collection: Dict[str, Any], item_page_size: int) -> str:
    for link in collection.get("links", []):
        if link.get("rel") == "items" and link.get("href"):
            return merge_query(link["href"], {"limit": item_page_size})
    raise RuntimeError(f"Collection {collection.get('id')} is missing an items link")


def next_link(payload: Dict[str, Any]) -> Optional[str]:
    for link in payload.get("links", []) or []:
        if link.get("rel") == "next" and link.get("href"):
            return link["href"]
    return None


def crawl(args: argparse.Namespace) -> None:
    api_url = normalize_api_url(args.api_url)
    collections_url = merge_query(
        f"{api_url}/collections",
        {
            "limit": args.collection_page_size,
            "bbox": args.bbox,
            "datetime": args.datetime,
        },
    )

    manifest_path = Path(args.manifest)
    collections_path = Path(args.collections)
    if args.overwrite:
        for path in (manifest_path, collections_path):
            if path.exists():
                path.unlink()

    seen_collections = 0
    seen_items = 0
    current_collections_url: Optional[str] = collections_url

    while current_collections_url:
        payload = request_json(current_collections_url, timeout=args.timeout, retries=args.retries)
        collections = payload.get("collections", []) or []
        eprint(f"[crawl] fetched {len(collections)} collections from {current_collections_url}")
        for collection in collections:
            if args.max_collections is not None and seen_collections >= args.max_collections:
                current_collections_url = None
                break

            collection_summary = normalize_collection(collection, api_url)
            write_jsonl_row(collections_path, collection_summary)
            seen_collections += 1

            current_items_url = collection_items_url(collection, args.item_page_size)
            while current_items_url:
                items_payload = request_json(current_items_url, timeout=args.timeout, retries=args.retries)
                features = items_payload.get("features", []) or []
                eprint(
                    f"[crawl] collection {collection_summary['collection_id']} "
                    f"returned {len(features)} items from {current_items_url}"
                )
                for feature in features:
                    if args.max_items is not None and seen_items >= args.max_items:
                        current_items_url = None
                        current_collections_url = None
                        break
                    row = normalize_item(feature, collection_summary)
                    write_jsonl_row(manifest_path, row)
                    seen_items += 1

                if args.sleep_seconds:
                    time.sleep(args.sleep_seconds)
                if args.max_items is not None and seen_items >= args.max_items:
                    break
                current_items_url = next_link(items_payload)

            if args.sleep_seconds:
                time.sleep(args.sleep_seconds)

        if current_collections_url is None:
            break
        current_collections_url = next_link(payload)

    eprint(
        f"[done] wrote {seen_collections} collections to {collections_path} "
        f"and {seen_items} items to {manifest_path}"
    )


def iter_manifest_rows(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL row at line {line_number} in {path}") from exc


def file_name_for_row(row: Dict[str, Any], naming: str, quality: str) -> str:
    if naming == "original" and row.get("original_file_name"):
        return str(row["original_file_name"])
    suffix = ".jpg"
    asset_url = (row.get("assets") or {}).get(quality, "")
    parsed = urlparse(asset_url)
    if parsed.path.endswith(".png"):
        suffix = ".png"
    return f"{row['item_id']}{suffix}"


def download_row(
    row: Dict[str, Any],
    output_dir: Path,
    quality: str,
    naming: str,
    sidecar_metadata: bool,
    timeout: float,
    retries: int,
) -> str:
    assets = row.get("assets") or {}
    asset_url = assets.get(quality)
    if not asset_url:
        raise RuntimeError(f"Row {row.get('item_id')} has no {quality} asset")

    file_name = file_name_for_row(row, naming=naming, quality=quality)
    destination = output_dir / str(row["collection_id"]) / file_name
    download_file(asset_url, destination, timeout=timeout, retries=retries)

    if sidecar_metadata:
        sidecar_path = destination.with_suffix(destination.suffix + ".json")
        sidecar_path.write_text(json.dumps(row, ensure_ascii=True, indent=2), encoding="utf-8")

    return str(destination)


def download_from_manifest(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(iter_manifest_rows(manifest_path))
    if args.limit is not None:
        rows = rows[: args.limit]

    total = len(rows)
    eprint(f"[download] scheduling {total} rows from {manifest_path}")

    completed = 0
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_row = {
            executor.submit(
                download_row,
                row=row,
                output_dir=output_dir,
                quality=args.quality,
                naming=args.naming,
                sidecar_metadata=args.sidecar_metadata,
                timeout=args.timeout,
                retries=args.retries,
            ): row
            for row in rows
        }
        for future in concurrent.futures.as_completed(future_to_row):
            row = future_to_row[future]
            try:
                destination = future.result()
                completed += 1
                if completed % 100 == 0 or completed == total:
                    eprint(f"[download] completed {completed}/{total}; last={destination}")
            except Exception as exc:
                failures += 1
                eprint(f"[error] failed to download {row.get('item_id')}: {exc}")

    eprint(f"[done] downloaded {completed} files with {failures} failures into {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crawl and download Panoramax ERP imagery")
    subparsers = parser.add_subparsers(dest="command", required=True)

    crawl_parser = subparsers.add_parser(
        "crawl",
        help="Crawl collections and items from a Panoramax instance into JSONL manifests",
    )
    crawl_parser.add_argument("--api-url", required=True, help="Panoramax instance API root, e.g. https://panoramax.ign.fr/api")
    crawl_parser.add_argument("--manifest", required=True, help="Output JSONL path for normalized item rows")
    crawl_parser.add_argument("--collections", required=True, help="Output JSONL path for normalized collection rows")
    crawl_parser.add_argument("--bbox", help="Optional bbox filter: minLon,minLat,maxLon,maxLat")
    crawl_parser.add_argument("--datetime", help="Optional STAC datetime filter")
    crawl_parser.add_argument("--collection-page-size", type=int, default=100, help="Collections fetched per page")
    crawl_parser.add_argument("--item-page-size", type=int, default=200, help="Items fetched per page per collection")
    crawl_parser.add_argument("--max-collections", type=int, help="Optional cap on crawled collections")
    crawl_parser.add_argument("--max-items", type=int, help="Optional cap on crawled items")
    crawl_parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    crawl_parser.add_argument("--retries", type=int, default=3, help="Retry count for HTTP requests")
    crawl_parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional delay between requests")
    crawl_parser.add_argument("--overwrite", action="store_true", help="Remove existing output files before crawling")
    crawl_parser.set_defaults(func=crawl)

    download_parser = subparsers.add_parser(
        "download",
        help="Download image assets listed in a JSONL manifest",
    )
    download_parser.add_argument("--manifest", required=True, help="Input JSONL manifest")
    download_parser.add_argument("--output-dir", required=True, help="Directory for downloaded images")
    download_parser.add_argument("--quality", choices=("hd", "sd", "thumb"), default="hd", help="Asset quality to download")
    download_parser.add_argument("--naming", choices=("item-id", "original"), default="item-id", help="Output file naming strategy")
    download_parser.add_argument("--workers", type=int, default=8, help="Concurrent download workers")
    download_parser.add_argument("--limit", type=int, help="Optional cap on number of rows to download")
    download_parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds")
    download_parser.add_argument("--retries", type=int, default=3, help="Retry count for asset downloads")
    download_parser.add_argument("--sidecar-metadata", action="store_true", help="Write a .json sidecar next to each image")
    download_parser.set_defaults(func=download_from_manifest)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
