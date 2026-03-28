import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import ee
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import Affine
from rasterio.windows import Window
from shapely.geometry import Polygon, box

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


ArrayLike = np.ndarray
Logger = logging.Logger


@retry(tries=3, delay=1, backoff=2, jitter=(0, 1))
def _init_earth_engine(server_url: str | None = None, logger: Logger | None = None) -> None:
    """Initialize Earth Engine with service account credentials."""
    service_account = os.environ.get('GEE_SERVICE_ACCOUNT')
    private_key_path = os.environ.get('GEE_SERVICE_ACCOUNT_KEY')
    
    if not service_account or not private_key_path:
        raise ValueError("GEE_SERVICE_ACCOUNT and GEE_SERVICE_ACCOUNT_KEY environment variables must be set")
    
    credentials = ee.ServiceAccountCredentials(service_account, private_key_path)

    if server_url:
        if logger:
            logger.debug("Initializing Earth Engine at custom URL %s", server_url)
        ee.Initialize(credentials, opt_url=server_url)
    else:
        if logger:
            logger.debug("Initializing Earth Engine at default URL")
        ee.Initialize(credentials)


DEFAULT_CHUNK_SIZE = 512
DEFAULT_PREPARE_WORKERS = 10
DEFAULT_DOWNLOAD_WORKERS = 15
DEFAULT_MAX_RETRIES = 4
DEFAULT_RETRY_DELAY_SECONDS = 2.0
DEFAULT_DOWNLOAD_NODATA = np.nan


@dataclass(frozen=True, slots=True)
class GoogleEarthEngineDownloadOptions:
    chunk_size: int = DEFAULT_CHUNK_SIZE
    prepare_workers: int = DEFAULT_PREPARE_WORKERS
    download_workers: int = DEFAULT_DOWNLOAD_WORKERS
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS
    nodata: float = DEFAULT_DOWNLOAD_NODATA


@dataclass(slots=True)
class DownloadedGoogleEarthEngineImage:
    collection_id: str
    image_id: str
    image_info: dict[str, Any]
    acquisition_time_utc: datetime
    local_datetime: datetime
    output_band_names: list[str]
    tiff_path: Path
    metadata_path: Path


@dataclass(slots=True)
class _PreparedImageJob:
    collection_id: str
    image_id: str
    image_info: dict[str, Any]
    acquisition_time_utc: datetime
    local_datetime: datetime
    grid: dict[str, Any]
    row0: int
    col0: int
    band_aliases: list[str]
    output_band_names: list[str]
    out_path: Path
    metadata_path: Path
    dst: rasterio.io.DatasetWriter
    tasks: list[tuple[int, int, int, int]]


def structured_to_hwc_array(raw: np.ndarray, bands: list[str]) -> np.ndarray:
    if getattr(raw.dtype, "names", None):
        return np.stack([raw[band] for band in bands], axis=-1)
    arr = np.asarray(raw)
    if arr.ndim == 2 and len(bands) == 1:
        arr = arr[:, :, None]
    return arr

def safe_identifier(value: str) -> str:
    return value.replace("/", "_").replace(":", "_").replace(" ", "_").replace(
        "=", "_"
    )


def _transform_bounds_to_image_crs(
    bounds: tuple[float, float, float, float], dst_crs: str
) -> Polygon:
    transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    bbox_polygon = box(*bounds)
    transformed = [
        transformer.transform(x, y) for x, y in bbox_polygon.exterior.coords
    ]
    polygon = Polygon(transformed)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        raise ValueError("Failed to transform bounds into the image CRS.")
    return polygon


def _get_image_grid_info(image_info: dict[str, Any]) -> dict[str, Any]:
    band0 = image_info["bands"][0]
    transform = band0["crs_transform"]
    width, height = band0["dimensions"]
    if transform[1] != 0 or transform[3] != 0:
        raise NotImplementedError("Only north-up images without shear are supported.")

    x_scale = float(transform[0])
    y_scale = float(transform[4])
    origin_x = float(transform[2])
    origin_y = float(transform[5])
    x2 = origin_x + x_scale * width
    y2 = origin_y + y_scale * height
    return {
        "crs": band0["crs"],
        "x_scale": x_scale,
        "y_scale": y_scale,
        "pixel_w": abs(x_scale),
        "pixel_h": abs(y_scale),
        "origin_x": origin_x,
        "origin_y": origin_y,
        "width": int(width),
        "height": int(height),
        "bbox": box(
            min(origin_x, x2),
            min(origin_y, y2),
            max(origin_x, x2),
            max(origin_y, y2),
        ),
    }


def _intersection_to_window(
    intersection: Polygon, grid: dict[str, Any], chunk_size: int
) -> tuple[int, int, int, int] | None:
    if intersection.is_empty:
        return None

    minx, miny, maxx, maxy = intersection.bounds
    col_min = max(0, math.floor((minx - grid["origin_x"]) / grid["pixel_w"]))
    col_max = min(
        grid["width"], math.ceil((maxx - grid["origin_x"]) / grid["pixel_w"])
    )
    row_min = max(0, math.floor((grid["origin_y"] - maxy) / grid["pixel_h"]))
    row_max = min(
        grid["height"], math.ceil((grid["origin_y"] - miny) / grid["pixel_h"])
    )
    if col_min >= col_max or row_min >= row_max:
        return None

    col0 = (col_min // chunk_size) * chunk_size
    row0 = (row_min // chunk_size) * chunk_size
    col1 = min(grid["width"], math.ceil(col_max / chunk_size) * chunk_size)
    row1 = min(grid["height"], math.ceil(row_max / chunk_size) * chunk_size)
    return row0, row1, col0, col1


def _chunk_bbox(
    row: int, col: int, chunk_h: int, chunk_w: int, grid: dict[str, Any]
) -> Polygon:
    x1 = grid["origin_x"] + col * grid["x_scale"]
    x2 = grid["origin_x"] + (col + chunk_w) * grid["x_scale"]
    y1 = grid["origin_y"] + row * grid["y_scale"]
    y2 = grid["origin_y"] + (row + chunk_h) * grid["y_scale"]
    return box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _build_chunk_tasks(
    row0: int,
    row1: int,
    col0: int,
    col1: int,
    grid: dict[str, Any],
    intersection: Polygon,
    chunk_size: int,
) -> list[tuple[int, int, int, int]]:
    tasks: list[tuple[int, int, int, int]] = []
    for row in range(row0, row1, chunk_size):
        for col in range(col0, col1, chunk_size):
            chunk_h = min(chunk_size, row1 - row)
            chunk_w = min(chunk_size, col1 - col)
            if _chunk_bbox(row, col, chunk_h, chunk_w, grid).intersects(intersection):
                tasks.append((row, col, chunk_h, chunk_w))
    return tasks


def _build_output_profile(
    grid: dict[str, Any],
    row0: int,
    col0: int,
    width: int,
    height: int,
    band_count: int,
    nodata: float,
) -> dict[str, Any]:
    transform = Affine(
        grid["x_scale"],
        0.0,
        grid["origin_x"] + col0 * grid["x_scale"],
        0.0,
        grid["y_scale"],
        grid["origin_y"] + row0 * grid["y_scale"],
    )
    return {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": band_count,
        "dtype": "float32",
        "crs": grid["crs"],
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }


def _initialize_output_file(
    out_path: Path,
    profile: dict[str, Any],
    image_info: dict[str, Any],
    band_aliases: list[str],
    chunk_size: int,
    nodata: float,
) -> rasterio.io.DatasetWriter:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dst = rasterio.open(out_path, "w", **profile)
    dst.descriptions = tuple(band_aliases)
    dst.update_tags(
        ee_id=str(image_info.get("id", "")),
        ee_version=str(image_info.get("version", "")),
        ee_properties_json=json.dumps(image_info.get("properties", {}), default=str),
        ee_full_image_info_json=json.dumps(image_info, default=str),
    )
    for index, band_meta in enumerate(image_info.get("bands", []), start=1):
        if index <= len(band_aliases):
            dst.update_tags(index, ee_band_info_json=json.dumps(band_meta, default=str))

    for row in range(0, profile["height"], chunk_size):
        window_h = min(chunk_size, profile["height"] - row)
        for col in range(0, profile["width"], chunk_size):
            window_w = min(chunk_size, profile["width"] - col)
            dst.write(
                np.full((profile["count"], window_h, window_w), nodata, dtype=np.float32),
                window=Window(col, row, window_w, window_h),
            )
    return dst


def _parse_gee_acquisition_time(image_info: dict[str, Any]) -> datetime:
    time_start = image_info.get("properties", {}).get("system:time_start")
    if time_start is None:
        raise ValueError(f"GEE image {image_info.get('id')} has no system:time_start")
    return datetime.fromtimestamp(float(time_start) / 1000.0, tz=timezone.utc)


def _estimate_local_datetime(
    acquisition_time_utc: datetime, bounds: tuple[float, float, float, float]
) -> datetime:
    center_longitude = (bounds[0] + bounds[2]) / 2
    return acquisition_time_utc.replace(tzinfo=None) + timedelta(
        hours=center_longitude / 15.0
    )


def _write_metadata_sidecar(metadata_path: Path, image_info: dict[str, Any]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(image_info, file, indent=2, default=str)


def _prepare_image_job(
    collection_id: str,
    image: ee.Image,
    bounds: tuple[float, float, float, float],
    raw_dir: Path,
    band_aliases: list[str],
    output_band_names: list[str],
    options: GoogleEarthEngineDownloadOptions,
) -> _PreparedImageJob | None:
    image_info = image.getInfo()
    image_id = image_info["id"]
    available_band_ids = {band["id"] for band in image_info["bands"]}
    missing_bands = [band for band in band_aliases if band not in available_band_ids]
    if missing_bands:
        raise ValueError(f"{collection_id} image {image_id} is missing bands: {missing_bands}")

    acquisition_time_utc = _parse_gee_acquisition_time(image_info)
    local_datetime = _estimate_local_datetime(acquisition_time_utc, bounds)
    grid = _get_image_grid_info(image_info)
    intersection = grid["bbox"].intersection(
        _transform_bounds_to_image_crs(bounds, grid["crs"])
    )
    if intersection.is_empty:
        return None

    window = _intersection_to_window(intersection, grid, options.chunk_size)
    if window is None:
        return None
    row0, row1, col0, col1 = window
    tasks = _build_chunk_tasks(
        row0=row0,
        row1=row1,
        col0=col0,
        col1=col1,
        grid=grid,
        intersection=intersection,
        chunk_size=options.chunk_size,
    )
    if not tasks:
        return None

    out_path = raw_dir / f"{safe_identifier(image_id)}.tif"
    metadata_path = Path(f"{out_path}.metadata.json")
    profile = _build_output_profile(
        grid=grid,
        row0=row0,
        col0=col0,
        width=col1 - col0,
        height=row1 - row0,
        band_count=len(band_aliases),
        nodata=options.nodata,
    )
    dst = _initialize_output_file(
        out_path=out_path,
        profile=profile,
        image_info=image_info,
        band_aliases=band_aliases,
        chunk_size=options.chunk_size,
        nodata=options.nodata,
    )
    return _PreparedImageJob(
        collection_id=collection_id,
        image_id=image_id,
        image_info=image_info,
        acquisition_time_utc=acquisition_time_utc,
        local_datetime=local_datetime,
        grid=grid,
        row0=row0,
        col0=col0,
        band_aliases=band_aliases,
        output_band_names=output_band_names,
        out_path=out_path,
        metadata_path=metadata_path,
        dst=dst,
        tasks=tasks,
    )


def _fetch_chunk(
    job: _PreparedImageJob,
    task: tuple[int, int, int, int],
    options: GoogleEarthEngineDownloadOptions,
) -> tuple[int, int, np.ndarray]:
    row, col, chunk_h, chunk_w = task
    grid = job.grid
    request = {
        "assetId": job.image_id,
        "fileFormat": "NUMPY_NDARRAY",
        "bandIds": job.band_aliases,
        "grid": {
            "dimensions": {"width": int(chunk_w), "height": int(chunk_h)},
            "crsCode": grid["crs"],
            "affineTransform": {
                "scaleX": grid["x_scale"],
                "shearX": 0,
                "translateX": grid["origin_x"] + col * grid["x_scale"],
                "shearY": 0,
                "scaleY": grid["y_scale"],
                "translateY": grid["origin_y"] + row * grid["y_scale"],
            },
        },
    }

    delay_seconds = options.retry_delay_seconds
    for attempt in range(1, options.max_retries + 1):
        try:
            raw = ee.data.getPixels(request)
            data = np.array(
                structured_to_hwc_array(raw, job.band_aliases),
                dtype=np.float32,
                copy=True,
            )
            data[data < 0] = np.nan
            return row, col, data
        except Exception:
            if attempt == options.max_retries:
                raise
            get_logger().warning(
                "Retrying GEE chunk fetch for %s (%d/%d)",
                job.image_id,
                attempt + 1,
                options.max_retries,
            )
            time.sleep(delay_seconds)
            delay_seconds *= 2
    raise RuntimeError("Unreachable chunk download retry branch.")


def download_google_earth_engine_images(
    collection_id: str,
    images: list[ee.Image],
    bounds: tuple[float, float, float, float],
    raw_dir: Path,
    band_aliases: list[str],
    output_band_names: list[str],
    options: GoogleEarthEngineDownloadOptions | None = None,
) -> list[DownloadedGoogleEarthEngineImage]:
    options = options or GoogleEarthEngineDownloadOptions()
    jobs: list[_PreparedImageJob] = []

    with ThreadPoolExecutor(
        max_workers=min(options.prepare_workers, max(1, len(images)))
    ) as executor:
        futures = [
            executor.submit(
                _prepare_image_job,
                collection_id,
                image,
                bounds,
                raw_dir,
                band_aliases,
                output_band_names,
                options,
            )
            for image in images
        ]
        for future in as_completed(futures):
            job = future.result()
            if job is not None:
                jobs.append(job)
    if not jobs:
        return []

    jobs_by_id = {job.image_id: job for job in jobs}
    tasks = [(job.image_id, task) for job in jobs for task in job.tasks]
    failed_downloads: list[str] = []

    try:
        with ThreadPoolExecutor(
            max_workers=min(options.download_workers, max(1, len(tasks)))
        ) as executor:
            futures = {
                executor.submit(_fetch_chunk, jobs_by_id[image_id], task, options): image_id
                for image_id, task in tasks
            }
            for future in as_completed(futures):
                image_id = futures[future]
                job = jobs_by_id[image_id]
                try:
                    row, col, data = future.result()
                    job.dst.write(
                        np.moveaxis(data, -1, 0),
                        window=Window(
                            col - job.col0,
                            row - job.row0,
                            data.shape[1],
                            data.shape[0],
                        ),
                    )
                except Exception:
                    failed_downloads.append(image_id)
                    get_logger().exception(
                        "Failed downloading GEE chunks for %s", image_id
                    )

        if failed_downloads:
            raise RuntimeError(
                "Failed downloading one or more Google Earth Engine chunks for "
                f"{sorted(set(failed_downloads))}"
            )

        downloads: list[DownloadedGoogleEarthEngineImage] = []
        for job in jobs:
            job.dst.close()
            _write_metadata_sidecar(job.metadata_path, job.image_info)
            downloads.append(
                DownloadedGoogleEarthEngineImage(
                    collection_id=job.collection_id,
                    image_id=job.image_id,
                    image_info=job.image_info,
                    acquisition_time_utc=job.acquisition_time_utc,
                    local_datetime=job.local_datetime,
                    output_band_names=job.output_band_names,
                    tiff_path=job.out_path,
                    metadata_path=job.metadata_path,
                )
            )
        return downloads
    finally:
        for job in jobs:
            try:
                if not job.dst.closed:
                    job.dst.close()
            except Exception:
                pass

