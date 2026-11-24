"""Build a Google Street View Static API hyperlapse MP4 from a GPX route."""

from __future__ import annotations

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests

GSV_METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
GSV_IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"
DEFAULT_GPX = r"c:\Users\qkins\Downloads\route.gpx"
DEFAULT_OUT = r"c:\Users\qkins\Desktop\Antigravity\Thesis\hyperlapse\route_hyperlapse_gsv.mp4"
EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = la2 - la1
    dlon = lo2 - lo1
    h = math.sin(dlat / 2.0) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(h)))


def compute_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlon = lo2 - lo1
    y = math.sin(dlon) * math.cos(la2)
    x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def ang_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def load_samples(gpx_path: Path, density: float) -> tuple[list[dict], float]:
    tree = ET.parse(gpx_path)
    root = tree.getroot()
    ns = {"g": "http://www.topografix.com/GPX/1/1"}
    trkpts = root.findall(".//g:trkpt", ns)
    if not trkpts:
        trkpts = root.findall(".//trkpt")
    if len(trkpts) < 2:
        raise ValueError(f"Expected at least 2 trackpoints in {gpx_path}, found {len(trkpts)}")

    raw_lat = np.array([float(p.attrib["lat"]) for p in trkpts], dtype=np.float64)
    raw_lon = np.array([float(p.attrib["lon"]) for p in trkpts], dtype=np.float64)

    cum_m = np.zeros(len(raw_lat), dtype=np.float64)
    for i in range(1, len(raw_lat)):
        cum_m[i] = cum_m[i - 1] + haversine_m(
            raw_lat[i - 1], raw_lon[i - 1], raw_lat[i], raw_lon[i]
        )

    total_m = float(cum_m[-1])
    step_m = 1000.0 / density
    sample_dists = np.arange(0.0, total_m, step_m, dtype=np.float64)
    if len(sample_dists) == 0 or sample_dists[-1] < total_m:
        sample_dists = np.append(sample_dists, total_m)

    s_lat = np.interp(sample_dists, cum_m, raw_lat)
    s_lon = np.interp(sample_dists, cum_m, raw_lon)

    n = len(sample_dists)
    samples: list[dict] = []
    for i in range(n):
        i0 = max(0, i - 1)
        i1 = min(n - 1, i + 1)
        brg = compute_bearing(s_lat[i0], s_lon[i0], s_lat[i1], s_lon[i1]) if i0 != i1 else 0.0
        samples.append(
            {
                "idx": i,
                "dist_m": float(sample_dists[i]),
                "lat": float(s_lat[i]),
                "lon": float(s_lon[i]),
                "route_bearing": brg,
            }
        )
    return samples, total_m


def fetch_metadata(
    samples: list[dict],
    api_key: str,
    cache_path: Path,
    force: bool,
) -> list[dict]:
    if cache_path.exists() and not force:
        with cache_path.open("r", encoding="utf-8") as f:
            cached = json.load(f)
        print(f"Loaded {len(cached)} cached Google Street View panoramas from {cache_path}")
        return cached

    thread_local = threading.local()
    abort_event = threading.Event()
    fatal_error: list[str] = []

    def get_session() -> requests.Session:
        if not hasattr(thread_local, "session"):
            thread_local.session = requests.Session()
        return thread_local.session

    def query_one(sample: dict) -> tuple[int, dict | None]:
        if abort_event.is_set():
            return sample["idx"], None
        session = get_session()
        params = {
            "location": f"{sample['lat']:.6f},{sample['lon']:.6f}",
            "radius": 50,
            "source": "outdoor",
            "key": api_key,
        }
        for attempt in range(5):
            if abort_event.is_set():
                return sample["idx"], None
            try:
                resp = session.get(GSV_METADATA_URL, params=params, timeout=15)
                if resp.status_code == 429 or resp.status_code >= 500:
                    time.sleep(0.5 * (2**attempt))
                    continue
                data = resp.json()
                status = data.get("status")
                if status in ("REQUEST_DENIED", "OVER_QUERY_LIMIT"):
                    msg = data.get("error_message") or status
                    fatal_error.append(f"Metadata API error {status}: {msg}")
                    abort_event.set()
                    return sample["idx"], None
                return sample["idx"], data
            except (requests.RequestException, ValueError):
                time.sleep(0.5 * (2**attempt))
        return sample["idx"], None

    print(f"Querying FREE Street View Metadata API for {len(samples)} route samples...")
    raw_results: list[dict | None] = [None] * len(samples)
    done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(query_one, s) for s in samples]
        for fut in as_completed(futures):
            idx, payload = fut.result()
            raw_results[idx] = payload
            done += 1
            if done % 400 == 0 or done == len(samples):
                print(f"Metadata progress: {done}/{len(samples)}")

    if fatal_error:
        raise RuntimeError(fatal_error[0])

    kept: list[dict] = []
    seen_panos: set[str] = set()
    skipped_non_google = 0
    skipped_dup = 0
    skipped_far = 0
    skipped_status = 0

    for sample, item in zip(samples, raw_results):
        if not item or item.get("status") != "OK":
            skipped_status += 1
            continue
        copyright_str = str(item.get("copyright") or "")
        if "google" not in copyright_str.lower():
            skipped_non_google += 1
            continue
        loc = item.get("location") or {}
        plat = loc.get("lat")
        plng = loc.get("lng")
        pano_id = str(item.get("pano_id") or "")
        if plat is None or plng is None or not pano_id:
            skipped_status += 1
            continue
        snap_m = haversine_m(sample["lat"], sample["lon"], float(plat), float(plng))
        if snap_m > 60.0:
            skipped_far += 1
            continue
        if pano_id in seen_panos:
            skipped_dup += 1
            continue
        seen_panos.add(pano_id)
        kept.append(
            {
                "pano_id": pano_id,
                "lat": float(plat),
                "lng": float(plng),
                "dist_m": round(sample["dist_m"], 2),
                "snap_m": round(snap_m, 2),
                "route_bearing": round(sample["route_bearing"], 2),
                "date": str(item.get("date") or ""),
                "copyright": copyright_str,
            }
        )

    # Compute snapped central-difference bearings and 3-point circular smoothing
    n_kept = len(kept)
    raw_headings: list[float] = []
    for i in range(n_kept):
        i0 = max(0, i - 1)
        i1 = min(n_kept - 1, i + 1)
        rb = kept[i]["route_bearing"]
        if i0 != i1:
            span_m = haversine_m(kept[i0]["lat"], kept[i0]["lng"], kept[i1]["lat"], kept[i1]["lng"])
            if 10.0 <= span_m <= 500.0:
                snapped_brg = compute_bearing(
                    kept[i0]["lat"], kept[i0]["lng"], kept[i1]["lat"], kept[i1]["lng"]
                )
                if ang_diff(snapped_brg, rb) <= 45.0:
                    raw_headings.append(snapped_brg)
                    continue
        raw_headings.append(rb)

    for i in range(n_kept):
        if n_kept == 1:
            sm_hdg = raw_headings[0]
        else:
            il = max(0, i - 1)
            ir = min(n_kept - 1, i + 1)
            weights = (0.25, 0.50, 0.25)
            angs = (raw_headings[il], raw_headings[i], raw_headings[ir])
            sx = sum(w * math.sin(math.radians(a)) for w, a in zip(weights, angs))
            cx = sum(w * math.cos(math.radians(a)) for w, a in zip(weights, angs))
            sm_hdg = (math.degrees(math.atan2(sx, cx)) + 360.0) % 360.0
        kept[i]["heading"] = round(sm_hdg, 1)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2)

    if kept:
        gaps = [kept[i]["dist_m"] - kept[i - 1]["dist_m"] for i in range(1, n_kept)]
        max_gap = max(gaps) if gaps else 0.0
        years = dict(sorted(collections.Counter(p["date"][:4] for p in kept).items()))
        print(
            f"Metadata summary: {n_kept} unique official Google panos "
            f"({kept[0]['dist_m']/1000.0:.2f}–{kept[-1]['dist_m']/1000.0:.2f} km, "
            f"largest gap={max_gap:.0f} m)"
        )
        print(
            f"Filtered out: status={skipped_status}, non-Google={skipped_non_google}, "
            f"far(>60m)={skipped_far}, duplicate_pano={skipped_dup}"
        )
        print(f"Capture years: {years}")
    else:
        print("Metadata summary: 0 official Google panos matched!")

    return kept


def frame_filename(frame: dict) -> str:
    hdg_int = int(round(frame["heading"])) % 360
    return f"{frame['pano_id']}_{hdg_int}.jpg"


def download_images(
    frames: list[dict],
    frames_dir: Path,
    api_key: str,
    width: int,
    height: int,
    fov: int,
    pitch: float,
    max_requests: int,
    workers: int,
) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
    todo: list[tuple[dict, Path]] = []
    for p in frames:
        dest = frames_dir / frame_filename(p)
        if dest.exists() and dest.stat().st_size > 2048:
            continue
        todo.append((p, dest))

    print(
        f"Image cache check: {len(frames) - len(todo)} already cached, "
        f"{len(todo)} new billable image requests needed (cap={max_requests})."
    )
    if len(todo) > max_requests:
        raise RuntimeError(
            f"Safety abort: {len(todo)} requests needed exceeds --max-requests={max_requests}. "
            "No billable image requests were made."
        )
    if not todo:
        return 0

    thread_local = threading.local()
    lock = threading.Lock()
    requests_made = 0
    failed = 0
    consecutive_403 = 0
    abort_event = threading.Event()
    abort_reason: list[str] = []

    def get_session() -> requests.Session:
        if not hasattr(thread_local, "session"):
            thread_local.session = requests.Session()
        return thread_local.session

    def fetch_one(item: tuple[dict, Path]) -> bool:
        nonlocal requests_made, failed, consecutive_403
        p, dest = item
        if abort_event.is_set():
            return False

        params = {
            "pano": p["pano_id"],
            "size": f"{width}x{height}",
            "heading": f"{p['heading']:.1f}",
            "pitch": f"{pitch}",
            "fov": f"{fov}",
            "source": "outdoor",
            "return_error_code": "true",
            "key": api_key,
        }
        session = get_session()
        for attempt in range(4):
            if abort_event.is_set():
                return False
            with lock:
                if requests_made >= max_requests:
                    abort_reason.append(f"Reached --max-requests={max_requests} hard cap.")
                    abort_event.set()
                    return False
                requests_made += 1

            try:
                resp = session.get(GSV_IMAGE_URL, params=params, timeout=20)
                if resp.status_code == 403:
                    with lock:
                        consecutive_403 += 1
                        if consecutive_403 >= 5:
                            abort_reason.append(
                                f"Aborted after 5 consecutive HTTP 403 responses: {resp.text[:200]}"
                            )
                            abort_event.set()
                    return False
                if resp.status_code == 429 or resp.status_code >= 500:
                    time.sleep(0.6 * (2**attempt))
                    continue
                if resp.status_code != 200 or len(resp.content) <= 2048:
                    with lock:
                        failed += 1
                    return False
                with lock:
                    consecutive_403 = 0
                tmp = dest.with_suffix(".tmp")
                tmp.write_bytes(resp.content)
                tmp.replace(dest)
                return True
            except requests.RequestException:
                time.sleep(0.6 * (2**attempt))

        with lock:
            failed += 1
        return False

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fetch_one, item) for item in todo]
        for fut in as_completed(futures):
            fut.result()
            completed += 1
            if completed % 200 == 0 or completed == len(todo):
                print(
                    f"Download progress: {completed}/{len(todo)} | "
                    f"billable_requests={requests_made} | failed={failed}"
                )

    if abort_reason:
        raise RuntimeError(abort_reason[0])

    print(f"Finished downloads: {requests_made} billable requests made, {failed} failed.")
    return requests_made


def create_preview_grid(
    samples: list[tuple[Image.Image, dict]], grid_path: Path
) -> None:
    if not samples:
        return
    tile_w, tile_h = 320, 180
    canvas = Image.new("RGB", (tile_w * 4, tile_h * 4), (18, 18, 18))
    font = ImageFont.load_default()

    indices = np.linspace(0, len(samples) - 1, min(16, len(samples)), dtype=int)
    for slot, idx in enumerate(indices):
        im, meta = samples[int(idx)]
        tile = im.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(tile)
        label = f"km {meta['dist_m']/1000.0:.1f} | {meta['date']}"
        draw.rectangle((4, 4, 155, 22), fill=(0, 0, 0))
        draw.text((8, 7), label, fill=(255, 255, 255), font=font)
        col = slot % 4
        row = slot // 4
        canvas.paste(tile, (col * tile_w, row * tile_h))

    grid_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(grid_path, quality=92)
    print(f"Saved preview grid to {grid_path}")


def encode_video(
    frames: list[dict],
    frames_dir: Path,
    out_path: Path,
    fps: int,
    width: int,
    height: int,
    hud: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_exe = None
    try:
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = None

    proc = None
    cv_writer = None
    if ffmpeg_exe:
        cmd = [
            ffmpeg_exe,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
    else:
        import cv2

        cv_writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not cv_writer.isOpened():
            raise RuntimeError("Neither ffmpeg nor OpenCV VideoWriter could be opened.")

    rendered = 0
    skipped = 0
    sample_stride = max(1, len(frames) // 32)
    preview_candidates: list[tuple[Image.Image, dict]] = []

    for i, meta in enumerate(frames):
        img_path = frames_dir / frame_filename(meta)
        if not img_path.exists():
            skipped += 1
            continue
        try:
            with Image.open(img_path) as raw:
                im = raw.convert("RGB")
        except Exception:
            skipped += 1
            continue

        if im.size != (width, height):
            im = im.resize((width, height), Image.Resampling.LANCZOS)

        if hud:
            draw = ImageDraw.Draw(im)
            label = f"km {meta['dist_m']/1000.0:.1f} | {meta['date']}"
            draw.rectangle((8, 8, 175, 28), fill=(0, 0, 0))
            draw.text((12, 12), label, fill=(255, 255, 255))

        if proc is not None and proc.stdin is not None:
            proc.stdin.write(im.tobytes())
        elif cv_writer is not None:
            cv_writer.write(np.asarray(im)[:, :, ::-1])

        if i % sample_stride == 0 or i == len(frames) - 1:
            preview_candidates.append((im.copy(), meta))

        rendered += 1
        if rendered % 300 == 0:
            print(f"Encoded {rendered} frames...")

    if proc is not None:
        if proc.stdin is not None:
            proc.stdin.close()
        stderr_out = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"FFmpeg failed with exit code {ret}:\n{stderr_out[-1000:]}")
    if cv_writer is not None:
        cv_writer.release()

    grid_path = out_path.parent / "preview_grid_gsv.jpg"
    create_preview_grid(preview_candidates, grid_path)
    print(
        f"Finished encoding {rendered} frames (skipped={skipped}) -> {out_path} "
        f"({rendered / fps:.2f}s @ {fps}fps)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a Google Street View Static API hyperlapse MP4 from a GPX route."
    )
    parser.add_argument("--gpx", default=DEFAULT_GPX, help="Input GPX route path")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output MP4 path")
    parser.add_argument(
        "--density", type=float, default=8.0, help="Target frames per km (default: 8.0)"
    )
    parser.add_argument("--fps", type=int, default=30, help="Output FPS (default: 30)")
    parser.add_argument("--width", type=int, default=640, help="Frame width <= 640 (default: 640)")
    parser.add_argument("--height", type=int, default=360, help="Frame height <= 640 (default: 360)")
    parser.add_argument("--fov", type=int, default=90, help="Field of view (default: 90)")
    parser.add_argument("--pitch", type=float, default=2.0, help="Camera pitch (default: 2.0)")
    parser.add_argument(
        "--max-requests",
        type=int,
        default=2500,
        help="Hard cap on new billable image requests (default: 2500)",
    )
    parser.add_argument("--workers", type=int, default=10, help="Download workers (default: 10)")
    parser.add_argument("--hud", action="store_true", help="Draw km and date overlay at top-left")
    parser.add_argument(
        "--force-metadata", action="store_true", help="Ignore cached metadata.json"
    )
    parser.add_argument(
        "--api-key", default=None, help="Google Maps API key (default: GOOGLE_MAPS_API_KEY env var)"
    )
    args = parser.parse_args()

    if args.width < 16 or args.height < 16 or args.width > 640 or args.height > 640:
        print("Error: Street View Static API requires width and height in [16, 640].", file=sys.stderr)
        return 1

    api_key = args.api_key or os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    if not api_key:
        print(
            "Error: Missing API key. Set GOOGLE_MAPS_API_KEY or pass --api-key.",
            file=sys.stderr,
        )
        return 1

    gpx_path = Path(args.gpx)
    out_path = Path(args.out)
    cache_dir = out_path.parent / "gsv_cache"
    metadata_path = cache_dir / "metadata.json"
    frames_dir = cache_dir / "frames"

    samples, total_m = load_samples(gpx_path, args.density)
    print(
        f"Loaded route {gpx_path}: {total_m/1000.0:.2f} km -> "
        f"{len(samples)} sample points @ {1000.0/args.density:.1f} m spacing"
    )

    frames = fetch_metadata(samples, api_key, metadata_path, args.force_metadata)
    if not frames:
        print("No Street View panoramas matched the route.", file=sys.stderr)
        return 1

    download_images(
        frames=frames,
        frames_dir=frames_dir,
        api_key=api_key,
        width=args.width,
        height=args.height,
        fov=args.fov,
        pitch=args.pitch,
        max_requests=args.max_requests,
        workers=args.workers,
    )
    encode_video(frames, frames_dir, out_path, args.fps, args.width, args.height, args.hud)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
