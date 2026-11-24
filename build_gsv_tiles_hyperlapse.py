"""Build a native 1080p Google Street View hyperlapse MP4 using Map Tiles API."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
from requests.adapters import HTTPAdapter

CREATE_SESSION_URL = "https://tile.googleapis.com/v1/createSession"
TILE_METADATA_URL = "https://tile.googleapis.com/v1/streetview/metadata"
TILE_IMAGE_URL = "https://tile.googleapis.com/v1/streetview/tiles"

DEFAULT_ROUTE_META = r"c:\Users\qkins\Desktop\Antigravity\Thesis\hyperlapse\gsv_cache\metadata.json"
DEFAULT_OUT = r"c:\Users\qkins\Desktop\Antigravity\Thesis\hyperlapse\route_hyperlapse_gsv_1080p.mp4"


def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def make_session(pool_size: int) -> requests.Session:
    s = requests.Session()
    adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def create_sv_session(http: requests.Session, api_key: str) -> str:
    resp = http.post(
        CREATE_SESSION_URL,
        params={"key": api_key},
        json={"mapType": "streetview", "language": "en-US", "region": "TH"},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    return str(data["session"])


def fetch_tile_metadata(
    route_frames: list[dict],
    http: requests.Session,
    session_token: str,
    api_key: str,
    cache_path: Path,
    force: bool,
) -> dict[str, dict]:
    if cache_path.exists() and not force:
        with cache_path.open("r", encoding="utf-8") as f:
            cached = json.load(f)
        if all(str(fr["pano_id"]) in cached for fr in route_frames):
            print(f"Loaded {len(cached)} Map Tiles metadata entries from {cache_path}")
            return cached

    existing: dict[str, dict] = {}
    if cache_path.exists() and not force:
        with cache_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)

    missing_ids = [
        str(fr["pano_id"]) for fr in route_frames if str(fr["pano_id"]) not in existing
    ]
    if not missing_ids:
        return existing

    print(f"Fetching FREE Map Tiles metadata for {len(missing_ids)} panoramas...")
    abort_event = threading.Event()
    fatal_err: list[str] = []

    def fetch_one(pid: str) -> tuple[str, dict | None]:
        if abort_event.is_set():
            return pid, None
        params = {"session": session_token, "key": api_key, "panoId": pid}
        for attempt in range(5):
            if abort_event.is_set():
                return pid, None
            try:
                r = http.get(TILE_METADATA_URL, params=params, timeout=15)
                if r.status_code in (401, 403):
                    fatal_err.append(f"HTTP {r.status_code} on metadata: {r.text[:250]}")
                    abort_event.set()
                    return pid, None
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(0.5 * (2**attempt))
                    continue
                if r.status_code != 200:
                    return pid, None
                d = r.json()
                roll = float(d.get("roll") or 0.0)
                if roll > 180.0:
                    roll -= 360.0
                return pid, {
                    "panoId": pid,
                    "imageWidth": int(d.get("imageWidth") or 16384),
                    "imageHeight": int(d.get("imageHeight") or 8192),
                    "tileWidth": int(d.get("tileWidth") or 512),
                    "tileHeight": int(d.get("tileHeight") or 512),
                    "heading": float(d.get("heading") or 0.0),
                    "tilt": float(d.get("tilt") if d.get("tilt") is not None else 90.0),
                    "roll": roll,
                    "copyright": str(d.get("copyright") or ""),
                    "date": str(d.get("date") or ""),
                }
            except (requests.RequestException, ValueError):
                time.sleep(0.5 * (2**attempt))
        return pid, None

    done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(fetch_one, pid) for pid in missing_ids]
        for fut in as_completed(futures):
            pid, info = fut.result()
            if info is not None:
                existing[pid] = info
            done += 1
            if done % 400 == 0 or done == len(missing_ids):
                print(f"Tile metadata progress: {done}/{len(missing_ids)}")

    if fatal_err:
        raise RuntimeError(fatal_err[0])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(existing, f)
    print(f"Saved {len(existing)} Map Tiles metadata entries to {cache_path}")
    return existing


def build_camera_rays(width: int, height: int, fov_deg: float) -> tuple[np.ndarray, np.ndarray]:
    fx = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    uu, vv = np.meshgrid(
        np.arange(width, dtype=np.float32) - (width - 1) / 2.0,
        (height - 1) / 2.0 - np.arange(height, dtype=np.float32),
    )
    v0 = np.stack([uu / fx, vv / fx, np.ones_like(uu)], axis=-1)
    v0 /= np.linalg.norm(v0, axis=-1, keepdims=True)

    # Coarse grid: every 8th pixel plus all 4 border edges for fast, exact tile discovery
    rows = np.unique(np.append(np.arange(0, height, 8), [height - 1]))
    cols = np.unique(np.append(np.arange(0, width, 8), [width - 1]))
    v0_coarse = np.concatenate(
        [
            v0[np.ix_(rows, cols)].reshape(-1, 3),
            v0[0, :, :],
            v0[-1, :, :],
            v0[:, 0, :],
            v0[:, -1, :],
        ],
        axis=0,
    )
    return v0, v0_coarse


def compute_rotation_matrix(
    target_hdg: float, pitch_deg: float, pano_hdg: float, pano_tilt: float, pano_roll: float
) -> np.ndarray:
    return (
        rot_z(pano_roll)
        @ rot_x(pano_tilt - 90.0)
        @ rot_y(-pano_hdg)
        @ rot_y(target_hdg)
        @ rot_x(-pitch_deg)
    ).astype(np.float32)


def project_rays(
    rays: np.ndarray, rot_mat: np.ndarray, eq_w: float, eq_h: float
) -> tuple[np.ndarray, np.ndarray]:
    vp = rays @ rot_mat.T
    lon = np.arctan2(vp[..., 0], vp[..., 2])
    lat = np.arcsin(np.clip(vp[..., 1], -1.0, 1.0))
    px = ((lon / (2.0 * math.pi) + 0.5) % 1.0) * eq_w
    py = np.clip((0.5 - lat / math.pi) * eq_h, 0.0, eq_h - 1.0)
    return px, py


def get_pano_grid_params(tile_meta: dict, target_zoom: int) -> tuple[int, float, float, int, int, int, int]:
    img_w = int(tile_meta["imageWidth"])
    img_h = int(tile_meta["imageHeight"])
    tile_w = int(tile_meta["tileWidth"])
    tile_h = int(tile_meta["tileHeight"])
    max_zoom = int(math.ceil(math.log2(max(1.0, img_w / tile_w))))
    use_z = min(target_zoom, max_zoom)
    scale = 2 ** (max_zoom - use_z)
    eq_w = img_w / scale
    eq_h = img_h / scale
    nx_total = int(math.ceil(int(round(eq_w)) / tile_w))
    ny_total = int(math.ceil(int(round(eq_h)) / tile_h))
    return use_z, eq_w, eq_h, tile_w, tile_h, nx_total, ny_total


def frame_cache_path(frames_dir: Path, idx: int, pano_id: str) -> Path:
    return frames_dir / f"{idx:04d}_{pano_id}.jpg"


def plan_tiles(
    route_frames: list[dict],
    tile_meta_map: dict[str, dict],
    frames_dir: Path,
    v0_coarse: np.ndarray,
    target_zoom: int,
    pitch_deg: float,
    max_tiles: int,
) -> list[dict]:
    jobs: list[dict] = []
    total_tiles = 0

    for idx, fr in enumerate(route_frames):
        pid = str(fr["pano_id"])
        out_jpg = frame_cache_path(frames_dir, idx, pid)
        if out_jpg.exists() and out_jpg.stat().st_size > 5120:
            continue
        tm = tile_meta_map.get(pid)
        if tm is None:
            continue

        use_z, eq_w, eq_h, tile_w, tile_h, nx_total, ny_total = get_pano_grid_params(
            tm, target_zoom
        )
        rot_mat = compute_rotation_matrix(
            target_hdg=float(fr["heading"]),
            pitch_deg=pitch_deg,
            pano_hdg=float(tm["heading"]),
            pano_tilt=float(tm["tilt"]),
            pano_roll=float(tm["roll"]),
        )
        px_c, py_c = project_rays(v0_coarse, rot_mat, eq_w, eq_h)

        px_nbr = (px_c + 1.0) % eq_w
        py_nbr = np.minimum(py_c + 1.0, eq_h - 1.0)

        tx_all = np.concatenate(
            [
                np.floor(px_c / tile_w).astype(int),
                np.floor(px_nbr / tile_w).astype(int),
            ]
        )
        ty_all = np.concatenate(
            [
                np.floor(py_c / tile_h).astype(int),
                np.floor(py_nbr / tile_h).astype(int),
            ]
        )
        tx_all = np.clip(tx_all, 0, nx_total - 1)
        ty_all = np.clip(ty_all, 0, ny_total - 1)

        tiles = sorted({(int(tx), int(ty)) for tx, ty in zip(tx_all, ty_all)})
        total_tiles += len(tiles)
        jobs.append(
            {
                "idx": idx,
                "pano_id": pid,
                "out_jpg": out_jpg,
                "use_z": use_z,
                "eq_w": eq_w,
                "eq_h": eq_h,
                "tile_w": tile_w,
                "tile_h": tile_h,
                "rot_mat": rot_mat,
                "tiles": tiles,
            }
        )

    avg_tiles = (total_tiles / len(jobs)) if jobs else 0.0
    print(
        f"Tile plan: {len(jobs)} frames to render ({len(route_frames) - len(jobs)} cached) | "
        f"total_tiles_needed={total_tiles} (avg {avg_tiles:.1f}/frame) | cap={max_tiles}"
    )
    if total_tiles > max_tiles:
        raise RuntimeError(
            f"Safety abort: {total_tiles} tiles needed exceeds --max-tiles={max_tiles}. "
            "No billable tile requests were made."
        )
    return jobs


def render_frames(
    jobs: list[dict],
    http: requests.Session,
    session_token: str,
    api_key: str,
    v0: np.ndarray,
    max_tiles: int,
    pano_workers: int,
    tile_workers: int,
) -> int:
    if not jobs:
        return 0

    lock = threading.Lock()
    tiles_fetched = 0
    consecutive_403 = 0
    abort_event = threading.Event()
    abort_reason: list[str] = []

    def fetch_tile(pid: str, z: int, tx: int, ty: int) -> tuple[int, int, np.ndarray | None]:
        nonlocal tiles_fetched, consecutive_403
        url = f"{TILE_IMAGE_URL}/{z}/{tx}/{ty}"
        params = {"session": session_token, "key": api_key, "panoId": pid}
        for attempt in range(4):
            if abort_event.is_set():
                return tx, ty, None
            with lock:
                if tiles_fetched >= max_tiles:
                    abort_reason.append(f"Reached --max-tiles={max_tiles} hard cap.")
                    abort_event.set()
                    return tx, ty, None
                tiles_fetched += 1
            try:
                r = http.get(url, params=params, timeout=20)
                if r.status_code == 403:
                    with lock:
                        consecutive_403 += 1
                        if consecutive_403 >= 10:
                            abort_reason.append(
                                f"Aborted after 10 consecutive HTTP 403 responses: {r.text[:250]}"
                            )
                            abort_event.set()
                    return tx, ty, None
                if r.status_code == 429:
                    with lock:
                        tiles_fetched = max(0, tiles_fetched - 1)
                    body_low = r.text.lower()
                    if "per day" in body_low or "perday" in body_low:
                        with lock:
                            if not abort_reason:
                                abort_reason.append(
                                    "Hit Google Map Tiles daily quota (15,000 tiles/day). "
                                    "Re-run after quota resets at midnight Pacific Time (15:00 Bangkok time) "
                                    "to resume from cached frames."
                                )
                            abort_event.set()
                        return tx, ty, None
                    time.sleep(0.8 * (2**attempt))
                    continue
                if r.status_code >= 500:
                    time.sleep(0.5 * (2**attempt))
                    continue
                if r.status_code != 200 or len(r.content) < 512:
                    return tx, ty, None
                with lock:
                    consecutive_403 = 0
                im = Image.open(io.BytesIO(r.content)).convert("RGB")
                return tx, ty, np.asarray(im, dtype=np.uint8)
            except (requests.RequestException, OSError):
                time.sleep(0.5 * (2**attempt))
        return tx, ty, None

    t0 = time.time()
    rendered = 0
    failed_frames = 0

    with ThreadPoolExecutor(max_workers=tile_workers) as tile_ex:
        def process_job(job: dict) -> bool:
            if abort_event.is_set():
                return False
            pid = job["pano_id"]
            z = job["use_z"]
            eq_w = job["eq_w"]
            eq_h = job["eq_h"]
            tile_w = job["tile_w"]
            tile_h = job["tile_h"]
            tiles = job["tiles"]

            eq_w_int = int(round(eq_w))
            eq_h_int = int(round(eq_h))
            y_min = min(ty for _, ty in tiles)
            y_max = max(ty for _, ty in tiles)
            band_h = min(eq_h_int - y_min * tile_h, (y_max - y_min + 1) * tile_h)
            band = np.zeros((band_h, eq_w_int + 1, 3), dtype=np.uint8)

            futs = [tile_ex.submit(fetch_tile, pid, z, tx, ty) for tx, ty in tiles]
            missing = 0
            for fut in as_completed(futs):
                tx, ty, tile_arr = fut.result()
                if tile_arr is None:
                    missing += 1
                    continue
                cx = tx * tile_w
                ry = (ty - y_min) * tile_h
                w_clip = min(tile_arr.shape[1], eq_w_int - cx)
                h_clip = min(tile_arr.shape[0], band_h - ry)
                if w_clip > 0 and h_clip > 0:
                    band[ry : ry + h_clip, cx : cx + w_clip] = tile_arr[:h_clip, :w_clip]

            if missing > 0 or abort_event.is_set():
                return False

            band[:, eq_w_int] = band[:, 0]
            px, py = project_rays(v0, job["rot_mat"], eq_w, eq_h)
            map_x = px.astype(np.float32)
            map_y = np.clip(py - y_min * tile_h, 0.0, band_h - 1.0).astype(np.float32)
            out_rgb = cv2.remap(
                band, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )

            out_im = Image.fromarray(out_rgb)
            draw = ImageDraw.Draw(out_im)
            w_out, h_out = out_im.size
            draw.text((w_out - 74, h_out - 20), "© Google", fill=(0, 0, 0))
            draw.text((w_out - 75, h_out - 21), "© Google", fill=(235, 235, 235))

            out_jpg: Path = job["out_jpg"]
            out_jpg.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_jpg.with_suffix(".tmp")
            out_im.save(tmp, format="JPEG", quality=95)
            tmp.replace(out_jpg)
            return True

        with ThreadPoolExecutor(max_workers=pano_workers) as pano_ex:
            pano_futs = [pano_ex.submit(process_job, j) for j in jobs]
            for fut in as_completed(pano_futs):
                ok = fut.result()
                if ok:
                    rendered += 1
                else:
                    failed_frames += 1
                total_done = rendered + failed_frames
                if total_done % 100 == 0 or total_done == len(jobs):
                    dt = max(0.1, time.time() - t0)
                    print(
                        f"Render progress: {total_done}/{len(jobs)} frames "
                        f"(failed={failed_frames}) | tiles_fetched={tiles_fetched} "
                        f"({tiles_fetched / dt:.1f} tiles/s, {dt:.0f}s)"
                    )

    if abort_reason:
        raise RuntimeError(abort_reason[0])

    print(
        f"Finished rendering {rendered} frames (failed={failed_frames}, total tiles fetched={tiles_fetched})."
    )
    return tiles_fetched


def create_preview_grid(
    samples: list[tuple[Image.Image, dict]], grid_path: Path
) -> None:
    if not samples:
        return
    tile_w, tile_h = 480, 270
    canvas = Image.new("RGB", (tile_w * 4, tile_h * 4), (18, 18, 18))
    font = ImageFont.load_default()

    indices = np.linspace(0, len(samples) - 1, min(16, len(samples)), dtype=int)
    for slot, idx in enumerate(indices):
        im, meta = samples[int(idx)]
        tile = im.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(tile)
        label = f"km {meta['dist_m']/1000.0:.1f} | {meta['date']}"
        draw.rectangle((6, 6, 170, 26), fill=(0, 0, 0))
        draw.text((10, 10), label, fill=(255, 255, 255), font=font)
        col = slot % 4
        row = slot // 4
        canvas.paste(tile, (col * tile_w, row * tile_h))

    grid_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(grid_path, quality=92)
    print(f"Saved preview grid to {grid_path}")


def encode_video(
    route_frames: list[dict],
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
            "19",
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
        cv_writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not cv_writer.isOpened():
            raise RuntimeError("Neither ffmpeg nor OpenCV VideoWriter could be opened.")

    rendered = 0
    skipped = 0
    sample_stride = max(1, len(route_frames) // 32)
    preview_candidates: list[tuple[Image.Image, dict]] = []

    for i, meta in enumerate(route_frames):
        jpg_path = frame_cache_path(frames_dir, i, str(meta["pano_id"]))
        if not jpg_path.exists():
            skipped += 1
            continue
        try:
            with Image.open(jpg_path) as raw:
                im = raw.convert("RGB")
        except Exception:
            skipped += 1
            continue

        if im.size != (width, height):
            im = im.resize((width, height), Image.Resampling.LANCZOS)

        if hud:
            draw = ImageDraw.Draw(im)
            label = f"km {meta['dist_m']/1000.0:.1f} | {meta['date']}"
            draw.rectangle((12, 12, 195, 34), fill=(0, 0, 0))
            draw.text((18, 17), label, fill=(255, 255, 255))

        if proc is not None and proc.stdin is not None:
            proc.stdin.write(im.tobytes())
        elif cv_writer is not None:
            cv_writer.write(np.asarray(im)[:, :, ::-1])

        if i % sample_stride == 0 or i == len(route_frames) - 1:
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

    grid_path = out_path.parent / "preview_grid_gsv_1080p.jpg"
    create_preview_grid(preview_candidates, grid_path)
    print(
        f"Finished encoding {rendered} frames (skipped={skipped}) -> {out_path} "
        f"({rendered / fps:.2f}s @ {fps}fps)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a native 1080p Google Street View hyperlapse via Map Tiles API."
    )
    parser.add_argument("--route-meta", default=DEFAULT_ROUTE_META, help="Route metadata JSON")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output 1080p MP4 path")
    parser.add_argument("--zoom", type=int, default=4, help="Street View tile zoom (default: 4)")
    parser.add_argument("--width", type=int, default=1920, help="Output width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Output height (default: 1080)")
    parser.add_argument("--fov", type=float, default=90.0, help="Horizontal FOV (default: 90.0)")
    parser.add_argument("--pitch", type=float, default=2.0, help="Camera pitch deg (default: 2.0)")
    parser.add_argument("--fps", type=int, default=30, help="Output FPS (default: 30)")
    parser.add_argument(
        "--max-tiles", type=int, default=45000, help="Hard cap on tile requests (default: 45000)"
    )
    parser.add_argument("--pano-workers", type=int, default=4, help="Concurrent panos (default: 4)")
    parser.add_argument(
        "--tile-workers", type=int, default=12, help="Concurrent tile downloads (default: 12)"
    )
    parser.add_argument("--hud", action="store_true", help="Overlay km and date in top-left")
    parser.add_argument(
        "--encode-only",
        action="store_true",
        help="Skip API calls and encode only already-cached 1080p frames",
    )
    parser.add_argument(
        "--force-metadata", action="store_true", help="Ignore cached tile_metadata.json"
    )
    parser.add_argument(
        "--api-key", default=None, help="Google Maps API key (default: GOOGLE_MAPS_API_KEY env)"
    )
    args = parser.parse_args()

    route_meta_path = Path(args.route_meta)
    with route_meta_path.open("r", encoding="utf-8") as f:
        route_frames = json.load(f)
    print(f"Loaded {len(route_frames)} route panoramas from {route_meta_path}")

    out_path = Path(args.out)
    cache_dir = out_path.parent / "gsv_tiles_cache"
    tile_meta_path = cache_dir / "tile_metadata.json"
    frames_dir = cache_dir / "frames"

    if not args.encode_only:
        api_key = args.api_key or os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
        if not api_key:
            print(
                "Error: Missing API key. Set GOOGLE_MAPS_API_KEY or pass --api-key.",
                file=sys.stderr,
            )
            return 1

        http = make_session(max(32, args.tile_workers + 8))
        session_token = create_sv_session(http, api_key)
        tile_meta_map = fetch_tile_metadata(
            route_frames, http, session_token, api_key, tile_meta_path, args.force_metadata
        )

        v0, v0_coarse = build_camera_rays(args.width, args.height, args.fov)
        jobs = plan_tiles(
            route_frames=route_frames,
            tile_meta_map=tile_meta_map,
            frames_dir=frames_dir,
            v0_coarse=v0_coarse,
            target_zoom=args.zoom,
            pitch_deg=args.pitch,
            max_tiles=args.max_tiles,
        )

        render_frames(
            jobs=jobs,
            http=http,
            session_token=session_token,
            api_key=api_key,
            v0=v0,
            max_tiles=args.max_tiles,
            pano_workers=args.pano_workers,
            tile_workers=args.tile_workers,
        )

    encode_video(
        route_frames=route_frames,
        frames_dir=frames_dir,
        out_path=out_path,
        fps=args.fps,
        width=args.width,
        height=args.height,
        hud=args.hud,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
