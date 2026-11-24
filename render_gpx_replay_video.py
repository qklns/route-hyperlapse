"""Render a standalone, frame-synced GPX Replay map MP4 (zero API keys)."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw
import requests

DEFAULT_GPX = r"c:\Users\qkins\Downloads\route.gpx"
DEFAULT_META = r"c:\Users\qkins\Desktop\Antigravity\Thesis\hyperlapse\gsv_cache\metadata.json"
DEFAULT_OUT = r"c:\Users\qkins\Desktop\Antigravity\Thesis\hyperlapse\gpx_replay_map.mp4"
EARTH_RADIUS_M = 6_371_000.0

STYLE_URLS = {
    "outdoors": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        "Map: Esri / OSM",
    ),
    "streets": (
        "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "Map: OSM Streets",
    ),
    "light": (
        "https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png",
        "Map: CARTO / OSM",
    ),
    "dark": (
        "https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png",
        "Map: CARTO / OSM",
    ),
    "satellite": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "Map: Esri Imagery",
    ),
}


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = la2 - la1
    dlon = lo2 - lo1
    h = math.sin(dlat / 2.0) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(h)))


def latlon_to_world_px(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    n = 2**zoom
    x = (lon + 180.0) / 360.0 * n * 256.0
    lat_rad = math.radians(max(-85.0511, min(85.0511, lat)))
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n * 256.0
    return x, y


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.strip().lstrip("#")
    if len(h) != 6:
        return (0, 230, 255)
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b, g, r)


def load_gpx_points(gpx_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tree = ET.parse(gpx_path)
    root = tree.getroot()
    ns = {"g": "http://www.topografix.com/GPX/1/1"}
    trkpts = root.findall(".//g:trkpt", ns)
    if not trkpts:
        trkpts = root.findall(".//trkpt")
    if len(trkpts) < 2:
        raise ValueError(f"Expected >=2 trackpoints in {gpx_path}")

    lats = np.array([float(p.attrib["lat"]) for p in trkpts], dtype=np.float64)
    lons = np.array([float(p.attrib["lon"]) for p in trkpts], dtype=np.float64)
    cum_m = np.zeros(len(lats), dtype=np.float64)
    for i in range(1, len(lats)):
        cum_m[i] = cum_m[i - 1] + haversine_m(lats[i - 1], lons[i - 1], lats[i], lons[i])
    return lats, lons, cum_m


def choose_zoom_and_crop(
    lats: np.ndarray, lons: np.ndarray, out_w: int, out_h: int
) -> tuple[int, float, float, float, float]:
    target_ar = out_w / out_h
    chosen = None
    for z in range(14, 6, -1):
        xs = np.array([latlon_to_world_px(la, lo, z)[0] for la, lo in zip(lats, lons)])
        ys = np.array([latlon_to_world_px(la, lo, z)[1] for la, lo in zip(lats, lons)])
        min_x, max_x = float(xs.min()), float(xs.max())
        min_y, max_y = float(ys.min()), float(ys.max())
        span_x = max(50.0, max_x - min_x)
        span_y = max(50.0, max_y - min_y)
        pad_x = span_x * 1.36
        pad_y = span_y * 1.24
        if pad_x / pad_y < target_ar:
            win_h = pad_y
            win_w = win_h * target_ar
        else:
            win_w = pad_x
            win_h = win_w / target_ar
        chosen = (z, min_x, max_x, min_y, max_y, win_w, win_h)
        if 900.0 <= win_h <= 2600.0 and 800.0 <= win_w <= 2600.0:
            break

    z, min_x, max_x, min_y, max_y, win_w, win_h = chosen
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    return z, cx - win_w / 2.0, cy - win_h / 2.0, cx + win_w / 2.0, cy + win_h / 2.0


def fetch_basemap(
    style: str,
    zoom: int,
    crop_x0: float,
    crop_y0: float,
    crop_x1: float,
    crop_y1: float,
    out_w: int,
    out_h: int,
    cache_root: Path,
) -> tuple[np.ndarray, str]:
    url_tpl, attribution = STYLE_URLS[style]
    tx0 = max(0, int(math.floor(crop_x0 / 256.0)))
    tx1 = min((2**zoom) - 1, int(math.floor(crop_x1 / 256.0)))
    ty0 = max(0, int(math.floor(crop_y0 / 256.0)))
    ty1 = min((2**zoom) - 1, int(math.floor(crop_y1 / 256.0)))

    tile_dir = cache_root / "tiles" / style
    tile_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "ThesisGPXReplay/1.0"})

    coords = [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]
    print(f"Preparing {len(coords)} basemap tiles ({style}, z={zoom})...")

    def get_tile(tx: int, ty: int) -> tuple[int, int, Image.Image | None]:
        p = tile_dir / f"{zoom}_{tx}_{ty}.png"
        if p.exists() and p.stat().st_size > 60:
            try:
                with Image.open(p) as im:
                    return tx, ty, im.convert("RGB")
            except Exception:
                pass
        url = url_tpl.format(z=zoom, x=tx, y=ty)
        for _ in range(3):
            try:
                r = session.get(url, timeout=15)
                if r.status_code == 200 and len(r.content) > 60:
                    im = Image.open(io.BytesIO(r.content)).convert("RGB")
                    p.write_bytes(r.content)
                    return tx, ty, im
            except Exception:
                pass
        return tx, ty, None

    loaded: dict[tuple[int, int], Image.Image] = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(get_tile, tx, ty) for tx, ty in coords]
        for fut in as_completed(futs):
            tx, ty, im = fut.result()
            if im is not None:
                loaded[(tx, ty)] = im

    if not loaded:
        if style != "streets":
            print(f"Warning: {style} tiles failed; falling back to CARTO Voyager.")
            return fetch_basemap(
                "streets", zoom, crop_x0, crop_y0, crop_x1, crop_y1, out_w, out_h, cache_root
            )
        raise RuntimeError("Failed to download basemap tiles.")

    sample_im = next(iter(loaded.values()))
    tile_px = sample_im.size[0]
    scale = tile_px / 256.0

    canvas_w = (tx1 - tx0 + 1) * tile_px
    canvas_h = (ty1 - ty0 + 1) * tile_px
    canvas = Image.new("RGB", (canvas_w, canvas_h), (230, 235, 238))
    for (tx, ty), im in loaded.items():
        if im.size != (tile_px, tile_px):
            im = im.resize((tile_px, tile_px), Image.Resampling.BILINEAR)
        canvas.paste(im, ((tx - tx0) * tile_px, (ty - ty0) * tile_px))

    cx0 = (crop_x0 - tx0 * 256.0) * scale
    cy0 = (crop_y0 - ty0 * 256.0) * scale
    cx1 = (crop_x1 - tx0 * 256.0) * scale
    cy1 = (crop_y1 - ty0 * 256.0) * scale
    cropped = canvas.crop((cx0, cy0, cx1, cy1)).resize((out_w, out_h), Image.Resampling.LANCZOS)
    return np.asarray(cropped, dtype=np.uint8)[:, :, ::-1].copy(), attribution


def draw_marker(
    img_bgr: np.ndarray, x: float, y: float, angle_rad: float, icon: str
) -> None:
    ix, iy = int(round(x)), int(round(y))
    if icon == "dot":
        cv2.circle(img_bgr, (ix, iy), 14, (25, 25, 25), -1, cv2.LINE_AA)
        cv2.circle(img_bgr, (ix, iy), 11, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img_bgr, (ix, iy), 7, (30, 60, 235), -1, cv2.LINE_AA)
        return

    if icon == "car" or icon == "bicycle":
        cv2.circle(img_bgr, (ix, iy), 20, (31, 41, 55), -1, cv2.LINE_AA)
        cv2.circle(img_bgr, (ix, iy), 17, (255, 255, 255), -1, cv2.LINE_AA)
        if icon == "bicycle":
            cv2.circle(img_bgr, (ix - 6, iy + 3), 4, (30, 30, 30), 2, cv2.LINE_AA)
            cv2.circle(img_bgr, (ix + 6, iy + 3), 4, (30, 30, 30), 2, cv2.LINE_AA)
            cv2.line(img_bgr, (ix - 6, iy + 3), (ix, iy - 5), (30, 30, 30), 2, cv2.LINE_AA)
            cv2.line(img_bgr, (ix, iy - 5), (ix + 6, iy + 3), (30, 30, 30), 2, cv2.LINE_AA)
        else:
            # Death Cage (car) silhouette
            body = np.array(
                [
                    [ix - 10, iy - 1],
                    [ix - 7, iy - 8],
                    [ix + 7, iy - 8],
                    [ix + 10, iy - 1],
                    [ix + 11, iy + 5],
                    [ix - 11, iy + 5],
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(img_bgr, [body], (24, 24, 27), cv2.LINE_AA)
            windshield = np.array(
                [[ix - 6, iy - 6], [ix + 6, iy - 6], [ix + 8, iy - 1], [ix - 8, iy - 1]],
                dtype=np.int32,
            )
            cv2.fillPoly(img_bgr, [windshield], (254, 242, 224), cv2.LINE_AA)
            cv2.circle(img_bgr, (ix - 7, iy + 2), 2, (0, 220, 255), -1, cv2.LINE_AA)
            cv2.circle(img_bgr, (ix + 7, iy + 2), 2, (0, 220, 255), -1, cv2.LINE_AA)
            cv2.rectangle(img_bgr, (ix - 10, iy + 5), (ix - 6, iy + 9), (24, 24, 27), -1)
            cv2.rectangle(img_bgr, (ix + 6, iy + 5), (ix + 10, iy + 9), (24, 24, 27), -1)
        return

    # Directional arrow beacon (default)
    cv2.circle(img_bgr, (ix, iy), 18, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img_bgr, (ix, iy), 18, (30, 30, 30), 2, cv2.LINE_AA)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    perp_x, perp_y = -sin_a, cos_a
    tip = (int(round(x + cos_a * 12)), int(round(y + sin_a * 12)))
    left = (int(round(x - cos_a * 8 + perp_x * 9)), int(round(y - sin_a * 8 + perp_y * 9)))
    tail = (int(round(x - cos_a * 3)), int(round(y - sin_a * 3)))
    right = (int(round(x - cos_a * 8 - perp_x * 9)), int(round(y - sin_a * 8 - perp_y * 9)))
    poly = np.array([tip, left, tail, right], dtype=np.int32)
    cv2.fillPoly(img_bgr, [poly], (36, 48, 225), cv2.LINE_AA)


def draw_hud(img_bgr: np.ndarray, dist_km: float, total_km: float) -> None:
    h, _ = img_bgr.shape[:2]
    text = f"km {dist_km:.1f} / {total_km:.1f} km"
    x0, y0, x1, y1 = 22, h - 78, 305, h - 22
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 24, 32), -1)
    cv2.addWeighted(overlay, 0.78, img_bgr, 0.22, 0.0, img_bgr)
    cv2.rectangle(img_bgr, (x0, y0), (x1, y1), (152, 136, 44), 2)
    cv2.putText(
        img_bgr, text, (x0 + 14, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA
    )
    frac = max(0.0, min(1.0, dist_km / total_km if total_km > 0 else 0.0))
    bar_x0, bar_y0, bar_x1, bar_y1 = x0 + 14, y1 - 14, x1 - 14, y1 - 8
    cv2.rectangle(img_bgr, (bar_x0, bar_y0), (bar_x1, bar_y1), (75, 85, 99), -1)
    fill_x = int(round(bar_x0 + frac * (bar_x1 - bar_x0)))
    if fill_x > bar_x0:
        cv2.rectangle(img_bgr, (bar_x0, bar_y0), (fill_x, bar_y1), (0, 230, 255), -1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render synced GPX Replay map MP4.")
    parser.add_argument("--gpx", default=DEFAULT_GPX, help="Input GPX route path")
    parser.add_argument("--route-meta", default=DEFAULT_META, help="GSV metadata.json for frame sync")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output MP4 path")
    parser.add_argument(
        "--style",
        choices=list(STYLE_URLS.keys()),
        default="streets",
        help="Map basemap style (default: streets)",
    )
    parser.add_argument("--width", type=int, default=960, help="Output width (default: 960)")
    parser.add_argument("--height", type=int, default=1080, help="Output height (default: 1080)")
    parser.add_argument("--fps", type=int, default=30, help="Output FPS (default: 30)")
    parser.add_argument(
        "--duration", type=float, default=67.7, help="Duration if --route-meta is omitted"
    )
    parser.add_argument("--line-color", default="#FFFF00", help="Route hex color (default: #FFFF00)")
    parser.add_argument("--line-width", type=int, default=4, help="Route line width (default: 4)")
    parser.add_argument(
        "--no-follow-track",
        action="store_true",
        help="Draw full route statically instead of progressive trail",
    )
    parser.add_argument(
        "--icon",
        choices=["dot", "arrow", "car", "bicycle"],
        default="car",
        help="Moving marker style (default: car)",
    )
    parser.add_argument(
        "--blink-period",
        type=float,
        default=1.0,
        help="Seconds per ON/OFF blink phase for ahead track (0 disables; default: 1.0)",
    )
    parser.add_argument("--hud", action="store_true", help="Show km progress HUD")
    args = parser.parse_args()

    out_path = Path(args.out)
    cache_root = out_path.parent / "gpx_replay_cache"

    lats, lons, cum_m = load_gpx_points(Path(args.gpx))
    total_m = float(cum_m[-1])
    total_km = total_m / 1000.0

    meta_path = Path(args.route_meta) if args.route_meta else None
    if meta_path and meta_path.exists():
        meta = json.load(meta_path.open("r", encoding="utf-8"))
        frame_dists_m = [float(item["dist_m"]) for item in meta]
        print(f"Syncing {len(frame_dists_m)} frames directly to {meta_path}")
    else:
        n_frames = max(2, int(round(args.duration * args.fps)))
        frame_dists_m = [float(i / (n_frames - 1) * total_m) for i in range(n_frames)]
        print(f"Generating {n_frames} uniform frames over {args.duration:.2f}s")

    zoom, crop_x0, crop_y0, crop_x1, crop_y1 = choose_zoom_and_crop(
        lats, lons, args.width, args.height
    )
    base_bgr, attribution = fetch_basemap(
        args.style, zoom, crop_x0, crop_y0, crop_x1, crop_y1, args.width, args.height, cache_root
    )

    sx = args.width / (crop_x1 - crop_x0)
    sy = args.height / (crop_y1 - crop_y0)
    pts_px = np.zeros((len(lats), 2), dtype=np.float64)
    for i, (la, lo) in enumerate(zip(lats, lons)):
        wx, wy = latlon_to_world_px(la, lo, zoom)
        pts_px[i, 0] = (wx - crop_x0) * sx
        pts_px[i, 1] = (wy - crop_y0) * sy

    line_bgr = hex_to_bgr(args.line_color)
    follow_track = not args.no_follow_track

    if not follow_track:
        cv2.polylines(
            base_bgr, [np.round(pts_px).astype(np.int32)], False, (30, 30, 30), args.line_width + 3, cv2.LINE_AA
        )
        cv2.polylines(
            base_bgr, [np.round(pts_px).astype(np.int32)], False, line_bgr, args.line_width, cv2.LINE_AA
        )

    import imageio_ffmpeg

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_exe,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{args.width}x{args.height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(args.fps),
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
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    n_frames = len(frame_dists_m)
    preview_indices = {
        min(10, n_frames - 1),
        min(675, n_frames - 1),
        min(1355, n_frames - 1),
        n_frames - 1,
    }
    if len(preview_indices) < 4 and n_frames >= 4:
        preview_indices = {int(x) for x in np.linspace(0, n_frames - 1, 4, dtype=int)}
    preview_frames: dict[int, np.ndarray] = {}

    for i, d_m in enumerate(frame_dists_m):
        frame = base_bgr.copy()
        seg_idx = int(np.searchsorted(cum_m, d_m, side="right")) - 1
        seg_idx = max(0, min(len(cum_m) - 2, seg_idx))
        seg_len = cum_m[seg_idx + 1] - cum_m[seg_idx]
        t = (d_m - cum_m[seg_idx]) / seg_len if seg_len > 1e-6 else 0.0
        t = max(0.0, min(1.0, t))

        cur_x = pts_px[seg_idx, 0] + t * (pts_px[seg_idx + 1, 0] - pts_px[seg_idx, 0])
        cur_y = pts_px[seg_idx, 1] + t * (pts_px[seg_idx + 1, 1] - pts_px[seg_idx, 1])

        # Lookahead tangent for smooth arrow heading
        la_idx = int(np.searchsorted(cum_m, min(total_m, d_m + 350.0)))
        lb_idx = int(np.searchsorted(cum_m, max(0.0, d_m - 350.0)))
        la_idx = min(len(pts_px) - 1, max(0, la_idx))
        lb_idx = min(len(pts_px) - 1, max(0, lb_idx))
        dx = pts_px[la_idx, 0] - pts_px[lb_idx, 0]
        dy = pts_px[la_idx, 1] - pts_px[lb_idx, 1]
        angle_rad = math.atan2(dy, dx) if (dx * dx + dy * dy) > 1e-4 else math.pi / 2.0

        if follow_track:
            t_sec = i / args.fps
            show_ahead = args.blink_period > 0 and (int(t_sec // args.blink_period) % 2 == 0)
            if show_ahead and (seg_idx + 1) < len(pts_px):
                ahead = np.vstack([[[cur_x, cur_y]], pts_px[seg_idx + 1 :]])
                ahead_i = np.round(ahead).astype(np.int32)
                cv2.polylines(
                    frame, [ahead_i], False, (25, 25, 25), args.line_width + 2, cv2.LINE_AA
                )
                cv2.polylines(
                    frame, [ahead_i], False, line_bgr, args.line_width, cv2.LINE_AA
                )

            trail = np.vstack([pts_px[: seg_idx + 1], [[cur_x, cur_y]]])
            trail_i = np.round(trail).astype(np.int32)
            cv2.polylines(frame, [trail_i], False, (25, 25, 25), args.line_width + 2, cv2.LINE_AA)
            cv2.polylines(frame, [trail_i], False, line_bgr, args.line_width, cv2.LINE_AA)

        draw_marker(frame, cur_x, cur_y, angle_rad, args.icon)
        if args.hud:
            draw_hud(frame, d_m / 1000.0, total_km)

        rgb = frame[:, :, ::-1]
        if proc.stdin is not None:
            proc.stdin.write(rgb.tobytes())

        if i in preview_indices:
            preview_frames[i] = rgb.copy()

        if (i + 1) % 400 == 0 or (i + 1) == n_frames:
            print(f"Rendered {i + 1}/{n_frames} map frames...")

    if proc.stdin is not None:
        proc.stdin.close()
    stderr_out = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"FFmpeg failed with exit code {ret}:\n{stderr_out[-1000:]}")

    # Save 2x2 verification collage
    ordered = [preview_frames[k] for k in sorted(preview_frames.keys())]
    if len(ordered) == 4:
        tw, th = args.width // 2, args.height // 2
        tiles = [
            np.asarray(Image.fromarray(fr).resize((tw, th), Image.Resampling.LANCZOS))
            for fr in ordered
        ]
        grid = np.vstack([np.hstack([tiles[0], tiles[1]]), np.hstack([tiles[2], tiles[3]])])
        grid_path = out_path.parent / "preview_gpx_replay.jpg"
        Image.fromarray(grid).save(grid_path, quality=92)
        print(f"Saved preview collage to {grid_path}")

    print(f"Finished {out_path} ({n_frames} frames, {n_frames / args.fps:.2f}s @ {args.fps}fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
