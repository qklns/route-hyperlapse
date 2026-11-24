"""Build a free street-level hyperlapse MP4 from a GPX route using KartaView v2.0."""

from __future__ import annotations

import argparse
import bisect
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import requests

KARTAVIEW_PHOTO_API = "https://api.kartaview.org/2.0/photo/"
DEFAULT_GPX = r"c:\Users\qkins\Downloads\route.gpx"
DEFAULT_OUT = r"c:\Users\qkins\Desktop\Antigravity\Thesis\hyperlapse\route_hyperlapse.mp4"
EARTH_RADIUS_M = 6_371_000.0
RESAMPLE_STEP_M = 25.0


@dataclass
class RouteData:
    lat: np.ndarray
    lon: np.ndarray
    dist_m: np.ndarray
    x: np.ndarray
    y: np.ndarray
    bearing: np.ndarray
    lat0_rad: float
    lon0_rad: float

    @property
    def total_dist_m(self) -> float:
        return float(self.dist_m[-1])

    def to_local_xy(self, lat: float, lon: float) -> tuple[float, float]:
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        x = EARTH_RADIUS_M * (lon_rad - self.lon0_rad) * math.cos(self.lat0_rad)
        y = EARTH_RADIUS_M * (lat_rad - self.lat0_rad)
        return x, y

    def project_onto_route(
        self, lat: float, lon: float, min_idx: int, max_idx: int
    ) -> tuple[int, float, float, float]:
        lo = max(0, min_idx)
        hi = min(len(self.lat), max_idx)
        if lo >= hi:
            lo = 0
            hi = len(self.lat)
        px, py = self.to_local_xy(lat, lon)
        dx = self.x[lo:hi] - px
        dy = self.y[lo:hi] - py
        d2 = dx * dx + dy * dy
        rel_idx = int(np.argmin(d2))
        best_idx = lo + rel_idx
        return (
            best_idx,
            float(self.dist_m[best_idx]),
            float(math.sqrt(d2[rel_idx])),
            float(self.bearing[best_idx]),
        )


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


def load_route(gpx_path: Path) -> RouteData:
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
    sample_dists = np.arange(0.0, total_m, RESAMPLE_STEP_M, dtype=np.float64)
    if len(sample_dists) == 0 or sample_dists[-1] < total_m:
        sample_dists = np.append(sample_dists, total_m)

    r_lat = np.interp(sample_dists, cum_m, raw_lat)
    r_lon = np.interp(sample_dists, cum_m, raw_lon)

    lat0_rad = math.radians(float(np.mean(r_lat)))
    lon0_rad = math.radians(float(np.mean(r_lon)))
    cos_lat0 = math.cos(lat0_rad)
    r_x = EARTH_RADIUS_M * (np.radians(r_lon) - lon0_rad) * cos_lat0
    r_y = EARTH_RADIUS_M * (np.radians(r_lat) - lat0_rad)

    n = len(r_lat)
    r_bearing = np.zeros(n, dtype=np.float64)
    for i in range(n):
        i0 = max(0, i - 2)
        i1 = min(n - 1, i + 2)
        if i0 == i1:
            r_bearing[i] = 0.0
        else:
            r_bearing[i] = compute_bearing(r_lat[i0], r_lon[i0], r_lat[i1], r_lon[i1])

    return RouteData(
        lat=r_lat,
        lon=r_lon,
        dist_m=sample_dists,
        x=r_x,
        y=r_y,
        bearing=r_bearing,
        lat0_rad=lat0_rad,
        lon0_rad=lon0_rad,
    )


def api_get_json(session: requests.Session, url: str, params: dict) -> dict:
    delay = 0.6
    for attempt in range(5):
        try:
            resp = session.get(url, params=params, timeout=20)
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(delay * (2**attempt))
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError):
            if attempt == 4:
                raise
            time.sleep(delay * (2**attempt))
    return {}


def extract_photo_url(item: dict) -> str:
    for key in ("imageLthUrl", "fileurlLTh", "imageProcUrl", "fileurlProc"):
        val = item.get(key)
        if val and isinstance(val, str) and val.startswith("http"):
            if "openstreetcam.org" in val and item.get("imageLthUrl"):
                continue
            return val
    return ""


def parse_float(val, default: float = 0.0) -> float:
    try:
        if val is None or val == "":
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def parse_int(val, default: int = 0) -> int:
    try:
        if val is None or val == "":
            return default
        return int(float(val))
    except (TypeError, ValueError):
        return default


def score_candidate_sequence(
    items: list[dict], route_brg: float, prev_date: str | None
) -> float:
    score = 0.0
    best_item_score = -1e9
    for it in items:
        s = 0.0
        proj = str(it.get("projection") or "").upper()
        w = parse_int(it.get("width"))
        h = parse_int(it.get("height"))
        hdg = parse_float(it.get("heading"), 0.0)
        date_str = str(it.get("dateAdded") or "")
        year = parse_int(date_str[:4], 0)

        if proj == "PLANE":
            s += 40.0
        elif proj == "SPHERE":
            s += 15.0

        if w > h:
            s += 25.0
        if w >= 1280:
            s += 15.0

        if abs(hdg) > 0.01:
            diff = ang_diff(hdg, route_brg)
            if diff <= 50.0:
                s += 60.0 - diff * 0.5
            else:
                s -= 40.0

        if 2020 <= year <= 2026:
            s += 20.0 + (year - 2020) * 2.0
        elif year > 0:
            s += 5.0

        if prev_date and date_str[:10] == prev_date:
            s += 35.0

        if s > best_item_score:
            best_item_score = s

    score += best_item_score
    return score


def fetch_sequence_items(
    session: requests.Session, sid: str, fetched_seqs: dict[str, list[dict]]
) -> list[dict]:
    if sid in fetched_seqs:
        return fetched_seqs[sid]

    all_items: list[dict] = []
    for page in range(1, 11):
        time.sleep(0.05)
        payload = api_get_json(
            session,
            KARTAVIEW_PHOTO_API,
            {"sequenceId": sid, "itemsPerPage": 150, "page": page},
        )
        res = payload.get("result") or {}
        batch = res.get("data") or []
        if not batch:
            break
        all_items.extend(batch)
        if not res.get("hasMoreData"):
            break

    all_items.sort(key=lambda x: parse_int(x.get("sequenceIndex"), 0))
    fetched_seqs[sid] = all_items
    return all_items


def compute_sequence_motion_bearings(items: list[dict]) -> list[float | None]:
    n = len(items)
    bearings: list[float | None] = [None] * n
    coords = [(parse_float(it.get("lat")), parse_float(it.get("lng"))) for it in items]
    for k in range(n):
        lat0, lon0 = coords[k]
        found = None
        for step in (1, 2, 3):
            if k + step < n:
                lat1, lon1 = coords[k + step]
                if haversine_m(lat0, lon0, lat1, lon1) >= 5.0:
                    found = compute_bearing(lat0, lon0, lat1, lon1)
                    break
        if found is None:
            for step in (1, 2, 3):
                if k - step >= 0:
                    lat1, lon1 = coords[k - step]
                    if haversine_m(lat1, lon1, lat0, lon0) >= 5.0:
                        found = compute_bearing(lat1, lon1, lat0, lon0)
                        break
        bearings[k] = found
    return bearings


def evaluate_sequence_on_route(
    items: list[dict],
    route: RouteData,
    cur_dist_m: float,
    cur_idx: int,
    max_perp_m: float = 70.0,
) -> tuple[list[dict], float]:
    n_route = len(route.lat)
    min_idx = max(0, cur_idx - 20)
    max_idx = min(n_route, cur_idx + 800)
    motion_bearings = compute_sequence_motion_bearings(items)

    accepted: list[dict] = []
    max_reached_m = cur_dist_m

    for k, it in enumerate(items):
        url = extract_photo_url(it)
        if not url:
            continue
        lat = parse_float(it.get("lat"), 999.0)
        lon = parse_float(it.get("lng"), 999.0)
        if abs(lat) > 90.0 or abs(lon) > 180.0:
            continue

        _, route_dist_m, perp_dist_m, route_bearing = route.project_onto_route(
            lat, lon, min_idx, max_idx
        )
        if perp_dist_m > max_perp_m or route_dist_m < cur_dist_m - 50.0:
            continue

        proj = str(it.get("projection") or "PLANE").upper()
        hdg = parse_float(it.get("heading"), 0.0)
        mot_brg = motion_bearings[k]

        if mot_brg is not None and ang_diff(mot_brg, route_bearing) > 55.0:
            continue
        if abs(hdg) > 0.01 and proj != "SPHERE" and ang_diff(hdg, route_bearing) > 55.0:
            continue
        if mot_brg is None and abs(hdg) <= 0.01:
            continue

        matched = {
            "id": str(it["id"]),
            "sequenceId": str(it.get("sequenceId", "")),
            "sequenceIndex": parse_int(it.get("sequenceIndex"), k),
            "route_dist_m": round(route_dist_m, 2),
            "perp_dist_m": round(perp_dist_m, 2),
            "route_bearing": round(route_bearing, 2),
            "heading": round(hdg, 2),
            "projection": proj,
            "width": parse_int(it.get("width"), 0),
            "height": parse_int(it.get("height"), 0),
            "dateAdded": str(it.get("dateAdded") or ""),
            "url": url,
        }
        accepted.append(matched)
        if route_dist_m > max_reached_m:
            max_reached_m = route_dist_m

    return accepted, max_reached_m


def discover_route_photos(
    route: RouteData, cache_path: Path, force: bool
) -> dict[str, dict]:
    if cache_path.exists() and not force:
        with cache_path.open("r", encoding="utf-8") as f:
            cached = json.load(f)
        print(f"Loaded {len(cached)} matched photos from {cache_path}")
        return cached

    session = requests.Session()
    session.headers.update({"User-Agent": "ThesisRouteHyperlapse/1.0"})

    fetched_seqs: dict[str, list[dict]] = {}
    evaluated_max_m: dict[str, float] = {}
    all_matched: dict[str, dict] = {}
    prev_date: str | None = None

    total_m = route.total_dist_m
    cur_dist_m = 0.0
    next_log_km = 0.0
    api_calls = 0

    while cur_dist_m <= total_m:
        cur_idx = min(len(route.lat) - 1, max(0, int(cur_dist_m / RESAMPLE_STEP_M)))
        lat = float(route.lat[cur_idx])
        lon = float(route.lon[cur_idx])
        route_brg = float(route.bearing[cur_idx])

        nearby_items: list[dict] = []
        for radius in (60, 150):
            time.sleep(0.05)
            api_calls += 1
            payload = api_get_json(
                session,
                KARTAVIEW_PHOTO_API,
                {"lat": lat, "lng": lon, "radius": radius, "itemsPerPage": 50},
            )
            nearby_items = (payload.get("result") or {}).get("data") or []
            if nearby_items:
                break

        by_seq: dict[str, list[dict]] = {}
        for it in nearby_items:
            sid = str(it.get("sequenceId") or "")
            if not sid:
                continue
            if sid in evaluated_max_m and evaluated_max_m[sid] <= cur_dist_m + 25.0:
                continue
            by_seq.setdefault(sid, []).append(it)

        ranked_sids = sorted(
            by_seq.keys(),
            key=lambda s: score_candidate_sequence(by_seq[s], route_brg, prev_date),
            reverse=True,
        )

        best_accepted: list[dict] = []
        best_max_m = cur_dist_m
        best_sid: str | None = None

        for sid in ranked_sids[:5]:
            was_cached = sid in fetched_seqs
            seq_items = fetch_sequence_items(session, sid, fetched_seqs)
            if not was_cached:
                api_calls += max(1, (len(seq_items) + 149) // 150)

            accepted, max_m = evaluate_sequence_on_route(
                seq_items, route, cur_dist_m, cur_idx, max_perp_m=70.0
            )
            if not accepted:
                accepted, max_m = evaluate_sequence_on_route(
                    seq_items, route, cur_dist_m, cur_idx, max_perp_m=120.0
                )

            evaluated_max_m[sid] = max(evaluated_max_m.get(sid, -1.0), max_m)
            for m in accepted:
                all_matched[m["id"]] = m

            if max_m > best_max_m:
                best_max_m = max_m
                best_accepted = accepted
                best_sid = sid
            if max_m - cur_dist_m >= 200.0:
                break

        if best_accepted and best_max_m > cur_dist_m + 50.0:
            if best_sid and best_accepted[0].get("dateAdded"):
                prev_date = best_accepted[0]["dateAdded"][:10]
            cur_dist_m = best_max_m + 50.0
        else:
            cur_dist_m += 250.0

        if cur_dist_m / 1000.0 >= next_log_km:
            print(
                f"Discovery: {min(cur_dist_m, total_m)/1000.0:6.1f} / {total_m/1000.0:.1f} km | "
                f"matched={len(all_matched)} | seqs={len(fetched_seqs)} | api_calls~={api_calls}"
            )
            next_log_km += 10.0

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(all_matched, f)
    print(f"Saved {len(all_matched)} matched photos to {cache_path}")
    return all_matched


def select_frames(
    all_matched: dict[str, dict],
    total_dist_m: float,
    density: float,
    out_path: Path,
) -> list[dict]:
    photos = sorted(all_matched.values(), key=lambda p: p["route_dist_m"])
    dists = [p["route_dist_m"] for p in photos]

    step_m = 1000.0 / density
    window_m = step_m * 0.65

    selected: list[dict] = []
    last_selected_dist_m = -1e9
    prev_seq_id: str | None = None
    prev_date: str | None = None

    target_d = 0.0
    while target_d <= total_dist_m:
        lo = bisect.bisect_left(dists, target_d - window_m)
        hi = bisect.bisect_right(dists, target_d + window_m)

        best_photo = None
        best_score = -1e9

        for idx in range(lo, hi):
            p = photos[idx]
            if p["route_dist_m"] <= last_selected_dist_m + 10.0:
                continue

            s = 0.0
            if prev_seq_id and p["sequenceId"] == prev_seq_id:
                s += 80.0
            elif prev_date and p["dateAdded"][:10] == prev_date:
                s += 25.0

            if p["projection"] == "PLANE":
                s += 40.0
            if p["width"] > p["height"]:
                s += 25.0

            s -= abs(p["route_dist_m"] - target_d) * 0.35
            s -= p["perp_dist_m"] * 0.4

            if s > best_score:
                best_score = s
                best_photo = p

        if best_photo is not None:
            selected.append(best_photo)
            last_selected_dist_m = best_photo["route_dist_m"]
            prev_seq_id = best_photo["sequenceId"]
            prev_date = best_photo["dateAdded"][:10] if best_photo["dateAdded"] else None

        target_d += step_m

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)

    if selected:
        gaps = [
            selected[i]["route_dist_m"] - selected[i - 1]["route_dist_m"]
            for i in range(1, len(selected))
        ]
        max_gap = max(gaps) if gaps else 0.0
        unique_seqs = len({p["sequenceId"] for p in selected})
        print(
            f"Selected {len(selected)} frames covering "
            f"{selected[0]['route_dist_m']/1000.0:.2f}–{selected[-1]['route_dist_m']/1000.0:.2f} km | "
            f"largest gap={max_gap:.0f} m | unique sequences={unique_seqs}"
        )
    else:
        print("Selected 0 frames!")
    return selected


def download_frames(selected: list[dict], frames_dir: Path, workers: int) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    thread_local = threading.local()

    def get_session() -> requests.Session:
        if not hasattr(thread_local, "session"):
            s = requests.Session()
            s.headers.update({"User-Agent": "ThesisRouteHyperlapse/1.0"})
            thread_local.session = s
        return thread_local.session

    todo = []
    for p in selected:
        dest = frames_dir / f"{p['id']}.jpg"
        if dest.exists() and dest.stat().st_size > 2048:
            continue
        todo.append((p["url"], dest))

    print(
        f"Downloading {len(todo)} frames ({len(selected) - len(todo)} already cached)..."
    )
    if not todo:
        return

    def fetch_one(url: str, dest: Path) -> bool:
        session = get_session()
        for attempt in range(4):
            try:
                r = session.get(url, timeout=20)
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(0.5 * (2**attempt))
                    continue
                if r.status_code != 200 or len(r.content) <= 2048:
                    return False
                tmp = dest.with_suffix(".tmp")
                tmp.write_bytes(r.content)
                tmp.replace(dest)
                return True
            except requests.RequestException:
                time.sleep(0.5 * (2**attempt))
        return False

    done = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fetch_one, u, d) for u, d in todo]
        for fut in as_completed(futures):
            done += 1
            if not fut.result():
                failed += 1
            if done % 200 == 0 or done == len(todo):
                print(f"Download progress: {done}/{len(todo)} (failed={failed})")


def process_frame_image(
    img_path: Path, frame_meta: dict, width: int, height: int, hud: bool
) -> Image.Image | None:
    try:
        with Image.open(img_path) as raw:
            im = ImageOps.exif_transpose(raw).convert("RGB")
    except Exception:
        return None

    iw, ih = im.size
    if iw < 64 or ih < 64:
        return None

    proj = frame_meta.get("projection", "PLANE")
    if proj == "SPHERE" and iw >= ih * 1.8:
        yaw_offset = (frame_meta["route_bearing"] - frame_meta["heading"]) % 360.0
        shift_px = int(round((yaw_offset / 360.0) * iw))
        arr = np.asarray(im)
        arr = np.roll(arr, shift=-shift_px, axis=1)
        im = Image.fromarray(arr)
        slice_w = max(16, int(round(iw * (90.0 / 360.0))))
        slice_h = min(ih, int(round(slice_w * (height / width))))
        cx = iw // 2
        cy = ih // 2
        x0 = max(0, cx - slice_w // 2)
        y0 = max(0, cy - slice_h // 2)
        im = im.crop((x0, y0, x0 + slice_w, y0 + slice_h))
    else:
        target_ar = width / height
        cur_ar = iw / ih
        if cur_ar < target_ar:
            new_h = int(round(iw / target_ar))
            excess = ih - new_h
            top = int(round(excess * 0.35))
            im = im.crop((0, top, iw, top + new_h))
        elif cur_ar > target_ar:
            new_w = int(round(ih * target_ar))
            left = (iw - new_w) // 2
            im = im.crop((left, 0, left + new_w, ih))

    im = im.resize((width, height), Image.Resampling.LANCZOS)

    mean_luma = float(np.asarray(im, dtype=np.float32).mean())
    if mean_luma < 18.0 or mean_luma > 245.0:
        return None

    if hud:
        draw = ImageDraw.Draw(im)
        label = f"km {frame_meta['route_dist_m']/1000.0:.1f} | {frame_meta['dateAdded'][:10]}"
        draw.rectangle((12, height - 38, 260, height - 12), fill=(0, 0, 0))
        draw.text((20, height - 33), label, fill=(255, 255, 255))

    return im


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
        label = f"km {meta['route_dist_m']/1000.0:.1f} | {meta['dateAdded'][:10]}"
        draw.rectangle((4, 4, 175, 22), fill=(0, 0, 0))
        draw.text((8, 7), label, fill=(255, 255, 255), font=font)
        col = slot % 4
        row = slot // 4
        canvas.paste(tile, (col * tile_w, row * tile_h))

    grid_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(grid_path, quality=92)
    print(f"Saved preview grid to {grid_path}")


def encode_video(
    selected: list[dict],
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
            "20",
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
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not cv_writer.isOpened():
            raise RuntimeError("Neither ffmpeg nor OpenCV VideoWriter could be opened.")

    rendered = 0
    skipped = 0
    sample_stride = max(1, len(selected) // 32)
    preview_candidates: list[tuple[Image.Image, dict]] = []

    for i, meta in enumerate(selected):
        img_path = frames_dir / f"{meta['id']}.jpg"
        if not img_path.exists():
            skipped += 1
            continue
        im = process_frame_image(img_path, meta, width, height, hud)
        if im is None:
            skipped += 1
            continue

        if proc is not None and proc.stdin is not None:
            proc.stdin.write(im.tobytes())
        elif cv_writer is not None:
            arr_bgr = np.asarray(im)[:, :, ::-1]
            cv_writer.write(arr_bgr)

        if i % sample_stride == 0 or i == len(selected) - 1:
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

    grid_path = out_path.parent / "preview_grid.jpg"
    create_preview_grid(preview_candidates, grid_path)
    print(
        f"Finished encoding {rendered} frames (skipped={skipped}) -> {out_path} "
        f"({rendered / fps:.2f}s @ {fps}fps)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a free street-level hyperlapse MP4 from a GPX route via KartaView."
    )
    parser.add_argument("--gpx", default=DEFAULT_GPX, help="Path to input GPX route")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Path to output MP4")
    parser.add_argument(
        "--density", type=float, default=8.0, help="Target frames per km (default: 8.0)"
    )
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS (default: 30)")
    parser.add_argument("--width", type=int, default=1280, help="Output width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Output height (default: 720)")
    parser.add_argument("--workers", type=int, default=10, help="Download workers (default: 10)")
    parser.add_argument("--hud", action="store_true", help="Overlay km and date on video")
    parser.add_argument(
        "--force-metadata", action="store_true", help="Ignore cached matched_photos.json"
    )
    args = parser.parse_args()

    gpx_path = Path(args.gpx)
    out_path = Path(args.out)
    cache_dir = out_path.parent / "cache"
    matched_path = cache_dir / "matched_photos.json"
    selected_path = cache_dir / "selected_frames.json"
    frames_dir = cache_dir / "frames"

    route = load_route(gpx_path)
    print(
        f"Loaded route {gpx_path}: {route.total_dist_m/1000.0:.2f} km "
        f"({len(route.lat)} samples @ {RESAMPLE_STEP_M:.0f}m)"
    )

    all_matched = discover_route_photos(route, matched_path, args.force_metadata)
    selected = select_frames(all_matched, route.total_dist_m, args.density, selected_path)
    if not selected:
        print("No frames matched the route.", file=sys.stderr)
        return 1

    download_frames(selected, frames_dir, args.workers)
    encode_video(selected, frames_dir, out_path, args.fps, args.width, args.height, args.hud)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
