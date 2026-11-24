"""Generate a minimal Video + Street Route Replay HTML viewer (zero API keys)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

DEFAULT_GPX = r"c:\Users\qkins\Downloads\route.gpx"
DEFAULT_META = r"c:\Users\qkins\Desktop\Antigravity\Thesis\hyperlapse\gsv_cache\metadata.json"
DEFAULT_OUT = r"c:\Users\qkins\Desktop\Antigravity\Thesis\hyperlapse\gpx_replay.html"
EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = la2 - la1
    dlon = lo2 - lo1
    h = math.sin(dlat / 2.0) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(h)))


def load_downsampled_route(gpx_path: Path, max_points: int = 1200) -> list[list[float]]:
    tree = ET.parse(gpx_path)
    root = tree.getroot()
    ns = {"g": "http://www.topografix.com/GPX/1/1"}
    trkpts = root.findall(".//g:trkpt", ns)
    if not trkpts:
        trkpts = root.findall(".//trkpt")
    if len(trkpts) < 2:
        raise ValueError(f"Expected >=2 trackpoints in {gpx_path}")

    raw_lat = [float(p.attrib["lat"]) for p in trkpts]
    raw_lon = [float(p.attrib["lon"]) for p in trkpts]

    cum_km = [0.0]
    for i in range(1, len(raw_lat)):
        cum_km.append(
            cum_km[-1] + haversine_m(raw_lat[i - 1], raw_lon[i - 1], raw_lat[i], raw_lon[i]) / 1000.0
        )

    total_km = cum_km[-1]
    n_out = min(max_points, len(raw_lat))
    out: list[list[float]] = []

    j = 0
    for k in range(n_out):
        target_km = (k / (n_out - 1)) * total_km
        while j + 1 < len(cum_km) and cum_km[j + 1] < target_km:
            j += 1
        if j + 1 < len(cum_km):
            seg = cum_km[j + 1] - cum_km[j]
            t = (target_km - cum_km[j]) / seg if seg > 1e-9 else 0.0
            lat = raw_lat[j] + t * (raw_lat[j + 1] - raw_lat[j])
            lon = raw_lon[j] + t * (raw_lon[j + 1] - raw_lon[j])
        else:
            lat, lon = raw_lat[-1], raw_lon[-1]
        out.append([round(lon, 6), round(lat, 6), round(target_km, 3)])
    return out


def load_frame_dists_km(meta_path: Path) -> list[float]:
    if not meta_path.exists():
        return []
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return [round(float(item["dist_m"]) / 1000.0, 3) for item in meta]


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover" />
  <title>Route Video &amp; Map</title>
  <link rel="stylesheet" href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" />
  <script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
  <style>
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      padding: 0;
      background: #ffffff;
      color: #1f2937;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      overflow-x: hidden;
    }
    body {
      padding: max(12px, env(safe-area-inset-top)) max(16px, env(safe-area-inset-right)) max(14px, env(safe-area-inset-bottom)) max(16px, env(safe-area-inset-left));
    }
    .selector-bar {
      max-width: 1440px;
      margin: 0 auto 12px;
      padding: 8px 12px;
      border: 1px solid #d1d5db;
      border-radius: 6px;
      background: #f9fafb;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .selector-group {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      font-size: 13.5px;
      min-width: 0;
      max-width: 100%;
    }
    .selector-group label.title {
      font-weight: 600;
      color: #111827;
    }
    select, .file-btn, .route-pill {
      min-height: 38px;
      padding: 6px 10px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: #ffffff;
      color: #1f2937;
      font-size: 13px;
      max-width: 100%;
      min-width: 0;
    }
    select, .file-btn {
      cursor: pointer;
    }
    .file-btn {
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      white-space: nowrap;
    }
    .file-btn:hover { background: #f3f4f6; }
    .file-btn input { display: none; }
    .route-pill {
      font-weight: 500;
      display: inline-flex;
      align-items: center;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .split-grid {
      max-width: 1440px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      align-items: stretch;
    }
    .pane {
      height: calc(100dvh - 96px);
      min-height: 360px;
      max-height: 760px;
      border: 1px solid #d1d5db;
      border-radius: 6px;
      overflow: hidden;
      background: #111111;
    }
    .pane.map-pane {
      background: #eae6df;
    }
    video {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      background: #111111;
    }
    #map {
      width: 100%;
      height: 100%;
      touch-action: pan-x pan-y;
    }
    @media (max-width: 767px), (max-width: 1024px) and (orientation: portrait) {
      .split-grid {
        grid-template-columns: 1fr;
        gap: 10px;
      }
      .pane.video-pane {
        aspect-ratio: 16 / 9;
        height: auto;
        min-height: 0;
        max-height: 36dvh;
      }
      .pane.map-pane {
        height: calc(100dvh - 36dvh - 135px);
        min-height: 260px;
        max-height: none;
      }
    }
    @media (max-width: 767px) {
      body {
        padding: max(8px, env(safe-area-inset-top)) max(10px, env(safe-area-inset-right)) max(10px, env(safe-area-inset-bottom)) max(10px, env(safe-area-inset-left));
      }
      .selector-bar {
        padding: 8px 10px;
        gap: 8px;
        margin-bottom: 10px;
      }
      select {
        font-size: 16px;
        flex: 1 1 160px;
      }
    }
  </style>
</head>
<body>
  <div class="selector-bar">
    <div class="selector-group">
      <label class="title" for="videoPreset">Video:</label>
      <select id="videoPreset">
        <option value="route_hyperlapse_gsv.mp4|full">Google Street View (640x360, Full Route)</option>
        <option value="route_hyperlapse.mp4|full">KartaView HD (1280x720, Full Route)</option>
        <option value="route_hyperlapse_gsv_1080p_part1.mp4|part1">Google Street View 1080p Part 1 (0–96 km)</option>
      </select>
      <label class="file-btn">
        Load Video
        <input id="localVideoInput" type="file" accept="video/*" />
      </label>
    </div>

    <div class="selector-group">
      <label class="title">Route:</label>
      <span id="routeNameLabel" class="route-pill">__ROUTE_NAME__ (__TOTAL_KM__ km)</span>
      <label class="file-btn">
        Load Route (.gpx)
        <input id="localGpxInput" type="file" accept=".gpx,application/gpx+xml" />
      </label>
    </div>
  </div>

  <div class="split-grid">
    <div class="pane video-pane">
      <video id="videoEl" controls playsinline webkit-playsinline preload="metadata" src="route_hyperlapse_gsv.mp4"></video>
    </div>
    <div class="pane map-pane">
      <div id="map"></div>
    </div>
  </div>

  <script>
    const INITIAL_ROUTE = __ROUTE_JSON__;
    const FRAME_DISTS_KM = __FRAME_DISTS_JSON__;
    const PART1_FRAMES = __PART1_FRAMES__;

    let routePts = INITIAL_ROUTE;
    let totalKm = routePts[routePts.length - 1][2];
    let currentDistKm = 0;
    let videoSyncMode = "full";

    const videoEl = document.getElementById("videoEl");

    function computeBounds(pts) {
      let minLon = Infinity, minLat = Infinity, maxLon = -Infinity, maxLat = -Infinity;
      for (const p of pts) {
        if (p[0] < minLon) minLon = p[0];
        if (p[1] < minLat) minLat = p[1];
        if (p[0] > maxLon) maxLon = p[0];
        if (p[1] > maxLat) maxLat = p[1];
      }
      return [[minLon, minLat], [maxLon, maxLat]];
    }

    const map = new maplibregl.Map({
      container: "map",
      style: "https://tiles.openfreemap.org/styles/bright",
      bounds: computeBounds(routePts),
      fitBoundsOptions: { padding: 36 },
      attributionControl: false
    });

    function sampleRouteAtKm(distKm) {
      const n = routePts.length;
      if (distKm <= routePts[0][2]) {
        return { lon: routePts[0][0], lat: routePts[0][1], idx: 0 };
      }
      if (distKm >= routePts[n - 1][2]) {
        return { lon: routePts[n - 1][0], lat: routePts[n - 1][1], idx: n - 1 };
      }
      let lo = 0, hi = n - 1;
      while (lo + 1 < hi) {
        const mid = (lo + hi) >> 1;
        if (routePts[mid][2] <= distKm) lo = mid;
        else hi = mid;
      }
      const p0 = routePts[lo], p1 = routePts[lo + 1];
      const seg = p1[2] - p0[2];
      const t = seg > 1e-9 ? (distKm - p0[2]) / seg : 0;
      return {
        lon: p0[0] + t * (p1[0] - p0[0]),
        lat: p0[1] + t * (p1[1] - p0[1]),
        idx: lo
      };
    }

    function createDeathCageIconData() {
      const size = 128;
      const canvas = document.createElement("canvas");
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, size, size);

      // Subtle white circular badge so the car pops cleanly on street map
      ctx.beginPath();
      ctx.arc(64, 64, 52, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255, 255, 255, 0.94)";
      ctx.fill();
      ctx.lineWidth = 6;
      ctx.strokeStyle = "#1f2937";
      ctx.stroke();

      // Death Cage (car) front silhouette
      ctx.fillStyle = "#111827";
      ctx.beginPath();
      ctx.moveTo(31, 63);
      ctx.lineTo(41, 41);
      ctx.lineTo(87, 41);
      ctx.lineTo(97, 63);
      ctx.lineTo(101, 79);
      ctx.lineTo(27, 79);
      ctx.closePath();
      ctx.fill();

      // Windshield
      ctx.fillStyle = "#e0f2fe";
      ctx.beginPath();
      ctx.moveTo(45, 46);
      ctx.lineTo(83, 46);
      ctx.lineTo(90, 60);
      ctx.lineTo(38, 60);
      ctx.closePath();
      ctx.fill();

      // Headlights
      ctx.fillStyle = "#facc15";
      ctx.beginPath(); ctx.arc(38, 70, 5, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(90, 70, 5, 0, Math.PI * 2); ctx.fill();

      // Tires
      ctx.fillStyle = "#111827";
      ctx.fillRect(31, 79, 14, 10);
      ctx.fillRect(83, 79, 14, 10);

      return ctx.getImageData(0, 0, size, size);
    }

    function getTraveledLineGeoJSON(distKm) {
      const sample = sampleRouteAtKm(distKm);
      const coords = routePts.slice(0, sample.idx + 1).map(p => [p[0], p[1]]);
      coords.push([sample.lon, sample.lat]);
      if (coords.length < 2) coords.push([sample.lon, sample.lat]);
      return {
        type: "Feature",
        geometry: { type: "LineString", coordinates: coords }
      };
    }

    function getAheadLineGeoJSON(distKm) {
      const sample = sampleRouteAtKm(distKm);
      const coords = [[sample.lon, sample.lat]];
      for (let i = sample.idx + 1; i < routePts.length; i++) {
        coords.push([routePts[i][0], routePts[i][1]]);
      }
      if (coords.length < 2) coords.push([sample.lon, sample.lat]);
      return {
        type: "Feature",
        geometry: { type: "LineString", coordinates: coords }
      };
    }

    function getMarkerGeoJSON(distKm) {
      const sample = sampleRouteAtKm(distKm);
      return {
        type: "Feature",
        geometry: { type: "Point", coordinates: [sample.lon, sample.lat] }
      };
    }

    let lastAheadVisible = null;

    map.on("load", () => {
      map.addSource("route-ahead", {
        type: "geojson",
        data: getAheadLineGeoJSON(0)
      });
      map.addSource("route-active", {
        type: "geojson",
        data: getTraveledLineGeoJSON(0)
      });
      map.addSource("route-marker-src", {
        type: "geojson",
        data: getMarkerGeoJSON(0)
      });

      map.addLayer({
        id: "route-ahead-casing",
        type: "line",
        source: "route-ahead",
        layout: { "line-cap": "round", "line-join": "round" },
        paint: {
          "line-color": "#111827",
          "line-width": 6.0,
          "line-opacity": 0.45
        }
      });

      map.addLayer({
        id: "route-ahead-line",
        type: "line",
        source: "route-ahead",
        layout: { "line-cap": "round", "line-join": "round" },
        paint: {
          "line-color": "#ffff00",
          "line-width": 4.0,
          "line-opacity": 1.0
        }
      });

      map.addLayer({
        id: "route-casing",
        type: "line",
        source: "route-active",
        layout: { "line-cap": "round", "line-join": "round" },
        paint: {
          "line-color": "#111827",
          "line-width": 6.0,
          "line-opacity": 0.45
        }
      });

      map.addLayer({
        id: "route-line",
        type: "line",
        source: "route-active",
        layout: { "line-cap": "round", "line-join": "round" },
        paint: {
          "line-color": "#ffff00",
          "line-width": 4.0
        }
      });

      map.addImage("death-cage", createDeathCageIconData());
      map.addLayer({
        id: "route-marker",
        type: "symbol",
        source: "route-marker-src",
        layout: {
          "icon-image": "death-cage",
          "icon-size": 1.5 * 0.22,
          "icon-allow-overlap": true,
          "icon-ignore-placement": true
        }
      });

      setProgress(0);
    });

    function progressToDistKm(prog) {
      const p = Math.max(0, Math.min(1, prog));
      if (videoSyncMode === "full" && FRAME_DISTS_KM.length > 1) {
        const f = p * (FRAME_DISTS_KM.length - 1);
        const i0 = Math.floor(f);
        const i1 = Math.min(FRAME_DISTS_KM.length - 1, i0 + 1);
        const t = f - i0;
        return FRAME_DISTS_KM[i0] + t * (FRAME_DISTS_KM[i1] - FRAME_DISTS_KM[i0]);
      }
      if (videoSyncMode === "part1" && FRAME_DISTS_KM.length >= PART1_FRAMES) {
        const f = p * (PART1_FRAMES - 1);
        const i0 = Math.floor(f);
        const i1 = Math.min(PART1_FRAMES - 1, i0 + 1);
        const t = f - i0;
        return FRAME_DISTS_KM[i0] + t * (FRAME_DISTS_KM[i1] - FRAME_DISTS_KM[i0]);
      }
      return p * totalKm;
    }

    function setProgress(prog) {
      currentDistKm = progressToDistKm(prog);
      const markerSrc = map.getSource("route-marker-src");
      if (markerSrc) markerSrc.setData(getMarkerGeoJSON(currentDistKm));
      const activeSrc = map.getSource("route-active");
      if (activeSrc) activeSrc.setData(getTraveledLineGeoJSON(currentDistKm));
      const aheadSrc = map.getSource("route-ahead");
      if (aheadSrc) aheadSrc.setData(getAheadLineGeoJSON(currentDistKm));
    }

    function tick(ts) {
      if (!videoEl.paused && !videoEl.ended && videoEl.duration > 0) {
        setProgress(videoEl.currentTime / videoEl.duration);
      }
      const showAhead = (Math.floor(ts / 1000) % 2) === 0;
      if (showAhead !== lastAheadVisible && map.getLayer("route-ahead-line")) {
        lastAheadVisible = showAhead;
        map.setPaintProperty("route-ahead-line", "line-opacity", showAhead ? 1.0 : 0.0);
        map.setPaintProperty("route-ahead-casing", "line-opacity", showAhead ? 0.45 : 0.0);
      }
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);

    window.addEventListener("resize", () => {
      map.resize();
    });
    window.addEventListener("orientationchange", () => {
      setTimeout(() => {
        map.resize();
        map.fitBounds(computeBounds(routePts), { padding: 36, duration: 250 });
      }, 180);
    });

    videoEl.addEventListener("timeupdate", () => {
      if (videoEl.duration > 0) setProgress(videoEl.currentTime / videoEl.duration);
    });
    videoEl.addEventListener("seeked", () => {
      if (videoEl.duration > 0) setProgress(videoEl.currentTime / videoEl.duration);
    });

    document.getElementById("videoPreset").addEventListener("change", (e) => {
      const [src, mode] = e.target.value.split("|");
      videoSyncMode = mode;
      videoEl.src = src;
      videoEl.load();
      setProgress(0);
    });

    document.getElementById("localVideoInput").addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (!file) return;
      videoSyncMode = "linear";
      videoEl.src = URL.createObjectURL(file);
      videoEl.load();
      setProgress(0);
    });

    document.getElementById("localGpxInput").addEventListener("change", async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const text = await file.text();
      const xml = new DOMParser().parseFromString(text, "application/xml");
      const pts = Array.from(xml.getElementsByTagName("trkpt"));
      if (pts.length < 2) {
        alert("Could not find at least 2 <trkpt> points in this GPX file.");
        return;
      }
      const parsed = [];
      let accKm = 0;
      let prevLat = null, prevLon = null;
      const toRad = Math.PI / 180;
      for (const p of pts) {
        const lat = parseFloat(p.getAttribute("lat"));
        const lon = parseFloat(p.getAttribute("lon"));
        if (prevLat !== null) {
          const dLat = (lat - prevLat) * toRad;
          const dLon = (lon - prevLon) * toRad;
          const h = Math.sin(dLat / 2) ** 2 + Math.cos(prevLat * toRad) * Math.cos(lat * toRad) * Math.sin(dLon / 2) ** 2;
          accKm += 6371 * 2 * Math.asin(Math.min(1, Math.sqrt(h)));
        }
        parsed.push([lon, lat, Number(accKm.toFixed(3))]);
        prevLat = lat; prevLon = lon;
      }
      routePts = parsed;
      totalKm = routePts[routePts.length - 1][2];
      videoSyncMode = "linear";
      document.getElementById("routeNameLabel").textContent = `${file.name} (${totalKm.toFixed(1)} km)`;
      map.fitBounds(computeBounds(routePts), { padding: 36 });
      setProgress(videoEl.duration > 0 ? videoEl.currentTime / videoEl.duration : 0);
    });
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build minimal Video + Street Route HTML.")
    parser.add_argument("--gpx", default=DEFAULT_GPX, help="Input GPX path")
    parser.add_argument("--meta", default=DEFAULT_META, help="Optional GSV metadata.json")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output HTML path")
    args = parser.parse_args()

    gpx_path = Path(args.gpx)
    meta_path = Path(args.meta)
    out_path = Path(args.out)

    route_pts = load_downsampled_route(gpx_path, max_points=1200)
    frame_dists = load_frame_dists_km(meta_path)
    total_km = route_pts[-1][2]

    html = HTML_TEMPLATE
    html = html.replace("__ROUTE_NAME__", gpx_path.name)
    html = html.replace("__TOTAL_KM__", f"{total_km:.1f}")
    html = html.replace("__ROUTE_JSON__", json.dumps(route_pts, separators=(",", ":")))
    html = html.replace("__FRAME_DISTS_JSON__", json.dumps(frame_dists, separators=(",", ":")))
    html = html.replace("__PART1_FRAMES__", str(min(768, len(frame_dists))))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(
        f"Generated {out_path} ({out_path.stat().st_size / 1024:.1f} KB) | "
        f"route={len(route_pts)} pts ({total_km:.2f} km) | frame_dists={len(frame_dists)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
