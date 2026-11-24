# Route Hyperlapse & GPX Replay

Synchronized street-level route hyperlapse video and interactive GPX route map viewer (Bangkok -> Prachuap Khiri Khan, 254.25 km).

- **Live Page:** https://qklns.github.io/route-hyperlapse/

## Included Files
- `index.html` / `gpx_replay.html` — Minimal responsive split-screen viewer (video on left, blinking FollowTrack Streets map with Death Cage icon on right; desktop, iPad, and mobile ready).
- `route_hyperlapse_gsv.mp4` — Street View hyperlapse MP4 (640x360, 30 fps, 2,031 frames).
- `gpx_replay_map.mp4` — Standalone synced GPX Replay map MP4 (960x1080, 30 fps, 2,031 frames).
- `build_gpx_replay_html.py` — Generates `gpx_replay.html` from any `.gpx` route.
- `render_gpx_replay_video.py` — Renders `gpx_replay_map.mp4` using free OSM Streets tiles.
- `build_gsv_hyperlapse.py` / `build_gsv_tiles_hyperlapse.py` / `build_hyperlapse.py` — Street-level hyperlapse builders (Google Street View Static API, Map Tiles API, and KartaView).
