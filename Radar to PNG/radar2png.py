"""
Ilmateenistuse radar (vaatlus + nowcast) ja välgud -> PNG andmetekstuurid + JSON UE jaoks.

Allikad (kõik avalikud, Keskkonnaagentuur, CC-BY 4.0) - samad, mida ilmateenistus.ee radarikaart kasutab:
  vaatlus   KAIA avaandmed, komposiit "comp.*.CAP.*.h5" (ODIM HDF5, mm/h), iga 5 min
            https://avaandmed.keskkonnaportaal.ee/swagger/
  nowcast   GeoServer WCS ilm:nowcasting (float32 GeoTIFF, EPSG:3301, mm/h), ~1.5 h ette, 5 min samm
            https://ilmgs.envir.ee/geoserver/ilm/wcs
  välgud    GeoServer WFS lightning:pikne (punktid: aeg, lat/lon, voolutugevus)
            https://gslightning.envir.ee/geoserver/lightning/wfs

Väljund (out_radar/):
  radar_<ts>.png    andmetekstuur (UE-s sRGB VÄLJAS, compression Masks/VectorDisplacementmap)
                      R = sademed log-skaalas: mm/h = exp(R/255 * ln(51)) - 1   (0..50 mm/h)
                      G = 255 kui piksel on radari katvuses, 0 kui andmed puuduvad
  preview_<ts>.png  värviline eelvaade ilmateenistuse legendi järgi
  meta.json         bounds, kodeering, kaadrid (kind = observed | nowcast)
  lightning.json    viimase 2 h välgud ala piires
  viewer.html       vaatur: ava brauseris (topeltklikk), liuguriga läbi kaadrite

Kõik pildid on samal EPSG:4326 (lat/lon) gridil, nii et UE-s piisab bounds'ist.
Käivita iga ~5 min järel. Vaatluskaadreid uuesti ei töödelda, nowcast arvutatakse iga kord uuesti.

pip install requests numpy h5py pyproj pillow rasterio
"""
import json
import math
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import h5py
    import numpy as np
    import rasterio
    import requests
    from PIL import Image
    from pyproj import Proj, Transformer
except ImportError as e:
    sys.exit(f"Puudub Pythoni teek: {e.name}\n"
             f"Paigalda: pip install requests numpy h5py pyproj pillow rasterio")

KAIA = "https://avaandmed.keskkonnaportaal.ee/api/lists/active"
WMS = "https://ilmgs.envir.ee/geoserver/ilm/wms"
WCS = "https://ilmgs.envir.ee/geoserver/ilm/wcs"
WFS_LIGHTNING = "https://gslightning.envir.ee/geoserver/lightning/wfs"

OUT = Path(__file__).resolve().parent / "out_radar"   # alati skripti kõrvale, sõltumata käivituskaustast
CACHE = OUT / "cache"
HISTORY = timedelta(hours=2)   # ajalugu slideri jaoks
SIZE = 1024                    # väljundpildi külg pikslites
MAX_RATE = 50.0                # mm/h, log-kodeeringu ülemine piir

# Ilmateenistuse legendi lävendid (mm/h) ja värvid eelvaate jaoks
LEGEND = [
    (0.05, (180, 215, 255)), (0.1, (130, 185, 250)), (0.3, (70, 140, 240)),
    (0.5, (30, 100, 220)), (1, (40, 170, 70)), (2, (120, 200, 40)),
    (4, (250, 220, 30)), (8, (250, 140, 20)), (16, (230, 40, 30)), (50, (200, 30, 200)),
]

session = requests.Session()


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------- väljundgrid ----------

class Grid:
    """Ühine lat/lon väljundgrid (rida 0 = põhi)."""

    def __init__(self, lon_min, lat_min, lon_max, lat_max):
        self.bounds = [float(lon_min), float(lat_min), float(lon_max), float(lat_max)]
        lons = lon_min + (np.arange(SIZE) + 0.5) / SIZE * (lon_max - lon_min)
        lats = lat_max - (np.arange(SIZE) + 0.5) / SIZE * (lat_max - lat_min)
        self.lon, self.lat = np.meshgrid(lons, lats)


def encode(rate, covered):
    """rate (mm/h) ja covered (bool) väljundgridil -> (andmetekstuur, eelvaade)."""
    rate = np.where(covered, rate, 0.0)
    data = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    data[..., 0] = np.clip(np.round(np.log1p(rate) / math.log1p(MAX_RATE) * 255), 0, 255)
    data[..., 1] = covered * 255
    data[..., 3] = 255
    preview = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    preview[covered] = (235, 235, 235, 255)
    for threshold, color in LEGEND:
        preview[(rate >= threshold) & covered] = (*color, 255)
    return data, preview


def save_frame(stamp, rate, covered):
    data, preview = encode(rate, covered)
    data_png, prev_png = OUT / f"radar_{stamp}.png", OUT / f"preview_{stamp}.png"
    Image.fromarray(data, "RGBA").save(data_png)
    Image.fromarray(preview, "RGBA").save(prev_png)
    return data_png.name, prev_png.name


# ---------- vaatlus: KAIA HDF5 ----------

def list_composites(since):
    body = {
        "filter": {"and": {"children": [
            {"greaterThan": {"field": "Timestamp", "value": iso(since)}},
            {"contains": {"field": "RMFileName", "value": "comp."}},
            {"contains": {"field": "RMFileName", "value": ".CAP."}},
        ]}},
        "pageSize": 500,
        "fields": ["Timestamp"],
        "includeFileMetadata": True,
    }
    r = session.post(f"{KAIA}/items/query", json=body, timeout=30)
    r.raise_for_status()
    items = []
    for doc in r.json()["documents"]:
        if not doc["fileMetadata"]:
            continue
        name = doc["fileMetadata"][0]["name"]
        # comp.20261015075003.CAP.1826.h5 -> failinimes on UTC aeg
        t = datetime.strptime(name.split(".")[1], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        items.append({"id": doc["id"], "name": name, "time": t})
    return sorted(items, key=lambda x: x["time"])


def read_h5(item):
    path = CACHE / item["name"]
    if not path.exists():
        r = session.get(f"{KAIA}/items/{item['id']}/files/0", timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    with h5py.File(path, "r") as f:
        what = f["dataset1/what"].attrs
        raw = f["dataset1/data1/data"][:].astype(np.float32)
        nodata = raw == float(what.get("nodata", 65535.0))
        rate = raw * float(what.get("gain", 1.0)) + float(what.get("offset", 0.0))
        where = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f["where"].attrs.items()}
    return rate, nodata, where


def h5_lookup(where, grid):
    """
    Lähtegrid on Mercatoris: x lineaarne UL->UR, y lineaarne UL->LL.
    NB: failis olev LR_lat ei klapi; kontrollitud radarite 250 km katvusringiga
    (UL->LL annab serva hälbeks ~0.5 km, LR ~9 km).
    """
    p = Proj(where["projdef"])
    x0, y_top = p(where["UL_lon"], where["UL_lat"])
    x1, _ = p(where["UR_lon"], where["UR_lat"])
    _, y_bot = p(where["LL_lon"], where["LL_lat"])
    nx, ny = int(where["xsize"]), int(where["ysize"])
    X, Y = p(grid.lon, grid.lat)
    col = ((X - x0) / (x1 - x0) * nx).astype(int)
    row = ((y_top - Y) / (y_top - y_bot) * ny).astype(int)
    inside = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
    return np.clip(row, 0, ny - 1), np.clip(col, 0, nx - 1), inside


# ---------- nowcast: GeoServer WCS ----------

def nowcast_times(after):
    """Võtab WMS GetCapabilities'ist nowcasting-kihi ajad, mis on hilisemad kui `after`."""
    r = session.get(WMS, params={"service": "WMS", "request": "GetCapabilities", "version": "1.1.1"}, timeout=60)
    r.raise_for_status()
    layer = re.search(r"<Name>nowcasting</Name>(.*?)</Layer>", r.text, re.S)
    extent = re.search(r'<Extent[^>]*name="time"[^>]*>(.*?)</Extent>', layer.group(1), re.S)
    times = [datetime.strptime(t.strip()[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
             for t in extent.group(1).split(",") if t.strip()]
    return sorted(t for t in times if t > after)


def read_nowcast(t):
    params = {
        "service": "WCS", "version": "2.0.1", "request": "GetCoverage",
        "coverageId": "ilm__nowcasting", "format": "image/tiff",
        "subset": f'time("{t.strftime("%Y-%m-%dT%H:%M:%S.000Z")}")',
    }
    r = session.get(WCS, params=params, timeout=180)
    r.raise_for_status()
    if not r.content.startswith((b"II*", b"MM\x00*")):
        raise RuntimeError(f"WCS ei tagastanud GeoTIFF-i: {r.text[:200]}")
    path = CACHE / "nowcast.tif"
    path.write_bytes(r.content)
    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.crs.to_string()


def tif_lookup(transform, crs, shape, grid):
    """Väljundgridi lat/lon -> GeoTIFF-i rida/veerg (EPSG:3301)."""
    to_src = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    X, Y = to_src.transform(grid.lon, grid.lat)
    col, row = ~transform * (X, Y)
    col, row = np.floor(col).astype(int), np.floor(row).astype(int)
    ny, nx = shape
    inside = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
    return np.clip(row, 0, ny - 1), np.clip(col, 0, nx - 1), inside


# ---------- välgud: GeoServer WFS ----------

def fetch_lightning(since, bounds):
    lon_min, lat_min, lon_max, lat_max = bounds
    cql = (f"date_timestamp AFTER {iso(since)} "
           f"AND latitude BETWEEN {lat_min} AND {lat_max} "
           f"AND longitude BETWEEN {lon_min} AND {lon_max}")
    params = {
        "service": "WFS", "version": "2.0.0", "request": "GetFeature",
        "typeNames": "lightning:pikne", "outputFormat": "application/json",
        "propertyName": "date_timestamp,latitude,longitude,peak_current,data_type,case",
        "CQL_FILTER": cql, "count": 10000, "sortBy": "date_timestamp D",  # limiidi korral värskeimad
    }
    r = session.get(WFS_LIGHTNING, params=params, timeout=90)
    r.raise_for_status()
    strikes = []
    for f in r.json()["features"]:
        p = f["properties"]
        strikes.append({
            "timestamp": p["date_timestamp"],
            "lat": float(p["latitude"]), "lon": float(p["longitude"]),
            "peak_current_kA": p["peak_current"],
            "polarity": p["case"],
            "type": p["data_type"],  # pilv-maa vs pilvesisene (vt ilmateenistuse legend)
        })
    return sorted(strikes, key=lambda s: s["timestamp"])


# ---------- vaatur ----------

def write_viewer(meta, lightning):
    """Kirjutab out_radar/viewer.html: liuguriga eelvaadete vaatur, andmed sisse kirjutatud
    (avaneb topeltklikiga, serverit pole vaja)."""
    html = (VIEWER_HTML
            .replace("__META__", json.dumps(meta, ensure_ascii=False))
            .replace("__LIGHTNING__", json.dumps(lightning))
            .replace("__LEGEND__", json.dumps([[v, list(c)] for v, c in LEGEND])))
    (OUT / "viewer.html").write_text(html, encoding="utf-8")


VIEWER_HTML = r'''<!doctype html>
<html lang="et">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Radari vaatur</title>
<style>
  :root {
    --bg: #111418; --panel: #1b2027; --text: #e6e9ee; --muted: #8a93a0;
    --obs: #4ea1ff; --now: #ffb547; --line: #2a313b;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 900px; margin: 0 auto; padding: 16px; }
  header { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
  h1 { font-size: 18px; margin: 0; }
  .updated { color: var(--muted); font-size: 12px; }
  .stage { position: relative; margin-top: 12px; background: #000; border-radius: 8px; overflow: hidden;
           aspect-ratio: 1 / 1; }
  .stage img, .stage svg { position: absolute; inset: 0; width: 100%; height: 100%; }
  .stage img { image-rendering: pixelated; }
  .badge { position: absolute; top: 10px; left: 10px; padding: 4px 10px; border-radius: 999px;
           font-weight: 600; font-size: 13px; color: #111; }
  .badge.observed { background: var(--obs); }
  .badge.nowcast { background: var(--now); }
  .clock { position: absolute; top: 10px; right: 10px; padding: 4px 10px; border-radius: 6px;
           background: rgba(0,0,0,.6); font-variant-numeric: tabular-nums; font-size: 15px; }
  .controls { display: flex; align-items: center; gap: 12px; margin-top: 12px; }
  button { background: var(--panel); color: var(--text); border: 1px solid var(--line); border-radius: 6px;
           padding: 6px 12px; font: inherit; cursor: pointer; min-width: 44px; }
  button:hover { border-color: var(--muted); }
  .track { position: relative; flex: 1; height: 28px; }
  .segments { position: absolute; left: 0; right: 0; top: 11px; height: 6px; display: flex; border-radius: 3px; overflow: hidden; }
  .segments span { flex: 1; }
  .segments .observed { background: var(--obs); }
  .segments .nowcast { background: var(--now); }
  input[type=range] { position: absolute; inset: 0; width: 100%; margin: 0; background: transparent;
                      -webkit-appearance: none; appearance: none; cursor: pointer; }
  input[type=range]::-webkit-slider-runnable-track { background: transparent; height: 28px; }
  input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 14px; height: 28px; border-radius: 4px;
                                            background: #fff; border: 2px solid #111; }
  input[type=range]::-moz-range-track { background: transparent; }
  input[type=range]::-moz-range-thumb { width: 12px; height: 24px; border-radius: 4px; background: #fff; border: 2px solid #111; }
  .ticks { display: flex; justify-content: space-between; color: var(--muted); font-size: 12px; margin: 4px 56px 0 56px; }
  .legend { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 14px; margin-top: 16px; font-size: 12px; color: var(--muted); }
  .scale { display: flex; gap: 0; }
  .sw { display: flex; flex-direction: column; align-items: center; width: 34px; }
  .sw span { width: 100%; height: 10px; }
  .sw small { margin-top: 2px; }
  .key { display: inline-flex; align-items: center; gap: 6px; }
  .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  .info { margin-top: 10px; font-size: 12px; color: var(--muted); }
  @media (max-width: 520px) { .ticks { margin: 4px 0 0 0; } }
</style>
</head>
<body>
<main>
  <header>
    <h1>Radari vaatur</h1>
    <span class="updated" id="updated"></span>
  </header>

  <div class="stage">
    <img id="frame" alt="radari kaader">
    <svg id="overlay" viewBox="0 0 1024 1024" preserveAspectRatio="none"></svg>
    <span class="badge" id="badge"></span>
    <span class="clock" id="clock"></span>
  </div>

  <div class="controls">
    <button id="play" title="Esita / peata (tühik)">▶</button>
    <div class="track">
      <div class="segments" id="segments"></div>
      <input type="range" id="slider" min="0" value="0" step="1" aria-label="aeg">
    </div>
    <button id="speed" title="Kiirus">1×</button>
  </div>
  <div class="ticks"><span id="tStart"></span><span id="tNow"></span><span id="tEnd"></span></div>

  <div class="legend">
    <span class="key"><span class="dot" style="background:var(--obs)"></span>vaatlus</span>
    <span class="key"><span class="dot" style="background:var(--now)"></span>prognoos (nowcast)</span>
    <span class="key"><span class="dot" style="background:#ffe600;border:1px solid #000"></span>välk (±5 min)</span>
    <span class="key"><span class="dot" style="background:#ebebeb"></span>radari katvus</span>
  </div>
  <div class="legend">
    <span>mm/h</span>
    <span class="scale" id="scale"></span>
  </div>
  <p class="info" id="info"></p>
</main>

<script>
const META = __META__;
const LIGHTNING = __LIGHTNING__;
const LEGEND = __LEGEND__;

const frames = META.frames;
const [lonMin, latMin, lonMax, latMax] = META.bounds;
const W = 1024, H = 1024;
const toPx = (lat, lon) => [(lon - lonMin) / (lonMax - lonMin) * W, (latMax - lat) / (latMax - latMin) * H];

const fmt = (iso, withDate) => new Date(iso).toLocaleString("et-EE", {
  timeZone: "Europe/Tallinn", hour: "2-digit", minute: "2-digit",
  ...(withDate ? { day: "numeric", month: "short" } : {})
});

const $ = id => document.getElementById(id);
const slider = $("slider"), img = $("frame"), overlay = $("overlay");

// eellae kõik kaadrid, et liugur oleks sujuv
frames.forEach(f => { const i = new Image(); i.src = f.preview; });

// ajatelje värvilised lõigud
$("segments").innerHTML = frames.map(f => `<span class="${f.kind}"></span>`).join("");
slider.max = frames.length - 1;
$("tStart").textContent = fmt(frames[0].timestamp);
$("tEnd").textContent = fmt(frames[frames.length - 1].timestamp);
const lastObs = frames.map(f => f.kind).lastIndexOf("observed");
if (lastObs >= 0) $("tNow").textContent = "viimane vaatlus " + fmt(frames[lastObs].timestamp);
$("updated").textContent = "uuendatud " + fmt(META.updated, true);

// legend
$("scale").innerHTML = LEGEND.map(([v, c]) =>
  `<span class="sw"><span style="background:rgb(${c})"></span><small>${String(v).replace(".", ",")}</small></span>`).join("");

// linnad orienteerumiseks
const CITIES = [
  ["Tallinn", 59.437, 24.754], ["Tartu", 58.378, 26.729], ["Pärnu", 58.385, 24.497],
  ["Narva", 59.377, 28.190], ["Kuressaare", 58.252, 22.485], ["Haapsalu", 58.943, 23.541],
  ["Rakvere", 59.346, 26.355], ["Viljandi", 58.364, 25.590], ["Võru", 57.834, 27.019],
  ["Paide", 58.886, 25.557], ["Helsinki", 60.170, 24.938], ["Riia", 56.949, 24.105],
];
const cityLayer = CITIES.map(([n, lat, lon]) => {
  const [x, y] = toPx(lat, lon);
  return `<circle cx="${x}" cy="${y}" r="4" fill="#111"/>
          <text x="${x + 7}" y="${y + 4}" font-size="15" fill="#111"
                stroke="#fff" stroke-width="3" paint-order="stroke">${n}</text>`;
}).join("");

function render(i) {
  const f = frames[i];
  img.src = f.preview;
  const badge = $("badge");
  badge.textContent = f.kind === "observed" ? "VAATLUS" : "PROGNOOS";
  badge.className = "badge " + f.kind;
  $("clock").textContent = fmt(f.timestamp, true);

  const t = Date.parse(f.timestamp);
  const near = LIGHTNING.strikes.filter(s => Math.abs(Date.parse(s.timestamp) - t) <= 5 * 60e3);
  const bolts = near.map(s => {
    const [x, y] = toPx(s.lat, s.lon);
    return `<circle cx="${x}" cy="${y}" r="5" fill="#ffe600" stroke="#000" stroke-width="1.5"/>`;
  }).join("");
  overlay.innerHTML = cityLayer + bolts;
  $("info").textContent = `Kaader ${i + 1}/${frames.length} · ${f.url} · välke ±5 min: ${near.length}`;
}

let timer = null, speeds = [1, 2, 4], speedIdx = 0;
function step() {
  slider.value = (+slider.value + 1) % frames.length;
  render(+slider.value);
}
function setPlaying(on) {
  clearInterval(timer); timer = null;
  if (on) timer = setInterval(step, 400 / speeds[speedIdx]);
  $("play").textContent = on ? "❚❚" : "▶";
}
$("play").onclick = () => setPlaying(!timer);
$("speed").onclick = () => {
  speedIdx = (speedIdx + 1) % speeds.length;
  $("speed").textContent = speeds[speedIdx] + "×";
  if (timer) setPlaying(true);
};
slider.oninput = () => { setPlaying(false); render(+slider.value); };
document.addEventListener("keydown", e => {
  if (e.code === "Space") { e.preventDefault(); setPlaying(!timer); }
  if (e.code === "ArrowRight") { setPlaying(false); slider.value = Math.min(+slider.value + 1, frames.length - 1); render(+slider.value); }
  if (e.code === "ArrowLeft") { setPlaying(false); slider.value = Math.max(+slider.value - 1, 0); render(+slider.value); }
});

// alusta viimasest vaatlusest
slider.value = Math.max(lastObs, 0);
render(+slider.value);
</script>
</body>
</html>
'''


# ---------- main ----------

def main():
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    print(f"Laen radariandmeid -> {OUT}")
    now = datetime.now(timezone.utc)
    since = now - HISTORY

    items = list_composites(since)
    if not items:
        raise SystemExit("KAIA ei tagastanud ühtegi komposiiti")

    # väljundgrid = komposiidi ulatus (Harku + Sürgavere katvus)
    _, _, where = read_h5(items[-1])
    grid = Grid(where["UL_lon"], where["LL_lat"], where["UR_lon"], where["UL_lat"])
    h5_idx = None

    frames = []
    for item in items:
        stamp = item["time"].strftime("%Y%m%dT%H%M%SZ")
        if not (OUT / f"radar_{stamp}.png").exists():
            rate, nodata, where = read_h5(item)
            if h5_idx is None:
                h5_idx = h5_lookup(where, grid)
            row, col, inside = h5_idx
            save_frame(stamp, rate[row, col], inside & ~nodata[row, col])
            print("observed", stamp)
        frames.append({"timestamp": iso(item["time"]), "kind": "observed",
                       "url": f"radar_{stamp}.png", "preview": f"preview_{stamp}.png"})

    # nowcast: kõik ajad pärast viimast vaatlust
    tif_idx = None
    for t in nowcast_times(items[-1]["time"]):
        try:
            data, transform, crs = read_nowcast(t)
        except Exception as e:
            print("nowcast skip", t, e)
            continue
        if tif_idx is None:
            tif_idx = tif_lookup(transform, crs, data.shape, grid)
        row, col, inside = tif_idx
        v = data[row, col]
        covered = inside & (v >= 0) & (v < 1000)   # 65535 / -9999 = andmed puuduvad
        stamp = t.strftime("%Y%m%dT%H%M%SZ")
        save_frame(stamp, v, covered)
        frames.append({"timestamp": iso(t), "kind": "nowcast",
                       "url": f"radar_{stamp}.png", "preview": f"preview_{stamp}.png"})
        print("nowcast ", stamp)

    try:
        strikes = fetch_lightning(since, grid.bounds)
    except Exception as e:
        print("lightning skip", e)
        strikes = []
    lightning = {"updated": iso(now), "since": iso(since), "strikes": strikes}
    (OUT / "lightning.json").write_text(json.dumps(lightning, indent=1))

    meta = {
        "layer": "precipitation",
        "source": "Keskkonnaagentuur (CC-BY 4.0): KAIA radari komposiit, GeoServer nowcasting",
        "updated": iso(now),
        "projection": "EPSG:4326",
        "bounds": grid.bounds,
        "size": [SIZE, SIZE],
        "encoding": {
            "R": f"mm/h = exp(R/255 * ln({MAX_RATE + 1:g})) - 1",
            "G": "255 = radari katvuses, 0 = andmed puuduvad",
        },
        "frames": frames,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    write_viewer(meta, lightning)

    # koristus: vanad kaadrid ja vahemälu
    keep = {f["url"] for f in frames} | {f["preview"] for f in frames}
    for p in OUT.glob("*.png"):
        if p.name not in keep:
            p.unlink()
    keep_h5 = {i["name"] for i in items}
    for p in CACHE.glob("*.h5"):
        if p.name not in keep_h5:
            p.unlink()
    print(f"{sum(f['kind'] == 'observed' for f in frames)} vaatlust, "
          f"{sum(f['kind'] == 'nowcast' for f in frames)} nowcasti, {len(strikes)} välku")
    print(f"Väljund: {OUT}")
    print(f"Vaatur:  {OUT / 'viewer.html'}")


if __name__ == "__main__":
    main()