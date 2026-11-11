"""
Radari PNG-d -> blobs.json: üks lihtne 0..1 väärtus iga linna/pilve (BP_WeatherBlob) jaoks.

Loeb radar2png.py väljundit (out_radar/meta.json + radar_*.png) ja kirjutab out_radar/blobs.json.
radar2png.py-d ei muuda. Käivita pärast radar2png.py-d (serve_weather.py teeb seda automaatselt).

blobs.json kuju (UE loeb "blobs" massiivi struktiks S_BlobWeather):
{
  "updated": "...", "timestamp": "...",          # viimane VAATLUS-kaader
  "blobs":  [ {"Name": "Tallinn", "Precipitation": 0.42, "CloudCover": 0.80, "RainMmH": 1.9, "Covered": true}, ... ],
  "frames": [ {"timestamp": "...", "kind": "observed"|"nowcast", "blobs": [...]}, ... ]   # ajasliideri jaoks (hiljem)
}

  Precipitation  0..1  vihma tugevus punktis (log-skaala, 0 = kuiv, 1 = >= MAX_RAIN mm/h)
  CloudCover     0..1  kui suur osa ümbrusest sajab (radar näeb ainult sajupilvi, mitte kuivi pilvi!)
  Covered        false = punkt jääb radari katvusest välja, väärtused on siis 0

pip install numpy pillow
"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent / "out_radar"

# Pilvede asukohad (maakonnakeskused). Name peab UE-s kattuma BP_WeatherBlob-i StationName'iga.
PLACES = [
    ("Tallinn",    59.437, 24.754),
    ("Tartu",      58.378, 26.729),
    ("Narva",      59.377, 28.190),
    ("Pärnu",      58.385, 24.497),
    ("Kuressaare", 58.253, 22.485),
    ("Kärdla",     58.997, 22.750),
    ("Haapsalu",   58.943, 23.541),
    ("Rakvere",    59.346, 26.355),
    ("Jõhvi",      59.359, 27.411),
    ("Paide",      58.885, 25.557),
    ("Rapla",      59.001, 24.793),
    ("Viljandi",   58.364, 25.590),
    ("Jõgeva",     58.746, 26.394),
    ("Põlva",      58.060, 27.069),
    ("Võru",       57.834, 27.019),
    ("Valga",      57.777, 26.047),
]

RAIN_RADIUS_KM = 8     # vihma tugevus = keskmine selle raadiuse sees
CLOUD_RADIUS_KM = 25   # pilvisus = mitu % sellest alast sajab
RAIN_THRESHOLD = 0.05  # mm/h, sellest alates loeme piksli "sajab"
MAX_RAIN = 10.0        # mm/h, millest alates Precipitation = 1


def decode(png_path, max_rate):
    """radar_*.png -> (mm/h massiiv, katvuse mask). R = log-skaalas sademed, G = katvus."""
    px = np.asarray(Image.open(png_path).convert("RGBA"))
    rate = np.expm1(px[..., 0] / 255.0 * math.log1p(max_rate))
    covered = px[..., 1] > 127
    return rate, covered


def window(lat, lon, radius_km, bounds, size):
    """Piksliaken (read, veerud) ümber punkti; rida 0 = põhi."""
    lon_min, lat_min, lon_max, lat_max = bounds
    w, h = size
    col = (lon - lon_min) / (lon_max - lon_min) * w
    row = (lat_max - lat) / (lat_max - lat_min) * h
    km_per_px_y = (lat_max - lat_min) / h * 111.32
    km_per_px_x = (lon_max - lon_min) / w * 111.32 * math.cos(math.radians(lat))
    ry, rx = radius_km / km_per_px_y, radius_km / km_per_px_x
    rows = np.arange(max(0, int(row - ry)), min(h, int(row + ry) + 1))
    cols = np.arange(max(0, int(col - rx)), min(w, int(col + rx) + 1))
    if len(rows) == 0 or len(cols) == 0:
        return None
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    inside = ((rr - row) / ry) ** 2 + ((cc - col) / rx) ** 2 <= 1.0   # ellips = ring maapinnal
    return rr[inside], cc[inside]


def blob_values(rate, covered, bounds, size):
    result = []
    for name, lat, lon in PLACES:
        entry = {"Name": name, "Precipitation": 0.0, "CloudCover": 0.0, "RainMmH": 0.0, "Covered": False}
        rain_win = window(lat, lon, RAIN_RADIUS_KM, bounds, size)
        cloud_win = window(lat, lon, CLOUD_RADIUS_KM, bounds, size)
        if rain_win is not None and cloud_win is not None:
            r_cov = covered[rain_win]
            c_cov = covered[cloud_win]
            if r_cov.any() and c_cov.any():
                mm = float(rate[rain_win][r_cov].mean())
                raining_share = float((rate[cloud_win][c_cov] >= RAIN_THRESHOLD).mean())
                precip = min(1.0, math.log1p(mm) / math.log1p(MAX_RAIN))
                entry.update(
                    Precipitation=round(precip, 3),
                    # sajab ümbruses -> pilves; vähemalt nii pilves kui vihm ise eeldab
                    CloudCover=round(max(min(1.0, raining_share * 1.5), precip), 3),
                    RainMmH=round(mm, 2),
                    Covered=True,
                )
        result.append(entry)
    return result


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    bounds, size = meta["bounds"], meta["size"]
    # max_rate loetakse kodeeringu valemist "mm/h = exp(R/255 * ln(51)) - 1"
    max_rate = float(meta["encoding"]["R"].split("ln(")[1].split(")")[0]) - 1

    frames = []
    for f in meta["frames"]:
        png = OUT / f["url"]
        if not png.exists():
            continue
        rate, covered = decode(png, max_rate)
        frames.append({"timestamp": f["timestamp"], "kind": f["kind"],
                       "blobs": blob_values(rate, covered, bounds, size)})
    if not frames:
        raise SystemExit("Ühtegi radari kaadrit ei leitud. Käivita enne radar2png.py")

    observed = [f for f in frames if f["kind"] == "observed"]
    current = observed[-1] if observed else frames[0]
    out = {
        "updated": meta["updated"],
        "timestamp": current["timestamp"],
        "blobs": current["blobs"],
        "frames": frames,
    }
    (OUT / "blobs.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    # Lihtne tekstiversioon UE jaoks (ei vaja JSON pluginat):
    #   Nimi;Precipitation;CloudCover|Nimi;Precipitation;CloudCover|...
    # UE: Parse Into Array "|" -> iga linn -> Parse Into Array ";" -> [0] nimi, [1] Precipitation, [2] CloudCover
    txt = "|".join(f"{b['Name']};{b['Precipitation']:.3f};{b['CloudCover']:.3f}" for b in current["blobs"])
    (OUT / "blobs.txt").write_text(txt, encoding="utf-8")

    print(f"blobs.json: {len(PLACES)} kohta, kaader {current['timestamp']} ({len(frames)} kaadrit kokku)")
    for b in current["blobs"]:
        flag = "" if b["Covered"] else "  (radari katvusest väljas)"
        print(f"  {b['Name']:<11} vihm {b['RainMmH']:5.2f} mm/h  Precipitation {b['Precipitation']:.2f}"
              f"  CloudCover {b['CloudCover']:.2f}{flag}")


if __name__ == "__main__":
    main()
