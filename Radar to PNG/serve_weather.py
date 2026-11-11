"""
Ilmaandmete "backend" UE jaoks: uuendab radarit iga 5 min järel ja jagab out_radar kausta üle HTTP.

  python serve_weather.py

Seejärel UE-s:   http://localhost:8000/blobs.json   (ja meta.json, radar_*.png, lightning.json)
Brauseris:       http://localhost:8000/viewer.html
Peata: Ctrl+C

Kasutab radar2png.py-d ja make_blobs.py-d samast kaustast, neid ei muuda.
pip install requests numpy h5py pyproj pillow rasterio
"""
import functools
import threading
import time
import traceback
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import make_blobs
import radar2png

PORT = 8000
EVERY_SECONDS = 5 * 60


def update_once():
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{stamp}] uuendan radarit...")
    try:
        radar2png.main()
    except SystemExit as e:          # radar2png lõpetab vea korral SystemExit'iga
        print(f"radar2png: {e}")
    except Exception:
        traceback.print_exc()
    try:
        make_blobs.main()
    except SystemExit as e:
        print(f"make_blobs: {e}")
    except Exception:
        traceback.print_exc()


def update_loop():
    while True:
        update_once()
        time.sleep(EVERY_SECONDS)


class Handler(SimpleHTTPRequestHandler):
    # ütle brauserile/UE-le, et tekst on UTF-8 (muidu "Pärnu" -> "PÃ¤rnu")
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".txt": "text/plain; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".html": "text/html; charset=utf-8",
    }

    def end_headers(self):
        # ära lase UE-l ega brauseril vana faili vahemälust võtta
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass  # vaikne: ära prindi iga päringut


def main():
    make_blobs.OUT.mkdir(exist_ok=True)
    threading.Thread(target=update_loop, daemon=True).start()
    handler = functools.partial(Handler, directory=str(make_blobs.OUT))
    server = ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    print(f"Server töötab: http://localhost:{PORT}/blobs.json   (peata: Ctrl+C)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nPeatatud.")


if __name__ == "__main__":
    main()
