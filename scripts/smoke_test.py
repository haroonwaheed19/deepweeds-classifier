"""Black-box check of a running service (local, CI container, or the public Space).

    python scripts/smoke_test.py --url https://<user>-deepweeds-classifier.hf.space
"""
import argparse
import sys
import time
from pathlib import Path

import requests

SAMPLES = Path(__file__).resolve().parent.parent / "service" / "samples"
EXPECTED = {"lantana.jpg": "Lantana", "negative.jpg": "Negative", "siam_weed.jpg": "Siam weed"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:7860")
    ap.add_argument("--wait", type=int, default=120, help="seconds to wait for /health (cold starts)")
    a = ap.parse_args()
    base = a.url.rstrip("/")

    deadline = time.time() + a.wait
    while True:
        try:
            r = requests.get(f"{base}/health", timeout=10)
            if r.status_code == 200 and r.json().get("status") == "ok":
                break
        except requests.RequestException:
            pass
        if time.time() > deadline:
            print("FAIL: /health did not become ok in time")
            return 1
        time.sleep(3)
    print("health ok")

    failures = 0
    for name, want in EXPECTED.items():
        with open(SAMPLES / name, "rb") as f:
            t0 = time.time()
            r = requests.post(f"{base}/predict", files={"file": (name, f, "image/jpeg")}, timeout=60)
        ms = (time.time() - t0) * 1000
        if r.status_code != 200:
            print(f"FAIL {name}: HTTP {r.status_code} {r.text[:120]}")
            failures += 1
            continue
        got = r.json()["label"]
        ok = got == want
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {got} ({r.json()['confidence']:.2f}), round trip {ms:.0f} ms")

    bad = requests.post(f"{base}/predict", files={"file": ("x.txt", b"hi", "text/plain")}, timeout=30)
    if bad.status_code != 415:
        print(f"FAIL: wrong content type returned {bad.status_code}, expected 415")
        failures += 1
    else:
        print("ok   invalid upload rejected with 415")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
