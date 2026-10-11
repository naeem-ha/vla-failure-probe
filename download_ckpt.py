"""Download gs://openpi-assets/<prefix> (default checkpoints/pi05_libero) into openpi's cache layout over public HTTPS.

openpi resolves gs://<bucket>/<path> to ~/.cache/openpi/<bucket>/<path>, so serve_policy.py
will pick this up without re-downloading.
"""
import json
import pathlib
import shutil
import urllib.parse
import urllib.request
import sys

BUCKET = "openpi-assets"
PREFIX = (sys.argv[1] if len(sys.argv) > 1 else "checkpoints/pi05_libero").rstrip("/") + "/"
dest_root = pathlib.Path("~/.cache/openpi").expanduser() / BUCKET

list_url = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?prefix={PREFIX}&fields=items(name,size),nextPageToken"
items, tok = [], None
while True:
    d = json.load(urllib.request.urlopen(list_url + (f"&pageToken={tok}" if tok else "")))
    items += d.get("items", [])
    tok = d.get("nextPageToken")
    if not tok:
        break

for it in items:
    name, size = it["name"], int(it["size"])
    if name.endswith("/"):
        continue
    out = dest_root / name
    if out.exists() and out.stat().st_size == size:
        print("skip", name)
        continue
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    url = f"https://storage.googleapis.com/{BUCKET}/{urllib.parse.quote(name)}"
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=16 << 20)
    tmp.rename(out)
    print(f"got {name} ({size / 1e9:.2f} GB)", flush=True)
print("CKPT_OK", dest_root / PREFIX)
