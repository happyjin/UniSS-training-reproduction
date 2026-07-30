"""Verify that the current Gradio share link is public and has no login gate."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--access-info", required=True)
    args = parser.parse_args()
    info = json.loads(Path(args.access_info).read_text(encoding="utf-8"))
    if info.get("auth_mode") != "public_no_login":
        raise SystemExit("unexpected auth mode")
    if info.get("username") is not None or info.get("password") is not None:
        raise SystemExit("public access file must not contain credentials")
    url = str(info.get("public_url") or "")
    if not url.startswith("https://"):
        raise SystemExit(f"invalid public URL: {url!r}")
    request = urllib.request.Request(url, headers={"User-Agent": "UniSS-public-smoke/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read(200_000).decode("utf-8", errors="replace")
        if response.status != 200:
            raise SystemExit(f"unexpected HTTP status: {response.status}")
    if "UniSS" not in body and "gradio" not in body.lower():
        raise SystemExit("public page did not look like the expected Gradio app")
    print(json.dumps({"public_url": url, "status": "ok", "auth": "none"}))


if __name__ == "__main__":
    main()
