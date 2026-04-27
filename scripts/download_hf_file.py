#!/usr/bin/env python3
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional


def remote_size(url: str) -> Optional[int]:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "AIRecordSummary/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None


def download(url: str, temp_path: Path, start: int) -> int:
    headers = {"User-Agent": "AIRecordSummary/1.0"}
    if start > 0:
        headers["Range"] = f"bytes={start}-"
    request = urllib.request.Request(url, headers=headers)
    written = start
    mode = "ab" if start > 0 else "wb"
    with urllib.request.urlopen(request, timeout=120) as response, temp_path.open(mode) as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            written += len(chunk)
    return written


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: download_hf_file.py <url> <output_path>", file=sys.stderr)
        return 2

    url = sys.argv[1]
    output_path = Path(sys.argv[2])
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    expected_size = remote_size(url)
    for attempt in range(1, 11):
        start = temp_path.stat().st_size if temp_path.exists() else 0
        try:
            written = download(url, temp_path, start)
            if expected_size is None or written >= expected_size:
                temp_path.replace(output_path)
                return 0
            print(f"download incomplete: {written}/{expected_size} bytes, retrying", file=sys.stderr)
        except Exception as exc:
            print(f"download attempt {attempt} failed: {exc}", file=sys.stderr)
        time.sleep(min(30, attempt * 3))

    print(f"failed to download after retries: {url}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
