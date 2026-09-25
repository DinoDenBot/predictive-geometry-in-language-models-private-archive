#!/usr/bin/env python3
"""Download and byte-verify every artifact in the frozen SmolLM2 pin."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from study3 import sha256_file


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cache", type=Path)
    source.add_argument("--source-directory", type=Path)
    parser.add_argument("--include-weights", action="store_true")
    args = parser.parse_args()
    pin = json.loads((ROOT / "model_pin.json").read_text())
    cache = (
        args.source_directory
        if args.source_directory is not None
        else args.cache / pin["repository"].replace("/", "--") / pin["revision"]
    )
    cache.mkdir(parents=True, exist_ok=True)
    verified: dict[str, dict[str, object]] = {}
    for name, expected in pin["files"].items():
        if name == "model.safetensors" and not args.include_weights:
            continue
        destination = cache / name
        if not destination.is_file() and args.source_directory is not None:
            raise FileNotFoundError(f"pinned artifact absent from source directory: {destination}")
        if not destination.is_file():
            url = (
                f"https://huggingface.co/{pin['repository']}/resolve/"
                f"{pin['revision']}/{name}"
            )
            temporary = destination.with_suffix(destination.suffix + ".partial")
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(destination)
        observed = sha256_file(destination)
        if observed != expected["sha256"]:
            raise RuntimeError(f"hash mismatch for {name}: {observed}")
        verified[name] = {"sha256": observed, "bytes": destination.stat().st_size}
    print(json.dumps({"revision": pin["revision"], "verified": verified}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
