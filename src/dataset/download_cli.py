"""CLI utility to download LIMA, Alpaca, and CodeAlpaca datasets."""

import argparse
import asyncio
import logging
import sys
from .loader import DatasetManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


async def main():
    parser = argparse.ArgumentParser(description="Download fine-tuning and instruction datasets.")
    parser.add_argument(
        "--source",
        choices=["all", "lima", "alpaca", "codealpaca"],
        default="all",
        help="Dataset to download (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="Force re-download even if cached.")
    args = parser.parse_args()

    manager = DatasetManager()
    sources = ["lima", "alpaca", "codealpaca"] if args.source == "all" else [args.source]

    print(f"[*] Starting download for dataset(s): {', '.join(sources)}")
    for src in sources:
        try:
            path = await manager.download_dataset(src, force=args.force)  # type: ignore
            print(f"[+] Downloaded {src} -> {path} ({path.stat().st_size:,} bytes)")
        except Exception as e:
            print(f"[-] Failed to download {src}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
