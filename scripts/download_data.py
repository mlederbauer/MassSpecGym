"""
Download MassSpecGym datasets from HuggingFace.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --include-molecules
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO)

from massspecgym.data.download import download_massspecgym_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MassSpecGym data from HuggingFace")
    parser.add_argument("--include-molecules", action="store_true", help="Also download molecule libraries")
    parser.add_argument("--include-retrieval", action="store_true", default=True, help="Download retrieval candidates")
    args = parser.parse_args()

    paths = download_massspecgym_data(
        include_retrieval=args.include_retrieval,
        include_molecules=args.include_molecules,
    )

    print(f"\nDownloaded {len(paths)} files:")
    for name, path in paths.items():
        print(f"  {name} -> {path}")
