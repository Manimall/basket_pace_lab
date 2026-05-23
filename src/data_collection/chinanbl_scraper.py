"""Shim — ChinaNBLCollector moved to src.data_collection.chinanbl.collector."""
from src.data_collection.chinanbl.collector import ChinaNBLCollector  # noqa: F401

__all__ = ["ChinaNBLCollector"]

if __name__ == "__main__":
    import argparse
    import logging
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout)
    parser = argparse.ArgumentParser(description="Collect ChinaNBL from Sofascore")
    parser.add_argument("--no-stats", action="store_true")
    args = parser.parse_args()
    ChinaNBLCollector(fetch_stats=not args.no_stats).run_sync()
