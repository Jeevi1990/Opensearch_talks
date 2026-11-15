"""Prepare the demo data: ensure the embedding model exists, then generate and embed the logs (cached in data/).

Usage:
    export OS_PASSWORD='<admin password>'
    .venv/bin/python prepare_data.py [--logs 10000] [--force]
"""

import argparse
import sys
import time

from log_intel import clients as c, dataset, ml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=int, default=10_000)
    ap.add_argument("--force", action="store_true", help="regenerate even if cached")
    a = ap.parse_args()

    clients = c.OpenSearchClients()
    model_id = ml.ensure_embedding_model(clients)
    print(f"embedding model: {model_id}")

    if dataset.load_dataset(a.logs) is not None and not a.force:
        print(f"already cached: {dataset.dataset_path(a.logs)}")
        return

    t = time.perf_counter()

    def progress(done, total):
        sys.stdout.write(f"\rembedding logs: {done}/{total}")
        sys.stdout.flush()

    dataset.build_dataset(clients, model_id, a.logs, progress=progress)
    print(f"\nsaved {dataset.dataset_path(a.logs)} in {time.perf_counter() - t:.1f}s")


if __name__ == "__main__":
    main()
