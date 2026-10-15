"""Reads the history rows of an offline W&B run (no upload needed)."""
import glob, json
from typing import Dict, Iterator

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore


def history(run_dir: str) -> Iterator[Dict[str, float]]:
    """Yields each logged history row of every .wandb file under run_dir."""
    for path in [p for p in glob.glob(run_dir + "/**/*.wandb", recursive=True) if "latest-run" not in p]:
        ds = datastore.DataStore()
        ds.open_for_scan(path)
        while True:
            try:
                data = ds.scan_data()
            except AssertionError:  # trailing partial record
                break
            if data is None:
                break
            rec = wandb_internal_pb2.Record()
            rec.ParseFromString(data)
            if rec.WhichOneof("record_type") == "history":
                yield {item.key or "/".join(item.nested_key): json.loads(item.value_json) for item in rec.history.item}
