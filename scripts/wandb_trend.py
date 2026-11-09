"""Average an offline W&B run's meta metrics over 25-update windows (custom .wandb block parser; wandb 0.31 has no datastore).
Usage: python scripts/wandb_trend.py <OUTPUT_ROOT of a meta run>  ->  one line per window: updates a - b query_loss=... ..."""
import sys, json, glob, struct, statistics as st
from wandb.proto import wandb_internal_pb2 as pb

f = sorted(glob.glob(sys.argv[1] + "/outputs/*/*/*/wandb/*run-*/run-*.wandb"))[-1]
data = open(f, "rb").read()
pos, block, records, buf = 7, 32768, [], b""
while pos + 7 <= len(data):
    if block - (pos % block) < 7:
        pos += block - (pos % block)
        continue
    _crc, length, kind = struct.unpack("<IHB", data[pos:pos + 7])
    chunk = data[pos + 7:pos + 7 + length]
    pos += 7 + length
    if kind == 0 and length == 0:
        continue
    buf = chunk if kind in (1, 2) else buf + chunk
    if kind in (1, 4):
        records.append(buf)
rows = []
for raw in records:
    r = pb.Record()
    try:
        r.ParseFromString(raw)
    except Exception:
        continue
    if r.WhichOneof("record_type") == "history":
        h = {i.key or "/".join(i.nested_key): i.value_json for i in r.history.item}
        if "train/meta/support_loss" in h:
            rows.append(h)
ks = ["train/meta/query_loss", "train/meta/support_loss", "train/meta/alpha_mean", "train/meta/alpha_max"]
v = lambda h, k: float(json.loads(h[k]))
for a in range(0, len(rows), 25):
    w = rows[a:a + 25]
    print("updates", a + 1, "-", a + len(w), " ".join(k.split("/")[-1] + "=" + format(st.mean(v(h, k) for h in w), ".4g") for k in ks))
