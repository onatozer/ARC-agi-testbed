"""Prints a markdown table comparing bench.py results against the first one given (the baseline).
usage: compare.py baseline.json other.json [more.json ...]"""
import json
import sys
from typing import Any, Dict, List


def _name(result: Dict[str, Any]) -> str:
    return result.get("label") or result["impl"]


def _micro_key(row: Dict[str, Any]) -> str:
    return f"{row['mode']} sims={row['sims']} depth={row['depth']}"


def main(paths: List[str]) -> None:
    results = [json.load(open(p)) for p in paths]
    base, others = results[0], results[1:]
    names = [_name(r) for r in results]
    print(f"Baseline: {names[0]} ({base['env']}, commit {base['commit']}, {base['time']}); speedup = baseline / candidate.\n")

    if "micro" in base:
        print("| search (micro) | " + " | ".join(f"{n} ms" for n in names) + " | " + " | ".join(f"{n} speedup" for n in names[1:]) + " |")
        print("|---|" + "---|" * (2 * len(names) - 1))
        base_rows = {_micro_key(r): r for r in base["micro"]}
        for key, base_row in base_rows.items():
            cells, speedups = [f"{base_row['ms_median']:.1f}"], []
            for other in others:
                row = {_micro_key(r): r for r in other.get("micro", [])}.get(key)
                if row is None:
                    cells.append("–"); speedups.append("–"); continue
                parity = row.get("parity")
                flag = "" if parity is None or parity["argmax_agree"] == 1.0 else f" (argmax agree {parity['argmax_agree']:.2f})"
                cells.append(f"{row['ms_median']:.1f}{flag}")
                speedups.append(f"{base_row['ms_median'] / row['ms_median']:.2f}x")
            print(f"| {key} | " + " | ".join(cells + speedups) + " |")
        print()
        print("| breakdown of the first micro row | " + " | ".join(names) + " |")
        print("|---|" + "---|" * len(names))
        for part in ("rollout", "expand", "step", "tree"):
            print(f"| {part} ms | " + " | ".join(f"{r['micro'][0]['breakdown_ms'].get(part, 0):.1f}" if r.get("micro") else "–" for r in results) + " |")
        print("| peak mem MB | " + " | ".join(f"{max(row['peak_mem_mb'] for row in r['micro']):.0f}" if r.get("micro") else "–" for r in results) + " |")
        print()

    if "train" in base:
        print("| training | " + " | ".join(names) + " | " + " | ".join(f"{n} speedup" for n in names[1:]) + " |")
        print("|---|" + "---|" * (2 * len(names) - 1))
        its = [r["train"]["it_s"] if r.get("train") else None for r in results]
        print("| it/s | " + " | ".join(f"{v:.2f}" if v else "–" for v in its) + " | " + " | ".join(f"{v / its[0]:.2f}x" if v else "–" for v in its[1:]) + " |")
        print("| step ms (median) | " + " | ".join(f"{r['train']['step_ms_median']:.0f}" if r.get("train") else "–" for r in results) + " | " + " | ".join("" for _ in others) + " |")
        print("| peak mem MB | " + " | ".join(f"{r['train']['peak_mem_mb']:.0f}" if r.get("train") else "–" for r in results) + " | " + " | ".join("" for _ in others) + " |")
        print()

    if "eval" in base:
        metric = base["eval"]["metric"]
        print("| eval | " + " | ".join(names) + " | " + " | ".join(f"{n} speedup" for n in names[1:]) + " |")
        print("|---|" + "---|" * (2 * len(names) - 1))
        secs = [r["eval"]["seconds"] if r.get("eval") else None for r in results]
        print("| seconds | " + " | ".join(f"{v:.1f}" if v else "–" for v in secs) + " | " + " | ".join(f"{secs[0] / v:.2f}x" if v else "–" for v in secs[1:]) + " |")
        accs = [r["eval"].get("accuracy") if r.get("eval") else None for r in results]
        print(f"| {metric} | " + " | ".join(f"{v:.3f}" if v is not None else "–" for v in accs) + " | " + " | ".join(f"{v - accs[0]:+.3f}" if v is not None and accs[0] is not None else "–" for v in accs[1:]) + " |")
        print("| puzzles | " + " | ".join(f"{r['eval'].get('puzzles') or '–'}" if r.get("eval") else "–" for r in results) + " | " + " | ".join("" for _ in others) + " |")
        print("| peak mem MB | " + " | ".join(f"{r['eval']['peak_mem_mb']:.0f}" if r.get("eval") else "–" for r in results) + " | " + " | ".join("" for _ in others) + " |")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
