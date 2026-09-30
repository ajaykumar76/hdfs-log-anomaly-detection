"""
Parse HDFS.log with Drain3, extract block IDs, and group template IDs
into per-block sequences.

Usage:
    python src\01_parse.py --limit 200000     # test mode, fast
    python src\01_parse.py                     # full run, ~11M lines
"""
import argparse
import re
import json
import time
from pathlib import Path
from collections import defaultdict

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

BLOCK_RE = re.compile(r"(blk_-?\d+)")

def main(limit):
    log_path = Path("data/raw/HDFS.log")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = TemplateMinerConfig()
    config.profiling_enabled = False
    miner = TemplateMiner(config=config)

    block_sequences = defaultdict(list)
    template_texts = {}

    n_lines = 0
    n_matched = 0
    start = time.time()

    with open(log_path, "r", errors="replace") as f:
        for line in f:
            n_lines += 1
            if limit and n_lines > limit:
                break

            parts = line.strip().split(" ", 5)
            if len(parts) < 6:
                continue
            message = parts[5]

            result = miner.add_log_message(message)
            template_id = result["cluster_id"]
            template_texts[template_id] = result["template_mined"]

            block_ids = set(BLOCK_RE.findall(message))
            for bid in block_ids:
                block_sequences[bid].append(template_id)
                n_matched += 1

            if n_lines % 200000 == 0:
                elapsed = time.time() - start
                print(f"  {n_lines:,} lines processed ({elapsed:.0f}s), "
                      f"{len(miner.drain.clusters)} templates found so far")

    elapsed = time.time() - start
    print(f"\nDone. {n_lines:,} lines processed in {elapsed:.0f}s")
    print(f"{len(miner.drain.clusters)} unique templates discovered")
    print(f"{len(block_sequences):,} unique block IDs found")
    print(f"{n_matched:,} block-ID mentions matched to a template")

    templates_out = out_dir / "templates.json"
    with open(templates_out, "w") as f:
        json.dump({str(k): v for k, v in template_texts.items()}, f, indent=2)
    print(f"Saved templates to {templates_out}")

    suffix = f"_test{limit}" if limit else ""
    seq_out = out_dir / f"block_sequences{suffix}.json"
    with open(seq_out, "w") as f:
        json.dump(block_sequences, f)
    print(f"Saved block sequences to {seq_out}")

    sample_block = next(iter(block_sequences))
    print(f"\nSample block '{sample_block}':")
    print(f"  Sequence: {block_sequences[sample_block]}")
    print(f"  Length: {len(block_sequences[sample_block])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0,
                        help="Only process this many lines (0 = full file)")
    args = parser.parse_args()
    main(args.limit)