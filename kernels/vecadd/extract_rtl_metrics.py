#!/usr/bin/env python3

import argparse
import re
from pathlib import Path


REPORT_RE = re.compile(
    r"Instructions:\s+(?P<instructions>\d+).*?"
    r"Cycles:\s+(?P<cycles>\d+).*?"
    r"with decoded insts:\s+(?P<decoded>\d+).*?"
    r"with dispatched insts:\s+(?P<dispatched>\d+).*?"
    r"with eligible insts:\s+(?P<eligible>\d+).*?"
    r"with issued insts:\s+(?P<issued>\d+).*?"
    r"dispatch stalls due to write-after-write:\s+(?P<waw>[0-9.]+).*?"
    r"dispatch stalls due to write-after-read:\s+(?P<war>[0-9.]+).*?"
    r"dispatch stalls due to scoreboard full:\s+(?P<scoreboard>[0-9.]+).*?"
    r"dispatch stalls due to RS full:\s+(?P<rs>[0-9.]+).*?"
    r"dispatch stalls due to busy LSU:\s+(?P<lsu>[0-9.]+).*?"
    r"issue stalls due to other busy FUs:\s+(?P<fu>[0-9.]+).*?"
    r"IPC:\s+(?P<ipc>[0-9.]+)",
    re.S,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Muon performance metrics from an RTL VCS log."
    )
    parser.add_argument("log", type=Path, help="Path to VCS .log file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    text = args.log.read_text()
    matches = list(REPORT_RE.finditer(text))
    if not matches:
        raise SystemExit(f"no Muon performance report found in {args.log}")

    print(f"log={args.log}")
    ipc_total = 0.0
    cycles_total = 0
    for core_id, match in enumerate(matches):
        groups = match.groupdict()
        ipc_total += float(groups["ipc"])
        cycles_total += int(groups["cycles"])
        print(
            "core={core} instructions={instructions} cycles={cycles} "
            "decoded={decoded} dispatched={dispatched} eligible={eligible} issued={issued} "
            "ipc={ipc} waw={waw} war={war} scoreboard={scoreboard} rs={rs} lsu={lsu} fu={fu}".format(
                core=core_id,
                **groups,
            )
        )

    avg_ipc = ipc_total / len(matches)
    avg_cycles = cycles_total / len(matches)
    print(f"summary cores={len(matches)} avg_ipc={avg_ipc:.3f} avg_cycles={avg_cycles:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
