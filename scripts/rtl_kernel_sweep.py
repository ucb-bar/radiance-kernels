#!/usr/bin/env python3
"""Submit RTL kernel sweeps with rtlq and collect Muon metrics as CSV."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_CONFIG = "RadianceSingleClusterConfig"
DEFAULT_KERNELS = ("vecadd", "saxpy", "sfilter", "nearn", "gaussian", "bfs")
DEFAULT_OCCUPANCIES = (1, 2, 4, 8)
DEFAULT_ILPS = (1, 2, 4, 8)
TERMINAL_STATUSES = {"done", "failed"}

REPORT_RE = re.compile(
    r"Instructions:\s+(?P<instructions>\d+).*?"
    r"Cycles:\s+(?P<cycles>\d+).*?"
    r"with decoded insts:\s+(?P<decoded_cycles>\d+).*?"
    r"with dispatched insts:\s+(?P<dispatched_cycles>\d+).*?"
    r"with eligible insts:\s+(?P<eligible_cycles>\d+).*?"
    r"with issued insts:\s+(?P<issued_cycles>\d+).*?"
    r"Warp occupancy with decoded insts:\s+(?P<decoded_warp_occ>[0-9.]+).*?"
    r"Warp occupancy with dispatched insts:\s+(?P<dispatched_warp_occ>[0-9.]+).*?"
    r"Warp occupancy with eligible insts:\s+(?P<eligible_warp_occ>[0-9.]+).*?"
    r"dispatch stalls due to write-after-write:\s+(?P<stall_waw>[0-9.]+).*?"
    r"dispatch stalls due to write-after-read:\s+(?P<stall_war>[0-9.]+).*?"
    r"dispatch stalls due to scoreboard full:\s+(?P<stall_scoreboard>[0-9.]+).*?"
    r"dispatch stalls due to RS full:\s+(?P<stall_rs>[0-9.]+).*?"
    r"dispatch stalls due to busy LSU:\s+(?P<stall_lsu>[0-9.]+).*?"
    r"issue stalls due to other busy FUs:\s+(?P<stall_fu>[0-9.]+).*?"
    r"IPC:\s+(?P<ipc>[0-9.]+)",
    re.S,
)

CSV_FIELDS = (
    "run_id",
    "job_id",
    "status",
    "config",
    "kernel",
    "variant",
    "phase",
    "occupancy",
    "core",
    "instructions",
    "cycles",
    "decoded_cycle_pct",
    "dispatched_cycle_pct",
    "eligible_cycle_pct",
    "issued_cycle_pct",
    "decoded_warp_occ",
    "dispatched_warp_occ",
    "eligible_warp_occ",
    "ipc",
    "stall_waw",
    "stall_war",
    "stall_scoreboard",
    "stall_rs",
    "stall_lsu",
    "stall_fu",
    "tag",
    "binary",
    "log",
    "trace_db",
    "verify_status",
    "verify_stdout",
    "error",
)


@dataclass(frozen=True)
class Target:
    kernel: str
    variant: str
    occupancy: int
    binary: Path

    @property
    def phase(self) -> str:
        return re.sub(r"_w\d+$", "", self.variant)


@dataclass(frozen=True)
class Job:
    job_id: int
    status: str
    config: str
    tag: str
    binary: Path
    returncode: int | None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def state_dir() -> Path:
    return Path(os.environ.get("RTLQ_HOME", "~/.rtlq")).expanduser().resolve()


def chipyard_dir() -> Path:
    return Path(os.environ.get("CHIPYARD_DIR", "/scratch/hansung/chipyard")).resolve()


def strip_soc_elf(path: Path) -> str:
    name = path.name
    suffix = ".soc.elf"
    if not name.endswith(suffix):
        raise ValueError(f"not a .soc.elf: {path}")
    return name[: -len(suffix)]


def discover_targets(
    kernels: Iterable[str],
    occupancies: Iterable[int],
    ilps: Iterable[int] | None = None,
) -> list[Target]:
    occs = tuple(int(occ) for occ in occupancies)
    ilp_values = None if ilps is None else tuple(int(ilp) for ilp in ilps)
    targets: list[Target] = []
    for kernel in kernels:
        kdir = repo_root() / "kernels" / kernel
        if not kdir.is_dir():
            raise SystemExit(f"unknown kernel directory: {kdir}")

        for occ in occs:
            if ilp_values is not None and kernel in {"saxpy", "nearn", "gemm_simt", "softmax", "gemv"}:
                paths = [kdir / f"kernel_ilp{ilp}_w{occ}.soc.elf" for ilp in ilp_values]
            elif ilp_values is not None:
                raise SystemExit(f"--ilps is only supported for saxpy/nearn/gemm_simt/softmax/gemv, not {kernel}")
            elif kernel in {"vecadd", "saxpy", "sfilter", "nearn"}:
                paths = [kdir / f"kernel_w{occ}.soc.elf"]
            elif kernel == "gaussian":
                paths = sorted(kdir.glob(f"fan[12]_t*_w{occ}.soc.elf"))
            elif kernel == "bfs":
                paths = sorted(kdir.glob(f"bfs[12]_l*_w{occ}.soc.elf"))
            else:
                raise SystemExit(f"no discovery rule for kernel: {kernel}")

            for path in paths:
                if not path.is_file():
                    raise SystemExit(f"missing ELF for {kernel} occupancy {occ}: {path}")
                targets.append(Target(kernel, strip_soc_elf(path), occ, path.resolve()))
    return targets


def run_id_default() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def tag_for(run_id: str, target: Target) -> str:
    return f"rk-{run_id}-{target.kernel}-{target.variant}"


def run_cmd(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def submit_targets(targets: list[Target], *, config: str, run_id: str, slots: int | None) -> list[int]:
    if slots is not None:
        proc = run_cmd(["rtlq", "slots", str(slots)])
        if proc.returncode != 0:
            raise SystemExit(proc.stderr.strip() or proc.stdout.strip())

    job_ids: list[int] = []
    for target in targets:
        tag = tag_for(run_id, target)
        proc = run_cmd(["rtlq", config, str(target.binary), tag])
        if proc.returncode != 0:
            raise SystemExit(proc.stderr.strip() or proc.stdout.strip())
        match = re.search(r"jobid:\s+(\d+)", proc.stdout)
        if not match:
            raise SystemExit(f"could not parse rtlq job id from:\n{proc.stdout}")
        job_id = int(match.group(1))
        job_ids.append(job_id)
        print(f"submitted job={job_id} kernel={target.kernel} variant={target.variant}", file=sys.stderr)
    return job_ids


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(state_dir() / "jobs.sqlite3")
    conn.row_factory = sqlite3.Row
    return conn


def read_jobs(job_ids: Iterable[int] | None = None, *, run_id: str | None = None) -> list[Job]:
    conn = db_connect()
    try:
        if job_ids is not None:
            ids = tuple(int(job_id) for job_id in job_ids)
            if not ids:
                return []
            placeholders = ",".join("?" for _ in ids)
            rows = conn.execute(
                f"SELECT id, status, config, suffix, binary, returncode FROM jobs WHERE id IN ({placeholders}) ORDER BY id",
                ids,
            ).fetchall()
        elif run_id is not None:
            rows = conn.execute(
                """
                SELECT id, status, config, suffix, binary, returncode
                FROM jobs
                WHERE suffix LIKE ?
                ORDER BY id
                """,
                (f"rk-{run_id}-%",),
            ).fetchall()
        else:
            raise ValueError("job_ids or run_id required")
    finally:
        conn.close()

    return [
        Job(
            job_id=int(row["id"]),
            status=str(row["status"]),
            config=str(row["config"]),
            tag=str(row["suffix"]),
            binary=Path(str(row["binary"])),
            returncode=None if row["returncode"] is None else int(row["returncode"]),
        )
        for row in rows
    ]


def wait_jobs(job_ids: list[int], *, poll_sec: float, timeout_sec: float | None) -> list[Job]:
    deadline = None if timeout_sec is None else time.time() + timeout_sec
    while True:
        proc = run_cmd(["rtlq", "list"])
        if proc.returncode != 0:
            raise SystemExit(proc.stderr.strip() or proc.stdout.strip())
        jobs = read_jobs(job_ids)
        status_counts: dict[str, int] = {}
        for job in jobs:
            status_counts[job.status] = status_counts.get(job.status, 0) + 1
        print("status " + " ".join(f"{k}={v}" for k, v in sorted(status_counts.items())), file=sys.stderr)
        if jobs and all(job.status in TERMINAL_STATUSES for job in jobs):
            return jobs
        if deadline is not None and time.time() >= deadline:
            return jobs
        time.sleep(poll_sec)


def target_from_job(job: Job) -> Target:
    parts = job.binary.parts
    try:
        kernel = parts[parts.index("kernels") + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(f"could not infer kernel from binary path: {job.binary}") from exc
    variant = strip_soc_elf(job.binary)
    match = re.search(r"_w(\d+)$", variant)
    if not match:
        raise SystemExit(f"could not infer occupancy from variant: {variant}")
    return Target(kernel, variant, int(match.group(1)), job.binary)


def output_dirs(config: str) -> list[Path]:
    root = chipyard_dir() / "sims" / "vcs" / "output"
    exact = root / f"chipyard.harness.TestHarness.{config}"
    if exact.is_dir():
        return [exact]
    return sorted(path for path in root.glob(f"*.{config}") if path.is_dir())


def resolve_log(job: Job) -> Path:
    binary_stem = job.binary.name[: -len(".elf")]
    pattern = f"{binary_stem}.{job.tag}.log"
    matches: list[Path] = []
    for directory in output_dirs(job.config):
        matches.extend(sorted(directory.rglob(pattern)))
    if not matches:
        raise FileNotFoundError(f"no VCS log matched {pattern}")
    return matches[-1]


def resolve_trace_db(job: Job) -> Path:
    binary_stem = job.binary.name[: -len(".elf")]
    pattern = f"{binary_stem}.{job.tag}.sqlite"
    matches: list[Path] = []
    for directory in output_dirs(job.config):
        matches.extend(sorted(directory.rglob(pattern)))
    if not matches:
        raise FileNotFoundError(f"no SQLite trace matched {pattern}")
    return matches[-1]


def radiance_elf_for(job: Job) -> Path:
    target = target_from_job(job)
    return job.binary.with_name(f"{target.variant}.radiance.elf")


def verify_trace(job: Job) -> tuple[str, str, str, str]:
    target = target_from_job(job)
    verifier = repo_root() / "kernels" / target.kernel / "verify_trace.py"
    if not verifier.is_file():
        raise FileNotFoundError(f"no verifier for kernel {target.kernel}: {verifier}")
    trace_db = resolve_trace_db(job)
    radiance_elf = radiance_elf_for(job)
    expected = repo_root() / "kernels" / target.kernel / "expected"
    proc = subprocess.run(
        [
            sys.executable,
            str(verifier),
            str(trace_db),
            str(radiance_elf),
            "--expected",
            str(expected),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    output = "\n".join(part for part in (proc.stdout.strip(), proc.stderr.strip()) if part)
    status = "pass" if proc.returncode == 0 else "fail"
    return status, str(trace_db), output, "" if proc.returncode == 0 else output


def parse_reports(log: Path) -> list[dict[str, str]]:
    text = log.read_text(encoding="utf-8", errors="replace")
    matches = list(REPORT_RE.finditer(text))
    if not matches:
        raise ValueError(f"no Muon performance report found in {log}")
    rows: list[dict[str, str]] = []
    for core_id, match in enumerate(matches):
        row = {key: value for key, value in match.groupdict().items()}
        row["core"] = str(core_id)
        cycles = int(row["cycles"])
        for source, dest in (
            ("decoded_cycles", "decoded_cycle_pct"),
            ("dispatched_cycles", "dispatched_cycle_pct"),
            ("eligible_cycles", "eligible_cycle_pct"),
            ("issued_cycles", "issued_cycle_pct"),
        ):
            row[dest] = f"{100.0 * int(row[source]) / cycles:.2f}" if cycles else ""
            del row[source]
        rows.append(row)
    return rows


def log_failure_summary(log: Path) -> str:
    text = log.read_text(encoding="utf-8", errors="replace")
    for pattern in (
        r"Assertion failed: (?P<message>[^\n]+)",
        r"Fatal: (?P<message>[^\n]+)",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group("message").strip()
    return ""


def collect_rows(jobs: Iterable[Job], *, run_id: str, verify_traces: bool = False) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for job in jobs:
        target = target_from_job(job)
        base = {
            "run_id": run_id,
            "job_id": str(job.job_id),
            "status": job.status,
            "config": job.config,
            "kernel": target.kernel,
            "variant": target.variant,
            "phase": target.phase,
            "occupancy": str(target.occupancy),
            "tag": job.tag,
            "binary": str(job.binary),
            "log": "",
            "trace_db": "",
            "verify_status": "",
            "verify_stdout": "",
        }
        verify_error = ""
        if verify_traces:
            try:
                verify_status, trace_db, verify_stdout, verify_error = verify_trace(job)
                base["trace_db"] = trace_db
                base["verify_status"] = verify_status
                base["verify_stdout"] = verify_stdout
            except (FileNotFoundError, ValueError, subprocess.SubprocessError) as exc:
                verify_error = str(exc)
                base["verify_status"] = "error"

        try:
            log = resolve_log(job)
            reports = parse_reports(log)
        except (FileNotFoundError, ValueError) as exc:
            row = {field: "" for field in CSV_FIELDS}
            row.update(base)
            row["core"] = "error"
            row["log"] = "" if isinstance(exc, FileNotFoundError) else str(log)
            row["error"] = str(exc)
            if not isinstance(exc, FileNotFoundError):
                failure = log_failure_summary(log)
                if failure:
                    row["error"] = f"{row['error']}; rtl: {failure}"
            if verify_error:
                row["error"] = f"{row['error']}; verify: {verify_error}"
            rows.append(row)
            continue

        for metrics in reports:
            row = {field: "" for field in CSV_FIELDS}
            row.update(base)
            row.update(metrics)
            row["log"] = str(log)
            if job.status != "done":
                row["error"] = f"rtlq status={job.status} returncode={job.returncode}; collected from artifacts"
            if verify_error:
                row["error"] = "; ".join(part for part in (row["error"], f"verify: {verify_error}") if part)
            rows.append(row)
    return rows


def write_csv(rows: list[dict[str, str]], path: Path | None) -> None:
    if path is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def add_target_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--kernels", nargs="+", default=list(DEFAULT_KERNELS), help="Kernel names to run")
    parser.add_argument("--occupancies", nargs="+", type=int, default=list(DEFAULT_OCCUPANCIES))
    parser.add_argument("--ilps", nargs="+", type=int, help="ILP values for saxpy/nearn kernel_ilp*_w* variants")
    parser.add_argument("--config", default=DEFAULT_CONFIG)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-targets", help="List discovered ELF targets")
    add_target_args(list_parser)

    submit = subparsers.add_parser("submit", help="Submit discovered targets to rtlq")
    add_target_args(submit)
    submit.add_argument("--run-id", default=run_id_default())
    submit.add_argument("--slots", type=int)

    wait = subparsers.add_parser("wait", help="Wait for jobs from a run id")
    wait.add_argument("--run-id", required=True)
    wait.add_argument("--poll-sec", type=float, default=15.0)
    wait.add_argument("--timeout-sec", type=float)

    collect = subparsers.add_parser("collect", help="Collect metrics from a run id")
    collect.add_argument("--run-id", required=True)
    collect.add_argument("--csv", type=Path, help="Output CSV path; stdout if omitted")
    collect.add_argument("--verify-traces", action="store_true", help="Run kernel verify_trace.py against RTL SQLite traces")

    run = subparsers.add_parser("run", help="Submit, wait, and collect in one command")
    add_target_args(run)
    run.add_argument("--run-id", default=run_id_default())
    run.add_argument("--slots", type=int)
    run.add_argument("--poll-sec", type=float, default=15.0)
    run.add_argument("--timeout-sec", type=float)
    run.add_argument("--csv", type=Path, help="Output CSV path; stdout if omitted")
    run.add_argument("--verify-traces", action="store_true", help="Run kernel verify_trace.py against RTL SQLite traces")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "list-targets":
        for target in discover_targets(args.kernels, args.occupancies, args.ilps):
            print(f"{target.kernel},{target.variant},{target.occupancy},{target.binary}")
        return 0

    if args.command == "submit":
        targets = discover_targets(args.kernels, args.occupancies, args.ilps)
        submit_targets(targets, config=args.config, run_id=args.run_id, slots=args.slots)
        print(args.run_id)
        return 0

    if args.command == "wait":
        jobs = read_jobs(run_id=args.run_id)
        if not jobs:
            raise SystemExit(f"no jobs found for run id: {args.run_id}")
        wait_jobs([job.job_id for job in jobs], poll_sec=args.poll_sec, timeout_sec=args.timeout_sec)
        return 0

    if args.command == "collect":
        jobs = read_jobs(run_id=args.run_id)
        if not jobs:
            raise SystemExit(f"no jobs found for run id: {args.run_id}")
        write_csv(collect_rows(jobs, run_id=args.run_id, verify_traces=args.verify_traces), args.csv)
        return 0

    if args.command == "run":
        targets = discover_targets(args.kernels, args.occupancies, args.ilps)
        job_ids = submit_targets(targets, config=args.config, run_id=args.run_id, slots=args.slots)
        jobs = wait_jobs(job_ids, poll_sec=args.poll_sec, timeout_sec=args.timeout_sec)
        write_csv(collect_rows(jobs, run_id=args.run_id, verify_traces=args.verify_traces), args.csv)
        return 0

    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
