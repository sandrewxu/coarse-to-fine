#!/usr/bin/env python3
"""Parse a throughput-sweep log directory and print a markdown summary.

Usage:
    python scripts/parse_rl_sweep.py logs/rl/joint/sweep_<TS>/

Reads each ``<probe>.log`` in the sweep dir, extracts the per-step metric
lines emitted by veRL (``step:N - actor/entropy:... - perf/throughput:... -
timing_s/gen:...``), averages steps 2..N (skipping step 1 to drop init
overhead), and prints a single markdown table comparing probes.
"""

import re
import sys
from pathlib import Path

# veRL's step-metric line is one giant " - "-joined line that begins with
# ``step:N - actor/entropy:...`` (after Ray's ``[actor pid]`` prefix). We
# anchor on that exact prefix to avoid false matches against ``timing_s/step:N``
# (a per-step duration field) or ``training/global_step:N`` (cumulative ID).
STEP_RE = re.compile(r"(?:^|\s)step:(\d+) - actor/entropy:")
KV_RE = re.compile(r"([A-Za-z_][\w/.@]*):([-\d.eE+]+)")


def parse_log(path: Path) -> dict[int, dict[str, float]]:
    """Return {step_num: {metric: value}} for all step-metric lines in the log."""
    steps: dict[int, dict[str, float]] = {}
    for line in path.read_text(errors="replace").splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        step_num = int(m.group(1))
        # Take the rest of the line from the match position onward so KV_RE
        # picks up every metric the line carries.
        rest = line[m.start():]
        kvs: dict[str, float] = {}
        for k, v in KV_RE.findall(rest):
            try:
                kvs[k] = float(v)
            except ValueError:
                continue
        if kvs:
            steps[step_num] = kvs
    return steps


def find_failure(path: Path) -> str | None:
    """Return a one-line failure summary, or None if no obvious failure."""
    text = path.read_text(errors="replace")
    for marker in ("AssertionError", "RuntimeError", "OutOfMemoryError", "Engine core initialization failed"):
        idx = text.find(marker)
        if idx == -1:
            continue
        # Take the line + a short context.
        line_end = text.find("\n", idx)
        return text[idx : line_end if line_end != -1 else idx + 200].strip()
    return None


def avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def fmt(x: float, ndigits: int = 2) -> str:
    if x != x:  # NaN
        return "—"
    return f"{x:.{ndigits}f}"


def summarise(probe_log: Path) -> dict:
    steps = parse_log(probe_log)
    fail = find_failure(probe_log)

    # Skip step 1 (init amortization). Use 2..N.
    measured = [s for s in sorted(steps) if s >= 2]
    if not measured:
        return {
            "name": probe_log.stem,
            "ok": False,
            "fail": fail or "no step:N metric lines parsed",
        }

    pick = lambda key: [steps[s].get(key, float("nan")) for s in measured]
    return {
        "name": probe_log.stem,
        "ok": True,
        "fail": fail,
        "n_steps": len(measured),
        "step_time": avg(pick("perf/time_per_step")),
        "tok_per_gpu_s": avg(pick("perf/throughput")),
        "tok_total_s": avg(pick("perf/throughput")) * 8,  # 8 GPUs
        "mfu": avg(pick("perf/mfu/actor")),
        "total_tokens": avg(pick("perf/total_num_tokens")),
        "gen": avg(pick("timing_s/gen")),
        "old_logp": avg(pick("timing_s/old_log_prob")),
        "update_actor": avg(pick("timing_s/update_actor")),
        "max_mem_gb": max(pick("perf/max_memory_allocated_gb"), default=float("nan")),
        "grad_norm": avg(pick("actor/grad_norm")),
        "score_mean": avg(pick("critic/score/mean")),
    }


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sweep_dir = Path(sys.argv[1])
    if not sweep_dir.is_dir():
        print(f"Not a directory: {sweep_dir}", file=sys.stderr)
        sys.exit(2)

    logs = sorted(sweep_dir.glob("*.log"))
    if not logs:
        print(f"No *.log files in {sweep_dir}", file=sys.stderr)
        sys.exit(2)

    rows = [summarise(p) for p in logs]

    print(f"# Sweep results — `{sweep_dir.name}`\n")
    print("| probe | ok | n | step (s) | tok/GPU/s | aggr tok/s | MFU | gen (s) | old_logp (s) | update (s) | peak mem (GB) | grad_norm | score |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if not r["ok"]:
            print(f"| {r['name']} | ✗ | — | — | — | — | — | — | — | — | — | — | — |  *{r['fail']}*")
            continue
        mfu_pct = r["mfu"] * 100 if r["mfu"] == r["mfu"] else float("nan")
        print(
            f"| {r['name']} | ✓ | {r['n_steps']} "
            f"| {fmt(r['step_time'])} "
            f"| {fmt(r['tok_per_gpu_s'], 1)} "
            f"| {fmt(r['tok_total_s'], 0)} "
            f"| {fmt(mfu_pct, 1)}% "
            f"| {fmt(r['gen'])} "
            f"| {fmt(r['old_logp'])} "
            f"| {fmt(r['update_actor'])} "
            f"| {fmt(r['max_mem_gb'], 1)} "
            f"| {fmt(r['grad_norm'], 1)} "
            f"| {fmt(r['score_mean'], 2)} |"
        )

    # Highlight winner by aggregate tok/s.
    ok_rows = [r for r in rows if r["ok"]]
    if ok_rows:
        winner = max(ok_rows, key=lambda r: r["tok_total_s"])
        print(f"\n**Best aggregate throughput:** `{winner['name']}` "
              f"at {fmt(winner['tok_total_s'], 0)} tok/s "
              f"({fmt(winner['tok_per_gpu_s'], 1)} per GPU)")


if __name__ == "__main__":
    main()
