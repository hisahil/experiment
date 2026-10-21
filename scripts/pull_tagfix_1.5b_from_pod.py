#!/usr/bin/env python3
"""Pull 1.5B tagfix JSONL + results from RunPod via json-line filter."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POD = "020gpg3omasfwu-64410d4f@ssh.runpod.io"
KEY = Path.home() / ".ssh/deploygpu_ed25519"
REMOTE = "/root/msc_thesis_experiment"

FILES = [
    ("data/1.5b/complete_dialogues.jsonl", "data/1.5b/complete_dialogues.jsonl"),
    ("data/1.5b/post_injection_runs.jsonl", "data/1.5b/post_injection_runs.jsonl"),
    ("data/1.5b/post_injection_runs.rejudged.jsonl", "data/1.5b/post_injection_runs.rejudged.jsonl"),
]


def fetch_jsonl(remote_rel: str, local: Path) -> int:
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(local.suffix + ".tmp")
    remote = f"{REMOTE}/{remote_rel}"
    script = f"""cat '{remote}'
exit
"""
    proc = subprocess.run(
        ["ssh", "-tt", "-i", str(KEY), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=300", POD],
        input=script,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=900,
    )
    lines = 0
    with tmp.open("w", encoding="utf-8") as out:
        for line in proc.stdout.replace("\r", "").splitlines():
            if line.startswith("{"):
                out.write(line + "\n")
                lines += 1
    if lines == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"no json lines fetched for {remote_rel}")
    tmp.replace(local)
    return lines


def fetch_results_dir() -> None:
    """Pull results/1.5b csv/json via line filter for text files."""
    dest = ROOT / "results" / "1.5b"
    dest.mkdir(parents=True, exist_ok=True)
    listing = subprocess.run(
        ["ssh", "-tt", "-i", str(KEY), "-o", "StrictHostKeyChecking=no", POD],
        input=f"ls -1 {REMOTE}/results/1.5b 2>/dev/null\nexit\n",
        capture_output=True,
        text=True,
        timeout=120,
    )
    names = []
    for line in listing.stdout.replace("\r", "").splitlines():
        name = line.strip()
        if name and not name.startswith(("#", "root@", "/")) and "." in name:
            names.append(name)
    for name in names:
        remote = f"{REMOTE}/results/1.5b/{name}"
        local = dest / name
        proc = subprocess.run(
            ["ssh", "-tt", "-i", str(KEY), "-o", "StrictHostKeyChecking=no", POD],
            input=f"cat '{remote}'\nexit\n",
            capture_output=True,
            text=True,
            timeout=300,
        )
        raw = proc.stdout.replace("\r", "")
        # keep printable text payload only
        body = []
        for line in raw.splitlines():
            if line.startswith(("{", "[", '"')) or "," in line or ":" in line:
                if not line.startswith(("root@", "#", "Enjoy", "For detailed")):
                    body.append(line)
        if body:
            local.write_text("\n".join(body) + "\n", encoding="utf-8")
            print(f"  results/{name}: {len(body)} lines")


def main() -> int:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = ROOT / f"backup_gpu_live/tagfix_1.5b_pull_{stamp}"
    backup.mkdir(parents=True, exist_ok=True)

    counts = {}
    for remote_rel, local_rel in FILES:
        local = ROOT / local_rel
        print(f"==> Fetch {remote_rel} ...")
        n = fetch_jsonl(remote_rel, local)
        counts[local_rel] = n
        backup_local = backup / local_rel
        backup_local.parent.mkdir(parents=True, exist_ok=True)
        backup_local.write_bytes(local.read_bytes())
        print(f"    {n} lines -> {local}")

    print("==> Fetch results/1.5b ...")
    fetch_results_dir()
    backup_results = backup / "results" / "1.5b"
    backup_results.mkdir(parents=True, exist_ok=True)
    for p in (ROOT / "results" / "1.5b").iterdir():
        if p.is_file():
            (backup_results / p.name).write_bytes(p.read_bytes())

    mirror = ROOT / "backup_gpu_live" / "1.5b" / "data" / "1.5b"
    mirror.mkdir(parents=True, exist_ok=True)
    for _, local_rel in FILES:
        src = ROOT / local_rel
        (mirror / src.name).write_bytes(src.read_bytes())

    fb = sum(
        1
        for line in (ROOT / "data/1.5b/post_injection_runs.rejudged.jsonl").open()
        if line.strip() and json.loads(line).get("judge_fallback")
    )
    print(f"counts={counts} fallbacks={fb}")
    ok = counts.get("data/1.5b/complete_dialogues.jsonl") == 5400 and counts.get(
        "data/1.5b/post_injection_runs.rejudged.jsonl"
    ) == 10800
    print(f"VERIFY={'OK' if ok and fb == 0 else 'FAIL'}")
    print(f"backup={backup}")
    return 0 if ok and fb == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
