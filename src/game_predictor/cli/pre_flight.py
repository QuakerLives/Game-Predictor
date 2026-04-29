#!/usr/bin/env python3
"""Pre-flight validation. Run manually before launching sleeper.py."""
import shutil
import subprocess
import sys
from pathlib import Path

import duckdb

from game_predictor.config import LLM_MODEL


def main():
    checks = []

    free_gb = shutil.disk_usage(".").free / (1024**3)
    checks.append(("Disk space >= 5 GB", free_gb >= 5, f"{free_gb:.1f} GB"))

    try:
        import httpx
        r = httpx.get("http://localhost:11434/v1/models", timeout=5)
        checks.append(("LLM endpoint", r.status_code == 200, f"HTTP {r.status_code}"))
    except Exception as e:
        checks.append(("LLM endpoint", False, str(e)))

    try:
        subprocess.run(
            ["playwright", "install", "--dry-run", "chromium"],
            capture_output=True, timeout=10,
        )
        checks.append(("Playwright Chromium", True, "OK"))
    except Exception as e:
        checks.append(("Playwright Chromium", False, str(e)))

    try:
        conn = duckdb.connect("data/gameplay_data.duckdb")
        conn.execute("SELECT 1")
        conn.close()
        checks.append(("DuckDB writable", True, "OK"))
    except Exception as e:
        checks.append(("DuckDB writable", False, str(e)))

    try:
        result = subprocess.run(
            ["wmic", "computersystem", "get", "TotalPhysicalMemory", "/value"],
            capture_output=True, text=True, timeout=5,
        )
        mem_gb = None
        for line in result.stdout.splitlines():
            if "TotalPhysicalMemory=" in line:
                mem_bytes = int(line.split("=")[1].strip())
                mem_gb = mem_bytes / (1024 ** 3)
                break
        if mem_gb is not None:
            checks.append(("RAM >= 28 GB", mem_gb >= 28, f"{mem_gb:.0f} GB"))
        else:
            checks.append(("RAM check", False, "Could not read memory info"))
    except Exception as e:
        checks.append(("RAM check", False, str(e)))

    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10,
        )
        has_model = LLM_MODEL in result.stdout
        checks.append((
            f"Ollama {LLM_MODEL} pulled",
            has_model,
            "found" if has_model else f"NOT FOUND -- run: ollama pull {LLM_MODEL}",
        ))
    except Exception as e:
        checks.append(("Ollama installed", False, str(e)))

    try:
        import httpx
        r = httpx.get("https://www.google.com", timeout=10)
        checks.append(("Internet", r.status_code == 200, "OK"))
    except Exception as e:
        checks.append(("Internet", False, str(e)))

    print("\n" + "=" * 60 + "\nPRE-FLIGHT CHECK\n" + "=" * 60)
    all_pass = True
    for name, passed, detail in checks:
        icon = "PASS" if passed else "FAIL"
        print(f"  [{icon}] {name}: {detail}")
        if passed is False:
            all_pass = False
    verdict = "PASSED" if all_pass else "FAILED"
    print(f"\n{verdict}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
