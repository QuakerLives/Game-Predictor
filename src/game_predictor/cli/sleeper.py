#!/usr/bin/env python3
"""
SLEEPER AGENT -- Autonomous watchdog for overnight runs.
TOP-LEVEL ENTRY POINT. Run this, then go to sleep.

Usage:
    uv run game-predictor-sleeper [--llm-base-url http://localhost:11434/v1]
                                  [--max-restarts 5] [--hang-timeout 600]
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import httpx

from game_predictor.config import LLM_MODEL

HEARTBEAT_FILE = Path("heartbeat.json")
PROD_LOG_FILE = Path("production_run.log")
DB_PATH = "data/gameplay_data.duckdb"
OLLAMA_HEALTH_URL = "http://localhost:11434/v1/models"
DEFAULT_HANG_TIMEOUT = 900
DEFAULT_MAX_RESTARTS = 8
TOTAL_TARGET = 10000
ALLOWED_ACTIONS = {
    "restart_production",
    "restart_ollama",
    "restart_browser",
    "switch_to_bing",
    "increase_delays",
    "skip_current_game",
    "clear_heartbeat_and_retry",
    "abort",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SLEEPER] %(message)s",
    handlers=[
        logging.FileHandler("sleeper.log", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("sleeper")


def read_heartbeat() -> dict | None:
    try:
        return json.loads(HEARTBEAT_FILE.read_text()) if HEARTBEAT_FILE.exists() else None
    except Exception:
        return None


def read_log_tail(n=80) -> str:
    try:
        return "\n".join(PROD_LOG_FILE.read_text().splitlines()[-n:]) if PROD_LOG_FILE.exists() else ""
    except Exception:
        return ""


def read_db_state() -> dict:
    try:
        c = duckdb.connect(DB_PATH, read_only=True)
        t = c.execute("SELECT COUNT(*) FROM gameplay_records").fetchone()[0]
        pg = dict(
            c.execute(
                "SELECT video_game_name, COUNT(*) FROM gameplay_records GROUP BY 1"
            ).fetchall()
        )
        c.close()
        return {"total": t, "per_game": pg}
    except Exception as e:
        return {"total": -1, "per_game": {}, "error": str(e)}


def check_ollama_health() -> bool:
    try:
        return httpx.get(OLLAMA_HEALTH_URL, timeout=10).status_code == 200
    except Exception:
        return False


def diagnose(
    exit_code, heartbeat, log_tail, db_state, ollama_ok, restart_count, url
) -> dict:
    prompt = f"""You are a systems diagnostician for a Python web-scraping pipeline.
The pipeline crashed or hung. Analyze these signals and recommend exactly ONE action.

SIGNALS:
- Exit code: {exit_code}
- Ollama healthy: {ollama_ok}
- Restart count: {restart_count}/5
- Heartbeat: {json.dumps(heartbeat, indent=2) if heartbeat else "MISSING"}
- DB state: {json.dumps(db_state, indent=2)}
- Last 80 log lines:
```
{log_tail[-4000:]}
```

ALLOWED ACTIONS:
- restart_production: Simple restart --resume. Transient crash.
- restart_ollama: Restart Ollama daemon. LLM failures / Ollama unhealthy.
- restart_browser: Clear Playwright state. Browser/page crash errors.
- switch_to_bing: Switch to Bing Images. Google CAPTCHA / rate-limit blocks.
- increase_delays: Double rate-limit delays. Rate-limiting errors.
- skip_current_game: Skip failing game. One game's sources consistently broken.
- clear_heartbeat_and_retry: Clear stale heartbeat. Stale heartbeat but recent log.
- abort: Stop entirely. Fundamental issue (disk full, no internet, 5th identical crash).

Return JSON only: {{"diagnosis": "...", "action": "action_name", "reasoning": "..."}}"""

    try:
        r = httpx.post(
            f"{url}/chat/completions",
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,
                "temperature": 0.2,
            },
            timeout=180,
        )
        text = r.json()["choices"][0]["message"]["content"]
        text = text.replace("```json", "").replace("```", "").strip()
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            text_lower = text.lower()
            action = "restart_production"
            for a in ALLOWED_ACTIONS:
                if a.replace("_", " ") in text_lower or a in text_lower:
                    action = a
                    break
            result = {"diagnosis": text[:200], "action": action, "reasoning": "parsed from malformed JSON"}
        if result.get("action") not in ALLOWED_ACTIONS:
            result["action"] = "restart_production"
        return result
    except Exception as e:
        logger.error(f"Diagnosis failed: {e}")
        if not ollama_ok:
            return {
                "diagnosis": "Ollama unreachable",
                "action": "restart_ollama",
                "reasoning": "LLM down",
            }
        return {
            "diagnosis": str(e),
            "action": "restart_production",
            "reasoning": "Fallback restart",
        }


def execute_action(action: str, llm_url: str) -> bool:
    logger.info(f"Executing: {action}")

    if action == "restart_production":
        return True

    if action == "restart_ollama":
        try:
            # Windows: taskkill; Unix fallback: pkill
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/IM", "ollama.exe"],
                    timeout=10, capture_output=True,
                )
            else:
                subprocess.run(["pkill", "-f", "ollama"], timeout=10, capture_output=True)
            time.sleep(3)
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(5)
            subprocess.run(
                ["ollama", "run", LLM_MODEL, "--keepalive", "12h"],
                input="hello",
                capture_output=True,
                text=True,
                timeout=120,
            )
            for _ in range(12):
                time.sleep(5)
                if check_ollama_health():
                    return True
            return False
        except Exception as e:
            logger.error(f"Ollama restart failed: {e}")
            return False

    if action == "restart_browser":
        import shutil
        tmp = Path(tempfile.gettempdir())
        for d in tmp.glob("playwright-*"):
            shutil.rmtree(d, ignore_errors=True)
        return True

    if action == "switch_to_bing":
        os.environ["SCRAPER_SEARCH_FALLBACK"] = "bing"
        return True

    if action == "increase_delays":
        cur = int(os.environ.get("SCRAPER_DELAY_MULTIPLIER", "1"))
        os.environ["SCRAPER_DELAY_MULTIPLIER"] = str(cur * 2)
        return True

    if action == "skip_current_game":
        hb = read_heartbeat()
        if hb and hb.get("per_game"):
            game = min(hb["per_game"], key=hb["per_game"].get)
            Path(f".skip_{game.replace(' ', '_').lower()}").write_text(
                f"Skipped by sleeper at {datetime.now().isoformat()}"
            )
        return True

    if action == "clear_heartbeat_and_retry":
        HEARTBEAT_FILE.unlink(missing_ok=True)
        return True

    if action == "abort":
        logger.error("ABORT -- human intervention required.")
        return False

    return True


def spawn(url):
    cmd = [sys.executable, "-m", "game_predictor.cli.run_production",
           "--llm-base-url", url, "--resume"]
    logger.info(f"Spawning: {' '.join(cmd)}")
    child_env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    child_env.pop("PLAYWRIGHT_BROWSERS_PATH", None)
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env=child_env,
    )
    logger.info(f"Child PID: {p.pid}")
    return p


def spawn_enrichment(url, max_duration="2h"):
    """Spawn run_production.py in --enrich-only mode."""
    cmd = [sys.executable, "-m", "game_predictor.cli.run_production",
           "--llm-base-url", url, "--enrich-only", "--max-duration", max_duration]
    logger.info(f"Spawning enrichment: {' '.join(cmd)}")
    child_env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    child_env.pop("PLAYWRIGHT_BROWSERS_PATH", None)
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env=child_env,
    )
    logger.info(f"Enrichment child PID: {p.pid}")
    return p


def monitor_enrichment(proc, deadline, hang_timeout):
    """Monitor an enrichment subprocess. Returns exit code or -1 on hang."""
    last_progress = datetime.now()
    last_enrichment_pct = -1

    while True:
        time.sleep(15)
        if datetime.now() >= deadline:
            logger.warning("Enrichment deadline reached — killing.")
            kill_child(proc)
            return -3

        exit_code = proc.poll()
        if exit_code is not None:
            if exit_code == 0:
                logger.info("Enrichment completed successfully.")
            else:
                logger.error(f"Enrichment crashed (exit {exit_code})")
            return exit_code

        hb = read_heartbeat()
        if hb:
            status = hb.get("status", "")
            if status == "enriching":
                pct = hb.get("enrichment_progress", 0)
                if pct > last_enrichment_pct:
                    last_enrichment_pct = pct
                    last_progress = datetime.now()
                elif (datetime.now() - last_progress).total_seconds() > hang_timeout:
                    logger.error(f"Enrichment HANG at {pct}%")
                    kill_child(proc)
                    return -1
                if int(time.time()) % 120 < 15:
                    detail = hb.get("enrichment_detail", {})
                    processed = detail.get("processed", "?")
                    total = detail.get("total", "?")
                    fields = detail.get("fields_updated_total", "?")
                    logger.info(
                        f"Enrichment OK | {processed}/{total} records "
                        f"({pct}%) | {fields} fields updated"
                    )
            elif status in ("enrichment_complete", "completed"):
                pass
            else:
                cur = hb.get("total_records", -1)
                if cur > last_enrichment_pct:
                    last_enrichment_pct = cur
                    last_progress = datetime.now()
                elif (datetime.now() - last_progress).total_seconds() > hang_timeout:
                    logger.error("Enrichment HANG (no progress)")
                    kill_child(proc)
                    return -1
        elif (datetime.now() - last_progress).total_seconds() > hang_timeout:
            logger.error("No heartbeat during enrichment")
            kill_child(proc)
            return -2


def kill_child(p):
    if p.poll() is not None:
        return
    p.terminate()
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait(10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-base-url", default="http://localhost:11434/v1")
    ap.add_argument("--max-restarts", type=int, default=DEFAULT_MAX_RESTARTS)
    ap.add_argument("--hang-timeout", type=int, default=DEFAULT_HANG_TIMEOUT)
    args = ap.parse_args()

    t0 = datetime.now()
    deadline = t0 + timedelta(hours=12)
    restarts = 0

    logger.info("=" * 70)
    logger.info(
        f"SLEEPER ACTIVE | deadline {deadline.isoformat()} | max restarts {args.max_restarts}"
    )
    logger.info("=" * 70)

    while restarts <= args.max_restarts:
        if datetime.now() >= deadline:
            break

        proc = spawn(args.llm_base_url)
        last_progress = datetime.now()
        last_count = -1
        exit_code = None

        while True:
            time.sleep(15)
            if datetime.now() >= deadline:
                kill_child(proc)
                break

            exit_code = proc.poll()
            if exit_code is not None:
                if exit_code == 0:
                    db = read_db_state()
                    if db["total"] >= TOTAL_TARGET:
                        logger.info(f"TARGET MET ({db['total']})")
                        return
                    logger.warning(f"Clean exit, {db['total']}/{TOTAL_TARGET}. Restarting.")
                else:
                    logger.error(f"Crash (exit {exit_code})")
                break

            hb = read_heartbeat()
            if hb:
                cur = hb.get("total_records", -1)
                if cur > last_count:
                    last_count = cur
                    last_progress = datetime.now()
                elif (datetime.now() - last_progress).total_seconds() > args.hang_timeout:
                    logger.error(f"HANG at {cur} records")
                    kill_child(proc)
                    exit_code = -1
                    break
                if int(time.time()) % 120 < 15:
                    logger.info(f"OK | {cur}/{TOTAL_TARGET}")
            elif (datetime.now() - last_progress).total_seconds() > args.hang_timeout:
                logger.error("No heartbeat")
                kill_child(proc)
                exit_code = -2
                break

        if datetime.now() >= deadline:
            break

        restarts += 1
        logger.info(f"\n  RESTART {restarts}/{args.max_restarts}")
        hb = read_heartbeat()
        log = read_log_tail()
        db = read_db_state()
        ollama_ok = check_ollama_health()
        logger.info(f"Ollama={ollama_ok} DB={db.get('total', '?')}")
        dx = diagnose(
            exit_code, hb, log, db, ollama_ok, restarts, args.llm_base_url
        )
        logger.info(f"Dx: {dx.get('diagnosis')} -> {dx.get('action')}")
        if not execute_action(
            dx.get("action", "restart_production"), args.llm_base_url
        ):
            if dx.get("action") == "abort":
                break
        time.sleep(30)

    # ── Phase 2: Enrichment Pass ──────────────────────────────────────────
    db_state = read_db_state()
    time_left = deadline - datetime.now()

    if db_state.get("total", 0) >= TOTAL_TARGET and time_left > timedelta(minutes=30):
        remaining_hours = time_left.total_seconds() / 3600
        logger.info("=" * 70)
        logger.info(
            f"PHASE 2: ENRICHMENT | {remaining_hours:.1f}h remaining"
        )
        logger.info("=" * 70)

        enrich_proc = spawn_enrichment(
            args.llm_base_url, f"{remaining_hours:.1f}h"
        )
        enrich_exit = monitor_enrichment(enrich_proc, deadline, args.hang_timeout)

        if enrich_exit != 0:
            logger.warning(
                f"Enrichment exited with code {enrich_exit}. "
                f"Re-run manually: uv run game-predictor-run --enrich-only"
            )
    elif db_state.get("total", 0) < TOTAL_TARGET:
        logger.warning(
            f"Production target not met ({db_state.get('total', 0)}/{TOTAL_TARGET}) "
            f"— skipping enrichment phase."
        )
    else:
        logger.info("Insufficient time remaining for enrichment phase.")

    db_final = read_db_state()
    logger.info("=" * 70)
    logger.info(
        f"SESSION DONE | restarts={restarts} | records={db_final.get('total', 0)}/{TOTAL_TARGET} | "
        f"elapsed={datetime.now() - t0}"
    )
    status = "MET" if db_final.get("total", 0) >= TOTAL_TARGET else "NOT MET"
    logger.info(status)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
