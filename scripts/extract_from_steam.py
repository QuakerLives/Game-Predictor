# Steam API → DuckDB. Usage: python extract_from_steam.py --key YOUR_KEY [--db steam_data.duckdb]

import argparse, asyncio, aiohttp, duckdb
from datetime import datetime, timezone

# Steam App IDs mapped to game names.
GAMES = {
    281990: "Stellaris",
    275850: "No Man's Sky",
    1172470: "Apex Legends",
    413150: "Stardew Valley",
    489830: "Skyrim Special Edition"
}

BASE = "https://api.steampowered.com"

# Each endpoint has a path and a lambda that builds its query params from (appid, key).
ENDPOINTS = {
    "news": (
        "/ISteamNews/GetNewsForApp/v2/",
        lambda a, k: {"appid": a, "count": 100, "maxlength": 0}
    ),
    "players": (
        "/ISteamUserStats/GetNumberOfCurrentPlayers/v1/",
        lambda a, k: {"appid": a}
    ),
    "achs": (
        "/ISteamUserStats/GetGlobalAchievementPercentagesForApp/v2/",
        lambda a, k: {"gameid": a, "key": k}
    ),
    "schema": (
        "/ISteamUserStats/GetSchemaForGame/v2/",
        lambda a, k: {"appid": a, "key": k}
    ),
}


# Fetches JSON from a single Steam API endpoint, returns {} on failure.
async def fetch(session, path, params):
    try:
        async with session.get(BASE + path, params=params,
                               timeout=aiohttp.ClientTimeout(total=15)) as r:
            # Return parsed JSON on success, log and return empty dict otherwise.
            return (await r.json()) if r.status == 200 else print(f"  WARN {r.status}: {path}") or {}
    except Exception as e:
        print(f"  FAIL {path}: {e}")
        return {}


# Transforms raw API JSON into a flat list of row dicts based on the endpoint kind.
def extract(kind, appid, raw):
    now = datetime.now(timezone.utc).isoformat()

    # News articles from the app's news feed.
    if kind == "news":
        return [{"appid": appid, "gid": str(i["gid"]), "title": i.get("title", ""),
                 "url": i.get("url", ""), "author": i.get("author", ""),
                 "contents": i.get("contents", ""), "feedlabel": i.get("feedlabel", ""),
                 "date_unix": i.get("date", 0)}
                for i in raw.get("appnews", {}).get("newsitems", [])]

    # Current player count snapshot with timestamp.
    if kind == "players":
        c = raw.get("response", {}).get("player_count")
        return [{"appid": appid, "player_count": c, "fetched_at": now}] if c is not None else []

    # Global achievement unlock percentages.
    if kind == "achs":
        return [{"appid": appid, "name": a["name"], "percent": round(float(a["percent"]), 2)}
                for a in raw.get("achievementpercentages", {}).get("achievements", [])]

    # Achievement metadata: display name, description, hidden flag, icon URL.
    if kind == "schema":
        return [{"appid": appid, "name": a["name"], "displayName": a.get("displayName", ""),
                 "description": a.get("description", ""), "hidden": a.get("hidden", 0),
                 "icon": a.get("icon", "")}
                for a in raw.get("game", {}).get("availableGameStats", {}).get("achievements", [])]


# Table definitions for the DuckDB schema.
DDL = """
CREATE TABLE IF NOT EXISTS games (appid INTEGER PRIMARY KEY, name VARCHAR);
CREATE TABLE IF NOT EXISTS news (appid INTEGER, gid VARCHAR, title VARCHAR, url VARCHAR, author VARCHAR, contents VARCHAR, feedlabel VARCHAR, date_unix BIGINT, PRIMARY KEY (appid, gid));
CREATE TABLE IF NOT EXISTS player_counts (appid INTEGER, player_count INTEGER, fetched_at VARCHAR);
CREATE TABLE IF NOT EXISTS achievements (appid INTEGER, name VARCHAR, percent DOUBLE, PRIMARY KEY (appid, name));
CREATE TABLE IF NOT EXISTS achievement_schema (appid INTEGER, name VARCHAR, displayName VARCHAR, description VARCHAR, hidden INTEGER, icon VARCHAR, PRIMARY KEY (appid, name));
"""

# Maps endpoint kind to its destination table.
TABLE_MAP = {"news": "news", "players": "player_counts", "achs": "achievements", "schema": "achievement_schema"}

# Tables with primary keys get INSERT OR REPLACE; others get plain INSERT.
PK_TABLES = {"games", "news", "achievements", "achievement_schema"}


# Inserts rows into the given table, using INSERT OR REPLACE for PK tables.
def upsert(con, table, rows):
    if not rows:
        return 0

    cols = list(rows[0].keys())
    ph = ",".join(["?"] * len(cols))  # placeholder string like "?,?,?"
    verb = "INSERT OR REPLACE" if table in PK_TABLES else "INSERT"
    con.executemany(
        f"{verb} INTO {table} ({','.join(cols)}) VALUES ({ph})",
        [tuple(r[c] for c in cols) for r in rows]
    )
    return len(rows)


# Main loop: create tables, then fetch and store data for every game × endpoint.
async def main(key, db_path):
    con = duckdb.connect(db_path)
    con.execute(DDL)  # Ensure all tables exist.
    upsert(con, "games", [{"appid": k, "name": v} for k, v in GAMES.items()])

    async with aiohttp.ClientSession() as s:
        for appid, name in GAMES.items():
            print(f"\nGAME {name} ({appid})")
            for kind, (path, params_fn) in ENDPOINTS.items():
                raw = await fetch(s, path, params_fn(appid, key))
                n = upsert(con, TABLE_MAP[kind], extract(kind, appid, raw))
                print(f"  {kind}: {n} rows")
            await asyncio.sleep(0.5)  # Rate-limit between games.

    # Print final row counts per table.
    print(f"\nSUCCESS {db_path}")
    for t in TABLE_MAP.values():
        print(f"  {t}: {con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]}")
    con.close()


# Entry point: parse CLI args and run.
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--key", required=True)  # Steam API key.
    p.add_argument("--db", default="steam_data.duckdb")  # Output database path.
    a = p.parse_args()
    asyncio.run(main(a.key, a.db))
