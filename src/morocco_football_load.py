"""
Dataset loader & initial explorer for all 4 Kaggle datasets.

Datasets needed (download from Kaggle and place in ./data/):
  1. results.csv          — International Football Results 1872–2026
       kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
  2. worldcup.csv         — FIFA World Cup 1930–2022
       kaggle.com/datasets/piterfm/fifa-football-world-cup
  3. wc2022_players.csv   — FIFA World Cup 2022 Complete Dataset
       kaggle.com/datasets/die9origephit/fifa-world-cup-2022-complete-dataset
  4. database.sqlite      — European Soccer Database (players + clubs)
       kaggle.com/datasets/hugomathien/soccer

Usage:
  pip install pandas numpy matplotlib seaborn kaggle
  python morocco_football_load.py
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = "./data"          # folder where you put the downloaded CSV/sqlite files
TEAM     = "Morocco"

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#f9f9f9",
                     "axes.grid": True, "grid.color": "#e0e0e0",
                     "grid.linestyle": "--", "grid.linewidth": 0.6})

print("=" * 65)
print("  MOROCCO FOOTBALL — Dataset Loader & Explorer")
print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# 1. INTERNATIONAL RESULTS 1872–2026   (results.csv)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/4] International Football Results (1872–2026)")
print("-" * 50)

results_path = os.path.join(DATA_DIR, "results.csv")
results = pd.read_csv(results_path, parse_dates=["date"])

print(f"  Total rows       : {len(results):,}")
print(f"  Date range       : {results['date'].min().date()} → {results['date'].max().date()}")
print(f"  Columns          : {list(results.columns)}")

# --- Filter Morocco matches ---
morocco = results[
    (results["home_team"] == TEAM) | (results["away_team"] == TEAM)
].copy()

# Normalise perspective to Morocco's side
morocco["mar_score"]  = np.where(morocco["home_team"] == TEAM,
                                  morocco["home_score"], morocco["away_score"])
morocco["opp_score"]  = np.where(morocco["home_team"] == TEAM,
                                  morocco["away_score"], morocco["home_score"])
morocco["opponent"]   = np.where(morocco["home_team"] == TEAM,
                                  morocco["away_team"], morocco["home_team"])
morocco["is_home"]    = morocco["home_team"] == TEAM
morocco["result"]     = np.select(
    [morocco["mar_score"] > morocco["opp_score"],
     morocco["mar_score"] == morocco["opp_score"]],
    ["W", "D"], default="L"
)
morocco["goal_diff"]  = morocco["mar_score"] - morocco["opp_score"]
morocco["decade"]     = (morocco["date"].dt.year // 10) * 10
morocco["year"]       = morocco["date"].dt.year

print(f"\n  Morocco matches  : {len(morocco):,}")
print(f"  Record (W-D-L)   : "
      f"{(morocco['result']=='W').sum()}-"
      f"{(morocco['result']=='D').sum()}-"
      f"{(morocco['result']=='L').sum()}")

# --- Preview ---
print("\n  First 3 Morocco rows:")
print(morocco[["date","opponent","mar_score","opp_score","result","tournament","is_home"]].head(3).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# 2. FIFA WORLD CUP 1930–2022   (worldcup.csv  or  matches.csv)
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n[2/4] FIFA World Cup 1930–2022")
print("-" * 50)

# The Kaggle dataset by piterfm has multiple CSVs; try common filenames
wc_candidates = ["Matches.csv", "matches.csv", "WorldCupMatches.csv", "worldcup_matches.csv"]
wc = None
for fname in wc_candidates:
    fpath = os.path.join(DATA_DIR, fname)
    if os.path.exists(fpath):
        wc = pd.read_csv(fpath)
        print(f"  Loaded           : {fname}")
        break

if wc is None:
    print("  ⚠  File not found. Place one of these in ./data/:")
    for f in wc_candidates:
        print(f"       {f}")
else:
    print(f"  Total rows       : {len(wc):,}")
    print(f"  Columns          : {list(wc.columns)}")

    # Normalise column names (dataset uses various casing)
    wc.columns = wc.columns.str.strip().str.lower().str.replace(" ", "_")
    home_col = [c for c in wc.columns if "home" in c and "team" in c][0]
    away_col = [c for c in wc.columns if "away" in c and "team" in c][0]

    wc_morocco = wc[
        (wc[home_col].str.contains(TEAM, na=False)) |
        (wc[away_col].str.contains(TEAM, na=False))
    ]
    print(f"\n  Morocco WC matches: {len(wc_morocco):,}")
    print(wc_morocco[[home_col, away_col]].head(8).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# 3. FIFA WORLD CUP 2022 — PLAYER STATS   (wc2022_players.csv)
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n[3/4] FIFA World Cup 2022 — Player & Match Stats")
print("-" * 50)

# This dataset has several files; try the player stats file first
wc22_candidates = [
    "player_stats.csv", "players_stats.csv", "FIFA World Cup 2022 Player Stats.csv",
    "players.csv", "wc2022_players.csv"
]
wc22 = None
for fname in wc22_candidates:
    fpath = os.path.join(DATA_DIR, fname)
    if os.path.exists(fpath):
        wc22 = pd.read_csv(fpath)
        print(f"  Loaded           : {fname}")
        break

if wc22 is None:
    # Try any CSV in data dir that might be WC2022 related
    all_csvs = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")] if os.path.isdir(DATA_DIR) else []
    print(f"  ⚠  Player stats CSV not found. CSVs in ./data/: {all_csvs}")
else:
    wc22.columns = wc22.columns.str.strip().str.lower().str.replace(" ", "_")
    print(f"  Total rows       : {len(wc22):,}")
    print(f"  Columns          : {list(wc22.columns)}")

    # Find team column
    team_col = next((c for c in wc22.columns if "team" in c or "country" in c or "nation" in c), None)
    if team_col:
        wc22_mar = wc22[wc22[team_col].str.contains(TEAM, na=False)]
        print(f"\n  Morocco players   : {len(wc22_mar):,} rows")
        print(wc22_mar.head(5).to_string(index=False))
    else:
        print("  Could not auto-detect team column — inspect wc22.columns")


# ══════════════════════════════════════════════════════════════════════════════
# 4. EUROPEAN SOCCER DATABASE   (database.sqlite)
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n[4/4] European Soccer Database (SQLite)")
print("-" * 50)

sqlite_path = os.path.join(DATA_DIR, "database.sqlite")
if not os.path.exists(sqlite_path):
    print(f"  ⚠  database.sqlite not found in {DATA_DIR}/")
else:
    conn = sqlite3.connect(sqlite_path)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    print(f"  Tables           : {tables['name'].tolist()}")

    # Load player attributes table
    players_df = pd.read_sql("SELECT * FROM Player_Attributes LIMIT 5", conn)
    print(f"\n  Player_Attributes columns:")
    print(f"  {list(players_df.columns)}")

    # Moroccan players in the squad (cross-reference by name)
    moroccan_players = [
        "Yassine Bounou", "Achraf Hakimi", "Hakim Ziyech",
        "Youssef En-Nesyri", "Sofiane Boufal", "Noussair Mazraoui",
        "Romain Saiss", "Azzedine Ounahi"
    ]
    # Search by last name for broader matching
    last_names = [name.split()[-1] for name in moroccan_players]
    like_clauses = " OR ".join([f"player_name LIKE '%{n}%'" for n in last_names])

    try:
        mar_players = pd.read_sql(
            f"SELECT DISTINCT player_name, birthday, height, weight FROM Player "
            f"WHERE {like_clauses}", conn
        )
        print(f"\n  Moroccan players found in DB ({len(mar_players)}):")
        print(mar_players.to_string(index=False))

        if len(mar_players) > 0:
            # Get latest FIFA attributes for found players
            pids = tuple(
                pd.read_sql(
                    f"SELECT player_api_id FROM Player WHERE {like_clauses}", conn
                )["player_api_id"].tolist()
            )
            if pids:
                attrs = pd.read_sql(f"""
                    SELECT p.player_name, pa.overall_rating, pa.potential,
                           pa.pace, pa.shooting, pa.passing, pa.dribbling,
                           pa.defending, pa.physic, pa.date
                    FROM Player p
                    JOIN Player_Attributes pa ON p.player_api_id = pa.player_api_id
                    WHERE p.player_api_id IN {pids if len(pids)>1 else f'({pids[0]})'}
                    ORDER BY pa.date DESC
                """, conn)
                # Keep latest entry per player
                attrs = attrs.drop_duplicates(subset="player_name", keep="first")
                print(f"\n  Latest FIFA attributes for Moroccan players:")
                print(attrs.to_string(index=False))
    except Exception as e:
        print(f"  DB query error: {e}")

    conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# QUICK VISUALISATION — Morocco win rate by decade
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n[Bonus] Generating quick EDA chart: Win rate by decade...")

if len(morocco) > 0:
    decade_stats = morocco.groupby("decade").apply(
        lambda g: pd.Series({
            "matches": len(g),
            "wins":    (g["result"] == "W").sum(),
            "draws":   (g["result"] == "D").sum(),
            "losses":  (g["result"] == "L").sum(),
            "win_rate": round((g["result"] == "W").mean() * 100, 1),
            "goals_per_match": round(g["mar_score"].mean(), 2),
        })
    ).reset_index()

    print("\n  Win rate by decade:")
    print(decade_stats[["decade","matches","wins","draws","losses","win_rate"]].to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Morocco National Team — Historical Overview", fontsize=14, fontweight="bold", y=1.01)

    # Chart 1: Win rate by decade
    ax = axes[0]
    bars = ax.bar(decade_stats["decade"].astype(str), decade_stats["win_rate"],
                  color="#C1272D", edgecolor="white", linewidth=0.8, width=0.6)
    ax.set_title("Win rate by decade (%)", fontsize=12)
    ax.set_xlabel("Decade")
    ax.set_ylabel("Win rate (%)")
    ax.set_ylim(0, 80)
    for bar, val in zip(bars, decade_stats["win_rate"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{val}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Chart 2: W/D/L stacked bars by decade
    ax2 = axes[1]
    w = decade_stats["wins"]
    d = decade_stats["draws"]
    l = decade_stats["losses"]
    x = range(len(decade_stats))
    ax2.bar(x, w, label="Win",  color="#2d6a4f", edgecolor="white")
    ax2.bar(x, d, bottom=w, label="Draw", color="#f4a261", edgecolor="white")
    ax2.bar(x, l, bottom=w+d, label="Loss", color="#C1272D", edgecolor="white")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(decade_stats["decade"].astype(str))
    ax2.set_title("Match results by decade", fontsize=12)
    ax2.set_xlabel("Decade")
    ax2.set_ylabel("Number of matches")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(DATA_DIR, "morocco_decade_overview.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved → {chart_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY EXPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 65)
print("  DATASET SUMMARY")
print("=" * 65)
print(f"  ✓ Dataset 1 (Results 1872–2026) : {len(morocco):,} Morocco matches")

if wc is not None:
    print(f"  ✓ Dataset 2 (World Cup history)  : {len(wc_morocco):,} Morocco WC matches")
else:
    print("  ✗ Dataset 2 (World Cup history)  : not loaded — add CSV to ./data/")

if wc22 is not None and team_col:
    print(f"  ✓ Dataset 3 (WC 2022 players)    : {len(wc22_mar):,} Morocco player rows")
else:
    print("  ✗ Dataset 3 (WC 2022 players)    : not loaded — add CSV to ./data/")

if os.path.exists(sqlite_path):
    print(f"  ✓ Dataset 4 (Euro Soccer DB)     : SQLite loaded successfully")
else:
    print("  ✗ Dataset 4 (Euro Soccer DB)     : not loaded — add database.sqlite to ./data/")

print("\n  morocco DataFrame saved as `morocco` — ready for Phase 2 EDA.")
print("  Next step: run the EDA notebook for deeper analysis.\n")

# Export Morocco filtered dataset for reuse
morocco.to_csv(os.path.join(DATA_DIR, "morocco_matches.csv"), index=False)
print(f"  Exported: ./data/morocco_matches.csv  ({len(morocco):,} rows)")