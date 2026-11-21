"""Preprocessing pipeline for NBA Clutch Shot Prediction.

Reconstructs pre-shot scoreboard state, computes chronological Bayesian-smoothed
player priors anchored to the 2024 training season, and exports 5 CSV in outputs:
  1. player_profiles_q1_q3.csv (for K-Means clustering)
  2. q4_shots_train_2024.csv (Q4 + OT training set)
  3. q4_shots_val_2025.csv (Q4 + OT validation set)
  4. q4_shots_test_2026.csv (Q4 + OT held-out test set)
  5. q4_shots_live_2027.csv (Q4 + OT early-season 2026-27 case study)
"""

from pathlib import Path
import numpy as np
import pandas as pd

# set smoothing weights 

PRIOR_WEIGHT_OVERALL = 50.0
PRIOR_WEIGHT_ZONE = 25.0
PRIOR_WEIGHT_INGAME = 5.0

# finding files
def locate_shot_files() -> tuple[list[Path], Path]:
    """Find shots_*.csv in court_zones.py."""
    repo_root = Path(__file__).resolve().parents[1]
    processed_dir = repo_root / "data" / "processed"
    local_dir = Path.cwd()

    for candidate in [processed_dir, local_dir]:
        files = sorted(candidate.glob("shots_*.csv"))
        if files:
            out_dir = repo_root / "data" / "modeling" if candidate == processed_dir else local_dir / "modeling_data"
            return files, out_dir

    raise FileNotFoundError("Could not find shots_*.csv in data/processed/ or current directory.")

# Cleaning text leakage and parsing clocks
def load_and_clean_raw_files(files: list[Path]) -> pd.DataFrame:
    """Load annotated shot CSVs, extract binary target/assist flags, and drop text leakage."""
    chunks = []
    for path in files:
        df_season = pd.read_csv(path, low_memory=False)

        # If raw files were loaded before court_zones.py ran, annotate on the fly
        if "basic_zone" not in df_season.columns:
            from court_zones import court_zones
            zones = [
                court_zones(x, y, v)
                for x, y, v in zip(df_season["x_legacy"], df_season["y_legacy"], df_season["shot_value"])
            ]
            df_season[["basic_zone", "zone_name", "zone_range"]] = zones

        df_season["sub_type"] = df_season["sub_type"].fillna("Unknown")
        df_season["shot_made"] = (df_season["shot_result"] == "Made").astype(np.int8)

        # Extract historical assist signal BEFORE dropping leaked description column
        df_season["_was_assisted"] = (
            df_season["description"].str.contains("AST)", regex=False, na=False).astype(np.int8)
        )

        # Immediately drop direct target-leakage text columns
        df_season = df_season.drop(columns=["action_type", "description", "shot_result"])
        chunks.append(df_season)

    df = pd.concat(chunks, ignore_index=True)

    # Parse ISO-style clock 'PT11M41.00S' -> seconds remaining in period
    clock_parts = df["clock"].str.slice(2, -1).str.split("M", expand=True)
    df["sec_remaining"] = (
        clock_parts[0].astype(np.float32) * 60.0 + clock_parts[1].astype(np.float32)
    ).round(2)

    # Guarantee strict chronological order within each game
    df["_row_order"] = np.arange(len(df), dtype=np.int32)
    df = df.sort_values(
        ["season", "game_id", "period", "sec_remaining", "_row_order"],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)

    return df.drop(columns=["clock", "_row_order"])

# Game State and Court Geometry 
def add_scoreboard_and_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct pre-shot score differential, home/away status, and spatial features."""
    same_game = df["game_id"] == df["game_id"].shift(1)

    # Forward-fill post-shot scores and shift by 1 shot so current outcome is never leaked
    pre_home = df.groupby("game_id")["score_home"].ffill().shift(1).where(same_game, 0.0).fillna(0.0)
    pre_away = df.groupby("game_id")["score_away"].ffill().shift(1).where(same_game, 0.0).fillna(0.0)

    # Identify home team_id per game 
    made_home = df[
        (df["shot_made"] == 1)
        & ((df["score_home"] - pre_home) >= df["shot_value"])
        & ((df["score_away"] - pre_away) == 0)
    ]
    home_team_map = made_home.drop_duplicates("game_id").set_index("game_id")["team_id"]

    # Drop post-shot score columns to eliminate NaN-pattern target leakage
    df = df.drop(columns=["score_home", "score_away"]) # so KNN and Keras can't cheat NaN values

    df["is_home"] = (df["team_id"] == df["game_id"].map(home_team_map)).astype(np.int8) # mark 1 if the shooter plays for the home team, 0 if away
    df["pre_shot_score_diff"] = np.where(
        df["is_home"] == 1, pre_home - pre_away, pre_away - pre_home
    ).astype(np.float32) # computes score margin from the shooter's perspective before the shot (+4 means the shooter's team leads by 4, -3 trailing by 3)

    # Period & official NBA clutch flags (Q4/OT, <= 5:00 left, score within 5 points)
    df["is_q4"] = (df["period"] == 4).astype(np.int8)
    df["is_ot"] = (df["period"] > 4).astype(np.int8)  
    df["is_clutch"] = (
        (df["period"] >= 4)
        & (df["sec_remaining"] <= 300.0)
        & (df["pre_shot_score_diff"].abs() <= 5.0)
    ).astype(np.int8)  # flags 4th quarter shots, overtime shots and official NBA clutch shots (4th quarter or OT, <= 5 minutes / 300 seconds left, score difference within 5 points)

    # Continuous geometry + non-destructive 16x19ft paint rectangle indicator
    x_ft = df["x_legacy"].astype(np.float32) / 10.0
    y_ft = df["y_legacy"].astype(np.float32) / 10.0 #converts tenths-of-foot coordinates into feet
    df["coord_radius_ft"] = np.hypot(x_ft, y_ft).round(2) #calculate exact Euclidean distance to the hoop in feet
    df["shot_angle_deg"] = np.degrees(np.arctan2(y_ft, x_ft)).round(2) #calculate continuous shooting angle in degrees (0 degree=right corner/sideline, 90=straight-on-top of the key, 180=left corner/sideline)
    df["in_paint_rect"] = (
        (df["x_legacy"].abs() <= 80) & (df["y_legacy"] >= -47.5) & (df["y_legacy"] <= 137.5)
    ).astype(np.int8) #Adds a 1/0 flag for whether the shot was inside the 16 ft x 19 ft NBA lane

    # Check keyword conditions, any jump shot that doesn not match the specialized moves default to Catch_And_Shoot_Jumpter
    sub_lower = df["sub_type"].str.lower()
    df["shot_mechanic"] = np.select(
        [
            sub_lower.str.contains("dunk", regex=False),
            sub_lower.str.contains("alley oop|tip|putback"),
            sub_lower.str.contains("layup|finger roll"),
            sub_lower.str.contains("floating", regex=False),
            sub_lower.str.contains("hook", regex=False),
            sub_lower.str.contains("step back", regex=False),
            sub_lower.str.contains("pullup|pull-up"),
            sub_lower.str.contains("fadeaway|turnaround"),
        ],
        [
            "Dunk",
            "Tip_Putback_Lob",
            "Layup",
            "Floater",
            "Hook",
            "Step_Back_Jumper",
            "Pullup_Jumper",
            "Turnaround_Fadeaway",
        ],
        default="Catch_And_Shoot_Jumper",
    )
    return df

# Historical Shooting Skill: Calculates what we knew about a shooter's skill before they released the current shot: their career/multi-season FG%, their FG% from that specific basic_zone, and their FG% earlier in that same game

def add_chronological_player_priors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute strictly pre-shot expanding FG% priors smoothed toward 2024 training averages."""

  #Calculates baseline league FG% (47.35%) and zone FG% averages using only the 2024 training season
    train_mask = df["season"] == 2024 
    league_fg_prior = float(df.loc[train_mask, "shot_made"].mean())
    zone_priors = df.loc[train_mask].groupby("basic_zone")["shot_made"].mean().to_dict()

    # 1. Career/multi-season expanding FG% prior to current shot
    p_cum_fga = df.groupby("person_id").cumcount().astype(np.float32) #Counts how many shots this person_id took strictly before the current row
    p_cum_fgm = (df.groupby("person_id")["shot_made"].cumsum() - df["shot_made"]).astype(np.float32) # adds up all makes including current shot, subtract remove current shot's outcome
    df["player_prior_fga"] = p_cum_fga.astype(np.int32) # Computes Bayesian-smoothed historical FG%:
    df["player_prior_fg_pct"] = (
        (p_cum_fgm + PRIOR_WEIGHT_OVERALL * league_fg_prior) / (p_cum_fga + PRIOR_WEIGHT_OVERALL)
    ).round(4)

    # 2. Player-by-zone expanding FG% prior to current shot, capturing whether this specific player is strong from that specific court zone
    pz_cum_fga = df.groupby(["person_id", "basic_zone"]).cumcount().astype(np.float32)
    pz_cum_fgm = (
        df.groupby(["person_id", "basic_zone"])["shot_made"].cumsum() - df["shot_made"]
    ).astype(np.float32)
    z_prior = df["basic_zone"].map(zone_priors).fillna(league_fg_prior).astype(np.float32)
    df["player_zone_prior_fg_pct"] = (
        (pz_cum_fgm + PRIOR_WEIGHT_ZONE * z_prior) / (pz_cum_fga + PRIOR_WEIGHT_ZONE)
    ).round(4)

    # 3. In-game 'hot hand' shooting prior to current shot
    ig_fga = df.groupby(["game_id", "person_id"]).cumcount().astype(np.float32)
    ig_fgm = (
        df.groupby(["game_id", "person_id"])["shot_made"].cumsum() - df["shot_made"]
    ).astype(np.float32)
    df["in_game_prior_fga"] = ig_fga.astype(np.int16)
    df["in_game_prior_fg_pct"] = (
        (ig_fgm + PRIOR_WEIGHT_INGAME * league_fg_prior) / (ig_fga + PRIOR_WEIGHT_INGAME)
    ).round(4) # Resets every game (groupby(["game_id", "person_id"])) to track how many shots the player has already taken tonight and their smoothed FG% so far in that game

    return df

# Unsupervised K-means Table: 
# takes only Quarters 1–3 and builds one summary row per (season, person_id) containing shot-zone percentages, shot-mechanic percentages, average shot distance, 3-point rate, Q1–Q3 FG%, and assisted make rate for K-Means.
def build_q1_q3_player_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Q1-Q3 player shooting profiles per season for K-Means."""
    q13 = df[df["period"].isin([1, 2, 3])].copy() #Filters strictly to Quarters 1, 2, and 3 so the player profiles used for K-Means clustering have zero overlap
    q13["is_three"] = (q13["shot_value"] == 3).astype(np.float32) #Creates a 1.0/0.0 indicator for 3-point attempts so its average equals the player's 3-point attempt rate.

    zone_dummies = pd.get_dummies(q13["basic_zone"], prefix="zone_share", dtype=np.float32) 
    mech_dummies = pd.get_dummies(q13["shot_mechanic"], prefix="mech_share", dtype=np.float32)

  #Joins the continuous columns and one-hot dummy columns horizontally (axis=1) and computes the mean of every column per (season, person_id).
    combined = pd.concat(
        [
            q13[["season", "person_id", "shot_distance", "in_paint_rect", "is_three", "shot_made"]],
            zone_dummies,
            mech_dummies,
        ],
        axis=1,
    )
    rate_cols = ["shot_distance", "in_paint_rect", "is_three", "shot_made"] + list(zone_dummies.columns) + list(mech_dummies.columns)
    means = combined.groupby(["season", "person_id"], as_index=False)[rate_cols].mean() 
  
  #Collects metadata and totals per player-season
    meta = q13.groupby(["season", "person_id"], as_index=False).agg(
        player_name=("player_name", "last"), #player name
        team_tricode=("team_tricode", "last"), #most recent team_tricode
        q13_fga=("shot_made", "count"), #Total Q1-Q3 attempts
        q13_fgm=("shot_made", "sum"), # total makes 
        q13_assisted_fgm=("_was_assisted", "sum"), #total assisted makes
    )
    profiles = meta.merge(means, on=["season", "person_id"])
    profiles["q13_assisted_fgm_rate"] = np.where(
        profiles["q13_fgm"] > 0, profiles["q13_assisted_fgm"] / profiles["q13_fgm"], 0.0
    ).round(4) # merges the totals with the averages and divides assisted makes by total makes, distinguishing sefl-creators from catch-and-finish/catch-and-shoot players

    return profiles.drop(columns=["q13_assisted_fgm"]).rename(
        columns={
            "shot_distance": "mean_shot_distance",
            "in_paint_rect": "paint_rect_share",
            "is_three": "three_pt_rate",
            "shot_made": "q13_fg_pct",
        }
    )

# script entry point 
def main():
    files, out_dir = locate_shot_files()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(files)} shot files. Running leakage-free preprocessing...")

    df = load_and_clean_raw_files(files)
    df = add_scoreboard_and_spatial_features(df)
    df = add_chronological_player_priors(df)

    # 1. Export Q1-Q3 K-Means player profiles
    profiles = build_q1_q3_player_profiles(df)
    profile_path = out_dir / "player_profiles_q1_q3.csv"
    profiles.to_csv(profile_path, index=False)
    print(f"Saved K-Means profiles: {profile_path} ({len(profiles):,} player-seasons)")

    # Drop internal helper column before saving shot-level modeling splits
    df = df.drop(columns=["_was_assisted"])

    # 2. Export Q4 + OT shot-level splits by season
    split_names = {
        2024: "q4_shots_train_2024.csv",
        2025: "q4_shots_val_2025.csv",
        2026: "q4_shots_test_2026.csv",
        2027: "q4_shots_live_2027.csv",
    }
    q4_ot = df[df["period"] >= 4]
    for season, filename in split_names.items():
        subset = q4_ot[q4_ot["season"] == season]
        if not subset.empty:
            dest = out_dir / filename
            subset.to_csv(dest, index=False)
            clutch_n = int(subset["is_clutch"].sum())
            print(f"Saved {filename}: {len(subset):,} Q4/OT shots ({clutch_n:,} strict clutch)")


if __name__ == "__main__":
    main()
