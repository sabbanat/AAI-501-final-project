"""Unsupervised K-Means Player Profiling for NBA Clutch Shot Prediction.

Fits StandardScaler + KMeans strictly on 2024 Q1-Q3 qualified player profiles
(>= 100 FGA), compares K = 2..8 via Inertia
and Silhouette score, generates PCA + Elbow diagnostic plots, and merges
player_cluster onto all Q4 shot splits for supervised classification.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


MIN_TRAIN_Q13_FGA = 100 #Only players with at least 100 Q1–Q3 shots in 2024 are used to train K-Means 
MIN_CURRENT_SEASON_FGA = 25 #Minimum Q1–Q3 shots required in a season before trusting a player's current-season cluster over their previous season's cluster.
SELECTED_K = 4

#shotting habit columns fed into StandardScaler and KMeans
CLUSTER_FEATURES = [
    "zone_share_Restricted Area",
    "zone_share_In the Paint (non-RA)",
    "zone_share_Midrange",
    "corner_3_share",
    "zone_share_Above the Break",
    "rim_finish_share",
    "self_created_jumper_share",
    "mech_share_Catch_And_Shoot_Jumper",
    "q13_assisted_fgm_rate",
    "q13_fg_pct",
]


def locate_modeling_dir() -> Path:
    """Locate data/modeling/ created by preprocess_data.py."""
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "data" / "modeling",
        Path.cwd() / "modeling_data",
        Path.cwd(),
    ]
    for folder in candidates:
        if (folder / "player_profiles_q1_q3.csv").exists():
            return folder
    raise FileNotFoundError(
        "Could not find player_profiles_q1_q3.csv. Run preprocess_data.py first."
    )

# combines related court zones and shot moves into 3 cleaner basketball features
def engineer_clustering_features(profiles: pd.DataFrame) -> pd.DataFrame:
    """Combine symmetric court corners and related shot-creation mechanics."""
    df = profiles.copy()
    df["corner_3_share"] = (
        df.get("zone_share_Left Corner 3", 0.0) + df.get("zone_share_Right Corner 3", 0.0) # adds the two corner columns together (defaulting to 0.0 if a column were ever missing).
    )
    df["self_created_jumper_share"] = (
        df.get("mech_share_Pullup_Jumper", 0.0)
        + df.get("mech_share_Step_Back_Jumper", 0.0)
        + df.get("mech_share_Turnaround_Fadeaway", 0.0) #sums Pull-up, Step-back, and Turnaround/Fadeaway shares, while rim_finish_share sums Dunk and Tip/Putback/Lob shares.
    )
    df["rim_finish_share"] = (
        df.get("mech_share_Dunk", 0.0) + df.get("mech_share_Tip_Putback_Lob", 0.0)
    )
    return df

# test K=2 through 8 using Inertia and Silhouette scores
# recording Inertia (how tightly packed points are inside each cluster and Silhouette score (how well-separated clusters are from each other)
def evaluate_k_range(X_train: np.ndarray, k_values: range) -> pd.DataFrame:
    """Compare K values using Within-Cluster Sum of Squares (Inertia) and Silhouette."""
    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X_train) #fits the clusters and assigns each of the 355 training players a cluster ID
        rows.append(
            {
                "K": k,
                "inertia": round(float(km.inertia_), 2),
                "silhouette_score": round(float(silhouette_score(X_train, labels)), 4),
            }
        )
    return pd.DataFrame(rows)

# looks cluster numbers (0-3) to consistent basketball role names
# looks at each cluster's actual shooting averages (centroids_df) and assigns fixed, meaningful IDs and names every single time.
def map_deterministic_archetypes(centroids_df: pd.DataFrame) -> dict[int, tuple[int, str]]:
    """Map arbitrary KMeans integer labels to consistent basketball archetype IDs."""
    remaining = set(centroids_df.index)

    # C2: Rim-Running Paint Bigs (highest Restricted Area share)
    c_big = centroids_df.loc[list(remaining), "zone_share_Restricted Area"].idxmax() #finds whichever cluster has the highest Restricted Area share
    remaining.remove(c_big)

    # C1: Off-Dribble Shot Creators (highest self-created jumper share)
    c_creator = centroids_df.loc[list(remaining), "self_created_jumper_share"].idxmax()
    remaining.remove(c_creator)

    # C0: 3-and-D / Perimeter Spacers (highest total 3PT share among remaining)
    rem_list = list(remaining)
    three_sum = (
        centroids_df.loc[rem_list, "corner_3_share"]
        + centroids_df.loc[rem_list, "zone_share_Above the Break"]
    )
    c_spacer = three_sum.idxmax()
    remaining.remove(c_spacer)

    # C3: Inside-Out / Stretch Scorers (remaining balanced cluster)
    c_versatile = remaining.pop()

    return {
        c_spacer: (0, "3-and-D / Perimeter Spacers"),
        c_creator: (1, "Off-Dribble Shot Creators"),
        c_big: (2, "Rim-Running Paint Bigs"),
        c_versatile: (3, "Inside-Out / Stretch Scorers"),
    }

# save the elbow/silhouette chart and 2D PCA player-archetype scatter plot
# Panel A plots Inertia and Silhouette score on dual Y-axes to visually justify picking K=4
# Panel B uses PCA to squish the 10-dimensional standardized feature space down into 2 dimensions (pc1 and pc2)
def save_diagnostic_plots(
    k_df: pd.DataFrame, train_df: pd.DataFrame, X_train: np.ndarray, fig_dir: Path
) -> None:
    """Generate and save Elbow/Silhouette curve and 2D PCA cluster visualization."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_train)
    var_exp = pca.explained_variance_ratio_ * 100.0

    plot_df = train_df.copy()
    plot_df["pc1"] = coords[:, 0]
    plot_df["pc2"] = coords[:, 1]
    plot_df["label"] = (
        "C" + plot_df["player_cluster"].astype(str) + ": " + plot_df["player_archetype"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2))

    # Panel A: Inertia & Silhouette across K
    ax1 = axes[0]
    l1 = ax1.plot(k_df["K"], k_df["inertia"], marker="o", linewidth=2.2, color="#1f77b4", label="Inertia (WCSS)")
    vline = ax1.axvline(SELECTED_K, linestyle="--", color="#555555", alpha=0.8, label=f"Selected K = {SELECTED_K}")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Within-Cluster Sum of Squares (Inertia)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xticks(k_df["K"])

    ax1_twin = ax1.twinx()  #adds a second right-hand Y-axis to Panel A because Inertia is in the thousands, Silhouette is a decimal
    l2 = ax1_twin.plot(k_df["K"], k_df["silhouette_score"], marker="s", linewidth=2.2, color="#d62728", label="Silhouette Score")
    ax1_twin.set_ylabel("Silhouette Score", color="#d62728")
    ax1_twin.tick_params(axis="y", labelcolor="#d62728")
    ax1_twin.grid(False)

    lines = l1 + l2 + [vline]
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right", frameon=True)
    ax1.set_title(f"A. K-Means Model Selection (2024 Train, N = {len(train_df)})", fontweight="bold")

    # Panel B: 2D PCA Projection
    ax2 = axes[1]
    palette = {
        "C0: 3-and-D / Perimeter Spacers": "#2ca02c",
        "C1: Off-Dribble Shot Creators": "#ff7f0e",
        "C2: Rim-Running Paint Bigs": "#1f77b4",
        "C3: Inside-Out / Stretch Scorers": "#9467bd",
    }
    sns.scatterplot(
        data=plot_df,
        x="pc1",
        y="pc2",
        hue="label",
        hue_order=list(palette.keys()),
        palette=palette,
        size="q13_fga",
        sizes=(30, 220),
        alpha=0.82,
        ax=ax2,
    )
    ax2.set_xlabel(f"PC1 ({var_exp[0]:.1f}% Variance: Perimeter 3PT ---> Interior Paint Finishing)")
    ax2.set_ylabel(f"PC2 ({var_exp[1]:.1f}% Variance: Assisted Role ---> Self-Created Jumpers)")
    ax2.set_title("B. NBA Shooting Archetypes via PCA (K = 4)", fontweight="bold")
    ax2.legend(loc="best", fontsize=8.5, frameon=True)

    plt.tight_layout()
    out_fig = fig_dir / "kmeans_diagnostics.png"
    plt.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"Saved diagnostic plot: {out_fig}")

# adds player_cluster and player_archetype onto all four Q4 shot CSVs
# Connects Unsupervised Learning model to Supervised Learning data 
# by attaching each shooter's player_cluster (0–3) and player_archetype name onto q4_shots_train_2024.csv, val_2025.csv, test_2026.csv, and live_2027.csv

def merge_clusters_into_q4_splits(modeling_dir: Path, profiles: pd.DataFrame) -> None:
    """Attach leakage-free player_cluster and player_archetype onto Q4 CSV splits."""
    archetype_lookup = (
        profiles[["player_cluster", "player_archetype"]]
        .drop_duplicates()
        .set_index("player_cluster")["player_archetype"]
        .to_dict()
    )
    qualified = profiles[profiles["q13_fga"] >= MIN_CURRENT_SEASON_FGA].sort_values(["person_id", "season"])

    split_files = [
        (2024, "q4_shots_train_2024.csv"),
        (2025, "q4_shots_val_2025.csv"),
        (2026, "q4_shots_test_2026.csv"),
        (2027, "q4_shots_live_2027.csv"),
    ]

    for season, fname in split_files:
        fpath = modeling_dir / fname
        if not fpath.exists():
            continue

        q4_df = pd.read_csv(fpath, low_memory=False)
        # Drop existing cluster columns if re-running script
        q4_df = q4_df.drop(columns=["player_cluster", "player_archetype"], errors="ignore")

        curr_map = (
            qualified[qualified["season"] == season]
            .set_index("person_id")["player_cluster"]
            .to_dict()
        )
        prior_map = (
            qualified[qualified["season"] < season]
            .groupby("person_id")["player_cluster"]
            .last()
            .to_dict()
        )
        any_curr_map = (
            profiles[profiles["season"] == season]
            .set_index("person_id")["player_cluster"]
            .to_dict()
        )

        # Priority: 1) Current season (>=25 Q1-Q3 FGA), 2) Prior season cluster, 3) Current low-FGA, 4) Balanced C3
        clusters = [
            curr_map.get(pid, prior_map.get(pid, any_curr_map.get(pid, 3)))
            for pid in q4_df["person_id"]
        ]
        q4_df["player_cluster"] = np.array(clusters, dtype=np.int8)
        q4_df["player_archetype"] = q4_df["player_cluster"].map(archetype_lookup)
        q4_df.to_csv(fpath, index=False)
        print(f"Updated {fname} with player_cluster ({len(q4_df):,} rows)")


def main():
    modeling_dir = locate_modeling_dir()
    profiles = pd.read_csv(modeling_dir / "player_profiles_q1_q3.csv")
    profiles = engineer_clustering_features(profiles)

    # 1. Fit scaler & K-Means strictly on 2024 training players (>= 100 Q1-Q3 FGA)
    train_mask = (profiles["season"] == 2024) & (profiles["q13_fga"] >= MIN_TRAIN_Q13_FGA)
    train_df = profiles[train_mask].copy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[CLUSTER_FEATURES])

    # 2. Evaluate K = 2..8
    k_df = evaluate_k_range(X_train, range(2, 9))
    k_df.to_csv(modeling_dir / "kmeans_k_comparison.csv", index=False)
    print("\nK-Means Evaluation on 2024 Train:")
    print(k_df.to_string(index=False))

    # 3. Fit final K = 4 model & assign deterministic archetype labels
    km_final = KMeans(n_clusters=SELECTED_K, random_state=42, n_init=20)
    train_df["raw_cluster"] = km_final.fit_predict(X_train)

    centroids_orig = train_df.groupby("raw_cluster")[CLUSTER_FEATURES].mean()
    archetype_map = map_deterministic_archetypes(centroids_orig)

    # Project learned clusters onto ALL player-seasons (2024-2027)
    X_all = scaler.transform(profiles[CLUSTER_FEATURES])
    raw_all = km_final.predict(X_all)
    profiles["player_cluster"] = [archetype_map[c][0] for c in raw_all]
    profiles["player_archetype"] = [archetype_map[c][1] for c in raw_all]

    train_df["player_cluster"] = [archetype_map[c][0] for c in train_df["raw_cluster"]]
    train_df["player_archetype"] = [archetype_map[c][1] for c in train_df["raw_cluster"]]

    out_profiles = modeling_dir / "player_profiles_clustered.csv"
    profiles.to_csv(out_profiles, index=False)
    print(f"\nSaved clustered profiles: {out_profiles}")

    # 4. Save diagnostic figures & merge cluster labels onto Q4 splits
    save_diagnostic_plots(k_df, train_df, X_train, modeling_dir / "figures")
    merge_clusters_into_q4_splits(modeling_dir, profiles)


if __name__ == "__main__":
    main()
