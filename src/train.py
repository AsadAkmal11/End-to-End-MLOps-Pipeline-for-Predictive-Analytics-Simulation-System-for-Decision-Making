"""
Advanced Training Script - PlayMaker Analytics (Cricket).

Loads all ball-by-ball cricket CSVs from the data/ directory, performs
feature engineering to derive historical win rates, head-to-head records,
toss impact, and venue advantages, then trains a Random Forest classifier
(with optional XGBoost upgrade) and exports the pipeline plus feature-
importance metadata for the UI dashboard.

Usage:
    python src/train.py

Outputs:
    models/cricket_model.pkl        — Joblib-serialised sklearn Pipeline
    models/feature_importance.json  — Top-N feature importances (JSON)
    models/match_metadata.json      — Teams / venues for the frontend
"""

import json
import os
import csv
import glob

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Optional XGBoost — graceful fallback to Random Forest if unavailable
# Catches ImportError AND runtime errors (e.g. missing DLL on Windows)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:  # noqa: BLE001
    XGBOOST_AVAILABLE = False

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
DATA_DIR = "data"
MODEL_PATH = os.path.join("models", "cricket_model.pkl")
IMPORTANCE_PATH = os.path.join("models", "feature_importance.json")
METADATA_PATH = os.path.join("models", "match_metadata.json")
MAX_FILES = None  # None = use entire dataset; set an int for quick tests
USE_XGBOOST = True  # Prefer XGBoost when available

IMPORTANCE_TOP_N = 15

# ---------------------------------------------------------------------------
# STEP 1: DATA LOADING
# ---------------------------------------------------------------------------


def load_raw_matches(data_dir=DATA_DIR, max_files=MAX_FILES):
    """
    Reads every CSV in data_dir and extracts match-level metadata rows.

    Each file encodes one innings of ball-by-ball data; we only need the
    'info' section at the top of each file.

    Returns:
        pd.DataFrame: One row per unique match (keyed by 'match_id').
    """
    pattern = os.path.join(data_dir, "*.csv")
    filepaths = sorted(glob.glob(pattern))
    if max_files:
        filepaths = filepaths[:max_files]

    records = []
    for fp in filepaths:
        match_id = os.path.splitext(os.path.basename(fp))[0]
        meta = {
            "match_id": match_id,
            "teams": [],
            "venue": None,
            "toss_winner": None,
            "toss_decision": None,
            "winner": None,
            "winner_runs": None,
            "winner_wickets": None,
            "season": None,
            "dates": [],
            "gender": None,
        }

        with open(fp, "r", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if not row or row[0] != "info":
                    if row and row[0] == "ball":
                        break
                    continue
                key = row[1] if len(row) > 1 else ""
                val = row[2] if len(row) > 2 else ""
                if key == "team":
                    meta["teams"].append(val)
                elif key == "venue":
                    meta["venue"] = val
                elif key == "toss_winner":
                    meta["toss_winner"] = val
                elif key == "toss_decision":
                    meta["toss_decision"] = val
                elif key == "winner":
                    meta["winner"] = val
                elif key == "winner_runs":
                    meta["winner_runs"] = val
                elif key == "winner_wickets":
                    meta["winner_wickets"] = val
                elif key == "season":
                    meta["season"] = val
                elif key == "date":
                    meta["dates"].append(val)
                elif key == "gender":
                    meta["gender"] = val

        # Only keep fully valid matches with a decisive winner
        if len(meta["teams"]) >= 2 and meta["venue"] and meta["winner"]:
            records.append({
                "match_id": meta["match_id"],
                "team1": meta["teams"][0],
                "team2": meta["teams"][1],
                "venue": meta["venue"],
                "toss_winner": meta["toss_winner"],
                "toss_decision": meta["toss_decision"],
                "winner": meta["winner"],
                "season": meta["season"],
                "match_date": meta["dates"][0] if meta["dates"] else None,
                "gender": meta["gender"] or "male",
            })

    df = pd.DataFrame(records).drop_duplicates(subset="match_id")

    # Sort chronologically so leakage-safe rolling features are correct
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values("match_date").reset_index(drop=True)

    print(
        f"[DATA]  Loaded {len(df)} valid matches from {len(filepaths)} files."
    )
    return df


# ---------------------------------------------------------------------------
# STEP 2: FEATURE ENGINEERING
# ---------------------------------------------------------------------------


def engineer_features(df):
    """
    Derives richer predictors from raw match metadata.

    Features produced
    -----------------
    team1_win_rate       : Global historical win rate for team1
    team2_win_rate       : Global historical win rate for team2
    h2h_win_rate         : team1's head-to-head win rate vs team2
    toss_impact_rate     : Rolling expanding-mean of toss→win correlation
    venue_t1_advantage   : Fraction of wins at this venue attributed to team1

    Binary target
    -------------
    1  →  team1 wins
    0  →  team2 wins
    """
    df = df[
        df["winner"].isin(df["team1"]) | df["winner"].isin(df["team2"])
    ].copy().reset_index(drop=True)

    # Binary target
    df["target"] = (df["winner"] == df["team1"]).astype(int)

    # ── Global win rate per team ────────────────────────────────────────── #
    all_teams = pd.concat([df["team1"], df["team2"]]).unique()
    team_wins = (
        df["winner"].value_counts().reindex(all_teams, fill_value=0)
    )
    team_games = {
        t: int((df["team1"] == t).sum() + (df["team2"] == t).sum())
        for t in all_teams
    }
    team_win_rate = {
        t: team_wins[t] / max(team_games[t], 1) for t in all_teams
    }
    df["team1_win_rate"] = df["team1"].map(team_win_rate)
    df["team2_win_rate"] = df["team2"].map(team_win_rate)

    # ── Head-to-head win rate (team1 vs team2, leakage-safe expanding avg) #
    def compute_h2h(df_in):
        """Rolling H2H: for each row, fraction of prior clashes won by team1."""
        results = []
        for idx, row in df_in.iterrows():
            t1, t2 = row["team1"], row["team2"]
            key = tuple(sorted([t1, t2]))
            prior = df_in.loc[
                :idx - 1,
                ["team1", "team2", "target"]
            ]
            # Matches between these two teams (either orientation)
            mask_fwd = (prior["team1"] == t1) & (prior["team2"] == t2)
            mask_rev = (prior["team1"] == t2) & (prior["team2"] == t1)
            fwd = prior.loc[mask_fwd, "target"].tolist()
            rev = [1 - x for x in prior.loc[mask_rev, "target"].tolist()]
            all_h2h = fwd + rev
            if all_h2h:
                results.append(sum(all_h2h) / len(all_h2h))
            else:
                results.append(0.5)  # no prior data → neutral
        return results

    df["h2h_win_rate"] = compute_h2h(df)

    # ── Toss-match correlation (expanding, leakage-safe) ───────────────── #
    df["toss_match_same"] = (df["toss_winner"] == df["winner"]).astype(int)
    expanding_mean = df["toss_match_same"].expanding().mean().shift(1)
    global_mean = df["toss_match_same"].mean()
    df["toss_impact_rate"] = expanding_mean.fillna(global_mean)

    # ── Venue advantage for team1 ──────────────────────────────────────── #
    venue_t1_wins = df[df["target"] == 1].groupby("venue").size()
    venue_total = df.groupby("venue").size()
    venue_advantage = (venue_t1_wins / venue_total).fillna(0.5)
    df["venue_t1_advantage"] = (
        df["venue"].map(venue_advantage).fillna(0.5)
    )

    print(
        f"[FEAT]  Feature engineering complete — dataset shape: {df.shape}"
    )
    return df


# ---------------------------------------------------------------------------
# STEP 3: COLUMN DEFINITIONS
# ---------------------------------------------------------------------------
CATEGORICAL_COLS = [
    "team1", "team2", "venue", "toss_winner", "toss_decision"
]
NUMERIC_COLS = [
    "team1_win_rate",
    "team2_win_rate",
    "h2h_win_rate",
    "toss_impact_rate",
    "venue_t1_advantage",
]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS


# ---------------------------------------------------------------------------
# STEP 4: MODEL TRAINING & EVALUATION
# ---------------------------------------------------------------------------


def build_pipeline():
    """Constructs the sklearn Pipeline with preprocessing + classifier."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
            (
                "num",
                "passthrough",
                NUMERIC_COLS,
            ),
        ]
    )

    if USE_XGBOOST and XGBOOST_AVAILABLE:
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        print("[MODEL] Using XGBoost classifier.")
    else:
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        if USE_XGBOOST and not XGBOOST_AVAILABLE:
            print(
                "[MODEL] XGBoost not installed; falling back to "
                "RandomForest. Run: pip install xgboost"
            )
        else:
            print("[MODEL] Using RandomForest classifier.")

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", clf)])


def evaluate_pipeline(pipeline, X, y):
    """
    Runs 5-fold Stratified Cross Validation and prints a full classification
    report. Returns mean CV accuracy and ROC-AUC.

    The pipeline is ALSO fit on all data here so that extract_feature_
    importance() can be called immediately after.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    roc_scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring="roc_auc"
    )

    print("\n" + "=" * 58)
    print("  MODEL EVALUATION REPORT")
    print("=" * 58)
    print(f"  CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  CV ROC-AUC  : {roc_scores.mean():.4f} ± {roc_scores.std():.4f}")

    # Full final fit on all data
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]
    train_acc = accuracy_score(y, y_pred)

    print("\n  Train-Set Classification Report:")
    print(classification_report(y, y_pred, target_names=["team2", "team1"]))
    print(f"  Train Accuracy : {train_acc:.4f}")
    print(f"  Train ROC-AUC  : {roc_auc_score(y, y_proba):.4f}")
    print("=" * 58 + "\n")

    return cv_scores.mean(), roc_scores.mean()


# ---------------------------------------------------------------------------
# STEP 5: FEATURE IMPORTANCE EXTRACTION
# ---------------------------------------------------------------------------


def extract_feature_importance(pipeline, top_n=IMPORTANCE_TOP_N):
    """
    Extracts feature importance data from the fitted estimator, mapping
    OHE-expanded + passthrough column names back to human-readable labels
    for the UI dashboard.

    Returns:
        dict: {'feature_names': [...], 'importances': [...]} (top_n entries,
              sorted descending, JSON-serialisable for the frontend).
    """
    ohe = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["cat"]
    )
    ohe_names = ohe.get_feature_names_out(CATEGORICAL_COLS).tolist()
    all_names = ohe_names + NUMERIC_COLS

    importances = pipeline.named_steps["model"].feature_importances_

    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [all_names[i] for i in indices]
    top_vals = [round(float(importances[i]), 6) for i in indices]

    return {"feature_names": top_names, "importances": top_vals}


# ---------------------------------------------------------------------------
# STEP 6: METADATA EXPORT (teams / venues for the frontend dropdown)
# ---------------------------------------------------------------------------


def export_match_metadata(df):
    """
    Saves the unique teams and venues encountered in the dataset to a JSON
    file so the Streamlit frontend can build its dropdowns dynamically
    instead of relying on a hard-coded list.

    Returns:
        dict: {'teams': [...], 'venues': [...]}
    """
    teams = sorted(
        set(df["team1"].unique().tolist() + df["team2"].unique().tolist())
    )
    venues = sorted(df["venue"].dropna().unique().tolist())
    meta = {"teams": teams, "venues": venues}

    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(
        f"[META]  Metadata -> {METADATA_PATH}  "
        f"({len(teams)} teams, {len(venues)} venues)"
    )
    return meta


# ---------------------------------------------------------------------------
# STEP 7: SAVE PIPELINE ARTIFACT
# ---------------------------------------------------------------------------


def save_artifacts(pipeline, importance_data):
    """Persists the model pipeline and feature-importance JSON to disk."""
    os.makedirs("models", exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)
    print(f"[SAVE]  Model pipeline -> {MODEL_PATH}")

    with open(IMPORTANCE_PATH, "w", encoding="utf-8") as fh:
        json.dump(importance_data, fh, indent=2)
    print(f"[SAVE]  Feature importance -> {IMPORTANCE_PATH}")


# ---------------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Load raw data
    raw_df = load_raw_matches()
    if raw_df.empty:
        print("[ABORT] No valid matches found. Check the data/ directory.")
        raise SystemExit(1)

    # 2. Export metadata before feature engineering (uses raw teams/venues)
    os.makedirs("models", exist_ok=True)
    export_match_metadata(raw_df)

    # 3. Feature engineering
    engineered_df = engineer_features(raw_df)

    # Drop rows where any feature or target is null
    engineered_df.dropna(subset=FEATURE_COLS + ["target"], inplace=True)

    X = engineered_df[FEATURE_COLS]
    y = engineered_df["target"]

    print(
        f"[DATA]  Final training set: {len(X)} samples, "
        f"{y.mean():.2%} class-1 (team1 wins)"
    )

    # 4. Build, evaluate, and fully fit pipeline
    pipe = build_pipeline()
    evaluate_pipeline(pipe, X, y)

    # Pipeline is now fully fit on all data (last call in evaluate_pipeline)
    # 5. Extract feature importances
    importance_data = extract_feature_importance(pipe, top_n=IMPORTANCE_TOP_N)

    # 6. Persist artifacts
    save_artifacts(pipe, importance_data)

    print("\n[DONE]  Training complete. Artifacts saved to models/")
