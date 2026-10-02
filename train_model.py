"""
train_model.py
==============
Machine Learning Pipeline for UpayPulse AI: Agent Cash Shortage Risk Prediction.

Problem Statement:
------------------
Predict whether an MFS (Mobile Financial Services) retail agent is at risk of
experiencing a cash shortage (cash stockout) during operational demand surges.

Data Sources:
-------------
- data/agents.csv       : MFS retail agent profiles & cash balances
- data/transactions.csv : Historical transactional records & financial volume

Pipeline Workflow:
------------------
1. Feature Engineering:
   - hourly_cashout_volume   : Average cash-out volume during active transacting hours (BDT)
   - avg_transaction_amount  : Mean transaction value across all transaction types (BDT)
   - current_cash_balance    : Physical cash in drawer/till held by the agent (BDT)
   - transaction_frequency   : Total count of consumer transactions handled by the agent
2. Target Definition:
   - shortage_risk           : Binary label (1 = at risk of cash depletion, 0 = safe liquidity buffer)
3. Model Training:
   - RandomForestClassifier  : Ensemble model with balanced class handling & regularization
4. Evaluation:
   - Accuracy, Precision, Recall, Confusion Matrix, Classification Report
5. Model Persistence:
   - models/liquidity_model.pkl
"""

import os
import sys
import pickle
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. PATH CONFIGURATION & REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "liquidity_model.pkl"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

FEATURE_COLUMNS = [
    "hourly_cashout_volume",
    "avg_transaction_amount",
    "current_cash_balance",
    "transaction_frequency",
]
TARGET_COLUMN = "shortage_risk"


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA INGESTION
# ─────────────────────────────────────────────────────────────────────────────
def load_data(data_dir: Path = DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load agents.csv and transactions.csv from the data directory.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]: (agents_df, transactions_df)
    """
    agents_path = data_dir / "agents.csv"
    txns_path = data_dir / "transactions.csv"

    if not agents_path.exists():
        raise FileNotFoundError(f"Missing agents dataset at: {agents_path}")
    if not txns_path.exists():
        raise FileNotFoundError(f"Missing transactions dataset at: {txns_path}")

    print(f"[1/5] Ingesting raw datasets from: {data_dir}")
    agents_df = pd.read_csv(agents_path)
    txns_df = pd.read_csv(txns_path)

    # Ensure timestamp parsing
    txns_df["timestamp"] = pd.to_datetime(txns_df["timestamp"])

    print(f"      -> Loaded {len(agents_df):,} agents and {len(txns_df):,} transactions.")
    return agents_df, txns_df


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def build_features(agents_df: pd.DataFrame, txns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer the 4 key liquidity features per agent:
    1. hourly_cashout_volume   : Mean cash-out volume processed per transacting hour (BDT)
    2. avg_transaction_amount  : Mean transaction size across all consumer activities (BDT)
    3. current_cash_balance    : Physical cash in drawer held by the agent (BDT)
    4. transaction_frequency   : Total count of transactions processed by the agent

    Returns:
    --------
    pd.DataFrame: Merged feature dataset indexed by agent_id.
    """
    print("[2/5] Engineering liquidity and transaction velocity features...")

    # Filter cash-out transactions
    cashouts = txns_df[txns_df["transaction_type"] == "cash_out"].copy()
    cashouts["hourly_bucket"] = cashouts["timestamp"].dt.floor("h")

    # Feature 1: Hourly cash-out volume (average BDT cash-out per active hour)
    hourly_cashout = (
        cashouts.groupby(["agent_id", "hourly_bucket"])["amount"]
        .sum()
        .groupby("agent_id")
        .mean()
        .rename("hourly_cashout_volume")
    )

    # Feature 2: Average transaction amount (across all transaction types)
    avg_txn_amt = (
        txns_df.groupby("agent_id")["amount"]
        .mean()
        .rename("avg_transaction_amount")
    )

    # Feature 3: Current cash balance (physical cash in hand from agents.csv)
    # Feature 4: Transaction frequency (total count of transactions handled)
    txn_freq = (
        txns_df.groupby("agent_id")
        .size()
        .rename("transaction_frequency")
    )

    # Merge features onto the agents roster
    features_df = (
        agents_df[["agent_id", "area", "cash_balance", "rating"]]
        .rename(columns={"cash_balance": "current_cash_balance"})
        .merge(hourly_cashout, on="agent_id", how="left")
        .merge(avg_txn_amt, on="agent_id", how="left")
        .merge(txn_freq, on="agent_id", how="left")
    )

    # Handle any cold-start/idle agents with no recorded cash-outs
    features_df["hourly_cashout_volume"] = features_df["hourly_cashout_volume"].fillna(0.0)
    features_df["avg_transaction_amount"] = features_df["avg_transaction_amount"].fillna(0.0)
    features_df["transaction_frequency"] = features_df["transaction_frequency"].fillna(0)

    print("      -> Features successfully engineered:")
    for col in FEATURE_COLUMNS:
        mean_val = features_df[col].mean()
        print(f"         * {col:<26}: Mean = {mean_val:,.2f}")

    return features_df


# ─────────────────────────────────────────────────────────────────────────────
# 4. TARGET DEFINITION
# ─────────────────────────────────────────────────────────────────────────────
def create_target(features_df: pd.DataFrame, operational_window_hours: float = 8.0) -> pd.DataFrame:
    """
    Define the binary target variable: shortage_risk.

    Business Logic:
    ---------------
    In Bangladesh MFS, an agent faces cash shortage risk if their current physical
    cash in hand is insufficient to sustain projected peak surge demand across an
    8-hour operational replenishment window.

    Formula:
    --------
    shortage_risk = 1  IF (current_cash_balance < hourly_cashout_volume * operational_window_hours)
                    0  OTHERWISE

    Returns:
    --------
    pd.DataFrame: Dataset containing feature matrix and target column.
    """
    print("[3/5] Constructing binary target variable: shortage_risk...")

    projected_demand = features_df["hourly_cashout_volume"] * operational_window_hours
    features_df[TARGET_COLUMN] = (features_df["current_cash_balance"] < projected_demand).astype(int)

    counts = features_df[TARGET_COLUMN].value_counts()
    pos_count = counts.get(1, 0)
    neg_count = counts.get(0, 0)
    pos_pct = (pos_count / len(features_df)) * 100

    print(f"      -> Target distribution: {pos_count} Shortage Risk ({pos_pct:.1f}%), {neg_count} Safe Liquidity ({100 - pos_pct:.1f}%)")
    return features_df


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL TRAINING & TUNING
# ─────────────────────────────────────────────────────────────────────────────
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 150,
    max_depth: int = 6,
    random_state: int = RANDOM_STATE
) -> RandomForestClassifier:
    """
    Train a regularized RandomForestClassifier for liquidity shortage risk.

    Parameters:
    -----------
    X_train      : Feature training matrix
    y_train      : Target training labels
    n_estimators : Number of trees in the forest
    max_depth    : Maximum depth of each decision tree

    Returns:
    --------
    RandomForestClassifier: Fitted scikit-learn model.
    """
    print(f"[4/5] Training RandomForestClassifier (n_estimators={n_estimators}, max_depth={max_depth})...")

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    return rf


# ─────────────────────────────────────────────────────────────────────────────
# 6. MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Evaluate model performance against test split.

    Metrics:
    --------
    - Accuracy
    - Precision
    - Recall
    - Confusion Matrix
    - ROC-AUC Score
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, digits=4)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": cm,
        "roc_auc": auc,
        "classification_report": report,
    }

    print("\n" + "=" * 68)
    print(" MODEL PERFORMANCE EVALUATION (TEST DATASET)")
    print("=" * 68)
    print(f" Accuracy  : {acc * 100:.2f}%")
    print(f" Precision : {prec * 100:.2f}%")
    print(f" Recall    : {rec * 100:.2f}%")
    print(f" ROC-AUC   : {auc:.4f}")

    print("\n Confusion Matrix:")
    print("                    Predicted: SAFE (0)   Predicted: RISK (1)")
    print(f"  Actual: SAFE (0)        {cm[0, 0]:<15}      {cm[0, 1]:<15}")
    print(f"  Actual: RISK (1)        {cm[1, 0]:<15}      {cm[1, 1]:<15}")

    print("\n Classification Report:")
    print(report)

    # Feature Importance Breakdown
    print(" Feature Importances:")
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for idx in sorted_idx:
        col = FEATURE_COLUMNS[idx]
        imp = importances[idx]
        bar = "#" * int(imp * 35)
        print(f"   * {col:<26}: {imp:.4f}  {bar}")

    print("=" * 68 + "\n")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 7. MODEL ARTIFACT PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────
def save_model_artifact(
    model: RandomForestClassifier,
    output_path: Path = MODEL_PATH,
    feature_names: list = FEATURE_COLUMNS,
    metrics: dict = None
) -> None:
    """
    Save the trained model and inference metadata to models/liquidity_model.pkl.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[5/5] Persisting trained model artifact to: {output_path}")

    artifact = {
        "model": model,
        "feature_names": feature_names,
        "target": TARGET_COLUMN,
        "metrics": metrics or {},
        "trained_at": pd.Timestamp.now().isoformat(),
        "model_type": "RandomForestClassifier",
    }

    with open(output_path, "wb") as f:
        pickle.dump(artifact, f)

    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"      -> Successfully saved models/liquidity_model.pkl ({file_size_kb:.1f} KB).")


# ─────────────────────────────────────────────────────────────────────────────
# 8. COMPLETE PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline() -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Execute the full end-to-end ML pipeline.
    """
    print("\n" + "=" * 68)
    print(" UpayPulse AI — Agent Liquidity Shortage Risk ML Pipeline")
    print("=" * 68)

    # 1. Ingest Data
    agents_df, txns_df = load_data(DATA_DIR)

    # 2. Build Features
    features_df = build_features(agents_df, txns_df)

    # 3. Create Target
    dataset_df = create_target(features_df, operational_window_hours=8.0)

    # Prepare Train/Test Split (Strict featurization & stratification)
    X = dataset_df[FEATURE_COLUMNS]
    y = dataset_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\n      Split: {len(X_train)} Training samples, {len(X_test)} Testing samples.")

    # 4. Train Model
    model = train_model(X_train, y_train, n_estimators=150, max_depth=6)

    # 5. Evaluate Model
    metrics = evaluate_model(model, X_test, y_test)

    # 6. Save Model Artifact
    save_model_artifact(model, MODEL_PATH, FEATURE_COLUMNS, metrics)

    return model, metrics


# ─────────────────────────────────────────────────────────────────────────────
# 9. CLI ENTRY POINT & INFERENCE SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    trained_model, eval_metrics = run_pipeline()

    # Inference sanity check
    print("Performing inference sanity check on reloaded artifact...")
    with open(MODEL_PATH, "rb") as f:
        loaded_artifact = pickle.load(f)

    loaded_model = loaded_artifact["model"]

    # Sample agent profile test
    # Case A: Low cash (৳12,000), high hourly cashout (৳8,500/hr) -> High Shortage Risk (1)
    # Case B: High cash (৳250,000), moderate hourly cashout (৳5,000/hr) -> Safe Liquidity (0)
    sample_agents = pd.DataFrame([
        {
            "hourly_cashout_volume": 8500.0,
            "avg_transaction_amount": 3200.0,
            "current_cash_balance": 12000.0,
            "transaction_frequency": 350
        },
        {
            "hourly_cashout_volume": 5000.0,
            "avg_transaction_amount": 2500.0,
            "current_cash_balance": 250000.0,
            "transaction_frequency": 120
        }
    ])[FEATURE_COLUMNS]

    sample_preds = loaded_model.predict(sample_agents)
    sample_probs = loaded_model.predict_proba(sample_agents)[:, 1]

    print("\nInference Validation Results:")
    print(f"  Agent A (Cash: BDT 12k, Hourly Demand: BDT 8.5k) -> Predicted Risk: {sample_preds[0]} ({'[SHORTAGE]' if sample_preds[0] == 1 else '[SAFE]'}, Prob: {sample_probs[0]:.2%})")
    print(f"  Agent B (Cash: BDT 250k, Hourly Demand: BDT 5k)  -> Predicted Risk: {sample_preds[1]} ({'[SHORTAGE]' if sample_preds[1] == 1 else '[SAFE]'}, Prob: {sample_probs[1]:.2%})")
    print("\nPipeline execution completed successfully.")
