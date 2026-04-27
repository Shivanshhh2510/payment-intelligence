"""
src/fraud_engine/registry.py
──────────────────────────────
MLflow Model Registry — version control + staging/production workflow.

Promotes models through: None → Staging → Production
The dashboard always loads from the Production-tagged model.
This is the difference between "I used MLflow for logging" and
"I implemented a model registry with promotion workflow."

Usage:
    python src/fraud_engine/registry.py --promote   ← promote latest to Production
    python src/fraud_engine/registry.py --list      ← list all versions
    python src/fraud_engine/registry.py --compare   ← compare Staging vs Production
"""

import os
import sys
import json
import argparse
import warnings
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore")

ARTIFACTS_DIR      = "mlflow_artifacts"
MODEL_NAME         = "fraud-detection-xgboost"
EXPERIMENT_NAME    = "fraud-detection-ieee-cis"
MLFLOW_TRACKING_URI = "mlruns"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()


def get_best_run() -> mlflow.entities.Run:
    """Return the run with highest test_auc from the experiment."""
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found. Run train.py first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_auc DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No runs found.")

    run = runs[0]
    print(f"Best run: {run.info.run_id}")
    print(f"  test_auc:  {run.data.metrics.get('test_auc', 'N/A'):.4f}")
    print(f"  test_f1:   {run.data.metrics.get('test_f1', 'N/A'):.4f}")
    return run


def register_model(run: mlflow.entities.Run) -> str:
    """Register model from run into MLflow Model Registry."""
    model_uri = f"runs:/{run.info.run_id}/xgboost_model"

    try:
        registered = mlflow.register_model(model_uri, MODEL_NAME)
        version = registered.version
        print(f"  Registered as '{MODEL_NAME}' version {version}")
        return version
    except Exception as e:
        print(f"  Registration note: {e}")
        # Get latest version if already registered
        versions = client.get_latest_versions(MODEL_NAME)
        if versions:
            return versions[-1].version
        raise


def promote_to_staging(version: str) -> None:
    client.transition_model_version_stage(
        name=MODEL_NAME, version=version, stage="Staging",
        archive_existing_versions=False,
    )
    print(f"  Version {version} → Staging")


def promote_to_production(version: str) -> None:
    client.transition_model_version_stage(
        name=MODEL_NAME, version=version, stage="Production",
        archive_existing_versions=True,   # archive previous Production
    )
    print(f"  Version {version} → Production")

    # Save production metadata
    meta = {
        "production_version": version,
        "model_name":         MODEL_NAME,
        "tracking_uri":       MLFLOW_TRACKING_URI,
    }
    with open(os.path.join(ARTIFACTS_DIR, "production_model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved → {ARTIFACTS_DIR}/production_model_meta.json")


def list_versions() -> None:
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["None","Staging","Production","Archived"])
        print(f"\n── Model Registry: {MODEL_NAME} ──────────────────────────────")
        for v in versions:
            print(f"  Version {v.version:>3}  Stage: {v.current_stage:<12}  "
                  f"Run: {v.run_id[:8]}…  Created: {v.creation_timestamp}")
    except Exception:
        print("  No versions registered yet.")


def compare_staging_production() -> None:
    """Print metrics comparison between Staging and Production versions."""
    for stage in ["Staging", "Production"]:
        try:
            versions = client.get_latest_versions(MODEL_NAME, stages=[stage])
            if not versions:
                print(f"  No {stage} model.")
                continue
            v    = versions[0]
            run  = client.get_run(v.run_id)
            metrics = run.data.metrics
            print(f"\n  {stage} (v{v.version}):")
            for k in ["test_auc","test_f1","test_precision","test_recall","val_auc"]:
                val = metrics.get(k, "N/A")
                print(f"    {k:<20} {val:.4f}" if isinstance(val, float) else f"    {k:<20} {val}")
        except Exception as e:
            print(f"  {stage}: {e}")


def load_production_model():
    """Load the Production-tagged model. Falls back to local joblib if registry empty."""
    try:
        model_uri = f"models:/{MODEL_NAME}/Production"
        model = mlflow.xgboost.load_model(model_uri)
        print(f"  Loaded Production model from registry: {MODEL_NAME}")
        return model
    except Exception:
        import joblib
        model = joblib.load(os.path.join(ARTIFACTS_DIR, "fraud_model.joblib"))
        print("  Loaded model from local artifact (registry not set up).")
        return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--promote",  action="store_true", help="Register + promote best run to Production")
    parser.add_argument("--list",     action="store_true", help="List all registered versions")
    parser.add_argument("--compare",  action="store_true", help="Compare Staging vs Production metrics")
    args = parser.parse_args()

    if args.list:
        list_versions()

    elif args.compare:
        compare_staging_production()

    elif args.promote:
        print("Finding best run …")
        run = get_best_run()
        print("\nRegistering model …")
        version = register_model(run)
        print("\nPromoting …")
        promote_to_staging(version)
        promote_to_production(version)
        print("\nFinal state:")
        list_versions()

    else:
        # Default: full promote workflow
        print("Finding best run …")
        run = get_best_run()
        version = register_model(run)
        promote_to_staging(version)
        promote_to_production(version)
        list_versions()