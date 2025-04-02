import os
from pathlib import Path
import yaml
import sys

# Add the project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.ml_pipeline import load_surv_data, tune_xgboost_stepwise

with open(os.path.join(PROJECT_ROOT, 'config.yml'), 'r') as f:
    experiment_config = yaml.safe_load(f)

experiment_config = experiment_config['experiments']['hyperparameter_tuning']

if __name__ == "__main__":
    print("Starting hyperparameter tuning...")

    # Load training data
    X_train, _, y_train, _ = load_surv_data(
        features_path=experiment_config['features_path'],
        outcomes_path=experiment_config['outcomes_path'],
        split_ratio=experiment_config['split_ratio'],
        random_state=experiment_config['random_state']
    )

    if experiment_config['method'] == 'stepwise':
        # Tune hyperparameters using stepwise strategy from Liu et al 2023
        best_params = tune_xgboost_stepwise(X_train, y_train, random_state=experiment_config['random_state'])
    else:
        raise ValueError(f"Invalid hyperparameter tuning method: {experiment_config['method']}")

    # Save best hyperparameters
    output_file = Path(os.path.join(PROJECT_ROOT, experiment_config['output_path']))

    if not output_file.exists():
        output_file.touch()

    with output_file.open("w") as f:
        yaml.dump(best_params, f, default_flow_style=False)

    print(f"Best hyperparameters saved to {output_file}")