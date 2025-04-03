import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

### Data loading ###

def load_surv_data(features_path, outcomes_path, split_ratio=0.8, random_state=42):
    """Load and prepare survival data for training and testing.

    This function loads feature and outcome data from feather or csv files, aligns them by patient ID,
    and splits them into training and test sets. (Validation set derived from training set e.g. by cross-validation)

    Args:
        features_path (str): Path to the features file (feather or csv) relative to project root
        outcomes_path (str): Path to the survival outcomes file (feather or csv) relative to project root
        split_ratio (float, optional): Ratio of data to use for training. Default is 0.8.
        random_state (int, optional): Random seed for reproducibility. Use same seed for all steps of the pipeline. Default is 42.

    Returns:
        tuple: Contains (X_train, X_test, y_train, y_test) where:
            - X_train, X_test: Training and test feature matrices
            - y_train, y_test: Training and test survival outcomes in (time, event) format

    Raises:
        ValueError: If features and outcomes have different lengths after alignment
    """
    # Load features & outcomes based on file extension
    features_path_full = os.path.join(PROJECT_ROOT, features_path)
    outcomes_path_full = os.path.join(PROJECT_ROOT, outcomes_path)
    
    if features_path.endswith('.feather'):
        features = pd.read_feather(features_path_full)
    elif features_path.endswith('.csv'):
        features = pd.read_csv(features_path_full)
    else:
        raise ValueError(f"Unsupported features file format: {features_path}")
        
    if outcomes_path.endswith('.feather'):
        outcomes = pd.read_feather(outcomes_path_full)
    elif outcomes_path.endswith('.csv'):
        outcomes = pd.read_csv(outcomes_path_full)
    else:
        raise ValueError(f"Unsupported outcomes file format: {outcomes_path}")

    # Only select the intersection of eids between features and outcomes
    common_eids = set(features['eid']).intersection(set(outcomes['eid']))
    features = features[features['eid'].isin(common_eids)]
    outcomes = outcomes[outcomes['eid'].isin(common_eids)]

    # Order by eid to ensure features and outcomes are aligned
    features = features.sort_values(by='eid')
    outcomes = outcomes.sort_values(by='eid')

    # Check that features and outcomes have same number of rows
    if len(features) != len(outcomes):
        raise ValueError(f"Features and outcomes have different lengths: features={len(features)}, outcomes={len(outcomes)}")
    print(f"\nSuccessfully loaded {len(features)} samples with matching features and outcomes")

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, outcomes, test_size=1-split_ratio, random_state=random_state)

    return X_train, X_test, y_train, y_test


### Stepwise hyperparameter tuning strategy from Liu et al 2023 ###
### https://github.com/xiaonanl1996/MLforBrCa/blob/main/Scripts/Stats_Analysis/JupyterNotebook/XGBoost-Cox.ipynb ###

def tune_xgboost_stepwise(X_train, y_train, random_state=42):
    """
    Tune XGBoost hyperparameters according to the stepwise strategy from Liu et al 2023.
    
    Parameters:
        X_train: array-like or DataFrame
            Training features.
        y_train: array-like or Series
            Survival outcome in (T, E) format.
        random_state: int, default=42
            Random seed for reproducibility.
            
    Returns:
        best_params_xgb: dict
            Dictionary containing the tuned hyperparameters.
    """

    # Cox negative log likelihood loss function can only be fed one y column, so have to encode censored individuals as negative T.
    y_train.loc[y_train['E'] == 0, 'T'] = -y_train['T']

    # Create DMatrix 
    dtrain = xgb.DMatrix(X_train, label=y_train['T'], enable_categorical=True)

    ### Step 1: Build baseline model with high learning rate ###

    print("Building baseline model...")

    # Create DMatrix for training and validation data
    Trees_train, Trees_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=random_state)
    dtrain = xgb.DMatrix(Trees_train, label=y_train['T'], enable_categorical=True)
    dval = xgb.DMatrix(Trees_val, label=y_val['T'], enable_categorical=True)
    
    # Set baseline parameters
    params = {
        'eta': 0.1,
        'max_depth': 3,
        'min_child_weight': 1,
        'gamma': 0,
        'subsample': 1,
        'colsample_bytree': 1,
        'lambda': 1,  # analogous to Ridge regression
        'alpha': 0,
        'objective': 'survival:cox',
        'tree_method': 'hist',
        'device': 'cuda'
    }
    
    # Train model with early stopping
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    num_round = 500
    xgb_baseline = xgb.train(
        params,
        dtrain,
        num_round,
        evallist,
        early_stopping_rounds=50,
        evals_result=None,
        verbose_eval=False
    )
    
    n_trees = xgb_baseline.best_iteration
    print(f"Number of trees: {n_trees}")
    
    ### Step 2: Tune max_depth and min_child_weight ###

    print("Tuning max_depth and min_child_weight...")

    gridsearch_params1 = [
        (max_depth, min_child_weight)
        for max_depth in range(1, 5)
        for min_child_weight in range(1, 6, 2)
    ]
    
    params = {
        "eta": 0.1,
        "max_depth": 5,
        "objective": "survival:cox",
        "min_child_weight": 1,
        "gamma": 0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "device": "cuda"
    }
    
    gsearch1_results = []
    for max_depth, min_child_weight in gridsearch_params1:
        print(f"CV with max_depth={max_depth}, min_child_weight={min_child_weight}")
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=n_trees,
            seed=random_state,
            nfold=5,
            metrics={'cox-nloglik'},
            early_stopping_rounds=10,
            verbose_eval=False,
        )
        mean_coxlik = cv_results['test-cox-nloglik-mean'].min()
        boost_rounds = cv_results['test-cox-nloglik-mean'].argmin()
        print(f"\tNegative Cox partial likelihood {mean_coxlik} for {boost_rounds} rounds")
        temp_df = pd.DataFrame({
            'max_depth': [max_depth],
            'min_child_weight': [min_child_weight],
            'best_rounds': [boost_rounds],
            'mean_cox_loss': [mean_coxlik]
        })
        gsearch1_results.append(temp_df)
    gsearch1_df = pd.concat(gsearch1_results, ignore_index=True)
    best_idx = gsearch1_df['mean_cox_loss'].idxmin()
    best_max_depth = gsearch1_df.loc[best_idx, 'max_depth']
    best_min_child_weight = gsearch1_df.loc[best_idx, 'min_child_weight']

    print(f"Best max_depth: {best_max_depth}")
    print(f"Best min_child_weight: {best_min_child_weight}")
    
    ### Step 3: Tune gamma ###

    print("Tuning gamma...")

    gridsearch_params2 = [gamma / 10.0 for gamma in range(0, 12, 2)]
    params = {
        "eta": 0.1,
        "max_depth": best_max_depth,
        "objective": "survival:cox",
        "min_child_weight": best_min_child_weight,
        "gamma": 0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "device": "cuda"
    }
    
    gsearch2_results = []
    for gamma in gridsearch_params2:
        print(f"CV with gamma={gamma}")
        params['gamma'] = gamma
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=n_trees,
            seed=random_state,
            nfold=5,
            metrics={'cox-nloglik'},
            early_stopping_rounds=10,
            verbose_eval=False,
        )
        mean_coxlik = cv_results['test-cox-nloglik-mean'].min()
        boost_rounds = cv_results['test-cox-nloglik-mean'].argmin()
        print(f"\tNegative Cox partial likelihood {mean_coxlik} for {boost_rounds} rounds")
        temp_df = pd.DataFrame({
            'gamma': [gamma],
            'best_rounds': [boost_rounds],
            'mean_cox_loss': [mean_coxlik]
        })
        gsearch2_results.append(temp_df)
    gsearch2_df = pd.concat(gsearch2_results, ignore_index=True)
    best_gamma = gsearch2_df.loc[gsearch2_df['mean_cox_loss'].idxmin(), 'gamma']

    print(f"Best gamma: {best_gamma}")
    
    ### Step 4: Tune subsample and colsample_bytree ###

    print("Tuning subsample and colsample_bytree...")

    gridsearch_params3 = [
        (subsample / 10.0, colsample / 10.0)
        for subsample in range(6, 10)
        for colsample in range(6, 10)
    ]
    params = {
        "eta": 0.1,
        "max_depth": best_max_depth,
        "objective": "survival:cox",
        "min_child_weight": best_min_child_weight,
        "gamma": best_gamma,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "device": "cuda"
    }
    
    gsearch3_results = []
    for subsample, colsample in gridsearch_params3:
        print(f"CV with subsample={subsample}, colsample_bytree={colsample}")
        params['subsample'] = subsample
        params['colsample_bytree'] = colsample
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=n_trees,
            seed=random_state,
            nfold=5,
            metrics={'cox-nloglik'},
            early_stopping_rounds=10,
            verbose_eval=False,
        )
        mean_coxlik = cv_results['test-cox-nloglik-mean'].min()
        boost_rounds = cv_results['test-cox-nloglik-mean'].argmin()
        print(f"\tNegative Cox partial likelihood {mean_coxlik} for {boost_rounds} rounds")
        temp_df = pd.DataFrame({
            'subsample': [subsample],
            'colsample_bytree': [colsample],
            'best_rounds': [boost_rounds],
            'mean_cox_loss': [mean_coxlik]
        })
        gsearch3_results.append(temp_df)
    gsearch3_df = pd.concat(gsearch3_results, ignore_index=True)
    best_subsample = gsearch3_df.loc[gsearch3_df['mean_cox_loss'].idxmin(), 'subsample']
    best_colsample_bytree = gsearch3_df.loc[gsearch3_df['mean_cox_loss'].idxmin(), 'colsample_bytree']

    print(f"Best subsample: {best_subsample}")
    print(f"Best colsample_bytree: {best_colsample_bytree}")
    
    ### Step 5: Tune regularization (lambda) ###

    print("Tuning regularization (lambda)...")

    gridsearch_params4 = [reg_lambda for reg_lambda in range(8, 24, 2)]
    params = {
        "eta": 0.1,
        "max_depth": best_max_depth,
        "objective": "survival:cox",
        "min_child_weight": best_min_child_weight,
        "gamma": best_gamma,
        "subsample": best_subsample,
        "colsample_bytree": best_colsample_bytree,
        "tree_method": "hist",
        "device": "cuda"
    }
    
    gsearch4_results = []
    for reg_lambda in gridsearch_params4:
        print(f"CV with lambda={reg_lambda}")
        params['lambda'] = reg_lambda
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=n_trees,
            seed=random_state,
            nfold=5,
            metrics={'cox-nloglik'},
            early_stopping_rounds=10,
            verbose_eval=False,
        )
        mean_coxlik = cv_results['test-cox-nloglik-mean'].min()
        boost_rounds = cv_results['test-cox-nloglik-mean'].argmin()
        print(f"\tNegative Cox partial likelihood {mean_coxlik} for {boost_rounds} rounds")
        temp_df = pd.DataFrame({
            'lambda': [reg_lambda],
            'best_rounds': [boost_rounds],
            'mean_cox_loss': [mean_coxlik]
        })
        gsearch4_results.append(temp_df)
    gsearch4_df = pd.concat(gsearch4_results, ignore_index=True)
    best_lambda = gsearch4_df.loc[gsearch4_df['mean_cox_loss'].idxmin(), 'lambda']

    print(f"Best lambda: {best_lambda}")
    
    ### Step 6: Tune learning rate (eta) ###

    print("Tuning learning rate (eta)...")

    gridsearch_params5 = [0.001, 0.003, 0.005, 0.007, 0.01]
    params = {
        "eta": 0.1,
        "max_depth": best_max_depth,
        "objective": "survival:cox",
        "min_child_weight": best_min_child_weight,
        "gamma": best_gamma,
        "subsample": best_subsample,
        "colsample_bytree": best_colsample_bytree,
        "lambda": best_lambda,
        "tree_method": "hist",
        "device": "cuda"
    }
    
    gsearch5_results = []
    for eta in gridsearch_params5:
        print(f"CV with eta={eta}")
        params['eta'] = eta
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=50000,
            seed=random_state,
            nfold=5,
            metrics={'cox-nloglik'},
            early_stopping_rounds=200,
            verbose_eval=False,
        )
        mean_coxlik = cv_results['test-cox-nloglik-mean'].min()
        boost_rounds = cv_results['test-cox-nloglik-mean'].argmin()
        print(f"\tNegative Cox partial likelihood {mean_coxlik} for {boost_rounds} rounds")
        temp_df = pd.DataFrame({
            'eta': [eta],
            'best_rounds': [boost_rounds],
            'mean_cox_loss': [mean_coxlik]
        })
        gsearch5_results.append(temp_df)
    gsearch5_df = pd.concat(gsearch5_results, ignore_index=True)
    best_eta = gsearch5_df.loc[gsearch5_df['mean_cox_loss'].idxmin(), 'eta']

    print(f"Best eta: {best_eta}")

    ### Compile the best hyperparameters found ###

    best_params_xgb = {
        "eta": float(best_eta),
        "max_depth": int(best_max_depth),
        "min_child_weight": int(best_min_child_weight),
        "gamma": float(best_gamma),
        "subsample": float(best_subsample),
        "colsample_bytree": float(best_colsample_bytree),
        "lambda": float(best_lambda),
    }
    
    print(f"Best hyperparameters: {best_params_xgb}")
    return best_params_xgb