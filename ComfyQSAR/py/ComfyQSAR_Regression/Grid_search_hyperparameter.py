import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error, make_scorer
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import multiprocessing
from .Data_Loader import create_text_container

class Hyperparameter_Grid_Search_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING",),
                "algorithm": (["xgboost", "random_forest", "decision_tree", "lightgbm", "svm", "ridge", "lasso", "elasticnet", "linear_regression"],),
                "advanced": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                # Common parameters
                "target_column": ("STRING", {"default": "value", "description": "Target column name in the dataset"}),
                "test_size": ("FLOAT", {"default": 0.2, "min": 0.1, "max": 0.5, "step": 0.05, "description": "Proportion of data used for testing"}),
                "num_cores": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1, "description": "Number of CPU cores to use"}),
                "cv_splits": ("INT", {"default": 5, "min": 3, "max": 10, "step": 1, "description": "Number of cross-validation folds"}),
                "verbose": ("INT", {"default": 1, "min": 0, "max": 2, "step": 1, "description": "Verbosity level"}),
                "random_state": ("INT", {"default": 42, "min": 0, "max": 999, "step": 1, "description": "Random seed for reproducibility"}),
                
                # Tree-based model parameters
                "n_estimators": ("STRING", {"default": "[50, 100, 200, 300, 500]", "description": "Number of trees in ensemble models (XGBoost, Random Forest, LightGBM)"}),
                "max_depth": ("STRING", {"default": "[None, 3, 5, 7, 10, 20]", "description": "Maximum depth of trees (XGBoost, Random Forest, Decision Tree, LightGBM)"}),
                "learning_rate": ("STRING", {"default": "[0.01, 0.05, 0.1]", "description": "Learning rate for boosting models (XGBoost, LightGBM)"}),
                
                # Detailed tree parameters
                "min_samples_split": ("STRING", {"default": "[2, 5, 10]", "description": "Minimum samples required to split (Random Forest, Decision Tree)"}),
                "min_samples_leaf": ("STRING", {"default": "[1, 2, 4]", "description": "Minimum samples required in leaf (Decision Tree)"}),
                "criterion": ("STRING", {"default": "['squared_error', 'friedman_mse']", "description": "Function to measure split quality (Decision Tree)"}),
                
                # LightGBM specific parameters
                "num_leaves": ("STRING", {"default": "[20, 30, 40]", "description": "Maximum number of leaves (LightGBM)"}),
                "subsample": ("STRING", {"default": "[0.7, 1.0]", "description": "Subsample ratio of training data (XGBoost, LightGBM)"}),
                "reg_alpha": ("STRING", {"default": "[0, 0.1, 1, 10]", "description": "L1 regularization (XGBoost, LightGBM)"}),
                "reg_lambda": ("STRING", {"default": "[0, 1, 10, 100]", "description": "L2 regularization (XGBoost, LightGBM)"}),
                
                # Linear model parameters
                "alpha": ("STRING", {"default": "[0.01, 0.1, 1, 10, 100]", "description": "Regularization strength (Ridge, Lasso, ElasticNet)"}),
                "l1_ratio": ("STRING", {"default": "[0.1, 0.5, 0.9]", "description": "Mixing parameter for ElasticNet"}),
                
                # SVM parameters
                "C": ("STRING", {"default": "[0.1, 1, 10, 100]", "description": "Regularization parameter (SVM)"}),
                "kernel": ("STRING", {"default": "['linear', 'rbf']", "description": "Kernel type (SVM)"}),
                "gamma": ("STRING", {"default": "['scale', 'auto']", "description": "Kernel coefficient (SVM)"}),
                "epsilon": ("STRING", {"default": "[0.01, 0.1]", "description": "Epsilon in epsilon-SVR model (SVM)"})
            },
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("MODEL_PATH", "SELECTED_DESCRIPTORS_PATH", "X_TEST_PATH", "Y_TEST_PATH")
    FUNCTION = "grid_search_regression_models"
    CATEGORY = "QSAR/REGRESSION/GRID SEARCH"
    OUTPUT_NODE = True
    
    def grid_search_regression_models(self, input_file, algorithm, advanced, 
                                      test_size=0.2, num_cores=4, cv_splits=5, verbose=1, 
                                      target_column="value", random_state=42,
                                      # Algorithm hyperparameters
                                      n_estimators="[50, 100, 200, 300, 500]",
                                      max_depth="[None, 3, 5, 7, 10, 20]",
                                      learning_rate="[0.01, 0.05, 0.1]",
                                      min_samples_split="[2, 5, 10]",
                                      min_samples_leaf="[1, 2, 4]",
                                      criterion="['squared_error', 'friedman_mse']",
                                      num_leaves="[20, 30, 40]",
                                      subsample="[0.7, 1.0]",
                                      reg_alpha="[0, 0.1, 1, 10]",
                                      reg_lambda="[0, 1, 10, 100]",
                                      alpha="[0.01, 0.1, 1, 10, 100]",
                                      l1_ratio="[0.1, 0.5, 0.9]",
                                      C="[0.1, 1, 10, 100]",
                                      kernel="['linear', 'rbf']",
                                      gamma="['scale', 'auto']",
                                      epsilon="[0.01, 0.1]"):
        
        os.makedirs("QSAR/Grid_Search_Hyperparameter", exist_ok=True)

        # String parameter parsing helper function
        def parse_param(param_str):
            try:
                # Handle None string
                if "None" in param_str:
                    param_str = param_str.replace("None", "None")
                parsed = eval(param_str)
                return parsed
            except Exception as e:
                print(f"Parameter parsing error: {str(e)}")
                # Return default value
                if "None" in param_str:
                    return [None]
                return [0]

        # Load dataset
        data = pd.read_csv(input_file)

        # Ensure target column exists
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")

        # Train-Test Split
        X = data.drop(columns=[target_column])  
        y = data[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Set the number of CPU cores
        available_cores = max(1, multiprocessing.cpu_count())  
        num_cores = min(num_cores, available_cores)  

        # Initialize parameter grid
        param_grid = {}

        # Define Algorithm & Hyperparameter Grid
        if algorithm == "xgboost":
            model = XGBRegressor(objective="reg:squarederror", random_state=random_state)
            model_abbr = "XGB"
            # XGBoost parameters
            param_grid = {
                'n_estimators': parse_param(n_estimators),
                'learning_rate': parse_param(learning_rate),
                'max_depth': parse_param(max_depth),
                'subsample': parse_param(subsample),
                'reg_alpha': parse_param(reg_alpha),
                'reg_lambda': parse_param(reg_lambda),
            }
            
        elif algorithm == "random_forest":
            model = RandomForestRegressor(random_state=random_state)
            model_abbr = "RF"
            # Random Forest parameters
            param_grid = {
                'n_estimators': parse_param(n_estimators),
                'max_depth': parse_param(max_depth),
                'min_samples_split': parse_param(min_samples_split),
                'min_samples_leaf': parse_param(min_samples_leaf),
            }
            
        elif algorithm == "decision_tree":
            model = DecisionTreeRegressor(random_state=random_state)
            model_abbr = "DT"
            # Decision Tree parameters
            param_grid = {
                'max_depth': parse_param(max_depth),
                'min_samples_split': parse_param(min_samples_split),
                'min_samples_leaf': parse_param(min_samples_leaf),
                'criterion': parse_param(criterion),
            }
            
        elif algorithm == "lightgbm":
            model = LGBMRegressor(random_state=random_state)
            model_abbr = "LGBM"
            # LightGBM parameters
            param_grid = {
                'n_estimators': parse_param(n_estimators),
                'learning_rate': parse_param(learning_rate),
                'num_leaves': parse_param(num_leaves),
                'max_depth': parse_param(max_depth),
                'reg_alpha': parse_param(reg_alpha),
                'reg_lambda': parse_param(reg_lambda),
            }
            
        elif algorithm == "svm":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", SVR())
            ])
            model_abbr = "SVM"
            # SVM parameters
            param_grid = {
                'reg__C': parse_param(C),
                'reg__kernel': parse_param(kernel),
                'reg__gamma': parse_param(gamma),
                'reg__epsilon': parse_param(epsilon),
            }
            
        elif algorithm == "ridge":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", Ridge(random_state=random_state))
            ])
            model_abbr = "Ridge"
            # Ridge parameters
            param_grid = {'reg__alpha': parse_param(alpha)}
            
        elif algorithm == "lasso":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", Lasso(random_state=random_state))
            ])
            model_abbr = "LASSO"
            # Lasso parameters
            param_grid = {'reg__alpha': parse_param(alpha)}
            
        elif algorithm == "elasticnet":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", ElasticNet(random_state=random_state))
            ])
            model_abbr = "EN"
            # ElasticNet parameters
            param_grid = {
                'reg__alpha': parse_param(alpha), 
                'reg__l1_ratio': parse_param(l1_ratio)
            }
            
        elif algorithm == "linear_regression":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", LinearRegression())
            ])
            model_abbr = "LR"
            param_grid = {}  # No hyperparameters to tune for Linear Regression
            
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}.")
        
        # Debug parameter grid
        param_grid_str = "\n".join([f"{k}: {v}" for k, v in param_grid.items()])
        print(f"Algorithm: {algorithm}")
        print(f"Parameter Grid:\n{param_grid_str}")
                
        # Perform GridSearchCV with multiprocessing
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        
        scoring = {
            'R2': 'r2',
            'MSE': 'neg_mean_squared_error'
        }
        
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, 
                                  refit="R2", return_train_score=True, 
                                  verbose=verbose, n_jobs=num_cores)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # Extract mean R2 and MSE from GridSearch results
        cv_results_df = pd.DataFrame(grid_search.cv_results_)
        mean_r2 = cv_results_df['mean_test_R2'].mean()
        mean_mse = abs(cv_results_df['mean_test_MSE'].mean())

        # Best Model & Evaluation
        predictions = best_model.predict(X_test)
        eval_results = {
            "r2_score": r2_score(y_test, predictions),
            "mse": mean_squared_error(y_test, predictions),
            "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
            "mae": mean_absolute_error(y_test, predictions),
        }

        # Save Best Model
        model_path = os.path.join("QSAR/Grid_Search_Hyperparameter", f"QSAR_Best_Model_{model_abbr}.pkl")
        joblib.dump(best_model, model_path)

        # Save Selected Descriptor Features
        descriptors_path = os.path.join("QSAR/Grid_Search_Hyperparameter", "Final_Selected_Descriptors.txt")
        with open(descriptors_path, "w") as f:
            f.write("\n".join(X_train.columns))

        # Save Test Set
        X_test_path = os.path.join("QSAR/Grid_Search_Hyperparameter", "X_test.csv")
        y_test_path = os.path.join("QSAR/Grid_Search_Hyperparameter", "y_test.csv")
        X_test.to_csv(X_test_path, index=False)
        pd.DataFrame(y_test, columns=[target_column]).to_csv(y_test_path, index=False)

        # Save best hyperparameters
        best_params_path = os.path.join("QSAR/Grid_Search_Hyperparameter", f"Best_Hyperparameters_{model_abbr}.txt")
        with open(best_params_path, "w") as f:
            for param, value in best_params.items():
                f.write(f"{param}: {value}\n")

        text_container = create_text_container(
        "🔹 Regression Model Training Completed 🔹",
        f"📌 Best Algorithm: {algorithm}",
        f"📊 Average R² Score (CV): {mean_r2:.4f}",
        f"📊 Average MSE (CV): {mean_mse:.4f}",
        f"📊 Test R² Score: {eval_results['r2_score']:.4f}",
        f"📊 Test MSE: {eval_results['mse']:.4f}",
        f"📊 Test RMSE: {eval_results['rmse']:.4f}",
        f"📊 Test MAE: {eval_results['mae']:.4f}",
        f"🔧 Best Parameters:" + "\n".join([f"  - {k}: {v}" for k, v in best_params.items()]),
        )
        
        return {"ui": {"text": text_container},
                "result": (str(model_path), str(descriptors_path), str(X_test_path), str(y_test_path))}
    
NODE_CLASS_MAPPINGS = {
    "Hyperparameter_Grid_Search_Regression": Hyperparameter_Grid_Search_Regression,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hyperparameter_Grid_Search_Regression": "Grid Search Hyperparameter (Regression)"
}