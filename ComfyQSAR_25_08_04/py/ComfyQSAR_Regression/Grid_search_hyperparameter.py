import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error # Removed make_scorer as not directly used here
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# Conditionally import XGBoost and LightGBM to avoid hard dependency
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None # Set to None if not installed
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None # Set to None if not installed

import multiprocessing
from sklearn.impute import SimpleImputer # Added for NaN/Inf handling

class Hyperparameter_Grid_Search_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        # Define available algorithms, checking for installed libraries
        available_algorithms = ["decision_tree", "random_forest", "svm", "ridge", "lasso", "elasticnet", "linear_regression"]
        if XGBRegressor:
             available_algorithms.insert(0, "xgboost") # Add if installed
        if LGBMRegressor:
             available_algorithms.insert(3, "lightgbm") # Add if installed

        return {
            "required": {
                "input_file": ("STRING",),
                "algorithm": (available_algorithms, {"default": available_algorithms[0]}), # Use dynamic list
                "target_column": ("STRING", {"default": "value"}), # Moved to required
                "advanced": ("BOOLEAN", {"default": False}), # Keep advanced toggle if UI uses it
            },
            "optional": {
                # Common parameters
                "test_size": ("FLOAT", {"default": 0.2, "min": 0.1, "max": 0.5, "step": 0.05}),
                "num_cores": ("INT", {"default": -1, "min": -1, "max": multiprocessing.cpu_count(), "step": 1}), # Default -1 (all cores)
                "cv_splits": ("INT", {"default": 5, "min": 3, "max": 10, "step": 1}),
                "verbose": ("INT", {"default": 0, "min": 0, "max": 3, "step": 1}), # Adjusted max for GridSearchCV verbosity levels
                "random_state": ("INT", {"default": 42, "min": 0, "max": 999, "step": 1}),

                # Hyperparameters (as strings for flexibility)
                "n_estimators": ("STRING", {"default": "[100, 200, 300]"}),
                "max_depth": ("STRING", {"default": "[-1, 5, 10, 20]"}), # -1 for unlimited
                "learning_rate": ("STRING", {"default": "[0.01, 0.1, 0.2]"}),
                "min_samples_split": ("STRING", {"default": "[2, 5, 10]"}),
                "min_samples_leaf": ("STRING", {"default": "[1, 3, 5]"}),
                "criterion": ("STRING", {"default": "['squared_error', 'friedman_mse']"}), # For DT/RF
                "num_leaves": ("STRING", {"default": "[20, 31, 40]"}), # For LGBM
                "subsample": ("STRING", {"default": "[0.8, 1.0]"}), # For XGB/LGBM
                "reg_alpha": ("STRING", {"default": "[0, 0.1, 1.0]"}), # For XGB/LGBM/Ridge/Lasso/EN (Lasso uses this name)
                "reg_lambda": ("STRING", {"default": "[0, 1.0, 10.0]"}), # For XGB/LGBM/Ridge (Ridge uses this name)
                "l1_ratio": ("STRING", {"default": "[0.1, 0.5, 0.9]"}), # For ElasticNet
                "C": ("STRING", {"default": "[0.1, 1, 10]"}), # For SVM
                "kernel": ("STRING", {"default": "['rbf', 'linear']"}), # For SVM
                "gamma": ("STRING", {"default": "['scale', 'auto']"}), # For SVM (rbf kernel)
                "epsilon": ("STRING", {"default": "[0.1, 0.2]"}), # For SVM
            },
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    # Updated names for consistency
    RETURN_NAMES = ("BEST_MODEL_PATH", "FINAL_DESCRIPTORS_PATH", "X_TEST_SET_PATH", "Y_TEST_SET_PATH")
    FUNCTION = "grid_search_regression_models"
    CATEGORY = "QSAR/REGRESSION/GRID SEARCH"
    OUTPUT_NODE = True
    
    # Simplified signature using **kwargs for optional params
    def grid_search_regression_models(self, input_file, algorithm, target_column, advanced=False, **kwargs):
        output_dir = "QSAR/Grid_Search_Hyperparameter"
        model_path, descriptors_path, X_test_path, y_test_path = "", "", "", "" # Default empty paths

        try:
            os.makedirs(output_dir, exist_ok=True)

            # --- Parameter Extraction and Defaults ---
            test_size = kwargs.get('test_size', 0.2)
            num_cores = kwargs.get('num_cores', -1)
            cv_splits = kwargs.get('cv_splits', 5)
            verbose = kwargs.get('verbose', 0) # Default to less verbose
            random_state = kwargs.get('random_state', 42)

            # --- Parameter Parsing Helper ---
            def parse_param(param_name, default_str_value):
                 param_str = kwargs.get(param_name, default_str_value)
                 try:
                     # Safely evaluate the string list/value
                     # Replace 'None' string with actual None before eval
                     if isinstance(param_str, str):
                          safe_str = param_str.replace("None", "None")
                          parsed = eval(safe_str)
                          # Ensure list for grid search (even if single value provided like "[100]")
                          if not isinstance(parsed, list):
                               return [parsed]
                          return parsed
                     else: # If already parsed (e.g., direct non-string input if UI changes)
                          if not isinstance(param_str, list):
                               return [param_str]
                          return param_str
                 except Exception as e:
                     # Fallback to evaluating the default string
                     try:
                          safe_default_str = default_str_value.replace("None", "None")
                          parsed_default = eval(safe_default_str)
                          if not isinstance(parsed_default, list):
                               return [parsed_default]
                          return parsed_default
                     except Exception as de:
                          return [] # Return empty list as a last resort

            # --- Data Loading and Preparation ---
            data = pd.read_csv(input_file)
            initial_rows, initial_cols = data.shape

            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in the dataset.")

            X = data.drop(columns=[target_column], errors='ignore')
            y = data[target_column]

            # Keep track of SMILES if present, but remove from features for modeling
            smiles_col_data = None
            if "SMILES" in X.columns:
                 smiles_col_data = X["SMILES"]
                 X = X.drop(columns=["SMILES"])

            X_numeric = X.select_dtypes(include=np.number)
            non_numeric_cols = X.columns.difference(X_numeric.columns)
            if not non_numeric_cols.empty:
                raise ValueError(f"Dropped non-numeric columns: {non_numeric_cols.tolist()}")
            X = X_numeric

            if X.empty:
                raise ValueError("No numeric feature columns remain after dropping target and non-numeric columns.")

            # Handle NaN/Inf (using median imputation)
            X = X.replace([np.inf, -np.inf], np.nan)
            if X.isnull().values.any():
                 imputer = SimpleImputer(strategy='median')
                 X_imputed = imputer.fit_transform(X)
                 X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            # Save final feature list *before* pipeline scaling
            final_descriptors_list = X_train.columns.tolist()


            # --- Model and Parameter Grid Setup ---
            pipeline_steps = []
            param_grid = {}
            model_abbr = ""

            # Add scaler for all models except potentially trees (though it usually doesn't hurt trees)
            pipeline_steps.append(("scaler", StandardScaler()))

            if algorithm == "xgboost":
                if not XGBRegressor: raise ImportError("XGBoost library is not installed.")
                pipeline_steps.append(("reg", XGBRegressor(objective="reg:squarederror", random_state=random_state, n_jobs=1))) # n_jobs in XGB is for internal threading
                model_abbr = "XGB"
                param_grid = {
                    'reg__n_estimators': parse_param('n_estimators', "[100, 200, 300]"),
                    'reg__learning_rate': parse_param('learning_rate', "[0.01, 0.1, 0.2]"),
                    'reg__max_depth': parse_param('max_depth', "[-1, 5, 10, 20]"),
                    'reg__subsample': parse_param('subsample', "[0.8, 1.0]"),
                    'reg__reg_alpha': parse_param('reg_alpha', "[0, 0.1, 1.0]"), # L1
                    'reg__reg_lambda': parse_param('reg_lambda', "[0, 1.0, 10.0]"), # L2
                }
            elif algorithm == "random_forest":
                 pipeline_steps.append(("reg", RandomForestRegressor(random_state=random_state, n_jobs=-1))) # n_jobs for sklearn RF
                 model_abbr = "RF"
                 param_grid = {
                     'reg__n_estimators': parse_param('n_estimators', "[100, 200, 300]"),
                     'reg__max_depth': parse_param('max_depth', "[-1, 5, 10, 20]"),
                     'reg__min_samples_split': parse_param('min_samples_split', "[2, 5, 10]"),
                     'reg__min_samples_leaf': parse_param('min_samples_leaf', "[1, 3, 5]"),
                     'reg__criterion': parse_param('criterion', "['squared_error', 'friedman_mse']"),
                 }
            elif algorithm == "decision_tree":
                 pipeline_steps.append(("reg", DecisionTreeRegressor(random_state=random_state)))
                 model_abbr = "DT"
                 param_grid = {
                     'reg__max_depth': parse_param('max_depth', "[-1, 5, 10, 20]"),
                     'reg__min_samples_split': parse_param('min_samples_split', "[2, 5, 10]"),
                     'reg__min_samples_leaf': parse_param('min_samples_leaf', "[1, 3, 5]"),
                     'reg__criterion': parse_param('criterion', "['squared_error', 'friedman_mse']"),
                 }
            elif algorithm == "lightgbm":
                 if not LGBMRegressor: raise ImportError("LightGBM library is not installed.")
                 pipeline_steps.append(("reg", LGBMRegressor(random_state=random_state, n_jobs=-1, verbosity=-1))) # n_jobs for LGBM
                 model_abbr = "LGBM"
                 param_grid = {
                     'reg__n_estimators': parse_param('n_estimators', "[100, 200, 300]"),
                     'reg__learning_rate': parse_param('learning_rate', "[0.01, 0.1, 0.2]"),
                     'reg__num_leaves': parse_param('num_leaves', "[20, 31, 40]"),
                     'reg__max_depth': parse_param('max_depth', "[-1, 5, 10, 20]"),
                     'reg__reg_alpha': parse_param('reg_alpha', "[0, 0.1, 1.0]"), # L1
                     'reg__reg_lambda': parse_param('reg_lambda', "[0, 1.0, 10.0]"), # L2
                     'reg__subsample': parse_param('subsample', "[0.8, 1.0]"),
                 }
            elif algorithm == "svm":
                 pipeline_steps.append(("reg", SVR()))
                 model_abbr = "SVM"
                 param_grid = {
                     'reg__C': parse_param('C', "[0.1, 1, 10]"),
                     'reg__kernel': parse_param('kernel', "['rbf', 'linear']"),
                     'reg__gamma': parse_param('gamma', "['scale', 'auto']"),
                     'reg__epsilon': parse_param('epsilon', "[0.1, 0.2]"),
                 }
            elif algorithm == "ridge":
                 pipeline_steps.append(("reg", Ridge(random_state=random_state)))
                 model_abbr = "Ridge"
                 param_grid = {'reg__alpha': parse_param('reg_lambda', "[0, 1.0, 10.0]")} # Ridge uses 'alpha' but we map from reg_lambda UI param
            elif algorithm == "lasso":
                 pipeline_steps.append(("reg", Lasso(random_state=random_state, max_iter=5000))) # Increase max_iter
                 model_abbr = "LASSO"
                 param_grid = {'reg__alpha': parse_param('reg_alpha', "[0, 0.1, 1.0]")} # Lasso uses 'alpha'
            elif algorithm == "elasticnet":
                 pipeline_steps.append(("reg", ElasticNet(random_state=random_state, max_iter=5000))) # Increase max_iter
                 model_abbr = "EN"
                 param_grid = {
                     'reg__alpha': parse_param('reg_alpha', "[0, 0.1, 1.0]"), # Combined penalty
                     'reg__l1_ratio': parse_param('l1_ratio', "[0.1, 0.5, 0.9]") # Mix ratio
                 }
            elif algorithm == "linear_regression":
                 pipeline_steps.append(("reg", LinearRegression()))
                 model_abbr = "LR"
                 param_grid = {} # No hyperparameters
            else:
                 raise ValueError(f"Unknown algorithm: {algorithm}.")

            # Create the full pipeline
            pipeline = Pipeline(pipeline_steps)

            # Filter empty param lists (can happen if parsing fails badly)
            param_grid = {k: v for k, v in param_grid.items() if v}
            if not param_grid and algorithm != "linear_regression":
                raise ValueError(f"Parameter grid for {algorithm} is empty after parsing. Check input strings. Running with defaults.")

            # --- Grid Search Execution ---
            # Use more standard scoring names
            scoring = {'R2': 'r2', 'Neg_MSE': 'neg_mean_squared_error'}
            cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

            grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring,
                                       refit="R2", return_train_score=True,
                                       verbose=verbose, n_jobs=num_cores, error_score='raise') # Raise error on failure

            try:
                grid_search.fit(X_train, y_train)
            except Exception as gs_e:
                import traceback
                error_detail = f"GridSearchCV failed: {str(gs_e)}\n{traceback.format_exc()}"
                raise RuntimeError(error_detail)

            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # --- Evaluation ---
            predictions = best_model.predict(X_test)
            eval_results = {
                "r2_score": r2_score(y_test, predictions),
                "mse": mean_squared_error(y_test, predictions),
                "mae": mean_absolute_error(y_test, predictions),
            }
            eval_results["rmse"] = np.sqrt(eval_results["mse"]) # Calculate RMSE

            # --- Save Artifacts ---
            # 1. Best Model
            model_filename = f"Regression_Best_Model_{model_abbr}.pkl"
            model_path = os.path.join(output_dir, model_filename)
            joblib.dump(best_model, model_path)

            # 2. Final Descriptor List (used for training)
            descriptors_filename = "Final_Selected_Descriptors_Regression.txt"
            descriptors_path = os.path.join(output_dir, descriptors_filename)
            with open(descriptors_path, "w") as f:
                 f.write("\n".join(final_descriptors_list))

            # 3. Test Set (Features and Target)
            X_test_filename = "Regression_X_test.csv"
            y_test_filename = "Regression_y_test.csv"
            X_test_path = os.path.join(output_dir, X_test_filename)
            y_test_path = os.path.join(output_dir, y_test_filename)
            X_test.to_csv(X_test_path, index=False)
            pd.DataFrame(y_test).to_csv(y_test_path, index=False) # Save y_test as DataFrame

            # 4. Best Hyperparameters
            best_params_filename = f"Best_Hyperparameters_{model_abbr}.txt"
            best_params_path = os.path.join(output_dir, best_params_filename)
            with open(best_params_path, "w") as f:
                 for param, value in best_params.items():
                      f.write(f"{param}: {value}\n")

            # --- Generate Summary ---
            cv_results_df = pd.DataFrame(grid_search.cv_results_)
            # Handle potential missing score keys robustly
            mean_r2_cv = cv_results_df['mean_test_R2'].iloc[grid_search.best_index_] if 'mean_test_R2' in cv_results_df.columns else 'N/A'
            mean_mse_cv = -cv_results_df['mean_test_Neg_MSE'].iloc[grid_search.best_index_] if 'mean_test_Neg_MSE' in cv_results_df.columns else 'N/A' # Use negative score

            # Best hyperparameters 형식화
            param_lines = [f"  - {k.replace('reg__', '')}: {v}" for k, v in best_params.items()]
            if not param_lines: param_lines = ["  - (Defaults Used)"]
            best_params_formatted = "\n".join(param_lines)
            
            log_message = f"Input File: {os.path.basename(input_file)}\nAlgorithm: {algorithm}\nCV Mean R² (best params): {mean_r2_cv:.4f}\nCV Mean MSE (best params): {mean_mse_cv:.4f}\nTest R² Score: {eval_results['r2_score']:.4f}\nTest MSE: {eval_results['mse']:.4f}\nTest RMSE: {eval_results['rmse']:.4f}\nTest MAE: {eval_results['mae']:.4f}\nBest Hyperparameters:\n{best_params_formatted}"

            return {"ui": {"text": log_message},
                    "result": (str(model_path), str(descriptors_path), str(X_test_path), str(y_test_path))}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}."
            return {"ui": {"text": error_msg}, "result": ("", "", "", "")}
        except ValueError as ve:
             error_msg = f"❌ Value Error: {str(ve)}"
             return {"ui": {"text": error_msg}, "result": ("", "", "", "")}
        except ImportError as ie:
             error_msg = f"❌ Import Error: {str(ie)}. Please ensure required libraries (e.g., xgboost, lightgbm) are installed."
             return {"ui": {"text": error_msg}, "result": ("", "", "", "")}
        except RuntimeError as rte: # Catch runtime errors like from GridSearchCV
             error_msg = f"❌ Runtime Error: {str(rte)}"
             return {"ui": {"text": error_msg}, "result": ("", "", "", "")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": error_msg}, "result": ("", "", "", "")}


NODE_CLASS_MAPPINGS = {
    "Hyperparameter_Grid_Search_Regression": Hyperparameter_Grid_Search_Regression,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Keep display name as is, maybe add "(Regression)" for extra clarity if needed elsewhere
    "Hyperparameter_Grid_Search_Regression": "Grid Search Hyperparameter (Regression)"
}