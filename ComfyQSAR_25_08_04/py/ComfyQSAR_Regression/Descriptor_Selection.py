import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class Feature_Selection_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING",),
                "method": (["Lasso", "RandomForest", "DecisionTree", "XGBoost", "LightGBM", "RFE", "SelectFromModel"], {"default": "Lasso"}),
                "target_column": ("STRING", {"default": "value"}),
            },
            "optional": {
                "advanced": ("BOOLEAN", {"default": False}),
                "n_features": ("INT", {"default": 10, "min": 1, "max": 1000, "step": 1}),
                "threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alpha": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "max_iter": ("INT", {"default": 1000, "min": 100, "max": 10000, "step": 100}),
                "n_estimators": ("INT", {"default": 100, "min": 10, "max": 1000, "step": 10}),
                "max_depth": ("INT", {"default": -1, "min": -1, "max": 50, "step": 1}),
                "min_samples_split": ("INT", {"default": 2, "min": 2, "max": 20, "step": 1}),
                "criterion": (["squared_error", "absolute_error", "friedman_mse"], {"default": "squared_error"}),
                "learning_rate": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 1.0, "step": 0.01}),
                "n_iterations": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "base_model_for_selection": (["Lasso", "RandomForest", "DecisionTree", "XGBoost", "LightGBM"], {"default": "RandomForest"}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("SELECTED_FEATURES_PATH",)
    FUNCTION = "select_descriptors"
    CATEGORY = "QSAR/REGRESSION/OPTIMIZATION(Model-based)"
    OUTPUT_NODE = True
    
    def select_descriptors(self, input_file, method, target_column,
                             advanced=False, n_features=10, threshold="0.01",
                             alpha=0.01, max_iter=1000,
                             n_estimators=100, max_depth=-1, min_samples_split=2, criterion="squared_error",
                             learning_rate=0.1, n_iterations=10, base_model_for_selection="RandomForest"):
        output_dir = "QSAR/Descriptor_Selection"
        output_file = ""
        initial_feature_count, final_feature_count = 0, 0
        model_name_for_log = method
        selected_columns = []

        try:
            os.makedirs(output_dir, exist_ok=True)

            data = pd.read_csv(input_file)
            initial_rows, initial_cols_total = data.shape

            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in the dataset.")

            X = data.drop(columns=[target_column], errors='ignore')
            y = data[target_column]

            # Drop non-numeric columns from features (except SMILES if present, though it shouldn't be used for fitting)
            smiles_col = None
            if "SMILES" in X.columns:
                 smiles_col = X["SMILES"]
                 X = X.drop(columns=["SMILES"])

            X_numeric = X.select_dtypes(include=np.number)
            non_numeric_cols = X.columns.difference(X_numeric.columns)
            if not non_numeric_cols.empty:
                 raise ValueError(f"Dropped non-numeric columns: {non_numeric_cols.tolist()}")
            X = X_numeric
            initial_feature_count = X.shape[1]

            if initial_feature_count == 0:
                raise ValueError("No numeric feature columns found after removing target and non-numeric columns.")

            # Handle potential NaNs/Infs in features (using median imputation as a default strategy)
            X = X.replace([np.inf, -np.inf], np.nan)
            if X.isnull().values.any():
                 imputer = SimpleImputer(strategy='median')
                 X_imputed = imputer.fit_transform(X)
                 X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)


            # --- Model Initialization ---
            model = None
            model_abbr = ""
            model_requires_scaling = False

            # Helper function to create model instance
            def create_model_instance(model_type_str):
                nonlocal model_abbr, model_requires_scaling
                model_instance = None
                _model_abbr = ""
                _requires_scaling = False

                # Use adjusted max_depth if -1 (None for sklearn)
                adj_max_depth = None if max_depth == -1 else max_depth

                if model_type_str == "RandomForest":
                    model_instance = RandomForestRegressor(n_estimators=n_estimators, max_depth=adj_max_depth,
                                                           min_samples_split=min_samples_split, criterion=criterion,
                                                           random_state=42, n_jobs=-1)
                    _model_abbr = "RF"
                elif model_type_str == "DecisionTree":
                    model_instance = DecisionTreeRegressor(max_depth=adj_max_depth, min_samples_split=min_samples_split,
                                                           criterion=criterion, random_state=42)
                    _model_abbr = "DT"
                elif model_type_str == "XGBoost":
                     try:
                          model_instance = XGBRegressor(n_estimators=n_estimators, max_depth=adj_max_depth,
                                                        learning_rate=learning_rate, random_state=42, n_jobs=-1,
                                                        tree_method='hist')
                          _model_abbr = "XGB"
                     except Exception as e:
                          raise ImportError(f"Failed to initialize XGBoost. Is it installed? Error: {e}")
                elif model_type_str == "LightGBM":
                     try:
                          model_instance = LGBMRegressor(n_estimators=n_estimators, max_depth=adj_max_depth,
                                                         learning_rate=learning_rate, random_state=42, n_jobs=-1, verbosity=-1)
                          _model_abbr = "LGBM"
                     except Exception as e:
                          raise ImportError(f"Failed to initialize LightGBM. Is it installed? Error: {e}")
                elif model_type_str == "Lasso":
                    model_instance = Lasso(alpha=alpha, max_iter=max_iter, random_state=42, tol=0.001)
                    _model_abbr = "LASSO"
                    _requires_scaling = True
                else:
                     raise ValueError(f"Invalid model type '{model_type_str}'.")

                model_abbr = _model_abbr
                model_requires_scaling = _requires_scaling
                return model_instance

            if method == "Lasso":
                model = create_model_instance("Lasso")
                model_name_for_log = f"Lasso (alpha={alpha})"
            elif method in ["RandomForest", "DecisionTree", "XGBoost", "LightGBM"]:
                 model = create_model_instance(method)
                 model_name_for_log = f"{method}"
            elif method in ["RFE", "SelectFromModel"]:
                 model = create_model_instance(base_model_for_selection)
                 model_name_for_log = f"{method} (Base: {base_model_for_selection})"
            else:
                 raise ValueError(f"Unknown feature selection method: {method}")


            # --- Data Scaling (if required) ---
            scaler = None
            if model_requires_scaling or method == "RFE":
                 scaler = StandardScaler()
                 X_scaled = scaler.fit_transform(X)
                 X_train = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            else:
                 X_train = X


            # --- Feature Selection Execution --- 

            if method == "Lasso":
                try:
                    model.fit(X_train, y)
                    importances = np.abs(model.coef_)
                    selected_mask = importances > 1e-6
                    selected_columns = X.columns[selected_mask]
                    if len(selected_columns) == 0:
                         raise ValueError("Lasso removed all features. Check alpha or data.")
                    else:
                         raise ValueError(f"Selected {len(selected_columns)} features based on non-zero coefficients.")

                except Exception as e:
                     raise RuntimeError(f"Error during Lasso fitting: {e}")

            elif method in ["RandomForest", "DecisionTree", "XGBoost", "LightGBM"]:
                if method in ["RandomForest", "DecisionTree"] and n_iterations > 1:
                     feature_importance_matrix = np.zeros((n_iterations, initial_feature_count))
                     for i in range(n_iterations):
                          iter_model = create_model_instance(method)
                          try:
                               iter_model.fit(X_train, y)
                               feature_importance_matrix[i] = iter_model.feature_importances_
                          except Exception as e:
                               raise RuntimeError(f"Error during {method} fitting (iteration {i+1}): {e}")
                     feature_importances = np.mean(feature_importance_matrix, axis=0)
                else:
                     try:
                          model.fit(X_train, y)
                          feature_importances = model.feature_importances_
                     except Exception as e:
                          raise RuntimeError(f"Error during {method} fitting: {e}")

                try:
                     thresh_val = float(threshold)
                except ValueError:
                     if threshold.lower() == "median":
                          thresh_val = np.median(feature_importances[feature_importances > 0])
                     elif threshold.lower() == "mean":
                          thresh_val = np.mean(feature_importances[feature_importances > 0])
                     else:
                          try:
                               perc = float(threshold.replace('%', ''))
                               thresh_val = np.percentile(feature_importances[feature_importances > 0], perc)
                          except:
                               raise ValueError(f"Invalid threshold string: '{threshold}'. Use float, 'median', 'mean', or 'N%'.")

                selected_mask = feature_importances >= thresh_val
                selected_columns = X.columns[selected_mask]

            elif method == "RFE":
                try:
                    actual_n_features = min(n_features, initial_feature_count)
                    if actual_n_features < n_features:
                        raise ValueError(f"Requested {n_features} features, but only {initial_feature_count} available. Selecting {actual_n_features}.")

                    selector = RFE(estimator=model, n_features_to_select=actual_n_features, step=0.1)
                    selector.fit(X_train, y)
                    selected_mask = selector.get_support()
                    selected_columns = X.columns[selected_mask]
                except Exception as e:
                    raise RuntimeError(f"Error during RFE execution: {e}")

            elif method == "SelectFromModel":
                try:
                     model.fit(X_train, y)

                     try:
                          thresh_val_sfm = float(threshold)
                          sfm_threshold_arg = thresh_val_sfm
                     except ValueError:
                          if hasattr(model, "feature_importances_"):
                               importances_sfm = model.feature_importances_
                          elif hasattr(model, "coef_"):
                               importances_sfm = np.abs(model.coef_)
                          else:
                               raise ValueError(f"Base model {base_model_for_selection} for SelectFromModel doesn't have feature_importances_ or coef_.")

                          if threshold.lower() == "median":
                               sfm_threshold_arg = "median"
                               median_val = np.median(importances_sfm[importances_sfm > 0])
                          elif threshold.lower() == "mean":
                               sfm_threshold_arg = "mean"
                               mean_val = np.mean(importances_sfm[importances_sfm > 0])
                          else:
                               try:
                                    if '%' in threshold:
                                         perc_sfm = float(threshold.replace('%',''))
                                         thresh_val_sfm = np.percentile(importances_sfm[importances_sfm > 0], perc_sfm)
                                         sfm_threshold_arg = thresh_val_sfm
                                    else:
                                         sfm_threshold_arg = threshold

                               except Exception as te:
                                     raise ValueError(f"Invalid threshold string '{threshold}': {te}. Use float, 'median', 'mean', 'N%', or 'factor*mean/median'.")

                     selector = SelectFromModel(estimator=model, threshold=sfm_threshold_arg, prefit=True)
                     selected_mask = selector.get_support()
                     selected_columns = X.columns[selected_mask]
                except Exception as e:
                     raise RuntimeError(f"Error during SelectFromModel execution: {e}")


            # --- Process Results ---
            final_feature_count = len(selected_columns)
            removed_features_count = initial_feature_count - final_feature_count

            if final_feature_count == 0:
                 raise ValueError("No features were selected by the chosen method and parameters.")
                 selected_features_df = pd.DataFrame({target_column: y})
                 if smiles_col is not None:
                      selected_features_df["SMILES"] = smiles_col
            else:
                 X_new = X[selected_columns]
                 selected_features_df = X_new.copy()
                 selected_features_df[target_column] = y.reset_index(drop=True)
                 if smiles_col is not None:
                      selected_features_df.insert(0, "SMILES", smiles_col.reset_index(drop=True))

            model_id_for_file = model_abbr if model_abbr else method
            if method in ["RFE", "SelectFromModel"]:
                model_id_for_file += f"_{base_model_for_selection}"

            filename = f"regression_selected_{method}_{model_id_for_file}_{initial_feature_count}_to_{final_feature_count}.csv"
            output_file = os.path.join(output_dir, filename)
            selected_features_df.to_csv(output_file, index=False)

            # --- Generate Summary ---
            summary_lines = [
                "🔹 **Feature Selection Completed (Regression)!** 🔹",
                f"Input File: {os.path.basename(input_file)}",
                f"Method: {method}",
            ]
            if method in ["RFE", "SelectFromModel"]:
                 summary_lines.append(f"Base Model: {base_model_for_selection}")
            if method == "Lasso":
                 summary_lines.append(f"Lasso Alpha: {alpha}")
            if method == "RFE":
                 summary_lines.append(f"Features to Select (n_features): {n_features} (Actual selected: {final_feature_count})")
            if method in ["SelectFromModel", "RandomForest", "DecisionTree", "XGBoost", "LightGBM"]:
                 summary_lines.append(f"Selection Threshold: {threshold}")

            summary_lines.extend([
                f"Initial Features: {initial_feature_count}",
                f"Selected Features: {final_feature_count}",
                f"Features Removed: {removed_features_count}",
                f"Output File: {output_file}",
                f"Selected Columns: {', '.join(selected_columns) if final_feature_count > 0 else 'None'}"
            ])

            log_message = "\n".join(summary_lines)

            return {"ui": {"text": log_message},
                    "result": (str(output_file),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}."
            return {"ui": {"text": error_msg}, "result": ("",)}
        except ValueError as ve:
            error_msg = f"❌ Value Error: {str(ve)}"
            return {"ui": {"text": error_msg}, "result": ("",)}
        except ImportError as ie:
             error_msg = f"❌ Import Error: {str(ie)}. Please ensure required libraries (e.g., xgboost, lightgbm) are installed."
             return {"ui": {"text": error_msg}, "result": ("",)}
        except RuntimeError as rte:
            error_msg = f"❌ Runtime Error during model fitting/selection: {str(rte)}"
            return {"ui": {"text": error_msg}, "result": ("",)}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": error_msg}, "result": ("",)}


NODE_CLASS_MAPPINGS = {
    "Feature_Selection_Regression": Feature_Selection_Regression
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Feature_Selection_Regression": "Feature Selection (Regression)"
}