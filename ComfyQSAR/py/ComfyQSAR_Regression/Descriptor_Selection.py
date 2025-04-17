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

# --- Common Utility Import ---
try:
    from .Data_Loader import send_progress, create_text_container
except ImportError:
    print("[ComfyQSAR Regression Selection] Warning: Could not import progress_utils. Progress updates might not work.")
    # Fallback functions
    def send_progress(message, progress=None, node_id=None):
        print(f"[Progress Fallback] {message}" + (f" ({progress}%)" if progress is not None else ""))
    def create_text_container(*lines):
        return "\n".join(lines)

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
                "threshold": ("STRING", {"default": "0.01"}),
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
    CATEGORY = "QSAR/REGRESSION/SELECTION"
    OUTPUT_NODE = True
    
    def select_descriptors(self, input_file, method, target_column,
                             advanced=False, n_features=10, threshold="0.01",
                             alpha=0.01, max_iter=1000,
                             n_estimators=100, max_depth=-1, min_samples_split=2, criterion="squared_error",
                             learning_rate=0.1, n_iterations=10, base_model_for_selection="RandomForest"):
        send_progress(f"🚀 Starting Feature Selection (Regression - Method: {method})...", 0)
        output_dir = "QSAR/Descriptor_Selection"
        output_file = ""
        initial_feature_count, final_feature_count = 0, 0
        model_name_for_log = method
        selected_columns = []

        try:
            os.makedirs(output_dir, exist_ok=True)
            send_progress(f"📂 Output directory checked/created: {output_dir}", 2)

            send_progress(f"⏳ Loading data from: {input_file}", 5)
            data = pd.read_csv(input_file)
            initial_rows, initial_cols_total = data.shape
            send_progress(f"   Data loaded ({initial_rows} rows, {initial_cols_total} columns).", 8)

            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in the dataset.")

            send_progress("⚙️ Separating features and target variable...", 10)
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
                 send_progress(f"   ⚠️ Warning: Dropping non-numeric columns from features: {list(non_numeric_cols)}", 11)
            X = X_numeric
            initial_feature_count = X.shape[1]
            send_progress(f"   Using {initial_feature_count} numeric features.", 12)

            if initial_feature_count == 0:
                raise ValueError("No numeric feature columns found after removing target and non-numeric columns.")

            # Handle potential NaNs/Infs in features (using median imputation as a default strategy)
            send_progress("⚙️ Handling potential NaN/Inf values in features (median imputation)...", 13)
            X = X.replace([np.inf, -np.inf], np.nan)
            if X.isnull().values.any():
                 imputer = SimpleImputer(strategy='median')
                 X_imputed = imputer.fit_transform(X)
                 X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
                 send_progress("   NaN/Inf imputation complete.", 14)
            else:
                 send_progress("   No NaN/Inf values found in numeric features.", 14)


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

            send_progress("⚙️ Initializing model for feature selection...", 15)
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
            send_progress(f"   Model initialized: {model_name_for_log}", 18)


            # --- Data Scaling (if required) ---
            scaler = None
            if model_requires_scaling or method == "RFE":
                 send_progress("⚖️ Scaling features using StandardScaler...", 19)
                 scaler = StandardScaler()
                 X_scaled = scaler.fit_transform(X)
                 X_train = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
                 send_progress("   Scaling complete.", 20)
            else:
                 X_train = X
                 send_progress("   Scaling not required for this model/method.", 20)


            # --- Feature Selection Execution ---
            send_progress(f"⏳ Performing feature selection using {model_name_for_log}...", 25)
            fit_progress_start = 25
            fit_progress_end = 75

            if method == "Lasso":
                try:
                    model.fit(X_train, y)
                    send_progress("   Model fitting complete.", fit_progress_end)
                    importances = np.abs(model.coef_)
                    selected_mask = importances > 1e-6
                    selected_columns = X.columns[selected_mask]
                    if len(selected_columns) == 0:
                         send_progress("   ⚠️ Warning: Lasso removed all features. Check alpha or data.", fit_progress_end + 1)
                    else:
                         send_progress(f"   Selected {len(selected_columns)} features based on non-zero coefficients.", fit_progress_end + 1)

                except Exception as e:
                     raise RuntimeError(f"Error during Lasso fitting: {e}")

            elif method in ["RandomForest", "DecisionTree", "XGBoost", "LightGBM"]:
                if method in ["RandomForest", "DecisionTree"] and n_iterations > 1:
                     send_progress(f"   Averaging feature importances over {n_iterations} iterations...", fit_progress_start + 5)
                     feature_importance_matrix = np.zeros((n_iterations, initial_feature_count))
                     for i in range(n_iterations):
                          iter_progress = int(fit_progress_start + 5 + (i / n_iterations) * (fit_progress_end - fit_progress_start - 10))
                          send_progress(f"      Iteration {i+1}/{n_iterations}...", iter_progress)
                          iter_model = create_model_instance(method)
                          try:
                               iter_model.fit(X_train, y)
                               feature_importance_matrix[i] = iter_model.feature_importances_
                          except Exception as e:
                               raise RuntimeError(f"Error during {method} fitting (iteration {i+1}): {e}")
                     feature_importances = np.mean(feature_importance_matrix, axis=0)
                     send_progress("   Finished averaging importances.", fit_progress_end - 5)
                else:
                     try:
                          model.fit(X_train, y)
                          send_progress("   Model fitting complete.", fit_progress_end - 5)
                          feature_importances = model.feature_importances_
                     except Exception as e:
                          raise RuntimeError(f"Error during {method} fitting: {e}")

                try:
                     thresh_val = float(threshold)
                     send_progress(f"   Selecting features with importance >= {thresh_val:.4f}", fit_progress_end)
                except ValueError:
                     if threshold.lower() == "median":
                          thresh_val = np.median(feature_importances[feature_importances > 0])
                          send_progress(f"   Selecting features with importance >= median ({thresh_val:.4f})", fit_progress_end)
                     elif threshold.lower() == "mean":
                          thresh_val = np.mean(feature_importances[feature_importances > 0])
                          send_progress(f"   Selecting features with importance >= mean ({thresh_val:.4f})", fit_progress_end)
                     else:
                          try:
                               perc = float(threshold.replace('%', ''))
                               thresh_val = np.percentile(feature_importances[feature_importances > 0], perc)
                               send_progress(f"   Selecting features with importance >= {perc:.0f}th percentile ({thresh_val:.4f})", fit_progress_end)
                          except:
                               raise ValueError(f"Invalid threshold string: '{threshold}'. Use float, 'median', 'mean', or 'N%'.")

                selected_mask = feature_importances >= thresh_val
                selected_columns = X.columns[selected_mask]
                send_progress(f"   Selected {len(selected_columns)} features based on importance threshold.", fit_progress_end + 1)

            elif method == "RFE":
                try:
                    actual_n_features = min(n_features, initial_feature_count)
                    if actual_n_features < n_features:
                        send_progress(f"   ⚠️ Warning: Requested {n_features} features, but only {initial_feature_count} available. Selecting {actual_n_features}.", fit_progress_start + 5)

                    send_progress(f"   Running RFE to select {actual_n_features} features...", fit_progress_start + 10)
                    selector = RFE(estimator=model, n_features_to_select=actual_n_features, step=0.1)
                    selector.fit(X_train, y)
                    send_progress("   RFE fitting complete.", fit_progress_end)
                    selected_mask = selector.get_support()
                    selected_columns = X.columns[selected_mask]
                    send_progress(f"   Selected {len(selected_columns)} features via RFE.", fit_progress_end + 1)
                except Exception as e:
                    raise RuntimeError(f"Error during RFE execution: {e}")

            elif method == "SelectFromModel":
                try:
                     send_progress(f"   Fitting base model ({base_model_for_selection}) for SelectFromModel...", fit_progress_start + 5)
                     model.fit(X_train, y)
                     send_progress("   Base model fitting complete.", int(fit_progress_start + (fit_progress_end - fit_progress_start)/2))

                     try:
                          thresh_val_sfm = float(threshold)
                          send_progress(f"   Using numeric threshold for selection: {thresh_val_sfm:.4f}", fit_progress_end - 5)
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
                               send_progress(f"   Using 'median' threshold ({median_val:.4f}) for selection.", fit_progress_end - 5)
                          elif threshold.lower() == "mean":
                               sfm_threshold_arg = "mean"
                               mean_val = np.mean(importances_sfm[importances_sfm > 0])
                               send_progress(f"   Using 'mean' threshold ({mean_val:.4f}) for selection.", fit_progress_end - 5)
                          else:
                               try:
                                    if '%' in threshold:
                                         perc_sfm = float(threshold.replace('%',''))
                                         thresh_val_sfm = np.percentile(importances_sfm[importances_sfm > 0], perc_sfm)
                                         sfm_threshold_arg = thresh_val_sfm
                                         send_progress(f"   Using {perc_sfm:.0f}th percentile threshold ({thresh_val_sfm:.4f}) for selection.", fit_progress_end - 5)
                                    else:
                                         sfm_threshold_arg = threshold
                                         send_progress(f"   Using scaled threshold string '{threshold}' for selection.", fit_progress_end - 5)

                               except Exception as te:
                                     raise ValueError(f"Invalid threshold string '{threshold}': {te}. Use float, 'median', 'mean', 'N%', or 'factor*mean/median'.")

                     selector = SelectFromModel(estimator=model, threshold=sfm_threshold_arg, prefit=True)
                     selected_mask = selector.get_support()
                     selected_columns = X.columns[selected_mask]
                     send_progress(f"   Selected {len(selected_columns)} features via SelectFromModel.", fit_progress_end + 1)
                except Exception as e:
                     raise RuntimeError(f"Error during SelectFromModel execution: {e}")


            # --- Process Results ---
            final_feature_count = len(selected_columns)
            removed_features_count = initial_feature_count - final_feature_count
            send_progress(f"📊 Feature selection processing complete. Initial: {initial_feature_count}, Final: {final_feature_count}, Removed: {removed_features_count}", 80)

            if final_feature_count == 0:
                 send_progress("   ⚠️ Warning: No features were selected by the chosen method and parameters.", 81)
                 selected_features_df = pd.DataFrame({target_column: y})
                 if smiles_col is not None:
                      selected_features_df["SMILES"] = smiles_col
            else:
                 X_new = X[selected_columns]
                 selected_features_df = X_new.copy()
                 selected_features_df[target_column] = y.reset_index(drop=True)
                 if smiles_col is not None:
                      selected_features_df.insert(0, "SMILES", smiles_col.reset_index(drop=True))

            send_progress("💾 Saving selected features data...", 85)
            model_id_for_file = model_abbr if model_abbr else method
            if method in ["RFE", "SelectFromModel"]:
                model_id_for_file += f"_{base_model_for_selection}"

            filename = f"regression_selected_{method}_{model_id_for_file}_{initial_feature_count}_to_{final_feature_count}.csv"
            output_file = os.path.join(output_dir, filename)
            selected_features_df.to_csv(output_file, index=False)
            send_progress(f"   Selected features saved to: {output_file}", 90)

            # --- Generate Summary ---
            send_progress("📝 Generating summary...", 95)
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
            text_container_content = create_text_container(*summary_lines)
            send_progress("🎉 Feature selection process finished successfully!", 100)

            return {"ui": {"text": text_container_content},
                    "result": (str(output_file),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": ("",)}
        except ValueError as ve:
            error_msg = f"❌ Value Error: {str(ve)}"
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": ("",)}
        except ImportError as ie:
             error_msg = f"❌ Import Error: {str(ie)}. Please ensure required libraries (e.g., xgboost, lightgbm) are installed."
             send_progress(error_msg)
             return {"ui": {"text": create_text_container(error_msg)}, "result": ("",)}
        except RuntimeError as rte:
            error_msg = f"❌ Runtime Error during model fitting/selection: {str(rte)}"
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": ("",)}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": ("",)}


NODE_CLASS_MAPPINGS = {
    "Feature_Selection_Regression": Feature_Selection_Regression
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Feature_Selection_Regression": "Feature Selection (Regression)"
}