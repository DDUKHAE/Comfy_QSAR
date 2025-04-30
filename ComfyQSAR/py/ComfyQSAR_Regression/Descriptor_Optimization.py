import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# --- Common Utility Import ---
try:
    from .Data_Loader import send_progress, create_text_container
except ImportError:
    print("[ComfyQSAR Regression Optimization] Warning: Could not import progress_utils. Progress updates might not work.")
    # Fallback functions
    def send_progress(message, progress=None, node_id=None):
        print(f"[Progress Fallback] {message}" + (f" ({progress}%)" if progress is not None else ""))
    def create_text_container(*lines):
        return "\n".join(lines)

class Remove_Low_Variance_Descriptors_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING",),
                "threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("LOW_VAR_FILTERED_PATH",)
    FUNCTION = "remove_low_variance_descriptors"
    CATEGORY = "QSAR/REGRESSION/OPTIMIZATION"
    OUTPUT_NODE = True

    @staticmethod
    def remove_low_variance_descriptors(input_file, threshold):
        send_progress("🚀 Starting Low Variance Descriptor Removal (Regression)...", 0)
        output_dir = "QSAR/Descriptor_Optimization"
        output_file = ""
        initial_feature_count, final_feature_count = 0, 0

        try:
            os.makedirs(output_dir, exist_ok=True)
            send_progress(f"📂 Output directory checked/created: {output_dir}", 5)

            send_progress(f"⏳ Loading data from: {input_file}", 10)
            data = pd.read_csv(input_file)
            initial_rows, initial_cols_total = data.shape
            send_progress(f"   Data loaded ({initial_rows} rows, {initial_cols_total} columns).", 15)

            if "value" not in data.columns:
                raise ValueError ("The dataset must contain a 'value' column.")

            send_progress("⚙️ Separating features, target ('value'), and SMILES columns...", 20)
            target_column = data["value"]
            smiles_column = data["SMILES"] if "SMILES" in data.columns else None
            feature_columns = data.drop(columns=["value"] + ([smiles_column.name] if smiles_column is not None else []))
            initial_feature_count = feature_columns.shape[1]
            send_progress(f"   Found {initial_feature_count} initial descriptors.", 25)

            send_progress("⚙️ Handling potential NaN/Inf values in descriptors...", 30)
            feature_columns = feature_columns.replace([np.inf, -np.inf], np.nan)
            cols_with_nan = feature_columns.columns[feature_columns.isnull().any()]
            if not cols_with_nan.empty:
                send_progress(f"   Imputing NaNs in {len(cols_with_nan)} columns using median...", 35)
                for col in cols_with_nan:
                    median_val = feature_columns[col].median()
                    if pd.isna(median_val): median_val = 0
                    feature_columns[col].fillna(median_val, inplace=True)
                send_progress("   NaN imputation complete.", 40)
            else:
                send_progress("   No NaN values found to impute.", 40)

            send_progress(f"📉 Applying Variance Threshold (threshold = {threshold:.3f})...", 50)
            selector = VarianceThreshold(threshold=threshold)
            try:
                selected_features_array = selector.fit_transform(feature_columns)
                retained_columns = feature_columns.columns[selector.get_support()]
                final_feature_count = len(retained_columns)
                removed_count = initial_feature_count - final_feature_count
                send_progress(f"   Variance filtering complete. Kept {final_feature_count} descriptors, removed {removed_count}.", 70)
            except ValueError as ve:
                 if "No feature in X meets the variance threshold" in str(ve):
                      error_msg = f"❌ Error: No descriptors met the variance threshold {threshold}. Try a lower threshold."
                      send_progress(error_msg, 70)
                      return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
                 elif "Input contains NaN" in str(ve):
                      error_msg = f"❌ Error: Input still contained NaN/Inf before variance thresholding, despite imputation attempt."
                      send_progress(error_msg, 70)
                      return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
                 else:
                      raise ve

            send_progress("📊 Creating new DataFrame with selected descriptors...", 75)
            df_retained = pd.DataFrame(selected_features_array, columns=retained_columns, index=data.index)
            df_retained["value"] = target_column
            if smiles_column is not None:
                df_retained["SMILES"] = smiles_column
            send_progress("   DataFrame created.", 80)

            send_progress("💾 Saving filtered data...", 85)
            output_file = os.path.join(output_dir, f"regression_low_variance_filtered_{initial_feature_count}_to_{final_feature_count}.csv")
            df_retained.to_csv(output_file, index=False)
            send_progress(f"   Filtered data saved to: {output_file}", 90)

            send_progress("📝 Generating summary...", 95)
            text_container_content = create_text_container(
                "🔹 **Low Variance Descriptor Removal Complete (Regression)!** 🔹",
                f"Input File: {os.path.basename(input_file)}",
                f"Variance Threshold: {threshold:.3f}",
                f"Initial Descriptors: {initial_feature_count}",
                f"Descriptors Removed: {removed_count}",
                f"Remaining Descriptors: {final_feature_count}",
                f"Output File: {output_file}"
            )
            send_progress("🎉 Low variance removal process finished.", 100)

            return {"ui": {"text": text_container_content},
                    "result": (str(output_file),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except ValueError as ve:
             error_msg = f"❌ Value Error: {str(ve)}"
             send_progress(error_msg)
             return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}

class Remove_High_Correlation_Features_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING",),
                "threshold": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01}),
                "mode": (["target_based","upper", "lower"], {"default": "target_based"}),
                "importance_model": (["lasso", "random_forest",], {"default": "lasso"})
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("HIGH_CORR_FILTERED_PATH",)
    FUNCTION = "remove_high_correlation_features"
    CATEGORY = "QSAR/REGRESSION/OPTIMIZATION"
    OUTPUT_NODE = True
    
    @staticmethod
    def remove_high_correlation_features(input_file, threshold, mode, importance_model):
        send_progress("🚀 Starting High Correlation Feature Removal (Regression)...", 0)
        output_dir = "QSAR/Descriptor_Optimization"
        output_file = ""
        initial_feature_count, final_feature_count = 0, 0

        try:
            os.makedirs(output_dir, exist_ok=True)
            send_progress(f"📂 Output directory checked/created: {output_dir}", 2)

            send_progress(f"⏳ Loading data from: {input_file}", 5)
            data = pd.read_csv(input_file)
            initial_rows, initial_cols_total = data.shape
            send_progress(f"   Data loaded ({initial_rows} rows, {initial_cols_total} columns).", 8)

            if "value" not in data.columns:
                raise ValueError("The dataset must contain a 'value' column.")

            send_progress("⚙️ Separating features, target ('value'), and SMILES columns...", 10)
            target_column = data["value"]
            smiles_column = data["SMILES"] if "SMILES" in data.columns else None
            feature_columns = data.drop(columns=["value"] + ([smiles_column.name] if smiles_column is not None else []))
            initial_feature_count = feature_columns.shape[1]
            send_progress(f"   Found {initial_feature_count} initial descriptors.", 12)

            send_progress("⚙️ Handling potential NaN/Inf values in descriptors...", 15)
            feature_columns = feature_columns.replace([np.inf, -np.inf], np.nan)
            cols_with_nan = feature_columns.columns[feature_columns.isnull().any()]
            if not cols_with_nan.empty:
                send_progress(f"   Imputing NaNs in {len(cols_with_nan)} columns using median...", 18)
                for col in cols_with_nan:
                    median_val = feature_columns[col].median()
                    if pd.isna(median_val): median_val = 0
                    feature_columns[col].fillna(median_val, inplace=True)
                send_progress("   NaN imputation complete.", 20)
            else:
                send_progress("   No NaN values found to impute.", 20)

            send_progress("📊 Calculating correlation matrix...", 25)
            correlation_matrix = feature_columns.corr()
            send_progress("   Correlation matrix calculated.", 30)

            to_remove = set()
            feature_target_corr = None
            feature_importance = {}
            importance_calc_success = False

            if mode == "target_based":
                send_progress("🎯 Calculating correlation with target variable...", 35)
                feature_target_corr = feature_columns.corrwith(target_column).abs().fillna(0)
                send_progress("   Target correlation calculated.", 40)

                send_progress(f"🧠 Calculating feature importance using '{importance_model}'...", 45)
                X, y = feature_columns, target_column
                try:
                    if importance_model == "lasso":
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        model = Lasso(random_state=42)
                        try:
                            model.fit(X_scaled, y)
                            importance_values = np.abs(model.coef_)
                            importance_calc_success = True
                            send_progress("      LASSO importance calculated (alpha=1.0).", 55)
                        except Exception as e1:
                            send_progress(f"      LASSO (alpha=1.0) failed: {e1}. Trying alpha=0.01...", 55)
                            model = Lasso(alpha=0.01, max_iter=2000, random_state=42)
                            model.fit(X_scaled, y)
                            importance_values = np.abs(model.coef_)
                            importance_calc_success = True
                            send_progress("      LASSO importance calculated (alpha=0.01).", 55)

                    elif importance_model == "random_forest":
                        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                        model.fit(X, y)
                        importance_values = model.feature_importances_
                        importance_calc_success = True
                        send_progress("      RandomForest importance calculated.", 55)

                    if importance_calc_success:
                        feature_importance = dict(zip(feature_columns.columns, importance_values))
                except Exception as model_e:
                    send_progress(f"   ⚠️ Warning: Failed to calculate importance with {importance_model} ({model_e}). Correlation tie-breaking disabled.", 55)
                    feature_importance = {}

                send_progress("   Identifying descriptors to remove based on correlation > {threshold:.2f} and target correlation/importance...", 60)
                rows, cols = np.where(np.abs(np.triu(correlation_matrix, k=1)) > threshold)
                for r, c in zip(rows, cols):
                    f1, f2 = correlation_matrix.columns[r], correlation_matrix.columns[c]
                    if f1 in to_remove or f2 in to_remove: continue

                    corr1 = feature_target_corr.get(f1, 0)
                    corr2 = feature_target_corr.get(f2, 0)
                    imp1 = feature_importance.get(f1, 0)
                    imp2 = feature_importance.get(f2, 0)

                    if corr1 > corr2: weaker = f2
                    elif corr2 > corr1: weaker = f1
                    elif imp1 > imp2: weaker = f2
                    elif imp2 > imp1: weaker = f1
                    else: weaker = f2

                    to_remove.add(weaker)
                send_progress(f"   Identified {len(to_remove)} descriptors to remove.", 75)

            else:
                send_progress(f"✂️ Identifying descriptors to remove based on '{mode}' triangle and correlation > {threshold:.2f}...", 60)
                mask = np.triu(np.ones(correlation_matrix.shape, dtype=bool), k=1) if mode == "upper" else \
                       np.tril(np.ones(correlation_matrix.shape, dtype=bool), k=-1)
                high_corr_matrix = correlation_matrix.where(mask)
                to_remove = {col for col in high_corr_matrix.columns if (high_corr_matrix[col].abs() > threshold).any()}
                send_progress(f"   Identified {len(to_remove)} descriptors to remove.", 75)

            send_progress("📊 Creating new DataFrame with selected descriptors...", 80)
            retained_columns = [col for col in feature_columns.columns if col not in to_remove]
            df_retained = feature_columns[retained_columns].copy()
            df_retained["value"] = target_column
            if smiles_column is not None:
                df_retained["SMILES"] = smiles_column
            final_feature_count = len(retained_columns)
            removed_count = initial_feature_count - final_feature_count
            send_progress(f"   DataFrame created. Kept {final_feature_count} descriptors.", 85)

            send_progress("💾 Saving filtered data...", 90)
            output_file = os.path.join(output_dir, f"regression_high_corr_filtered_{initial_feature_count}_to_{final_feature_count}.csv")
            df_retained.to_csv(output_file, index=False)
            send_progress(f"   Filtered data saved to: {output_file}", 94)

            send_progress("📝 Generating summary...", 95)
            text_container_content = create_text_container(
                "🔹 **High Correlation Feature Removal Complete (Regression)!** 🔹",
                f"Input File: {os.path.basename(input_file)}",
                f"Correlation Threshold: {threshold:.2f}",
                f"Mode: {mode}",
                f"Importance Model (target_based): {importance_model if mode=='target_based' else 'N/A'}",
                f"Initial Descriptors: {initial_feature_count}",
                f"Descriptors Removed: {removed_count}",
                f"Remaining Descriptors: {final_feature_count}",
                f"Output File: {output_file}"
            )
            send_progress("🎉 High correlation removal process finished.", 100)

            return {"ui": {"text": text_container_content},
                    "result": (str(output_file),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except ValueError as ve:
             error_msg = f"❌ Value Error: {str(ve)}"
             send_progress(error_msg)
             return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}

class Descriptor_Optimization_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING",),
                "variance_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "correlation_threshold": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01}),
                "correlation_mode": (["target_based", "upper", "lower"], {"default": "target_based"}),
                "importance_model": (["lasso", "random_forest",], {"default": "lasso"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("OPTIMIZED_DATA_PATH",)
    FUNCTION = "optimize_descriptors"
    CATEGORY = "QSAR/REGRESSION/OPTIMIZATION"
    OUTPUT_NODE = True
    
    @staticmethod
    def optimize_descriptors(input_file, variance_threshold, correlation_threshold, correlation_mode, importance_model):
        send_progress("🚀 Starting Integrated Descriptor Optimization (Regression)...", 0)
        output_dir = "QSAR/Descriptor_Optimization"
        output_file = ""
        initial_feature_count, count_after_variance, final_feature_count = 0, 0, 0
        variance_removed, correlation_removed = 0, 0

        try:
            os.makedirs(output_dir, exist_ok=True)
            send_progress(f"📂 Output directory checked/created: {output_dir}", 2)

            send_progress(f"⏳ Loading data from: {input_file}", 5)
            data = pd.read_csv(input_file)
            initial_rows, initial_cols_total = data.shape
            send_progress(f"   Data loaded ({initial_rows} rows, {initial_cols_total} columns).", 8)

            if "value" not in data.columns:
                raise ValueError("The dataset must contain a 'value' column.")

            send_progress("⚙️ Separating features, target ('value'), and SMILES columns...", 10)
            target_column = data["value"]
            smiles_column = data["SMILES"] if "SMILES" in data.columns else None
            feature_columns = data.drop(columns=["value"] + ([smiles_column.name] if smiles_column is not None else []))
            initial_feature_count = feature_columns.shape[1]
            send_progress(f"   Found {initial_feature_count} initial descriptors.", 12)

            send_progress("⚙️ Handling potential NaN/Inf values in descriptors...", 15)
            feature_columns = feature_columns.replace([np.inf, -np.inf], np.nan)
            cols_with_nan = feature_columns.columns[feature_columns.isnull().any()]
            if not cols_with_nan.empty:
                send_progress(f"   Imputing NaNs in {len(cols_with_nan)} columns using median...", 18)
                for col in cols_with_nan:
                    median_val = feature_columns[col].median()
                    if pd.isna(median_val): median_val = 0
                    feature_columns[col].fillna(median_val, inplace=True)
                send_progress("   NaN imputation complete.", 20)
            else:
                send_progress("   No NaN values found to impute.", 20)

            send_progress(f"➡️ Step 1: Removing Low Variance Descriptors (Threshold: {variance_threshold:.3f})...", 25)
            selector = VarianceThreshold(threshold=variance_threshold)
            try:
                selector.fit(feature_columns)
                retained_columns_var = feature_columns.columns[selector.get_support()]
                feature_columns = feature_columns[retained_columns_var].copy()
                count_after_variance = feature_columns.shape[1]
                variance_removed = initial_feature_count - count_after_variance
                send_progress(f"   Low variance step complete. Kept {count_after_variance} descriptors, removed {variance_removed}.", 40)
            except ValueError as ve:
                 if "No feature in X meets the variance threshold" in str(ve):
                      error_msg = f"❌ Error (Step 1): No descriptors met variance threshold {variance_threshold}. Stopping."
                      send_progress(error_msg, 40)
                      return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
                 else: raise ve

            send_progress(f"➡️ Step 2: Removing High Correlation Descriptors (Threshold: {correlation_threshold:.2f}, Mode: {correlation_mode})...", 45)
            if count_after_variance <= 1:
                send_progress("   Skipping correlation removal (<= 1 descriptor remaining).", 50)
                correlation_removed = 0
                df_final = feature_columns.copy()
                df_final["value"] = target_column
                if smiles_column is not None: df_final["SMILES"] = smiles_column
            else:
                correlation_matrix = feature_columns.corr()
                send_progress("   Correlation matrix calculated.", 50)
                to_remove = set()
                feature_target_corr = None
                feature_importance = {}
                importance_calc_success = False

                if correlation_mode == "target_based":
                    send_progress("   Calculating target correlation...", 55)
                    feature_target_corr = feature_columns.corrwith(target_column).abs().fillna(0)
                    send_progress(f"   Calculating feature importance ('{importance_model}')...", 60)
                    X, y = feature_columns, target_column
                    try:
                        if importance_model == "lasso":
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)
                            model = Lasso(random_state=42)
                            try:
                                model.fit(X_scaled, y)
                                importance_values = np.abs(model.coef_)
                                importance_calc_success = True
                                send_progress("      LASSO importance calculated (alpha=1.0).", 65)
                            except Exception as e1:
                                send_progress(f"      LASSO (alpha=1.0) failed: {e1}. Trying alpha=0.01...", 65)
                                model = Lasso(alpha=0.01, max_iter=2000, random_state=42)
                                model.fit(X_scaled, y)
                                importance_values = np.abs(model.coef_)
                                importance_calc_success = True
                                send_progress("      LASSO importance calculated (alpha=0.01).", 65)

                        elif importance_model == "random_forest":
                            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                            model.fit(X, y)
                            importance_values = model.feature_importances_
                            importance_calc_success = True
                            send_progress("      RandomForest importance calculated.", 65)

                        if importance_calc_success:
                            feature_importance = dict(zip(feature_columns.columns, importance_values))
                    except Exception as model_e:
                        send_progress(f"      ⚠️ Warning: Failed importance calculation ({model_e}). Correlation tie-breaking disabled.", 65)
                        feature_importance = {}

                    send_progress("   Identifying descriptors to remove (target_based)...", 70)
                    rows, cols = np.where(np.abs(np.triu(correlation_matrix, k=1)) > correlation_threshold)
                    for r, c in zip(rows, cols):
                        f1, f2 = correlation_matrix.columns[r], correlation_matrix.columns[c]
                        if f1 in to_remove or f2 in to_remove: continue
                        corr1, corr2 = feature_target_corr.get(f1, 0), feature_target_corr.get(f2, 0)
                        imp1, imp2 = feature_importance.get(f1, 0), feature_importance.get(f2, 0)
                        if corr1 > corr2: weaker = f2
                        elif corr2 > corr1: weaker = f1
                        elif imp1 > imp2: weaker = f2
                        elif imp2 > imp1: weaker = f1
                        else: weaker = f2
                        to_remove.add(weaker)
                    send_progress(f"   Identified {len(to_remove)} descriptors to remove.", 75)

                else:
                    send_progress(f"   Identifying descriptors to remove ('{correlation_mode}' mode)...", 70)
                    mask = np.triu(np.ones(correlation_matrix.shape, dtype=bool), k=1) if correlation_mode == "upper" else \
                           np.tril(np.ones(correlation_matrix.shape, dtype=bool), k=-1)
                    high_corr_matrix = correlation_matrix.where(mask)
                    to_remove = {col for col in high_corr_matrix.columns if (high_corr_matrix[col].abs() > correlation_threshold).any()}
                    send_progress(f"   Identified {len(to_remove)} descriptors to remove.", 75)

                final_columns_list = [col for col in feature_columns.columns if col not in to_remove]
                df_final = feature_columns[final_columns_list].copy()
                df_final["value"] = target_column.reset_index(drop=True)
                if smiles_column is not None:
                    df_final["SMILES"] = smiles_column.reset_index(drop=True)
                correlation_removed = count_after_variance - len(final_columns_list)

            final_feature_count = df_final.shape[1] - (1 + (1 if smiles_column is not None else 0))
            send_progress(f"   High correlation step complete. Kept {final_feature_count} descriptors, removed {correlation_removed}.", 85)

            send_progress("💾 Saving final optimized data...", 90)
            if smiles_column is not None:
                final_cols_order = ["SMILES"] + [col for col in df_final.columns if col not in ["SMILES", "value"]] + ["value"]
                df_final = df_final[final_cols_order]

            output_file = os.path.join(output_dir, f"regression_optimized_descriptors_{initial_feature_count}_to_{final_feature_count}.csv")
            df_final.to_csv(output_file, index=False)
            send_progress(f"   Optimized data saved to: {output_file}", 94)

            send_progress("📝 Generating final summary...", 95)
            text_container_content = create_text_container(
                "🔹 **Integrated Descriptor Optimization Completed (Regression)!** 🔹",
                f"Input File: {os.path.basename(input_file)}",
                f"Initial Descriptors: {initial_feature_count}",
                "--- Processing Steps ---",
                f"1. Low Variance Removal (Thresh: {variance_threshold:.3f}): Removed {variance_removed} (Kept: {count_after_variance})",
                f"2. High Correlation Removal (Thresh: {correlation_threshold:.2f}, Mode: {correlation_mode}, Importance: {importance_model if correlation_mode=='target_based' else 'N/A'}): Removed {correlation_removed}",
                "--- Final Output ---",
                f"Final Descriptor Count: {final_feature_count}",
                f"Output File: {output_file}",
                f"Final Data Shape: {df_final.shape[0]} rows, {df_final.shape[1]} columns"
            )
            send_progress("🎉 Integrated optimization finished successfully!", 100)

            return {"ui": {"text": text_container_content},
                    "result": (str(output_file),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except ValueError as ve:
             error_msg = f"❌ Value Error: {str(ve)}"
             send_progress(error_msg)
             return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}

# Node Registration (Updated)
NODE_CLASS_MAPPINGS = {
    "Remove_Low_Variance_Features_Regression": Remove_Low_Variance_Descriptors_Regression,
    "Remove_High_Correlation_Features_Regression": Remove_High_Correlation_Features_Regression,
    "Descriptor_Optimization_Regression": Descriptor_Optimization_Regression
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Remove_Low_Variance_Features_Regression": "Remove Low Variance Features (Regression)",
    "Remove_High_Correlation_Features_Regression": "Remove High Correlation Features (Regression)",
    "Descriptor_Optimization_Regression": "Descriptor Optimization (Integrated) (Regression)"
}