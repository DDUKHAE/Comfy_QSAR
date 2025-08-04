import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

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
    CATEGORY = "QSAR/REGRESSION/OPTIMIZATION(Filter-based)"
    OUTPUT_NODE = True

    @staticmethod
    def remove_low_variance_descriptors(input_file, threshold):

        output_dir = "QSAR/Descriptor_Optimization"
        output_file = ""
        initial_feature_count, final_feature_count = 0, 0

        try:
            os.makedirs(output_dir, exist_ok=True)

            data = pd.read_csv(input_file)
            initial_rows, initial_cols_total = data.shape

            if "value" not in data.columns:
                raise ValueError ("The dataset must contain a 'value' column.")

            target_column = data["value"]
            smiles_column = data["SMILES"] if "SMILES" in data.columns else None
            feature_columns = data.drop(columns=["value"] + ([smiles_column.name] if smiles_column is not None else []))
            initial_feature_count = feature_columns.shape[1]

            feature_columns = feature_columns.replace([np.inf, -np.inf], np.nan)
            cols_with_nan = feature_columns.columns[feature_columns.isnull().any()]
            if not cols_with_nan.empty:
                for col in cols_with_nan:
                    median_val = feature_columns[col].median()
                    if pd.isna(median_val): median_val = 0
                    feature_columns[col].fillna(median_val, inplace=True)

            selector = VarianceThreshold(threshold=threshold)
            try:
                selected_features_array = selector.fit_transform(feature_columns)
                retained_columns = feature_columns.columns[selector.get_support()]
                final_feature_count = len(retained_columns)
                removed_count = initial_feature_count - final_feature_count
            except ValueError as ve:
                 if "No feature in X meets the variance threshold" in str(ve):
                      error_msg = f"❌ Error: No descriptors met the variance threshold {threshold}. Try a lower threshold."
                      return {"ui": {"text": error_msg}, "result": (",")}
                 elif "Input contains NaN" in str(ve):
                      error_msg = f"❌ Error: Input still contained NaN/Inf before variance thresholding, despite imputation attempt."
                      return {"ui": {"text": error_msg}, "result": (",")}
                 else:
                      raise ve

            df_retained = pd.DataFrame(selected_features_array, columns=retained_columns, index=data.index)
            df_retained["value"] = target_column
            if smiles_column is not None:
                df_retained["SMILES"] = smiles_column

            output_file = os.path.join(output_dir, f"regression_low_variance_filtered_{initial_feature_count}_to_{final_feature_count}.csv")
            df_retained.to_csv(output_file, index=False)

            log_message = (
                "========================================\n"
                "🔹 **Low Variance Descriptor Removal Completed!** 🔹\n"
                "========================================\n"
                f"✅ Input File: {os.path.basename(input_file)}\n"
                f"✅ Variance Threshold: > {threshold*100:.0f}% per descriptor\n"
                f"✅ Initial Descriptors: {initial_feature_count}\n"
                f"✅ Descriptors Removed: {removed_count}\n"
                f"✅ Remaining Descriptors: {final_feature_count}\n"
                f"💾 Output File: {output_file}\n"
                "========================================"
            )

            return {"ui": {"text": log_message},
                    "result": (str(output_file),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}."
            return {"ui": {"text": error_msg}, "result": (",")}
        except ValueError as ve:
             error_msg = f"❌ Value Error: {str(ve)}"
             return {"ui": {"text": error_msg}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": error_msg}, "result": (",")}

class Remove_High_Correlation_Features_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING",),
                "threshold": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01}),
                "correlation_mode": (["target_based","upper", "lower"], {"default": "target_based"}),
                "importance_model": (["lasso", "random_forest",], {"default": "lasso"})
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("HIGH_CORR_FILTERED_PATH",)
    FUNCTION = "remove_high_correlation_features"
    CATEGORY = "QSAR/REGRESSION/OPTIMIZATION(Filter-based)"
    OUTPUT_NODE = True
    
    @staticmethod
    def remove_high_correlation_features(input_file, threshold, correlation_mode, importance_model):
        output_dir = "QSAR/Descriptor_Optimization"
        output_file = ""
        initial_feature_count, final_feature_count = 0, 0

        try:
            os.makedirs(output_dir, exist_ok=True)

            data = pd.read_csv(input_file)
            initial_rows, initial_cols_total = data.shape

            if "value" not in data.columns:
                raise ValueError("The dataset must contain a 'value' column.")

            target_column = data["value"]
            smiles_column = data["SMILES"] if "SMILES" in data.columns else None
            feature_columns = data.drop(columns=["value"] + ([smiles_column.name] if smiles_column is not None else []))
            initial_feature_count = feature_columns.shape[1]

            feature_columns = feature_columns.replace([np.inf, -np.inf], np.nan)
            cols_with_nan = feature_columns.columns[feature_columns.isnull().any()]
            if not cols_with_nan.empty:
                for col in cols_with_nan:
                    median_val = feature_columns[col].median()
                    if pd.isna(median_val): median_val = 0
                    feature_columns[col].fillna(median_val, inplace=True)

            correlation_matrix = feature_columns.corr()

            to_remove = set()
            feature_target_corr = None
            feature_importance = {}
            importance_calc_success = False

            if correlation_mode == "target_based":
                feature_target_corr = feature_columns.corrwith(target_column).abs().fillna(0)

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
                        except Exception as e1:
                            model = Lasso(alpha=0.01, max_iter=2000, random_state=42)
                            model.fit(X_scaled, y)
                            importance_values = np.abs(model.coef_)
                            importance_calc_success = True

                    elif importance_model == "random_forest":
                        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                        model.fit(X, y)
                        importance_values = model.feature_importances_
                        importance_calc_success = True

                    if importance_calc_success:
                        feature_importance = dict(zip(feature_columns.columns, importance_values))
                except Exception as model_e:
                    feature_importance = {}

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

            else:
                mask = np.triu(np.ones(correlation_matrix.shape, dtype=bool), k=1) if correlation_mode == "upper" else \
                       np.tril(np.ones(correlation_matrix.shape, dtype=bool), k=-1)
                high_corr_matrix = correlation_matrix.where(mask)
                to_remove = {col for col in high_corr_matrix.columns if (high_corr_matrix[col].abs() > threshold).any()}

            retained_columns = [col for col in feature_columns.columns if col not in to_remove]
            df_retained = feature_columns[retained_columns].copy()
            df_retained["value"] = target_column
            if smiles_column is not None:
                df_retained["SMILES"] = smiles_column
            final_feature_count = len(retained_columns)
            removed_count = initial_feature_count - final_feature_count

            output_file = os.path.join(output_dir, f"regression_high_corr_filtered_{initial_feature_count}_to_{final_feature_count}.csv")
            df_retained.to_csv(output_file, index=False)

            log_message = (
                "========================================\n"
                "🔹 **High Correlation Feature Removal Completed!** 🔹\n"
                "========================================\n"
                f"✅ Input File: {os.path.basename(input_file)}\n"
                f"✅ Correlation Threshold: > {threshold*100:.0f}% per descriptor\n"
                f"✅ Initial Descriptors: {initial_feature_count}\n"
                f"✅ Descriptors Removed: {removed_count}\n"
                f"✅ Remaining Descriptors: {final_feature_count}\n"
                f"💾 Output File: {output_file}\n"
                "========================================"
            )

            return {"ui": {"text": log_message},
                    "result": (str(output_file),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}."
            return {"ui": {"text": error_msg}, "result": (",")}
           
        except ValueError as ve:
             error_msg = f"❌ Value Error: {str(ve)}"
             return {"ui": {"text": error_msg}, "result": (",")}

        except Exception as e:
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": error_msg}, "result": (",")}

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
    CATEGORY = "QSAR/REGRESSION/OPTIMIZATION(Filter-based)"
    OUTPUT_NODE = True
    
    @staticmethod
    def optimize_descriptors(input_file, variance_threshold, correlation_threshold, correlation_mode, importance_model):
        output_dir = "QSAR/Descriptor_Optimization"
        output_file = ""
        initial_feature_count, count_after_variance, final_feature_count = 0, 0, 0
        variance_removed, correlation_removed = 0, 0

        try:
            os.makedirs(output_dir, exist_ok=True)

            data = pd.read_csv(input_file)
            initial_rows, initial_cols_total = data.shape

            if "value" not in data.columns:
                raise ValueError("The dataset must contain a 'value' column.")

            target_column = data["value"]
            smiles_column = data["SMILES"] if "SMILES" in data.columns else None
            feature_columns = data.drop(columns=["value"] + ([smiles_column.name] if smiles_column is not None else []))
            initial_feature_count = feature_columns.shape[1]

            feature_columns = feature_columns.replace([np.inf, -np.inf], np.nan)
            cols_with_nan = feature_columns.columns[feature_columns.isnull().any()]
            if not cols_with_nan.empty:
                for col in cols_with_nan:
                    median_val = feature_columns[col].median()
                    if pd.isna(median_val): median_val = 0
                    feature_columns[col].fillna(median_val, inplace=True)

            selector = VarianceThreshold(threshold=variance_threshold)
            try:
                selector.fit(feature_columns)
                retained_columns_var = feature_columns.columns[selector.get_support()]
                feature_columns = feature_columns[retained_columns_var].copy()
                count_after_variance = feature_columns.shape[1]
                variance_removed = initial_feature_count - count_after_variance
            except ValueError as ve:
                 if "No feature in X meets the variance threshold" in str(ve):
                      error_msg = f"❌ Error (Step 1): No descriptors met variance threshold {variance_threshold}. Stopping."
                      return {"ui": {"text": error_msg}, "result": (",")}
                 else: raise ve

            if count_after_variance <= 1:
                correlation_removed = 0
                df_final = feature_columns.copy()
                df_final["value"] = target_column
                if smiles_column is not None: df_final["SMILES"] = smiles_column
            else:
                correlation_matrix = feature_columns.corr()
                to_remove = set()
                feature_target_corr = None
                feature_importance = {}
                importance_calc_success = False

                if correlation_mode == "target_based":
                    feature_target_corr = feature_columns.corrwith(target_column).abs().fillna(0)
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
                            except Exception as e1:
                                model = Lasso(alpha=0.01, max_iter=2000, random_state=42)
                                model.fit(X_scaled, y)
                                importance_values = np.abs(model.coef_)
                                importance_calc_success = True

                        elif importance_model == "random_forest":
                            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                            model.fit(X, y)
                            importance_values = model.feature_importances_
                            importance_calc_success = True

                        if importance_calc_success:
                            feature_importance = dict(zip(feature_columns.columns, importance_values))
                    except Exception as model_e:
                        feature_importance = {}

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

                else:
                    mask = np.triu(np.ones(correlation_matrix.shape, dtype=bool), k=1) if correlation_mode == "upper" else \
                           np.tril(np.ones(correlation_matrix.shape, dtype=bool), k=-1)
                    high_corr_matrix = correlation_matrix.where(mask)
                    to_remove = {col for col in high_corr_matrix.columns if (high_corr_matrix[col].abs() > correlation_threshold).any()}

                final_columns_list = [col for col in feature_columns.columns if col not in to_remove]
                df_final = feature_columns[final_columns_list].copy()
                df_final["value"] = target_column.reset_index(drop=True)
                if smiles_column is not None:
                    df_final["SMILES"] = smiles_column.reset_index(drop=True)
                correlation_removed = count_after_variance - len(final_columns_list)

            final_feature_count = df_final.shape[1] - (1 + (1 if smiles_column is not None else 0))

            if smiles_column is not None:
                final_cols_order = ["SMILES"] + [col for col in df_final.columns if col not in ["SMILES", "value"]] + ["value"]
                df_final = df_final[final_cols_order]

            output_file = os.path.join(output_dir, f"regression_optimized_descriptors_{initial_feature_count}_to_{final_feature_count}.csv")
            df_final.to_csv(output_file, index=False)

            log_message = (
                "========================================\n"
                "🔹 **Integrated Descriptor Optimization Completed!** 🔹\n"
                "========================================\n"
                f"✅ Input File: {os.path.basename(input_file)}\n"
                f"✅ Variance Threshold: {variance_threshold:.3f}\n"
                f"✅ Correlation Threshold: {correlation_threshold:.2f}\n"
                f"✅ Correlation Mode: {correlation_mode}\n"
                f"✅ Importance Model: {importance_model if correlation_mode=='target_based' else 'N/A'}\n"
                f"✅ Variance Removed: {variance_removed}\n"
                f"✅ Correlation Removed: {correlation_removed}\n"
                f"✅ Final Descriptor Count: {final_feature_count}\n"
                f"💾 Output File: {output_file}\n"
                "========================================"
            )

            return {"ui": {"text": log_message},
                    "result": (str(output_file),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}."
            return {"ui": {"text": error_msg}, "result": (",")}
        except ValueError as ve:
             error_msg = f"❌ Value Error: {str(ve)}"
             return {"ui": {"text": error_msg}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": error_msg}, "result": (",")}

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