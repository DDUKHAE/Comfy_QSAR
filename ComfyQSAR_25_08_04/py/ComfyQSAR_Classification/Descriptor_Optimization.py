import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class Remove_Low_Variance_Features_Classification:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_file": ("STRING", {"tooltip": "Path to the input file"}),
                "threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Variance threshold"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("LOW_VAR_FILTERED_PATH",) # Updated name
    FUNCTION = "remove_low_variance"
    CATEGORY = "QSAR/CLASSIFICATION/OPTIMIZATION(Filter-based)"
    OUTPUT_NODE = True

    def remove_low_variance(self, input_file, threshold=0.05):

        output_dir = "QSAR/Optimization"
        output_file = ""
        initial_count, final_count = 0, 0

        try:
            os.makedirs(output_dir, exist_ok=True)

            df = pd.read_csv(input_file)
            initial_rows, initial_cols_total = df.shape

            if "Label" not in df.columns:
                raise ValueError("The dataset must contain a 'Label' column.")

            # Separate features and target
            target_column = df["Label"]
            feature_columns = df.drop(columns=["Label"])

            selector = VarianceThreshold(threshold=threshold)

            try:
                selected_features_array = selector.fit_transform(feature_columns)
                retained_columns = feature_columns.columns[selector.get_support()] # Get remaining column names
                initial_count = feature_columns.shape[1]
                final_count = len(retained_columns)
            except ValueError as ve:
                if "No feature in X meets the variance threshold" in str(ve):
                    error_msg = f"❌ Error: No features met the variance threshold {threshold}. Try a lower threshold."
                    return {"ui": {"text": error_msg}, "result": (",")}
                else:
                    raise ve # Re-raise other ValueErrors

            # Create new DataFrame
            df_retained = pd.DataFrame(selected_features_array, columns=retained_columns, index=df.index) # Preserve index
            df_retained["Label"] = target_column # Add Label back

            output_file = os.path.join(output_dir, f"low_variance_filtered_{initial_count}_to_{final_count}.csv")
            df_retained.to_csv(output_file, index=False)

            log_message = (
                "========================================\n"
                "🔹 **Low Variance Feature Removal Done!** 🔹\n"
                "========================================\n"
                f"📊 Initial Features: {initial_count}\n"
                f"📉 Remaining Features: {final_count}\n"
                f"🗑️ Removed: {initial_count - final_count}\n"
                f"💾 Saved: {output_file}\n"
                "========================================"
            )

            return {
                "ui": {"text": log_message},
                "result": (str(output_file),)
            }

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}. Please check input file path."
            return {"ui": {"text": error_msg}, "result": (",")}
        except ValueError as ve:
             error_msg = f"❌ Value Error: {str(ve)}"
             return {"ui": {"text": error_msg}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred during low variance removal: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": error_msg}, "result": (",")}


class Remove_High_Correlation_Features_Classification:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_file": ("STRING", {"tooltip": "Path to the input file"}),
                "threshold": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01,
                                        "tooltip": "Correlation threshold"}),
                "correlation_mode": (["target_based", "upper", "lower"], 
                         {"tooltip": "Correlation mode"}),
                "importance_model": (["lasso", "random_forest"], 
                                     {"tooltip": "Importance model"}),
                "alpha": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01,
                                    "tooltip": "LASSO alpha parameter"}),
                "max_iter": ("INT", {"default": 1000, "min": 100, "max": 10000, "step": 100,
                                    "tooltip": "LASSO max_iter parameter"}),
                "n_estimators": ("INT", {"default": 200, "min": 100, "max": 1000, "step": 100,
                                         "tooltip": "RandomForest n_estimators"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("HIGH_CORR_FILTERED_PATH",) # Updated name
    FUNCTION = "remove_high_correlation"
    CATEGORY = "QSAR/CLASSIFICATION/OPTIMIZATION(Filter-based)"
    OUTPUT_NODE = True

    def remove_high_correlation(self, input_file, threshold=0.95, correlation_mode="target_based", importance_model="lasso", alpha=0.01, max_iter=1000, n_estimators=200):

        output_dir = "QSAR/Optimization"
        output_file = ""
        initial_count, final_count = 0, 0

        try:
            os.makedirs(output_dir, exist_ok=True)

            df = pd.read_csv(input_file)

            if "Label" not in df.columns:
                raise ValueError("The dataset must contain a 'Label' column.")

            target_column = df["Label"]
            feature_columns = df.drop(columns=["Label"])

            correlation_matrix = feature_columns.corr()
            to_remove = set()

            if correlation_mode == "target_based":
                feature_target_corr = feature_columns.corrwith(target_column).abs().fillna(0)
                feature_importance = {}

                try:
                    X, y = feature_columns, target_column
                    if importance_model == "lasso":
                        model = Lasso(alpha=alpha, max_iter=max_iter, random_state=42)
                        model.fit(X, y)
                        importance_values = np.abs(model.coef_)
                    elif importance_model == "random_forest":
                        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                        model.fit(X, y)
                        importance_values = model.feature_importances_

                    feature_importance = dict(zip(feature_columns.columns, importance_values))

                except Exception as model_e:
                    feature_importance = {}

                # Find highly correlated pairs and decide which to remove
                rows, cols = np.where(np.abs(np.triu(correlation_matrix, k=1)) > threshold)
                for row, col in zip(rows, cols):
                    f1 = correlation_matrix.columns[row]
                    f2 = correlation_matrix.columns[col]

                    if feature_target_corr[f1] > feature_target_corr[f2]:
                        weaker = f2
                    elif feature_target_corr[f1] < feature_target_corr[f2]:
                        weaker = f1
                    else:
                        weaker = f2 if feature_importance.get(f1, 0) > feature_importance.get(f2, 0) else f1

                    to_remove.add(weaker)
            else:
                tri = np.triu(correlation_matrix, k=1) if correlation_mode == "upper" else np.tril(correlation_matrix, k=-1)
                rows, cols = np.where(np.abs(tri) > threshold)
                for row, col in zip(rows, cols):
                    f1 = correlation_matrix.columns[row]
                    f2 = correlation_matrix.columns[col]
                    to_remove.add(f2 if correlation_mode == "upper" else f1)

            retained_columns = [c for c in feature_columns.columns if c not in to_remove]
            df_retained = feature_columns[retained_columns].copy()
            df_retained["Label"] = target_column

            initial_count = feature_columns.shape[1]
            final_count = len(retained_columns)
            output_file = os.path.join(output_dir, f"high_correlation_filtered_{initial_count}_to_{final_count}.csv")

            df_retained.to_csv(output_file, index=False)

            log_message = (
                "========================================\n"
                "🔹 **High Correlation Feature Removal Done!** 🔹\n"
                "========================================\n"
                f"📊 Initial Features: {initial_count}\n"
                f"📉 Remaining Features: {final_count}\n"
                f"🗑️ Removed: {initial_count - final_count}\n"
                f"💾 Saved: {output_file}\n"
                "========================================"
            )
            
            return {
                "ui": {"text": log_message},
                "result": (str(output_file),)
            }

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}. Please check input file path."
            return {"ui": {"text": error_msg}, "result": (",")}
        except ValueError as ve:
            error_msg = f"❌ Value Error: {str(ve)}"
            return {"ui": {"text": error_msg}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred during high correlation removal: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": error_msg}, "result": (",")}


class Descriptor_Optimization_Classification:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_file": ("STRING", {"tooltip": "Path to the input file"}),
                "variance_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                                                 "tooltip": "Variance threshold"}),
                "correlation_threshold": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01,
                                                    "tooltip": "Correlation threshold"}),
                "correlation_mode": (["target_based", "upper", "lower"],
                                     {"tooltip": "Correlation mode"}),
                "importance_model": (["lasso", "random_forest"],
                                     {"tooltip": "Importance model"}),
                "alpha": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01,
                                    "tooltip": "LASSO alpha parameter"}),
                "max_iter": ("INT", {"default": 1000, "min": 100, "max": 10000, "step": 100,
                                     "tooltip": "LASSO max_iter parameter"}),
                "n_estimators": ("INT", {"default": 200, "min": 100, "max": 1000, "step": 100,
                                         "tooltip": "RandomForest n_estimators"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("OPTIMIZED_DESC_PATH",)
    FUNCTION = "optimize_descriptors"
    CATEGORY = "QSAR/CLASSIFICATION/OPTIMIZATION(Filter-based)"
    OUTPUT_NODE = True

    def optimize_descriptors(self, input_file, variance_threshold, correlation_threshold, correlation_mode, importance_model, alpha=0.01, max_iter=1000, n_estimators=200):
        output_dir = "QSAR/Optimization"
        output_file = ""
        try:
            os.makedirs(output_dir, exist_ok=True)

            df = pd.read_csv(input_file)
            if "Label" not in df.columns:
                raise ValueError("The dataset must contain a 'Label' column.")

            target_column = df["Label"]
            feature_columns = df.drop(columns=["Label"])
            selector = VarianceThreshold(threshold=variance_threshold)
            try:
                selected_features_array = selector.fit_transform(feature_columns)
                retained_columns = feature_columns.columns[selector.get_support()]
                initial_count = feature_columns.shape[1]
                after_var_count = len(retained_columns)
            except ValueError as ve:
                if "No feature in X meets the variance threshold" in str(ve):
                    error_msg = f"❌ Error: No features met the variance threshold {variance_threshold}. Try a lower threshold."
                    return {"ui": {"text": error_msg}, "result": (",")}
                else:
                    raise ve

            df_retained = pd.DataFrame(selected_features_array, columns=retained_columns, index=df.index)
            df_retained["Label"] = target_column

            feature_columns_corr = df_retained.drop(columns=["Label"])
            correlation_matrix = feature_columns_corr.corr()
            to_remove = set()

            if correlation_mode == "target_based":
                feature_target_corr = feature_columns_corr.corrwith(target_column).abs().fillna(0)
                feature_importance = {}
                try:
                    X, y = feature_columns_corr, target_column
                    if importance_model == "lasso":
                        model = Lasso(alpha=alpha, max_iter=max_iter, random_state=42)
                        model.fit(X, y)
                        importance_values = np.abs(model.coef_)
                    elif importance_model == "random_forest":
                        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                        model.fit(X, y)
                        importance_values = model.feature_importances_
                    feature_importance = dict(zip(feature_columns_corr.columns, importance_values))
                except Exception as model_e:
                    feature_importance = {}

                rows, cols = np.where(np.abs(np.triu(correlation_matrix, k=1)) > correlation_threshold)
                for row, col in zip(rows, cols):
                    f1 = correlation_matrix.columns[row]
                    f2 = correlation_matrix.columns[col]
                    if feature_target_corr[f1] > feature_target_corr[f2]:
                        weaker = f2
                    elif feature_target_corr[f1] < feature_target_corr[f2]:
                        weaker = f1
                    else:
                        weaker = f2 if feature_importance.get(f1, 0) > feature_importance.get(f2, 0) else f1
                    to_remove.add(weaker)
            else:
                tri = np.triu(correlation_matrix, k=1) if correlation_mode == "upper" else np.tril(correlation_matrix, k=-1)
                rows, cols = np.where(np.abs(tri) > correlation_threshold)
                for row, col in zip(rows, cols):
                    f1 = correlation_matrix.columns[row]
                    f2 = correlation_matrix.columns[col]
                    to_remove.add(f2 if correlation_mode == "upper" else f1)

            retained_columns_final = [c for c in feature_columns_corr.columns if c not in to_remove]
            df_final = feature_columns_corr[retained_columns_final].copy()
            df_final["Label"] = target_column

            final_count = len(retained_columns_final)
            output_file = os.path.join(output_dir, f"optimized_filtered_{initial_count}_to_{final_count}.csv")
            df_final.to_csv(output_file, index=False)

            log_message = (
                "========================================\n"
                "🔹 **Descriptor Optimization Done!** 🔹\n"
                "========================================\n"
                f"📊 Initial Features: {initial_count}\n"
                f"📉 After Variance Filter: {after_var_count}\n"
                f"📉 Remaining Features: {final_count}\n"
                f"🗑️ Removed: {initial_count - final_count}\n"
                f"💾 Saved: {output_file}\n"
                "========================================"
            )

            return {
                "ui": {"text": log_message},
                "result": (str(output_file),)
            }

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}. Please check input file path."
            return {"ui": {"text": error_msg}, "result": (",")}
        except ValueError as ve:
            error_msg = f"❌ Value Error: {str(ve)}"
            return {"ui": {"text": error_msg}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred during descriptor optimization: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": error_msg}, "result": (",")}
        
# Node Registration
NODE_CLASS_MAPPINGS = {
    "Remove_Low_Variance_Features_Classification": Remove_Low_Variance_Features_Classification,
    "Remove_High_Correlation_Features_Classification": Remove_High_Correlation_Features_Classification,
    "Descriptor_Optimization_Classification": Descriptor_Optimization_Classification
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Remove_Low_Variance_Features_Classification": "Remove Low Variance Features (Classification)", # Updated
    "Remove_High_Correlation_Features_Classification": "Remove High Correlation Features (Classification)", # Updated
    "Descriptor_Optimization_Classification": "Descriptor Optimization (Integrated) (Classification)" # Updated
} 