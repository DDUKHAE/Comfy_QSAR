import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# --- Common Utility Import ---
try:
    from .Data_Loader import send_progress, create_text_container
except ImportError:
    print("[ComfyQSAR Descriptor Optimization] Warning: Could not import progress_utils. Progress updates might not work.")
    # Fallback functions
    def send_progress(message, progress=None, node_id=None):
        print(f"[Progress Fallback] {message}" + (f" ({progress}%)" if progress is not None else ""))
    def create_text_container(*lines):
        return "\n".join(lines)

class Remove_Low_Variance_Features_Classification:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_file": ("STRING",),
                "threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("LOW_VAR_FILTERED_PATH",) # Updated name
    FUNCTION = "remove_low_variance"
    CATEGORY = "QSAR/CLASSIFICATION/OPTIMIZATION"
    OUTPUT_NODE = True

    def remove_low_variance(self, input_file, threshold=0.05):
        """
        Remove low variance features from a dataset.
        """
        send_progress("🚀 Starting Low Variance Feature Removal...", 0)
        output_dir = "QSAR/Optimization"
        output_file = ""
        initial_count, final_count = 0, 0

        try:
            os.makedirs(output_dir, exist_ok=True)
            send_progress(f"📂 Output directory checked/created: {output_dir}", 5)

            send_progress(f"⏳ Loading data from: {input_file}", 10)
            df = pd.read_csv(input_file)
            initial_rows, initial_cols_total = df.shape
            send_progress(f"   Data loaded ({initial_rows} rows, {initial_cols_total} columns).", 15)

            if "Label" not in df.columns:
                raise ValueError("The dataset must contain a 'Label' column.")

            # Separate features and target
            send_progress("⚙️ Separating features and target ('Label') column...", 20)
            target_column = df["Label"]
            feature_columns = df.drop(columns=["Label"])
            initial_count = feature_columns.shape[1]
            send_progress(f"   Found {initial_count} initial features.", 25)

            # Handle NaN/Inf before variance calculation
            send_progress("⚙️ Handling potential NaN/Inf values in features...", 30)
            feature_columns = feature_columns.replace([np.inf, -np.inf], np.nan)
            cols_with_nan = feature_columns.columns[feature_columns.isnull().any()]
            if not cols_with_nan.empty:
                send_progress(f"   Imputing NaNs in {len(cols_with_nan)} columns using median...", 35)
                for col in cols_with_nan:
                    median_val = feature_columns[col].median()
                    if pd.isna(median_val): median_val = 0 # Handle cases where median is NaN
                    feature_columns[col].fillna(median_val, inplace=True)
                send_progress("   NaN imputation complete.", 40)
            else:
                send_progress("   No NaN values found to impute.", 40)

            # Apply Variance Threshold
            send_progress(f"📉 Applying Variance Threshold (threshold = {threshold:.3f})...", 50)
            selector = VarianceThreshold(threshold=threshold)
            try:
                selected_features_array = selector.fit_transform(feature_columns)
                retained_columns = feature_columns.columns[selector.get_support()]
                final_count = len(retained_columns)
                variance_removed = initial_count - final_count
                send_progress(f"   Variance filtering complete. Kept {final_count} features, removed {variance_removed}.", 70)
            except ValueError as ve:
                # Catch errors like "No feature in X meets the variance threshold"
                 if "No feature in X meets the variance threshold" in str(ve):
                      error_msg = f"❌ Error: No features met the variance threshold {threshold}. Try a lower threshold."
                      send_progress(error_msg, 70)
                      return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
                 else:
                      raise ve # Re-raise other ValueErrors

            # Create new DataFrame
            send_progress("📊 Creating new DataFrame with selected features...", 75)
            df_retained = pd.DataFrame(selected_features_array, columns=retained_columns, index=df.index) # Preserve index
            df_retained["Label"] = target_column # Add Label back
            send_progress("   DataFrame created.", 80)


            send_progress("💾 Saving filtered data...", 85)
            output_file = os.path.join(output_dir, f"low_variance_filtered_{initial_count}_to_{final_count}.csv")
            df_retained.to_csv(output_file, index=False)
            send_progress(f"   Filtered data saved to: {output_file}", 90)

            send_progress("📝 Generating summary...", 95)
            text_container_content = create_text_container(
                "🔹 **Low Variance Feature Removal Completed!** 🔹",
                f"Input File: {os.path.basename(input_file)}",
                f"Variance Threshold: {threshold:.3f}",
                f"Initial Features: {initial_count}",
                f"Features Removed: {variance_removed}",
                f"Remaining Features: {final_count}",
                f"Output File: {output_file}"
            )
            send_progress("🎉 Low variance removal process finished.", 100)

            return {
                "ui": {"text": text_container_content},
                "result": (str(output_file),)
            }

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}. Please check input file path."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except ValueError as ve:
             error_msg = f"❌ Value Error: {str(ve)}"
             send_progress(error_msg)
             return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred during low variance removal: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}


class Remove_High_Correlation_Features_Classification:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_file": ("STRING",),
                "threshold": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01}),
                "mode": (["target_based", "upper", "lower"], {"default": "target_based"}),
                "importance_model": (["lasso", "random_forest"], {"default": "lasso"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("HIGH_CORR_FILTERED_PATH",) # Updated name
    FUNCTION = "remove_high_correlation"
    CATEGORY = "QSAR/CLASSIFICATION/OPTIMIZATION"
    OUTPUT_NODE = True

    def remove_high_correlation(self, input_file, threshold=0.95, mode="target_based", importance_model="lasso"):
        """
        Remove highly correlated features from a classification dataset.
        """
        send_progress("🚀 Starting High Correlation Feature Removal...", 0)
        output_dir = "QSAR/Optimization"
        output_file = ""
        initial_count, final_count = 0, 0

        try:
            os.makedirs(output_dir, exist_ok=True)
            send_progress(f"📂 Output directory checked/created: {output_dir}", 2)

            send_progress(f"⏳ Loading data from: {input_file}", 5)
            df = pd.read_csv(input_file)
            initial_rows, initial_cols_total = df.shape
            send_progress(f"   Data loaded ({initial_rows} rows, {initial_cols_total} columns).", 8)

            if "Label" not in df.columns:
                raise ValueError("The dataset must contain a 'Label' column.")

            send_progress("⚙️ Separating features and target ('Label') column...", 10)
            target_column = df["Label"]
            feature_columns = df.drop(columns=["Label"])
            initial_count = feature_columns.shape[1]
            send_progress(f"   Found {initial_count} initial features.", 12)

            # Handle NaN/Inf
            send_progress("⚙️ Handling potential NaN/Inf values in features...", 15)
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

            # Calculate correlation matrix
            send_progress("📊 Calculating correlation matrix...", 25)
            correlation_matrix = feature_columns.corr()
            send_progress("   Correlation matrix calculated.", 30)

            to_remove = set()
            feature_target_corr = None
            feature_importance = {}

            if mode == "target_based":
                send_progress("🎯 Calculating correlation with target variable...", 35)
                feature_target_corr = feature_columns.corrwith(target_column).abs().fillna(0) # Fill NaN correlations with 0
                send_progress("   Target correlation calculated.", 40)

                send_progress(f"🧠 Calculating feature importance using '{importance_model}'...", 45)
                X, y = feature_columns, target_column
                try:
                    if importance_model == "lasso":
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
                        # Simplified LASSO attempt - use default alpha=1.0 first
                        try:
                            model = Lasso(random_state=42)
                            model.fit(X_scaled_df, y)
                            importance_values = np.abs(model.coef_)
                            send_progress("   LASSO training successful.", 55)
                        except Exception as e_lasso:
                             send_progress(f"   Warning: LASSO failed ({e_lasso}). Falling back to RandomForest.", 55)
                             importance_model = "random_forest" # Switch model type
                             # Re-raise if needed, or continue to RF below

                    # Use RandomForest (either chosen or fallback)
                    if importance_model == "random_forest":
                        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # Use 100 estimators, all cores
                        model.fit(X, y) # RF often handles unscaled data okay
                        importance_values = model.feature_importances_
                        send_progress("   RandomForest training successful.", 55)

                    feature_importance = dict(zip(feature_columns.columns, importance_values))

                except Exception as model_e:
                    send_progress(f"   ⚠️ Warning: Failed to train {importance_model} model ({model_e}). Proceeding without importance tie-breaking.", 55)
                    feature_importance = {} # Empty dict, won't be used for tie-breaking

                # Find highly correlated pairs and decide which to remove
                send_progress(f"✂️ Identifying features to remove based on correlation > {threshold:.2f} and target correlation/importance...", 60)
                rows, cols = np.where(np.abs(np.triu(correlation_matrix, k=1)) > threshold)
                for r, c in zip(rows, cols):
                    f1 = correlation_matrix.columns[r]
                    f2 = correlation_matrix.columns[c]

                    # If already decided to remove one, skip
                    if f1 in to_remove or f2 in to_remove:
                        continue

                    corr1 = feature_target_corr.get(f1, 0)
                    corr2 = feature_target_corr.get(f2, 0)
                    imp1 = feature_importance.get(f1, 0)
                    imp2 = feature_importance.get(f2, 0)

                    # Prioritize higher target correlation, then higher importance
                    if corr1 > corr2: weaker = f2
                    elif corr2 > corr1: weaker = f1
                    elif imp1 > imp2: weaker = f2
                    elif imp2 > imp1: weaker = f1
                    else: weaker = f2 # Arbitrarily remove f2 if all else equal

                    to_remove.add(weaker)
                send_progress(f"   Identified {len(to_remove)} features to remove based on target correlation/importance.", 75)

            else: # "upper" or "lower" mode
                send_progress(f"✂️ Identifying features to remove based on '{mode}' triangle and correlation > {threshold:.2f}...", 60)
                # Create a boolean mask for the upper or lower triangle
                mask = np.triu(np.ones(correlation_matrix.shape, dtype=bool), k=1) if mode == "upper" else \
                       np.tril(np.ones(correlation_matrix.shape, dtype=bool), k=-1)

                # Find columns where any correlation in the selected triangle exceeds the threshold
                high_corr_matrix = correlation_matrix.where(mask) # Apply mask
                to_remove = {col for col in high_corr_matrix.columns if (high_corr_matrix[col].abs() > threshold).any()}

                # Alternative approach for 'upper'/'lower' (iterative removal, might be slower but sometimes preferred):
                # corr_pairs = correlation_matrix.abs().unstack()
                # sorted_pairs = corr_pairs.sort_values(kind="quicksort", ascending=False)
                # high_corr_pairs = sorted_pairs[sorted_pairs > threshold]
                # for (f1, f2), corr_val in high_corr_pairs.items():
                #      if f1 == f2 or f1 in to_remove or f2 in to_remove: continue
                #      # Decide based on mode (remove f2 in upper, f1 in lower) - needs care if not target-based
                #      remove_candidate = f2 if mode == "upper" else f1
                #      # Simple removal based on name order if no other criteria
                #      # remove_candidate = max(f1, f2) # Or min(f1, f2)
                #      to_remove.add(remove_candidate)

                send_progress(f"   Identified {len(to_remove)} features to remove based on '{mode}' mode.", 75)


            send_progress("📊 Creating new DataFrame with selected features...", 80)
            retained_columns = [c for c in feature_columns.columns if c not in to_remove]
            df_retained = feature_columns[retained_columns].copy() # Added .copy()
            df_retained["Label"] = target_column # Add Label back
            final_count = len(retained_columns)
            correlation_removed = initial_count - final_count
            send_progress(f"   DataFrame created. Kept {final_count} features.", 85)

            send_progress("💾 Saving filtered data...", 90)
            output_file = os.path.join(output_dir, f"high_corr_filtered_{initial_count}_to_{final_count}.csv")
            df_retained.to_csv(output_file, index=False)
            send_progress(f"   Filtered data saved to: {output_file}", 94)

            send_progress("📝 Generating summary...", 95)
            text_container_content = create_text_container(
                "🔹 **High Correlation Feature Removal Completed!** 🔹",
                f"Input File: {os.path.basename(input_file)}",
                f"Correlation Threshold: {threshold:.2f}",
                f"Mode: {mode}",
                f"Importance Model (if target_based): {importance_model if mode=='target_based' else 'N/A'}",
                f"Initial Features: {initial_count}",
                f"Features Removed: {correlation_removed}",
                f"Remaining Features: {final_count}",
                f"Output File: {output_file}"
            )
            send_progress("🎉 High correlation removal process finished.", 100)

            return {
                "ui": {"text": text_container_content},
                "result": (str(output_file),)
            }

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}. Please check input file path."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except ValueError as ve:
             error_msg = f"❌ Value Error: {str(ve)}"
             send_progress(error_msg)
             return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred during high correlation removal: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}


class Descriptor_Optimization_Classification:
    @classmethod
    def INPUT_TYPES(s):
         return {
            "required": {
                "input_file": ("STRING",),
                "variance_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "correlation_threshold": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01}),
                "correlation_mode": (["target_based", "upper", "lower"], {"default": "target_based"}),
                "importance_model": (["lasso", "random_forest"], {"default": "lasso"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("OPTIMIZED_DESC_PATH",) # Updated name
    FUNCTION = "optimize_descriptors"
    CATEGORY = "QSAR/CLASSIFICATION/OPTIMIZATION"
    OUTPUT_NODE = True

    def optimize_descriptors(self, input_file, variance_threshold, correlation_threshold, correlation_mode, importance_model):
        """
        Optimize descriptors based on variance and correlation.
        """
        send_progress("🚀 Starting Integrated Descriptor Optimization...", 0)
        output_dir = "QSAR/Optimization"
        output_file = ""
        initial_count, count_after_variance, final_count = 0, 0, 0
        variance_removed, correlation_removed = 0, 0

        try:
            os.makedirs(output_dir, exist_ok=True)
            send_progress(f"📂 Output directory checked/created: {output_dir}", 2)

            send_progress(f"⏳ Loading data from: {input_file}", 5)
            df = pd.read_csv(input_file)
            initial_rows, initial_cols_total = df.shape
            send_progress(f"   Data loaded ({initial_rows} rows, {initial_cols_total} columns).", 8)

            if "Label" not in df.columns:
                raise ValueError("The dataset must contain a 'Label' column.")

            send_progress("⚙️ Separating features and target ('Label') column...", 10)
            target_column = df["Label"]
            feature_columns = df.drop(columns=["Label"])
            initial_count = feature_columns.shape[1]
            send_progress(f"   Found {initial_count} initial features.", 12)

            # Handle NaN/Inf (applied before both steps)
            send_progress("⚙️ Handling potential NaN/Inf values in features...", 15)
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


            # --- Step 1: Remove low variance features ---
            send_progress(f"➡️ Step 1: Removing Low Variance Features (Threshold: {variance_threshold:.3f})...", 25)
            selector = VarianceThreshold(threshold=variance_threshold)
            try:
                selector.fit(feature_columns) # Fit selector
                retained_columns_var = feature_columns.columns[selector.get_support()]
                feature_columns = feature_columns[retained_columns_var].copy() # Filter and copy
                count_after_variance = feature_columns.shape[1]
                variance_removed = initial_count - count_after_variance
                send_progress(f"   Low variance step complete. Kept {count_after_variance} features, removed {variance_removed}.", 40)
            except ValueError as ve:
                 if "No feature in X meets the variance threshold" in str(ve):
                      error_msg = f"❌ Error (Step 1): No features met the variance threshold {variance_threshold}. Stopping."
                      send_progress(error_msg, 40)
                      return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
                 else: raise ve


            # --- Step 2: Remove high correlation features ---
            send_progress(f"➡️ Step 2: Removing High Correlation Features (Threshold: {correlation_threshold:.2f}, Mode: {correlation_mode})...", 45)
            if count_after_variance <= 1: # Skip correlation if only 1 or 0 features left
                 send_progress("   Skipping correlation removal (<= 1 feature remaining).", 50)
                 correlation_removed = 0
                 df_final = feature_columns.copy() # Use the result from variance step
                 df_final["Label"] = target_column # Add label back
            else:
                 correlation_matrix = feature_columns.corr()
                 send_progress("   Correlation matrix calculated.", 50)
                 to_remove = set()
                 feature_target_corr = None
                 feature_importance = {}

                 if correlation_mode == "target_based":
                     send_progress("   Calculating target correlation...", 55)
                     feature_target_corr = feature_columns.corrwith(target_column).abs().fillna(0)
                     send_progress(f"   Calculating feature importance ('{importance_model}')...", 60)
                     # --- Importance Calculation Logic (Copied & adapted from High Corr class) ---
                     X, y = feature_columns, target_column
                     try:
                         if importance_model == "lasso":
                             scaler = StandardScaler()
                             X_scaled = scaler.fit_transform(X)
                             X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
                             try:
                                 model = Lasso(random_state=42)
                                 model.fit(X_scaled_df, y)
                                 importance_values = np.abs(model.coef_)
                                 send_progress("      LASSO training successful.", 65)
                             except Exception as e_lasso:
                                  send_progress(f"      Warning: LASSO failed ({e_lasso}). Falling back to RandomForest.", 65)
                                  importance_model = "random_forest" # Switch

                         if importance_model == "random_forest":
                             model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                             model.fit(X, y)
                             importance_values = model.feature_importances_
                             send_progress("      RandomForest training successful.", 65)

                         feature_importance = dict(zip(feature_columns.columns, importance_values))
                     except Exception as model_e:
                          send_progress(f"      ⚠️ Warning: Failed to train {importance_model} ({model_e}). No importance tie-breaking.", 65)
                          feature_importance = {}
                     # --- End Importance Calculation ---

                     send_progress("   Identifying features to remove (target_based)...", 70)
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
                     send_progress(f"   Identified {len(to_remove)} features to remove.", 75)

                 else: # "upper" or "lower" mode
                     send_progress(f"   Identifying features to remove ('{correlation_mode}' mode)...", 70)
                     mask = np.triu(np.ones(correlation_matrix.shape, dtype=bool), k=1) if correlation_mode == "upper" else \
                            np.tril(np.ones(correlation_matrix.shape, dtype=bool), k=-1)
                     high_corr_matrix = correlation_matrix.where(mask)
                     to_remove = {col for col in high_corr_matrix.columns if (high_corr_matrix[col].abs() > correlation_threshold).any()}
                     send_progress(f"   Identified {len(to_remove)} features to remove.", 75)

                 # Apply removal
                 final_columns = [c for c in feature_columns.columns if c not in to_remove]
                 df_final = feature_columns[final_columns].copy()
                 df_final["Label"] = target_column # Add Label back
                 correlation_removed = count_after_variance - len(final_columns)


            final_count = df_final.shape[1] - 1 # Subtract Label column for feature count
            send_progress(f"   High correlation step complete. Kept {final_count} features, removed {correlation_removed}.", 85)

            # --- Save final data ---
            send_progress("💾 Saving final optimized data...", 90)
            output_file = os.path.join(output_dir, f"optimized_descriptors_{initial_count}_to_{final_count}.csv")
            df_final.to_csv(output_file, index=False)
            send_progress(f"   Optimized data saved to: {output_file}", 94)

            # --- Generate Summary ---
            send_progress("📝 Generating final summary...", 95)
            text_container_content = create_text_container(
                "🔹 **Integrated Descriptor Optimization Completed!** 🔹",
                f"Input File: {os.path.basename(input_file)}",
                f"Initial Features: {initial_count}",
                "--- Processing Steps ---",
                f"1. Low Variance Removal (Thresh: {variance_threshold:.3f}): Removed {variance_removed} (Kept: {count_after_variance})",
                f"2. High Correlation Removal (Thresh: {correlation_threshold:.2f}, Mode: {correlation_mode}, Importance: {importance_model if correlation_mode=='target_based' else 'N/A'}): Removed {correlation_removed}",
                "--- Final Output ---",
                f"Final Feature Count: {final_count}",
                f"Output File: {output_file}"
            )
            send_progress("🎉 Integrated optimization finished successfully!", 100)

            return {
                "ui": {"text": text_container_content},
                "result": (str(output_file),)
            }

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}. Please check input file path."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except ValueError as ve:
             error_msg = f"❌ Value Error: {str(ve)}"
             send_progress(error_msg)
             return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred during integrated optimization: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}


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