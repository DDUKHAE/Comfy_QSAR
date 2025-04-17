import os
import pandas as pd
import numpy as np
import itertools
import multiprocessing
from multiprocessing import Pool
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
# from ..utils.progress_utils import create_text_container # Now imported below

# --- Common Utility Import ---
try:
    from .Data_Loader import send_progress, create_text_container
except ImportError:
    print("[ComfyQSAR Regression Combination] Warning: Could not import progress_utils. Progress updates might not work.")
    # Fallback functions
    def send_progress(message, progress=None, node_id=None):
        print(f"[Progress Fallback] {message}" + (f" ({progress}%)" if progress is not None else ""))
    def create_text_container(*lines):
        return "\n".join(lines)

# --- Evaluation Functions (for multiprocessing) ---

def evaluate_combination_rf(X_subset, y):
    """Evaluates RandomForestRegressor for a subset."""
    mse, r2 = np.inf, -np.inf # Default scores in case of error
    try:
        if y.ndim > 1 and y.shape[1] == 1: y = y.ravel()

        # Robust NaN/Inf handling using median imputation
        if np.any(np.isnan(X_subset)) or np.any(np.isinf(X_subset)):
            imputer = SimpleImputer(strategy='median')
            # Fit on the subset itself (or a larger sample if needed, but subset median is typical)
            try:
                 # Need at least 2 samples to fit imputer if fitting per subset
                 if X_subset.shape[0] >= 2:
                      X_subset = imputer.fit_transform(X_subset)
                 else: # Fallback for very small subsets
                      X_subset = np.nan_to_num(X_subset, nan=0.0) # Impute with 0 if too small
            except ValueError: # Handle cases like all-NaN columns
                 X_subset = np.nan_to_num(X_subset, nan=0.0)

        # Check for constant features *after* imputation
        if X_subset.shape[1] > 0 and np.all(X_subset == X_subset[0,:], axis=0).all():
            return mse, r2 # Cannot train on constant features

        if X_subset.shape[0] < 2: return mse, r2 # Cannot split

        X_train, X_eval, y_train, y_eval = train_test_split(X_subset, y, test_size=0.2, random_state=42)

        if X_train.shape[0] < 1 or X_eval.shape[0] < 1: return mse, r2 # Split resulted in empty

        # Simple RF model for evaluation speed
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=101, n_jobs=1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_eval)

        mse = mean_squared_error(y_eval, y_pred)
        r2 = r2_score(y_eval, y_pred)

    except ValueError as ve: # Catch specific sklearn errors
        # print(f"Eval RF ValueError: {ve} on subset shape {X_subset.shape}") # Optional debug
        pass
    except Exception as e:
        # print(f"Eval RF Error: {e} on subset shape {X_subset.shape}") # Optional debug
        pass
    return mse, r2

def evaluate_combination_lr(X_subset, y_scaled): # Renamed for clarity
    """Evaluates LinearRegression for a subset (assumes X and y are scaled)."""
    mse, r2 = np.inf, -np.inf # Default scores
    try:
        if y_scaled.ndim > 1 and y_scaled.shape[1] == 1: y_scaled = y_scaled.ravel()

        # Robust NaN/Inf handling (median imputation) - should ideally happen *before* scaling,
        # but applied here defensively on the subset if needed.
        if np.any(np.isnan(X_subset)) or np.any(np.isinf(X_subset)):
             imputer = SimpleImputer(strategy='median')
             try:
                  if X_subset.shape[0] >= 2:
                       X_subset = imputer.fit_transform(X_subset)
                  else:
                       X_subset = np.nan_to_num(X_subset, nan=0.0)
             except ValueError:
                  X_subset = np.nan_to_num(X_subset, nan=0.0)

        if X_subset.shape[1] > 0 and np.all(X_subset == X_subset[0,:], axis=0).all():
             return mse, r2

        if X_subset.shape[0] < 2: return mse, r2

        X_train, X_eval, y_train, y_eval = train_test_split(X_subset, y_scaled, test_size=0.2, random_state=42)

        if X_train.shape[0] < 1 or X_eval.shape[0] < 1: return mse, r2

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_eval)

        mse = mean_squared_error(y_eval, y_pred)
        r2 = r2_score(y_eval, y_pred)

    except ValueError as ve:
        # print(f"Eval LR ValueError: {ve} on subset shape {X_subset.shape}") # Optional debug
        pass
    except Exception as e:
        # print(f"Eval LR Error: {e} on subset shape {X_subset.shape}") # Optional debug
        pass
    return mse, r2

def evaluate_combination_wrapper_rf(args):
    """Wrapper for RF evaluation for multiprocessing."""
    X_subset, y, feature_comb = args
    mse, r2 = evaluate_combination_rf(X_subset, y)
    return feature_comb, mse, r2

def evaluate_combination_wrapper_lr(args): # Renamed for clarity
    """Wrapper for LR evaluation for multiprocessing."""
    X_subset, y_scaled, feature_comb = args
    mse, r2 = evaluate_combination_lr(X_subset, y_scaled)
    return feature_comb, mse, r2

# --- Combined Node Class ---

class Regression_Feature_Combination_Search:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_csv_path": ("STRING", {"default": "input.csv"}),
                "evaluation_model": (["RandomForest", "LinearRegression"], {"default": "RandomForest"}),
                "max_features": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1}),
                "num_cores": ("INT", {"default": -1, "min": -1, "max": multiprocessing.cpu_count(), "step": 1}),
                "top_n": ("INT", {"default": 3, "min": 1, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("BEST_FEATURE_SET_PATH",)
    FUNCTION = "find_best_combinations" # Renamed function
    CATEGORY = "QSAR/REGRESSION/COMBINATION"
    OUTPUT_NODE = True

    def find_best_combinations(self, input_csv_path, evaluation_model, max_features, num_cores, top_n):
        model_abbr = "RF" if evaluation_model == "RandomForest" else "LR"
        send_progress(f"🚀 Starting Feature Combination Search ({evaluation_model})...", 0)
        output_dir = f"QSAR/Descriptor_Combination_{model_abbr}" # Dynamic output dir
        output_file_path = ""
        best_per_size_path = ""

        try:
            os.makedirs(output_dir, exist_ok=True)
            send_progress(f"📂 Output directory set: {output_dir}", 5)

            send_progress(f"⏳ Loading data from: {input_csv_path}", 10)
            df = pd.read_csv(input_csv_path)
            if "value" not in df.columns:
                raise ValueError("Input CSV must contain a 'value' column.")
            # Optionally keep SMILES if present, but don't use for modeling
            smiles_col = df["SMILES"] if "SMILES" in df.columns else None
            df_features = df.drop(columns=["value"] + (["SMILES"] if smiles_col is not None else []))
            df_target = df["value"]
            feature_names = df_features.columns.tolist()
            send_progress(f"   Data loaded ({len(df)} rows, {len(feature_names)} features).", 15)


            # --- Preprocessing (NaN/Inf Imputation) ---
            send_progress("⚙️ Preprocessing data (handling global NaN/Inf)...", 16)
            X_full = df_features.replace([np.inf, -np.inf], np.nan).copy()
            y_full = df_target.values # Keep as 1D array

            # Impute NaNs in features globally *before* scaling or splitting
            if X_full.isnull().values.any():
                 imputer = SimpleImputer(strategy='median')
                 X_imputed = imputer.fit_transform(X_full)
                 X_full = pd.DataFrame(X_imputed, columns=feature_names, index=X_full.index)
                 send_progress("   Global NaN imputation complete (using median).", 18)
            else:
                 send_progress("   No global NaNs found to impute.", 18)

            # --- Data Preparation (Scaling for LR, conversion to numpy) ---
            X_eval_data = None
            y_eval_data = None
            wrapper_func = None

            if evaluation_model == "LinearRegression":
                 send_progress("⚖️ Scaling features (X) and target (y) using MinMaxScaler for Linear Regression...", 19)
                 scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
                 X_scaled = scaler_X.fit_transform(X_full)
                 # Reshape y for scaler, then flatten for evaluation func
                 y_scaled = scaler_y.fit_transform(y_full.reshape(-1, 1)).ravel()
                 X_eval_data = X_scaled
                 y_eval_data = y_scaled
                 wrapper_func = evaluate_combination_wrapper_lr
                 send_progress("   Scaling complete.", 20)
            else: # RandomForest
                 X_eval_data = X_full.values # Use imputed numpy array
                 y_eval_data = y_full # Use original 1D numpy array
                 wrapper_func = evaluate_combination_wrapper_rf
                 send_progress("   Data prepared for RandomForest (no scaling needed).", 20)


            # --- Combination Generation and Evaluation ---
            all_results = []
            if num_cores == -1:
                actual_cores = multiprocessing.cpu_count()
                core_msg = "all available"
            else:
                actual_cores = min(num_cores, multiprocessing.cpu_count())
                core_msg = f"{actual_cores}"
            send_progress(f"⚙️ Using {core_msg} CPU cores for evaluation.", 21)

            loop_start_progress = 22
            loop_end_progress = 85
            loop_total_progress = loop_end_progress - loop_start_progress
            max_f = min(max_features, X_eval_data.shape[1])

            total_combinations_evaluated = 0
            total_valid_results = 0

            for num_features in range(1, max_f + 1):
                current_loop_iter_progress = loop_start_progress + ((num_features - 1) / max_f) * loop_total_progress
                send_progress(f"⏳ Analyzing combinations of {num_features} features...", int(current_loop_iter_progress))

                try:
                    feature_indices_combinations = list(itertools.combinations(range(X_eval_data.shape[1]), num_features))
                    num_combinations = len(feature_indices_combinations)
                    send_progress(f"   Generated {num_combinations} combinations.", int(current_loop_iter_progress + 1))

                    if num_combinations == 0: continue

                    # Prepare arguments for the correct wrapper function
                    task_args = [(X_eval_data[:, list(comb)], y_eval_data, comb) for comb in feature_indices_combinations]

                    results_for_n_features = []
                    send_progress(f"   Evaluating {num_combinations} combinations using {actual_cores} cores...", int(current_loop_iter_progress + 2))

                    # Decide whether to use pool or run sequentially
                    if num_combinations < actual_cores * 2 or actual_cores <= 1:
                         results_for_n_features = [wrapper_func(args) for args in task_args]
                    else:
                         try:
                             # Set start method for better compatibility across platforms
                             start_method = "fork" if hasattr(os, "fork") else "spawn"
                             ctx = multiprocessing.get_context(start_method)
                             with ctx.Pool(processes=actual_cores) as pool:
                                 # Adjust chunksize calculation
                                 chunksize = max(1, min(500, num_combinations // (actual_cores * 2))) # Smaller chunks can be better
                                 # Use map_async for potentially better memory management
                                 async_result = pool.map_async(wrapper_func, task_args, chunksize=chunksize)
                                 pool.close()
                                 pool.join() # Wait for completion
                                 results_for_n_features = async_result.get() # Get results
                         except (MemoryError, TimeoutError) as pool_mem_err:
                              send_progress(f"   ⚠️ Pool Error ({type(pool_mem_err).__name__}), falling back to single process...", int(current_loop_iter_progress + 3))
                              results_for_n_features = [wrapper_func(args) for args in task_args]
                         except Exception as pool_e:
                              send_progress(f"   ⚠️ Multiprocessing pool failed ({pool_e}), falling back to single process...", int(current_loop_iter_progress + 3))
                              results_for_n_features = [wrapper_func(args) for args in task_args]

                    # Collect valid results
                    valid_results_count_iter = 0
                    for feature_comb_indices, mse, r2 in results_for_n_features:
                        if r2 > -np.inf : # Check if evaluation was successful (not default error score)
                            all_results.append({
                                "Num_Features": len(feature_comb_indices),
                                "Feature_Indices": feature_comb_indices,
                                "Features": [feature_names[i] for i in feature_comb_indices],
                                "R2_Score": r2, # Consistent naming
                                "MSE": mse # Note: MSE is scaled for LR
                            })
                            valid_results_count_iter += 1
                    total_combinations_evaluated += num_combinations
                    total_valid_results += valid_results_count_iter
                    send_progress(f"   Finished evaluation for {num_features} features. Got {valid_results_count_iter} valid results.", int(loop_start_progress + (num_features / max_f) * loop_total_progress))

                except MemoryError:
                     error_msg = f"❌ MemoryError: Not enough RAM to generate/process combinations of {num_features} features. Try reducing 'max_features' or provide more RAM."
                     send_progress(error_msg)
                     # Optionally save partial results if desired before breaking
                     break # Stop processing further feature counts
                except Exception as comb_e:
                     # Log other errors but try to continue if possible
                     send_progress(f"   ⚠️ Error analyzing {num_features} features: {comb_e}", int(loop_start_progress + (num_features / max_f) * loop_total_progress))


            # --- Processing and Saving Results ---
            if not all_results:
                raise ValueError("No valid feature combinations found or evaluated successfully. Check input data and parameters.")

            send_progress("📊 Processing results...", loop_end_progress + 2) # 87%

            # Find best combination per size
            best_per_feature_count = defaultdict(lambda: {"R2_Score": -np.inf})
            for res in all_results:
                n = res['Num_Features']
                if res['R2_Score'] > best_per_feature_count[n]['R2_Score']:
                    best_per_feature_count[n] = res

            if best_per_feature_count:
                 best_per_size_df = pd.DataFrame(list(best_per_feature_count.values())) # Convert values view to list
                 best_per_size_df = best_per_size_df.sort_values(by="Num_Features") # Sort by number of features
                 best_per_size_path = os.path.join(output_dir, f"{model_abbr}_Best_combination_per_size.csv")
                 send_progress(f"💾 Saving best combinations per size to: {best_per_size_path}", 90)
                 best_per_size_df.to_csv(best_per_size_path, index=False)
            else:
                 send_progress("   ⚠️ No best combination found for any feature size.", 90)
                 best_per_size_path = "N/A"


            # Sort all results by R² and save top N feature sets
            sorted_results = sorted(all_results, key=lambda x: x["R2_Score"], reverse=True)
            send_progress(f"💾 Saving top {top_n} optimal feature sets...", 92)

            optimal_feature_paths = []
            best_features_list_overall = []
            for i, result in enumerate(sorted_results[:top_n], start=1):
                 selected_columns = result["Features"] + ["value"] # Use names from result dict
                 # Reconstruct the DataFrame slice using original data
                 df_selected = df[selected_columns].copy()
                 if smiles_col is not None: # Add SMILES back if it existed
                      df_selected.insert(0, "SMILES", smiles_col)

                 output_path = os.path.join(output_dir, f"{model_abbr}_Optimal_Feature_Set_rank{i}.csv")
                 df_selected.to_csv(output_path, index=False)
                 optimal_feature_paths.append(output_path)
                 if i == 1:
                     best_features_list_overall = result["Features"]
                     output_file_path = output_path # The rank 1 path is the main output

            if not output_file_path: # Handle case where no results or top_n=0
                 raise ValueError("Could not determine the best feature set (no valid results or top_n=0).")
            send_progress("   Saving complete.", 95)

            # --- Generate Final Summary ---
            best_result = sorted_results[0]
            summary_lines = [
                f"🔹 **{evaluation_model} Feature Combination Search Completed!** 🔹",
                f"Input File: {os.path.basename(input_csv_path)}",
                f"Evaluation Model: {evaluation_model}",
                f"✅ Best R² Score: {best_result['R2_Score']:.4f} (with {len(best_features_list_overall)} features)",
                f"   - MSE{'(scaled)' if evaluation_model=='LinearRegression' else ''}: {best_result['MSE']:.4f}",
                f"   - Features: {', '.join(best_features_list_overall)}",
                f"💾 Best Set File: {output_file_path}",
                f"💾 Per-Size Best Combinations: {best_per_size_path}",
                f"💾 Top {len(optimal_feature_paths)} Sets Saved in: {output_dir}",
                f"🔍 Total Combinations Evaluated: {total_combinations_evaluated} (Valid Results: {total_valid_results})"
            ]

            text_container_content = create_text_container(*summary_lines)
            send_progress("🎉 Feature combination search finished successfully!", 100)

            return {"ui": {"text": text_container_content},
                    "result": (str(output_file_path),)} # Return path to the best (rank 1) set

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",",)} # Return empty string on error
        except ValueError as ve:
            error_msg = f"❌ Value Error: {str(ve)}."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",",)}
        except MemoryError as me:
             # Catch memory errors specifically during the main process (pool errors handled above)
             error_msg = f"❌ Memory Error: {str(me)}. The process ran out of memory. Try reducing 'max_features' or provide more RAM."
             send_progress(error_msg)
             # Consider saving partial results here if implemented
             return {"ui": {"text": create_text_container(error_msg)}, "result": (",",)}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",",)}


# --- Node Registration (Updated) ---
NODE_CLASS_MAPPINGS = {
    "Regression_Feature_Combination_Search": Regression_Feature_Combination_Search, # Combined class
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Regression_Feature_Combination_Search": "Feature Combination Search (Regression)", # Combined display name
}

# Keep evaluation helper functions (no changes needed there unless logic requires it)
# def evaluate_combination_rf(X_subset, y): ...
# def evaluate_combination_lr(X_subset, y_scaled): ...
# def evaluate_combination_wrapper_rf(args): ...
# def evaluate_combination_wrapper_lr(args): ...
