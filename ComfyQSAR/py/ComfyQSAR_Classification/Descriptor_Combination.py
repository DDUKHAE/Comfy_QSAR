import os
import pandas as pd
import numpy as np
import itertools
import multiprocessing
from multiprocessing import Pool
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 공통 유틸리티 임포트 ---
try:
    from .Data_Loader import create_text_container, send_progress
except ImportError:
    print("[ComfyQSAR Descriptor Combination] Warning: Could not import progress_utils. Progress updates might not work.")
    # 대체 함수 정의
    def send_progress(message, progress=None, node_id=None):
        print(f"[Progress Fallback] {message}" + (f" ({progress}%)" if progress is not None else ""))
    # 대체 create_text_container 정의
    def create_text_container(*lines):
        return "\n".join(lines)

# 모듈 레벨 함수로 이동 - 멀티프로세싱에서 사용할 함수들
def evaluate_combination_cls(args):
    """
    Evaluates a classification model for a single feature combination.
    Args: tuple (X_subset, y, feature_comb)
    """
    X_subset, y, feature_comb = args
    try:
        # 기본 데이터 분할
        X_train, X_eval, y_train, y_eval = train_test_split(X_subset, y, test_size=0.2, stratify=y, random_state=42)

        # 스케일링 적용
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_eval_scaled = scaler.transform(X_eval)

        # 무한값, NaN 검사 및 처리 (분할 및 스케일링 후에도 확인)
        if np.isnan(X_train_scaled).any() or np.isinf(X_train_scaled).any() or not np.all(np.isfinite(X_train_scaled)):
            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=1e6, neginf=-1e6) # Use large finite numbers
            X_train_scaled = np.clip(X_train_scaled, -1e6, 1e6)
        if np.isnan(X_eval_scaled).any() or np.isinf(X_eval_scaled).any() or not np.all(np.isfinite(X_eval_scaled)):
            X_eval_scaled = np.nan_to_num(X_eval_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
            X_eval_scaled = np.clip(X_eval_scaled, -1e6, 1e6)

        # 데이터 유효성 검사 (모든 값이 동일한 경우 등)
        if X_train_scaled.shape[0] < 2 or X_eval_scaled.shape[0] < 1 or np.all(X_train_scaled == X_train_scaled[0,:], axis=0).all():
             # print(f"Warning: Insufficient or constant data for combination {feature_comb}. Skipping.")
             return feature_comb, 0.0 # 평가 불가

        # 모델 학습 및 평가
        model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42) # Try different solver
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_eval_scaled)
        acc = accuracy_score(y_eval, y_pred)

    except ValueError as ve:
        # 특정 ValueError (예: 입력 데이터 문제) 처리
        # print(f"ValueError during model evaluation for {feature_comb}: {str(ve)}")
        acc = 0.0
    except Exception as e:
        # 기타 예외 처리
        # print(f"Error during model evaluation for combination {feature_comb}: {str(e)}")
        acc = 0.0  # Assign 0 score on error
    
    return feature_comb, acc

class Feature_Combination_Search:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_file": ("STRING",),
                "max_features": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1}),
                "num_cores": ("INT", {"default": 4, "min": 1, "max": multiprocessing.cpu_count(), "step": 1}), # Max to cpu_count
                "top_n": ("INT", {"default": 3, "min": 1, "max": 100, "step": 1})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("BEST_FEATURE_SET",)
    FUNCTION = "find_best_combinations"
    CATEGORY = "QSAR/CLASSIFICATION/COMBINATION"
    OUTPUT_NODE = True
    
    def find_best_combinations(self, input_file, max_features, num_cores, top_n):
        """
        Find the best combinations of descriptors for classification model.
        """
        send_progress("🚀 Starting Feature Combination Search...", 0)

        output_dir = "QSAR/Combination"
        os.makedirs(output_dir, exist_ok=True)
        send_progress(f"📂 Output directory created: {output_dir}", 5)

        try:
            send_progress(f"⏳ Loading data from: {input_file}", 10)
            df = pd.read_csv(input_file)
        except Exception as e:
            error_msg = f"❌ Error loading input file: {str(e)}"
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": ("",)}

        # --- Data Preprocessing ---
        send_progress("⚙️ Preprocessing data (handling inf/NaN)...", 15)
        df_processed = df.copy()
        
        if "Label" not in df_processed.columns:
            error_msg = "❌ Target column 'Label' not found in the dataset."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": ("",)}

        nan_inf_cols = []
        feature_cols = [col for col in df_processed.columns if col != "Label"]

        for col in feature_cols:
            # Ensure column is numeric before processing
            if pd.api.types.is_numeric_dtype(df_processed[col]):
                 original_nan_count = df_processed[col].isnull().sum()
                 original_inf_count = np.isinf(df_processed[col]).sum()

                 # Replace inf with NaN first
                 df_processed[col] = df_processed[col].replace([np.inf, -np.inf], np.nan)

                 if df_processed[col].isnull().any():
                     median_val = df_processed[col].median()
                     # If median is NaN (e.g., all NaNs), fill with 0
                     if pd.isna(median_val):
                         median_val = 0
                     df_processed[col] = df_processed[col].fillna(median_val)
                     # Log if NaNs or Infs were actually present and filled
                     if original_nan_count > 0 or original_inf_count > 0:
                          nan_inf_cols.append(f"{col} (filled with {median_val:.2f})")
            else:
                 # Handle non-numeric columns if necessary, e.g., drop or encode
                 print(f"Warning: Non-numeric column '{col}' found and skipped during NaN/inf handling.")
                 # Or drop: df_processed.drop(columns=[col], inplace=True)
                 # Update feature_cols if dropped: feature_cols.remove(col)


        if nan_inf_cols:
            send_progress(f"   NaN/Inf values handled in columns: {', '.join(nan_inf_cols)}", 20)
        else:
             send_progress("   No NaN/Inf values found or handled.", 20)


        # Ensure X and y use the potentially modified df_processed and feature_cols
        feature_names = [col for col in df_processed.columns if col != "Label" and col in feature_cols] # Update feature_names based on processed columns
        if not feature_names:
             error_msg = "❌ No valid numeric feature columns found after preprocessing."
             send_progress(error_msg)
             return {"ui": {"text": create_text_container(error_msg)}, "result": ("",)}

        X = df_processed[feature_names].values
        y = df_processed["Label"].values

        if X.shape[1] == 0: # Double check if features remain
             error_msg = "❌ No features remaining after preprocessing."
             send_progress(error_msg)
             return {"ui": {"text": create_text_container(error_msg)}, "result": ("",)}

        send_progress(f"🔢 Features ready for combination search: {X.shape[1]}", 25)


        # --- Combination Evaluation Loop ---
        all_results = []
        available_cores = min(num_cores, multiprocessing.cpu_count())
        send_progress(f"🧠 Using {available_cores} cores for analysis.", 26)

        # Define progress range for the loop itself
        loop_start_progress = 27
        loop_end_progress = 85
        loop_total_progress = loop_end_progress - loop_start_progress

        for num_features in range(1, min(max_features, X.shape[1]) + 1): # Ensure max_features doesn't exceed available
            current_loop_iter_progress = loop_start_progress + ((num_features -1) / max_features) * loop_total_progress
            send_progress(f"⏳ Analyzing combinations of {num_features} features...", current_loop_iter_progress)

            try:
                feature_indices_combinations = list(itertools.combinations(range(X.shape[1]), num_features))
                num_combinations = len(feature_indices_combinations)
                send_progress(f"   Generated {num_combinations} combinations.", current_loop_iter_progress + 1)

                if num_combinations == 0:
                    continue # Skip if no combinations generated (shouldn't happen with check above)

                # Prepare arguments for multiprocessing
                task_args = [(X[:, list(comb)], y, comb) for comb in feature_indices_combinations]

                # Evaluate combinations
                results_for_n_features = []
                if num_combinations < available_cores * 2 or available_cores <= 1: # Simple heuristic: Use single process for small tasks
                    # print(f"   Using single process for {num_combinations} combinations.") # Optional debug
                    for args in task_args:
                        results_for_n_features.append(evaluate_combination_cls(args))
                else:
                    # print(f"   Using {available_cores} processes for {num_combinations} combinations.") # Optional debug
                    try:
                        with Pool(processes=available_cores) as pool:
                            # Use map_async with chunksize for potentially better performance on large iterables
                            chunksize = max(1, num_combinations // (available_cores * 4)) # Heuristic chunksize
                            async_result = pool.map_async(evaluate_combination_cls, task_args, chunksize=chunksize)
                            pool.close() # No more tasks
                            pool.join() # Wait for completion
                            results_for_n_features = async_result.get() # Get results
                    except Exception as pool_e:
                        # Fallback to single process if pooling fails
                        print(f"Warning: Multiprocessing pool failed ({str(pool_e)}), falling back to single process.")
                        results_for_n_features = [evaluate_combination_cls(args) for args in task_args]

                # Collect results
                for feature_comb_indices, acc in results_for_n_features:
                     if acc > 0: # Only add valid results
                          all_results.append({
                               "Num_Features": len(feature_comb_indices),
                               "Feature_Indices": feature_comb_indices,
                               "Best Features": [feature_names[i] for i in feature_comb_indices],
                               "Accuracy": acc
                          })

            except MemoryError:
                 error_msg = f"❌ MemoryError: Not enough memory to generate combinations for {num_features} features. Try reducing 'max_features'."
                 send_progress(error_msg)
                 # Optionally break or return here depending on desired behavior
                 break # Stop further iterations if memory runs out
            except Exception as comb_e:
                 print(f"Error during combination analysis for {num_features} features: {comb_e}")
                 # Continue to the next number of features if possible

            # Update progress after completing analysis for num_features
            current_progress = loop_start_progress + (num_features / max_features) * loop_total_progress
            send_progress(f"   Finished analysis for {num_features} features.", current_progress)

        # --- Process Results ---
        if not all_results:
            error_message = "❌ No valid feature combinations found or evaluated successfully."
            send_progress(error_message)
            return {"ui": {"text": create_text_container(error_message)}, "result": ("",)}

        send_progress("📊 Processing results...", loop_end_progress + 2) # 87%

        # Find best combination per feature count
        best_per_feature_count = {}
        for n in range(1, max_features + 1):
            candidates = [res for res in all_results if res['Num_Features'] == n]
            if candidates:
                best_per_feature_count[n] = max(candidates, key=lambda x: x['Accuracy'])

        best_per_size_df = pd.DataFrame(best_per_feature_count.values())
        best_per_size_path = os.path.join(output_dir, "Best_combination_per_size.csv")
        send_progress(f"💾 Saving best combinations per size to: {best_per_size_path}", 90)
        best_per_size_df.to_csv(best_per_size_path, index=False)

        # Save top N overall combinations
        optimal_feature_paths = []
        best_features_list_overall = [] # Changed name to avoid conflict
        output_file_path = "" # Initialize

        sorted_results = sorted(all_results, key=lambda x: x["Accuracy"], reverse=True)
        send_progress(f"💾 Saving top {top_n} optimal feature sets...", 92)

        for i, result in enumerate(sorted_results[:top_n], start=1):
            selected_columns = result["Best Features"] + ["Label"]
            df_selected = df_processed[selected_columns]
            output_path = os.path.join(output_dir, f"Optimal_Feature_Set_rank{i}.csv")
            df_selected.to_csv(output_path, index=False)
            optimal_feature_paths.append(output_path)
            
            if i == 1:
                best_features_list_overall = result["Best Features"]
                output_file_path = os.path.join(output_dir, "Best_Optimal_Feature_Set.csv") # Use the rank 1 file
                # No need to save again, just assign the path
                # df_selected.to_csv(output_file_path, index=False)

        if not output_file_path and optimal_feature_paths: # Handle case where top_n=0 or results < top_n
             output_file_path = optimal_feature_paths[0]
             best_features_list_overall = sorted_results[0]["Best Features"]
        elif not optimal_feature_paths: # Handle case where no results were valid
             error_message = "❌ Could not determine the best feature set."
             send_progress(error_message)
             return {"ui": {"text": create_text_container(error_message)}, "result": ("",)}


        send_progress("   Saving complete.", 95)

        # Prepare result text
        best_result = sorted_results[0]
        text_container_content = create_text_container(
            f"🔹 **Classification Feature Combination Search Completed!** 🔹",
            f"✅ Best Accuracy: {best_result['Accuracy']:.4f}",
            f"✅ Optimal Feature Set ({len(best_features_list_overall)} features): {', '.join(best_features_list_overall)}",
            f"💾 Best Set File: {output_file_path}",
            f"💾 Per-Size Best Combinations: {best_per_size_path}",
            f"💾 Top {len(optimal_feature_paths)} Sets Saved in: {output_dir}",
            f"🔍 Total Combinations Evaluated: {len(all_results)}"
        )
        send_progress("🎉 Feature combination search finished successfully!", 100)

        return {
            "ui": {"text": text_container_content},
            "result": (str(output_file_path),) # Return the path to the best file
        }

# Node registration
NODE_CLASS_MAPPINGS = {
    "Feature_Combination_Search_Classification": Feature_Combination_Search # Renamed class
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Feature_Combination_Search_Classification": "Feature Combination Search (Classification)" # Renamed display name
} 