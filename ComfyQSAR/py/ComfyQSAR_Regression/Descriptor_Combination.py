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

def evaluate_combination_rf(X_subset, y):

    X_train, X_eval, y_train, y_eval = train_test_split(X_subset, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=101, n_jobs=1)
    model.fit(X_train, y_train.ravel())  # Flatten target (y) for compatibility
    y_pred = model.predict(X_eval)

    mse = mean_squared_error(y_eval, y_pred)
    r2 = r2_score(y_eval, y_pred)

    return mse, r2

def evaluate_combination(X_subset, y):
    X_train, X_eval, y_train, y_eval = train_test_split(X_subset, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)

    mse = mean_squared_error(y_eval, y_pred)
    r2 = r2_score(y_eval, y_pred)

    return mse, r2

def evaluate_combination_wrapper_rf(args):
    X_subset, y, feature_comb = args
    mse, r2 = evaluate_combination_rf(X_subset, y)
    return feature_comb, mse, r2

def evaluate_combination_wrapper(args):
    X_subset, y_scaled, feature_comb = args
    mse, r2 = evaluate_combination(X_subset, y_scaled)
    return feature_comb, mse, r2

class Get_Best_Descriptor_Combinations_RF:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_csv": ("STRING", {"default": "input.csv"}),
                "max_features": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1}),
                "num_cores": ("INT", {"default": 6, "min": 1, "max": 100, "step": 1}),
                "top_n": ("INT", {"default": 3, "min": 1, "max": 100, "step": 1}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("DATA",)
    FUNCTION = "run"
    CATEGORY = "QSAR/REGRESSION/COMBINATION"
    OUTPUT_NODE = True

    def get_best_descriptor_combinations(self, input_csv, max_features, num_cores, top_n):
        # Load the input CSV file
        os.makedirs("QSAR/Descriptor_Combination", exist_ok=True)

        df = pd.read_csv(input_csv)
        if "value" not in df.columns:
            raise ValueError("❌ Error: The dataset must contain a 'value' column.")

        X = df.drop(columns=["value"]).values
        y = df["value"].values.reshape(-1, 1)
        feature_names = df.drop(columns=["value"]).columns.tolist()

        all_results = []

        available_cores = max(1, multiprocessing.cpu_count())
        num_cores = min(num_cores, available_cores)

        for num_features in range(1, max_features + 1):
            print(f"🔎 Searching best combination for {num_features} features...")
            feature_combinations = list(itertools.combinations(range(X.shape[1]), num_features))
            task_args = [(X[:, list(comb)], y, comb) for comb in feature_combinations]
            with Pool(num_cores) as pool:
                results = pool.map(evaluate_combination_wrapper_rf, task_args)
            for feature_comb, mse, r2 in results:
                all_results.append({
                    "Num_Features": len(feature_comb),
                    "Best Features": [feature_names[i] for i in feature_comb],
                    "R² Score": r2,
                    "MSE": mse
                })

        # Save best combination per size
        best_per_size_dict = defaultdict(lambda: {"R² Score": -np.inf})
        for entry in all_results:
            n = entry["Num_Features"]
            if entry["R² Score"] > best_per_size_dict[n]["R² Score"]:
                best_per_size_dict[n] = entry

        best_per_size_df = pd.DataFrame(best_per_size_dict.values())
        best_per_size_path = os.path.join("QSAR/Descriptor_Combination", "Best_combination_per_size_results.csv")
        best_per_size_df.to_csv(best_per_size_path, index=False)

        # Save top-N optimal feature sets
        optimal_feature_paths = []
        best_features = None
        for i, result in enumerate(sorted(all_results, key=lambda x: x["R² Score"], reverse=True)[:top_n], start=1):
            selected_columns = result["Best Features"] + ["value"]
            df_selected = df[selected_columns]
            output_path = os.path.join("QSAR/Descriptor_Combination", f"Optimal_Feature_Set_rank{i}.csv")
            df_selected.to_csv(output_path, index=False)
            optimal_feature_paths.append(output_path)
            if i == 1:
                best_features = result["Best Features"]
                df_selected.to_csv(os.path.join("QSAR/Descriptor_Combination", "Best_Optimal_Feature_Set.csv"), index=False)

        text_container = (
        "========================================\n"
        "🔹 **Feature Selection Completed!** 🔹\n"
        "========================================\n"
        f"💾 **Optimal feature set:** {best_features}\n"
        f"📊 **Top R² Score:** {sorted(all_results, key=lambda x: x['R² Score'], reverse=True)[0]['R² Score']:.4f}\n"
        f"📊 **Top MSE:** {sorted(all_results, key=lambda x: x['R² Score'], reverse=True)[0]['MSE']:.4f}\n"
        f"💾 Saved Per-Size Best Combinations: {best_per_size_path}\n"
        f"💾 Saved Optimal Feature Set: {optimal_feature_paths[0]}\n"
        "========================================\n"
        )

        return {"ui": {"text": text_container},
                "result": (str(os.path.join("QSAR/Descriptor_Combination", "Best_Optimal_Feature_Set.csv")))}
        
class Get_Best_Descriptor_Combinations:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_csv": ("STRING", {"default": "input.csv"}),
            },

            "optional": {
                "max_features": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1}),
                "num_cores": ("INT", {"default": 6, "min": 1, "max": 100, "step": 1}),
                "top_n": ("INT", {"default": 3, "min": 1, "max": 100, "step": 1}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("DATA",)
    FUNCTION = "get_best_descriptor_combinations"
    CATEGORY = "QSAR/REGRESSION/COMBINATION"
    OUTPUT_NODE = True
    
    def get_best_descriptor_combinations(self, input_csv, max_features, num_cores, top_n):
        # Load the input CSV file
        os.makedirs("QSAR/Descriptor_Combination", exist_ok=True)
        
        df = pd.read_csv(input_csv)
        if "value" not in df.columns:
            raise ValueError("❌ Error: The dataset must contain a 'value' column.")

        X = df.drop(columns=["value"]).values
        y = df["value"].values.reshape(-1, 1)
        feature_names = df.drop(columns=["value"]).columns.tolist()

        # Scaling
        scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)

        all_results = []

        available_cores = max(1, multiprocessing.cpu_count())
        num_cores = min(num_cores, available_cores)

        for num_features in range(1, max_features + 1):
            feature_combinations = list(itertools.combinations(range(X.shape[1]), num_features))
            task_args = [(X_scaled[:, list(comb)], y_scaled, comb) for comb in feature_combinations]
            with Pool(num_cores) as pool:
                results = pool.map(evaluate_combination_wrapper, task_args)
            for feature_comb, mse, r2 in results:
                all_results.append({
                    "Num_Features": len(feature_comb),
                    "Best Features": [feature_names[i] for i in feature_comb],
                    "R² Score": r2,
                    "MSE": mse
                })

        # Save best combination per size
        best_per_size_dict = defaultdict(lambda: {"R² Score": -np.inf})
        for entry in all_results:
            n = entry["Num_Features"]
            if entry["R² Score"] > best_per_size_dict[n]["R² Score"]:
                best_per_size_dict[n] = entry

        best_per_size_df = pd.DataFrame(best_per_size_dict.values())
        best_per_size_path = os.path.join("QSAR/Descriptor_Combination", "Best_combination_per_size_results.csv")
        best_per_size_df.to_csv(best_per_size_path, index=False)

        # Save top-N optimal feature sets
        optimal_feature_paths = []
        best_features = None
        for i, result in enumerate(sorted(all_results, key=lambda x: x["R² Score"], reverse=True)[:top_n], start=1):
            selected_columns = result["Best Features"] + ["value"]
            df_selected = df[selected_columns]
            output_path = os.path.join("QSAR/Descriptor_Combination", f"Optimal_Feature_Set_rank{i}.csv")
            df_selected.to_csv(output_path, index=False)
            optimal_feature_paths.append(output_path)
            if i == 1:
                best_features = result["Best Features"]
                df_selected.to_csv(os.path.join("QSAR/Descriptor_Combination", "Best_Optimal_Feature_Set.csv"), index=False)

        text_container = (
        "========================================\n"
        "🔹 **Feature Selection Completed!** 🔹\n"
        "========================================\n"
        f"✅ **Optimal feature set:** {best_features}\n"
        f"📊 **Top R² Score:** {sorted(all_results, key=lambda x: x['R² Score'], reverse=True)[0]['R² Score']:.4f}\n"
        f"📊 **Top MSE:** {sorted(all_results, key=lambda x: x['R² Score'], reverse=True)[0]['MSE']:.4f}\n"
        f"💾 Saved Per-Size Best Combinations: {best_per_size_path}\n"
        f"💾 Saved Optimal Feature Set: {optimal_feature_paths[0]}\n"
        "========================================\n"
        )

        return {"ui": {"text": text_container},
                "result": (str(os.path.join("QSAR/Descriptor_Combination", "Best_Optimal_Feature_Set.csv")))}
    
NODE_CLASS_MAPPINGS = {
    "Get_Best_Descriptor_Combinations_RF": Get_Best_Descriptor_Combinations_RF,
    "Get_Best_Descriptor_Combinations": Get_Best_Descriptor_Combinations,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Get_Best_Descriptor_Combinations_RF": "Get Best Descriptor Combinations (Random Forest)",
    "Get_Best_Descriptor_Combinations": "Get Best Descriptor Combinations (Linear Regression)",
}
