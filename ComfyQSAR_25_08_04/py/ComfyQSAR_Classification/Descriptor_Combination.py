import os
import pandas as pd
import itertools
import multiprocessing
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def evaluate_combination_cls(X_subset, y):
    X_train, X_eval, y_train, y_eval = train_test_split(X_subset, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_eval_scaled)
    acc = accuracy_score(y_eval, y_pred)
    return acc

def evaluate_combination_wrapper_cls(args):
    X_subset, y, feature_comb = args
    acc = evaluate_combination_cls(X_subset, y)
    return feature_comb, acc

class Feature_Combination_Search:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_file": ("STRING", {"tooltip": "Path to the input file"}),
                "max_features": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1,
                                          "tooltip": "Max features"}),
                "num_cores": ("INT", {"default": 4, "min": 1, "max": multiprocessing.cpu_count(), "step": 1,
                                       "tooltip": "Number of cores"}),
                "top_n": ("INT", {"default": 3, "min": 1, "max": 100, "step": 1,
                                  "tooltip": "Top N"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("BEST_FEATURE_SET",)
    FUNCTION = "find_best_combinations"
    CATEGORY = "QSAR/CLASSIFICATION/COMBINATION"
    OUTPUT_NODE = True
    
    def find_best_combinations(self, input_file, max_features, num_cores, top_n):
        output_dir = "QSAR/Combination"
        os.makedirs(output_dir, exist_ok=True)

        try:
            df = pd.read_csv(input_file)
        except Exception as e:
            error_msg = f"❌ Error loading input file: {str(e)}"
            return {"ui": {"text": error_msg}, "result": ("",)}
     
        if "Label" not in df.columns:
            error_msg = "❌ Target column 'Label' not found in the dataset."
            return {"ui": {"text": error_msg}, "result": ("",)}

        X = df.drop(columns=["Label"]).values
        y = df["Label"].values
        feature_names = df.drop(columns=["Label"]).columns.tolist()

        all_results = []
        available_cores = min(num_cores, multiprocessing.cpu_count())
        print(f"🖥️ Using {available_cores} CPU cores for parallel processing!")

        for num_features in range(1, max_features + 1):
            print(f"🔎 Searching best combination for {num_features} features...")
            feature_combinations = list(itertools.combinations(range(X.shape[1]), num_features))
            task_args = [(X[:, list(comb)], y, comb) for comb in feature_combinations]
            with Pool(available_cores) as pool:
                results = pool.map(evaluate_combination_wrapper_cls, task_args)
            for feature_comb, acc in results:
                all_results.append({
                    "Num_Features": len(feature_comb),
                    "Feature_Indices": feature_comb,
                    "Best Features": [feature_names[i] for i in feature_comb],
                    "Accuracy": acc
                })

        best_per_feature_count = {}
        for n in range(1, max_features + 1):
            candidates = [res for res in all_results if res['Num_Features'] == n]
            if candidates:
                best_per_feature_count[n] = max(candidates, key=lambda x: x['Accuracy'])

        best_per_size_df = pd.DataFrame(best_per_feature_count.values())
        best_per_size_path = os.path.join(output_dir, "Best_combination_per_size.csv")
        best_per_size_df.to_csv(best_per_size_path, index=False)

        optimal_feature_paths = []
        best_features = None
        for i, result in enumerate(sorted(all_results, key=lambda x: x["Accuracy"], reverse=True)[:top_n], start=1):
            selected_columns = result["Best Features"] + ["Label"]
            df_selected = df[selected_columns]
            output_path = os.path.join(output_dir, f"Optimal_Feature_Set_rank{i}.csv")
            df_selected.to_csv(output_path, index=False)
            optimal_feature_paths.append(output_path)
            if i == 1:
                best_features = result["Best Features"]
                output_file = os.path.join(output_dir, "Best_Optimal_Feature_Set.csv")
                df_selected.to_csv(output_file, index=False)

        log_message = (
            "========================================\n"
            "🔹 Classification Feature Combination Search Completed! 🔹\n"
            "========================================\n"
            f"✅ Best Accuracy: {sorted(all_results, key=lambda x: x['Accuracy'], reverse=True)[0]['Accuracy']:.4f}\n"
            f"✅ Optimal Feature Set: {best_features}\n"
            f"💾 Saved Per-Size Best Combinations: {best_per_size_path}\n"
            f"💾 Saved Top Feature Set: {optimal_feature_paths[0]}\n"
            "========================================\n"
        )


        return {
            "ui": {"text": log_message},
            "result": (str(output_file),)
        }

# Node registration
NODE_CLASS_MAPPINGS = {
    "Feature_Combination_Search_Classification": Feature_Combination_Search
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Feature_Combination_Search_Classification": "Feature Combination Search (Classification)"
} 