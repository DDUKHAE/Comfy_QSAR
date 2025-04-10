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

# 모듈 레벨 함수로 이동 - 멀티프로세싱에서 사용할 함수들
def evaluate_combination_cls(args):
    """
    단일 특성 조합에 대한 분류 모델 평가
    args: (X_subset, y, feature_comb)의 튜플
    """
    X_subset, y, feature_comb = args
    X_train, X_eval, y_train, y_eval = train_test_split(X_subset, y, test_size=0.2, stratify=y, random_state=42)

    # 스케일링 적용
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    # 무한값, NaN 검사 및 처리
    if np.isnan(X_train_scaled).any() or np.isinf(X_train_scaled).any():
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    if np.isnan(X_eval_scaled).any() or np.isinf(X_eval_scaled).any():
        X_eval_scaled = np.nan_to_num(X_eval_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_eval_scaled)
        acc = accuracy_score(y_eval, y_pred)
    except Exception as e:
        print(f"모델 평가 중 오류 발생: {str(e)}")
        acc = 0.0  # 오류 발생 시 0점 처리
    
    return feature_comb, acc

class Feature_Combination_Search:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_file": ("STRING",),
                "max_features": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1}),
                "num_cores": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),
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
        os.makedirs("QSAR/Combination", exist_ok=True)
        df = pd.read_csv(input_file)

        # 결측치와 무한값 처리
        df_processed = df.copy()
        
        # Label 열 분리
        if "Label" not in df_processed.columns:
            raise ValueError("데이터셋에 'Label' 열이 없습니다.")
        
        # 무한값 및 NaN 처리
        for col in df_processed.columns:
            if col != "Label":
                df_processed[col] = df_processed[col].replace([np.inf, -np.inf], np.nan)
                if df_processed[col].isnull().any():
                    median_val = df_processed[col].median()
                    if np.isnan(median_val):  # 열 전체가 NaN인 경우
                        median_val = 0
                    df_processed[col] = df_processed[col].fillna(median_val)

        X = df_processed.drop(columns=["Label"]).values
        y = df_processed["Label"].values
        feature_names = df_processed.drop(columns=["Label"]).columns.tolist()

        all_results = []
        available_cores = min(num_cores, multiprocessing.cpu_count())
        print(f"총 {available_cores}개의 코어를 사용하여 분석을 시작합니다...")

        for num_features in range(1, max_features + 1):
            print(f"특성 {num_features}개 조합 분석 중...")
            feature_combinations = list(itertools.combinations(range(X.shape[1]), num_features))
            
            # 작업 수가 적으면 병렬 처리하지 않음
            if len(feature_combinations) < 10:
                results = []
                for comb in feature_combinations:
                    result = evaluate_combination_cls((X[:, list(comb)], y, comb))
                    results.append(result)
            else:
                # 멀티프로세싱 풀 생성
                task_args = [(X[:, list(comb)], y, comb) for comb in feature_combinations]
                
                try:
                    with Pool(processes=available_cores) as pool:
                        results = pool.map(evaluate_combination_cls, task_args)
                except Exception as e:
                    print(f"병렬 처리 중 오류 발생: {str(e)}")
                    # 오류 발생 시 단일 프로세스로 대체
                    results = []
                    for args in task_args:
                        result = evaluate_combination_cls(args)
                        results.append(result)
                
            for feature_comb, acc in results:
                all_results.append({
                    "Num_Features": len(feature_comb),
                    "Feature_Indices": feature_comb,
                    "Best Features": [feature_names[i] for i in feature_comb],
                    "Accuracy": acc
                })

        # 특성 수별 최적 조합 찾기
        best_per_feature_count = {}
        for n in range(1, max_features + 1):
            candidates = [res for res in all_results if res['Num_Features'] == n]
            if candidates:
                best_per_feature_count[n] = max(candidates, key=lambda x: x['Accuracy'])

        # 특성 수별 최적 조합 저장
        best_per_size_df = pd.DataFrame(best_per_feature_count.values())
        best_per_size_path = os.path.join("QSAR/Combination", "Best_combination_per_size.csv")
        best_per_size_df.to_csv(best_per_size_path, index=False)

        # 상위 N개 최적 조합 저장
        optimal_feature_paths = []
        best_features = None
        
        # 결과가 없는 경우 처리
        if not all_results:
            error_message = "특성 조합 평가 중 오류가 발생했습니다. 결과가 없습니다."
            print(error_message)
            output_file = os.path.join("QSAR/Combination", "error.txt")
            with open(output_file, 'w') as f:
                f.write(error_message)
            return {
                "ui": {"text": error_message},
                "result": (str(output_file),)
            }
            
        for i, result in enumerate(sorted(all_results, key=lambda x: x["Accuracy"], reverse=True)[:top_n], start=1):
            selected_columns = result["Best Features"] + ["Label"]
            df_selected = df_processed[selected_columns]
            output_path = os.path.join("QSAR/Combination", f"Optimal_Feature_Set_rank{i}.csv")
            df_selected.to_csv(output_path, index=False)
            optimal_feature_paths.append(output_path)
            
            if i == 1:
                best_features = result["Best Features"]
                output_file = os.path.join("QSAR/Combination", "Best_Optimal_Feature_Set.csv")
                df_selected.to_csv(output_file, index=False)

        # 로그 메시지 생성
        best_result = sorted(all_results, key=lambda x: x["Accuracy"], reverse=True)[0]
        text_container = (
            "========================================\n"
            "🔹 Classification Feature Combination Search Completed! 🔹\n"
            "========================================\n"
            f"✅ Best Accuracy: {best_result['Accuracy']:.4f}\n"
            f"✅ Optimal Feature Set: {best_features}\n"
            f"📊 Number of Features: {len(best_features)}\n"
            f"💾 Saved Per-Size Best Combinations: {best_per_size_path}\n"
            f"💾 Saved Top Feature Set: {optimal_feature_paths[0]}\n"
            f"🔍 Total Combinations Evaluated: {len(all_results)}\n"
            "========================================"
        )

        return {
            "ui": {"text": text_container},
            "result": (str(output_file),)
        }

# 노드 등록
NODE_CLASS_MAPPINGS = {
    "Feature_Combination_Search": Feature_Combination_Search
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Feature_Combination_Search": "Feature Combination Search"
} 