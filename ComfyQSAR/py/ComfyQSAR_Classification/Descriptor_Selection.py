import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from .Data_Loader import create_text_container

class Feature_Selection_Classification:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_file": ("STRING",),
                "method": (["Lasso", "RandomForest", "DecisionTree", "XGBoost", "LightGBM", "RFE", "SelectFromModel"],),
                "advanced": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                # Model selection and basic parameters
                "model": (["Lasso", "RandomForest", "DecisionTree", "XGBoost", "LightGBM"],{"default": "None", "description": "Model to use with RFE or SelectFromModel methods"}),
                "target_column": ("STRING", {"default": "Label", "description": "Target column name in the dataset"}),
                "n_features": ("INT", {"default": 10, "min": 1, "max": 1000, "step": 1, "description": "Number of features to select"}),
                "threshold": ("STRING", {"default": "percentile(90)", "description": "Threshold for feature selection (percentile or absolute value)"}),
                
                # Lasso related parameters
                "alpha": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 10.0, "step": 0.001, "description": "Regularization strength for Lasso"}),
                "max_iter": ("INT", {"default": 1000, "min": 100, "max": 10000, "step": 100, "description": "Maximum number of iterations"}),
                
                # Tree-based model parameters
                "n_estimators": ("INT", {"default": 100, "min": 10, "max": 1000, "step": 10, "description": "Number of trees in ensemble models"}),
                "max_depth": ("INT", {"default": 5, "min": 1, "max": 50, "step": 1, "description": "Maximum depth of trees"}),
                "min_samples_split": ("INT", {"default": 2, "min": 2, "max": 20, "step": 1, "description": "Minimum samples required to split a node"}),
                "criterion": (["gini", "entropy"], {"description": "Function to measure quality of split"}),
                "learning_rate": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 1.0, "step": 0.01, "description": "Learning rate for boosting models"}),
                
                # Additional parameters
                "n_iterations": ("INT", {"default": 100, "min": 10, "max": 1000, "step": 10, "description": "Number of iterations for stability analysis"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("OUTPUT_FILE",)
    FUNCTION = "select_features"
    CATEGORY = "QSAR/CLASSIFICATION/SELECTION"
    OUTPUT_NODE = True
    
    def select_features(self, input_file, advanced, target_column="Label", method="lasso", n_features=10, threshold="percentile(90)",
                      model=None, alpha=0.01, max_iter=1000, n_estimators=100, max_depth=5, min_samples_split=2, 
                      criterion="gini", learning_rate=0.1, n_iterations=100):
        """
        Feature selection supporting strategy-method and model-based approaches.
        """
        os.makedirs("QSAR/Selection", exist_ok=True)
        df = pd.read_csv(input_file)
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")

        X_raw = df.drop(columns=[target_column])
        y = df[target_column]

        # 무한값 및 NaN 처리
        print("데이터 전처리: 무한값 및 NaN 처리 중...")
        X = X_raw.replace([np.inf, -np.inf], np.nan)
        
        # NaN 값을 중앙값으로 대체
        for col in X.columns:
            if X[col].isnull().any():
                median_val = X[col].median()
                if np.isnan(median_val):  # 열 전체가 NaN인 경우
                    median_val = 0
                X[col] = X[col].fillna(median_val)
        
        # 특수 값 처리: 매우 큰 값을 감지하고 대체
        for col in X.columns:
            # 매우 큰 값(예: float32 범위를 초과하는 값) 확인
            if X[col].abs().max() > 1e30:
                # 문제가 있는 열을 표준화하거나 값 범위 제한
                X[col] = np.clip(X[col], -1e30, 1e30)
                print(f"열 '{col}'에 매우 큰 값이 있어 범위를 제한했습니다.")

        initial_feature_count = X.shape[1]

        model_abbr = None
        feature_importances = None

        def get_model(model_name):
            if model_name == "random_forest":
                return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                          min_samples_split=min_samples_split, criterion=criterion), "RF"
            elif model_name == "decision_tree":
                return DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                          criterion=criterion), "DT"
            elif model_name == "xgboost":
                return XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                 learning_rate=learning_rate, random_state=42, eval_metric="logloss"), "XGB"
            elif model_name == "lightgbm":
                return LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                  learning_rate=learning_rate, random_state=42), "LGBM"
            elif model_name == "lasso":
                return LogisticRegression(penalty='l1', solver='saga', C=1/alpha, max_iter=max_iter, random_state=42), "LASSO"
            else:
                raise ValueError(f"Invalid model name '{model_name}'.")

        try:
            if method == "lasso":
                model, model_abbr = get_model("lasso")
                
                # 스케일링 적용
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                model.fit(X_scaled, y)
                selected_columns = X.columns[model.coef_[0] != 0]
                X_new = X[selected_columns]

            elif method == "rfe":
                if model is None:
                    raise ValueError("'rfe' method requires a model.")
                
                # 스케일링 적용
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                model, model_abbr = get_model(model)
                print(f"RFE 특성 선택 시작: 모델={model_abbr}, 선택할 특성 수={n_features}")
                
                try:
                    selector = RFE(estimator=model, n_features_to_select=n_features, step=0.05)
                    selector.fit(X_scaled, y)
                    selected_columns = X.columns[selector.get_support()]
                    X_new = X[selected_columns]
                    print(f"RFE 특성 선택 완료: {len(selected_columns)}개 특성 선택됨")
                except Exception as e:
                    print(f"RFE 오류 발생: {e}")
                    print("대체 전략 시도: SelectFromModel 사용")
                    
                    # RFE 실패 시 SelectFromModel로 대체
                    model.fit(X_scaled, y)
                    if hasattr(model, "feature_importances_"):
                        feature_importances = model.feature_importances_
                    elif hasattr(model, "coef_"):
                        feature_importances = np.abs(model.coef_[0])
                    
                    # 특성 중요도를 기준으로 상위 n_features 선택
                    threshold = np.sort(feature_importances)[-min(n_features, len(feature_importances))]
                    selector = SelectFromModel(estimator=model, threshold=threshold, prefit=True)
                    selected_columns = X.columns[selector.get_support()]
                    X_new = X[selected_columns]
                    print(f"대체 전략 완료: {len(selected_columns)}개 특성 선택됨")

            elif method == "select_from_model":
                if model is None:
                    raise ValueError("'select_from_model' method requires a model.")
                
                # 스케일링 적용
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                model, model_abbr = get_model(model)
                model.fit(X_scaled, y)

                if hasattr(model, "feature_importances_"):
                    feature_importances = model.feature_importances_
                elif hasattr(model, "coef_"):
                    feature_importances = np.abs(model.coef_[0])
                else:
                    raise ValueError(f"The model {model} does not provide feature importances.")

                if isinstance(threshold, str) and "percentile" in threshold:
                    percentile_value = int(threshold.replace("percentile(", "").replace(")", ""))
                    threshold = np.percentile(feature_importances, percentile_value)

                selector = SelectFromModel(estimator=model, threshold=threshold, prefit=True)
                selected_columns = X.columns[selector.get_support()]
                X_new = X[selected_columns]

            elif method in ["random_forest", "decision_tree"]:
                feature_importance_matrix = np.zeros((n_iterations, X.shape[1]))

                # 스케일링 적용
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                for i in range(n_iterations):
                    model_iter, model_abbr = get_model(method)
                    model_iter.fit(X_scaled, y)
                    feature_importance_matrix[i] = model_iter.feature_importances_

                feature_importances = np.mean(feature_importance_matrix, axis=0)

                if isinstance(threshold, str) and "percentile" in threshold:
                    percentile_value = int(threshold.replace("percentile(", "").replace(")", ""))
                    threshold = np.percentile(feature_importances, percentile_value)
                else:
                    threshold = float(threshold)

                important_indices = np.where(feature_importances >= threshold)[0]
                selected_columns = X.columns[important_indices]
                X_new = X[selected_columns]

            elif method in ["xgboost", "lightgbm"]:
                # 스케일링 적용
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                model, model_abbr = get_model(method)
                model.fit(X_scaled, y)
                feature_importances = model.feature_importances_

                if isinstance(threshold, str) and "percentile" in threshold:
                    percentile_value = int(threshold.replace("percentile(", "").replace(")", ""))
                    threshold = np.percentile(feature_importances, percentile_value)
                else:
                    threshold = float(threshold)

                important_indices = np.where(feature_importances >= threshold)[0]
                selected_columns = X.columns[important_indices]
                X_new = X[selected_columns]

            else:
                raise ValueError(f"Unsupported method '{method}'")

        except Exception as e:
            error_msg = f"특성 선택 중 오류 발생: {str(e)}"
            print(error_msg)
            
            # 오류 발생 시 간단한 전략으로 대체
            print("오류로 인해 기본 상관관계 기반 특성 선택으로 대체합니다.")
            
            # 타겟과의 상관관계 계산
            correlations = []
            for col in X.columns:
                try:
                    corr = abs(X[col].corr(pd.Series(y)))
                    if np.isnan(corr):
                        corr = 0
                    correlations.append((col, corr))
                except:
                    correlations.append((col, 0))
            
            # 상관관계 기준 정렬
            correlations.sort(key=lambda x: x[1], reverse=True)
            
            # 상위 n_features 선택
            selected_columns = [item[0] for item in correlations[:min(n_features, len(correlations))]]
            X_new = X[selected_columns]
            model_abbr = "CORR"
            
            print(f"상관관계 기반 선택 완료: {len(selected_columns)}개 특성 선택됨")
            
        # 결과 생성
        final_feature_count = len(selected_columns)
        removed_features = initial_feature_count - final_feature_count

        selected_features = X_new.copy()
        selected_features["Label"] = y.reset_index(drop=True)

        filename = f"features_{method}_{model_abbr}_{initial_feature_count}_{final_feature_count}.csv"
        output_file = os.path.join("QSAR/Selection", filename)
        selected_features.to_csv(output_file, index=False)

        text_container = create_text_container(
            "🔹 Feature Selection Completed! 🔹",
            f"📌 Method: {method} ({model_abbr})",
            f"📊 Initial Features: {initial_feature_count}",
            f"📉 Selected Features: {final_feature_count}",
            f"🗑️ Removed: {removed_features}",
        )

        return {
            "ui": {"text": text_container},
            "result": (str(output_file),)
        }

# 노드 등록
NODE_CLASS_MAPPINGS = {
    "Feature_Selection_Classification": Feature_Selection_Classification
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Feature_Selection_Classification": "Feature Selection(Classification)"
} 