import os
import joblib
import numpy as np
import pandas as pd
import multiprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from .Data_Loader import create_text_container
class Hyperparameter_Grid_Search_Classification:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_file": ("STRING",),
                "algorithm": (["xgboost", "random_forest", "decision_tree", "lightgbm", "logistic", "lasso", "svm"],),
                "advanced": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                # 공통 파라미터
                "target_column": ("STRING", {"default": "Label"}),
                "test_size": ("FLOAT", {"default": 0.2, "min": 0.1, "max": 0.5, "step": 0.05}),
                "num_cores": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),
                "cv_splits": ("INT", {"default": 5, "min": 3, "max": 10, "step": 1}),
                "verbose": ("INT", {"default": 1, "min": 0, "max": 2, "step": 1}),
                "random_state": ("INT", {"default": 42, "min": 0, "max": 999, "step": 1}),
                
                # 트리 기반 알고리즘 파라미터 (XGBoost, Random Forest, LightGBM)
                "n_estimators": ("STRING", {"default": "[100, 200, 300]", "description": "tree number (XGBoost, Random Forest, LightGBM)"}),
                "max_depth": ("STRING", {"default": "[3, 5, 7]", "description": "tree max depth (XGBoost, Random Forest, Decision Tree, LightGBM)"}),
                "learning_rate": ("STRING", {"default": "[0.01, 0.05, 0.1]", "description": "learning rate (XGBoost, LightGBM)"}),
                
                # 트리 세부 파라미터
                "min_samples_split": ("STRING", {"default": "[2, 5, 10]", "description": "min samples split (Random Forest, Decision Tree)"}),
                "min_samples_leaf": ("STRING", {"default": "[1, 2, 4]", "description": "min samples leaf (Decision Tree)"}),
                "criterion": ("STRING", {"default": "['gini', 'entropy']", "description": "criterion (Decision Tree)"}),
                
                # LightGBM 전용 파라미터
                "subsample": ("STRING", {"default": "[0.6, 0.8, 1.0]", "description": "subsampling ratio (LightGBM)"}),
                "reg_alpha": ("STRING", {"default": "[0.1, 1, 10]", "description": "L1 regularization (LightGBM)"}),
                "reg_lambda": ("STRING", {"default": "[1, 10, 100]", "description": "L2 regularization (LightGBM)"}),
                
                # 선형 모델 파라미터 (Logistic, Lasso, SVM)
                "C": ("STRING", {"default": "[0.01, 0.1, 1, 10, 100]", "description": "inverse of regularization strength (Logistic, Lasso, SVM)"}),
                "penalty": ("STRING", {"default": "['l2']", "description": "regularization type (Logistic)"}),
                "solver": ("STRING", {"default": "['lbfgs']", "description": "optimization algorithm (Logistic)"}),
                
                # SVM 전용 파라미터
                "kernel": ("STRING", {"default": "['linear', 'rbf']", "description": "kernel function (SVM)"}),
                "gamma": ("STRING", {"default": "['scale', 'auto']", "description": "kernel coefficient (SVM)"})
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("MODEL_PATH", "DESCRIPTORS_PATH", "X_TEST_PATH", "Y_TEST_PATH",)
    FUNCTION = "perform_grid_search"
    CATEGORY = "QSAR/CLASSIFICATION/GRID SEARCH"
    OUTPUT_NODE = True
    
    def perform_grid_search(self, input_file, algorithm, advanced,
                          test_size=0.2, num_cores=4, cv_splits=5, verbose=1, 
                          target_column="Label", random_state=42,
                          # 알고리즘별 하이퍼파라미터
                          n_estimators="[100, 200, 300]",
                          learning_rate="[0.01, 0.05, 0.1]",
                          max_depth="[3, 5, 7]",
                          min_samples_split="[2, 5, 10]",
                          min_samples_leaf="[1, 2, 4]",
                          criterion="['gini', 'entropy']",
                          subsample="[0.6, 0.8, 1.0]",
                          reg_alpha="[0.1, 1, 10]",
                          reg_lambda="[1, 10, 100]",
                          C="[0.01, 0.1, 1, 10, 100]",
                          penalty="['l2']",
                          solver="['lbfgs']",
                          kernel="['linear', 'rbf']",
                          gamma="['scale', 'auto']"):
        """
        Perform grid search for model hyperparameter optimization.
        """
        os.makedirs("QSAR/Model", exist_ok=True)
        
        # 문자열 파라미터 파싱 헬퍼 함수
        def parse_param(param_str):
            try:
                # None 문자열 처리
                if "None" in param_str:
                    param_str = param_str.replace("None", "None")
                parsed = eval(param_str)
                return parsed
            except Exception as e:
                print(f"파라미터 파싱 오류: {str(e)}")
                # 기본값 반환
                if "None" in param_str:
                    return [None]
                return [0]

        # 데이터 로드
        data = pd.read_csv(input_file)

        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        X = data.drop(columns=[target_column])
        y = data[target_column]

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state, stratify=y)

        # CPU 코어 수 설정
        available_cores = max(1, multiprocessing.cpu_count())  
        num_cores = min(num_cores, available_cores)  

        # 파라미터 그리드 초기화
        param_grid = {}

        # 모델과 하이퍼파라미터 그리드 설정
        if algorithm == "xgboost":
            model = XGBClassifier(eval_metric="logloss", random_state=random_state)
            model_abbr = "XGB"
            # 사용자 지정 파라미터 파싱 - XGBoost 관련 파라미터
            param_grid = {
                'n_estimators': parse_param(n_estimators),
                'learning_rate': parse_param(learning_rate),
                'max_depth': parse_param(max_depth),
            }
            
        elif algorithm == "random_forest":
            model = RandomForestClassifier(random_state=random_state)
            model_abbr = "RF"
            # Random Forest 관련 파라미터
            param_grid = {
                'n_estimators': parse_param(n_estimators),
                'max_depth': parse_param(max_depth),
                'min_samples_split': parse_param(min_samples_split),
            }
            
        elif algorithm == "decision_tree":
            model = DecisionTreeClassifier(random_state=random_state)
            model_abbr = "DT"
            # Decision Tree 관련 파라미터
            param_grid = {
                'max_depth': parse_param(max_depth),
                'min_samples_split': parse_param(min_samples_split),
                'min_samples_leaf': parse_param(min_samples_leaf),
                'criterion': parse_param(criterion),
            }
            
        elif algorithm == "lightgbm":
            model = LGBMClassifier(random_state=random_state)
            model_abbr = "LGBM"
            # LightGBM 관련 파라미터
            param_grid = {
                'n_estimators': parse_param(n_estimators),
                'learning_rate': parse_param(learning_rate),
                'max_depth': parse_param(max_depth),
                'subsample': parse_param(subsample),
                'reg_alpha': parse_param(reg_alpha),
                'reg_lambda': parse_param(reg_lambda),
            }
            
        elif algorithm == "logistic":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=random_state))
            ])
            model_abbr = "LogReg"
            # Logistic Regression 관련 파라미터
            param_grid = {
                'clf__C': parse_param(C),
                'clf__penalty': parse_param(penalty),
                'clf__solver': parse_param(solver),
            }
            
        elif algorithm == "lasso":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=random_state))
            ])
            model_abbr = "LASSO"
            # Lasso 관련 파라미터
            param_grid = {
                'clf__C': parse_param(C)
            }
            
        elif algorithm == "svm":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(probability=True, random_state=random_state))
            ])
            model_abbr = "SVM"
            # SVM 관련 파라미터
            param_grid = {
                'clf__C': parse_param(C),
                'clf__kernel': parse_param(kernel),
                'clf__gamma': parse_param(gamma),
            }
            
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # 파라미터 그리드 확인 및 디버깅
        param_grid_str = "\n".join([f"{k}: {v}" for k, v in param_grid.items()])
        print(f"Algorithm: {algorithm}")
        print(f"Parameter Grid:\n{param_grid_str}")

        # 평가 지표 설정
        scoring = {
            'accuracy': 'accuracy',
            'f1': make_scorer(f1_score),
            'roc_auc': 'roc_auc',
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
        }

        # 교차 검증 설정
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        
        # 그리드 서치 수행
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring,
                                refit='accuracy', return_train_score=True,
                                verbose=verbose, n_jobs=num_cores)

        grid_search.fit(X_train, y_train)

        # 최적 모델 평가
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        predictions = best_model.predict(X_test)
        eval_results = {
            "accuracy": accuracy_score(y_test, predictions),
            "f1_score": f1_score(y_test, predictions),
            "roc_auc": roc_auc_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
        }

        # 파일 저장
        model_path = os.path.join("QSAR/Model", f"Best_Classifier_{model_abbr}.pkl")
        joblib.dump(best_model, model_path)

        descriptors_path = os.path.join("QSAR/Model", "Final_Selected_Descriptors.txt")
        with open(descriptors_path, "w") as f:
            f.write("\n".join(X_train.columns))

        # 테스트 데이터 저장
        X_test_path = os.path.join("QSAR/Model", "X_test.csv")
        y_test_path = os.path.join("QSAR/Model", "y_test.csv")
            
        X_test.to_csv(X_test_path, index=False)
        pd.DataFrame(y_test, columns=[target_column]).to_csv(y_test_path, index=False)

        # 최적 하이퍼파라미터 저장
        best_params_path = os.path.join("QSAR/Model", f"Best_Hyperparameters_{model_abbr}.txt")
        with open(best_params_path, "w") as f:
            for param, value in best_params.items():
                f.write(f"{param}: {value}\n")

        # 결과 로그 생성
        text_container = create_text_container(
            "🔹 Classification Model Training Complete 🔹",
            f"📌 Best Algorithm: {algorithm}",
            f"📊 Accuracy: {eval_results['accuracy']:.4f}",
            f"📊 F1 Score: {eval_results['f1_score']:.4f}",
            f"📊 ROC AUC: {eval_results['roc_auc']:.4f}",
            f"📊 Precision: {eval_results['precision']:.4f}",
            f"📊 Recall: {eval_results['recall']:.4f}",
            f"🔧 Best Parameters:" + "\n".join([f"  - {k}: {v}" for k, v in best_params.items()]),
        )

        return {
            "ui": {"text": text_container},
            "result": (str(model_path), str(descriptors_path), str(X_test_path), str(y_test_path),)
        }

# 노드 등록
NODE_CLASS_MAPPINGS = {
    "Hyperparameter_Grid_Search_Classification": Hyperparameter_Grid_Search_Classification
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hyperparameter_Grid_Search_Classification": "Grid Search Hyperparameter (Classification)"
} 