import os
import pandas as pd
import joblib
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score)
from .Data_Loader import create_text_container

class Model_Validation_Classification:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING",),
                "selected_descriptors_path": ("STRING",),
                "X_test_path": ("STRING",),
                "y_test_path": ("STRING",),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("EVALUATION_PATH", "PREDICTION_PATH",)
    FUNCTION = "validate_model"
    CATEGORY = "QSAR/CLASSIFICATION/VALIDATION"
    OUTPUT_NODE = True
    
    def validate_model(self, model_path, X_test_path, y_test_path, selected_descriptors_path):
        """
        Validate a trained classification model on the test set.
        """
        os.makedirs("QSAR/Validation", exist_ok=True)

        # 모델 로드
        model = joblib.load(model_path)

        # 테스트 데이터 로드
        X_test = pd.read_csv(X_test_path)            
        y_test = pd.read_csv(y_test_path)
        # 선택된 디스크립터 로드
        with open(selected_descriptors_path, "r") as f:
            selected_features = [line.strip() for line in f.readlines()]

        # 테스트 데이터에서 선택된 특성만 사용
        X_test = X_test[selected_features]

        # 예측 수행
        y_pred = model.predict(X_test)

        # 확률값 예측 시도
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            test_roc_auc = roc_auc_score(y_test, y_proba)
        except Exception:
            y_proba = None
            test_roc_auc = None

        # 분류 메트릭 계산
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred)
        test_recall = recall_score(y_test, y_pred)
        test_specificity = recall_score(y_test, y_pred, pos_label=0) if len(set(y_test)) == 2 else None

        # 예측 결과 저장
        pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        
        # 예측 확률 추가 (가능한 경우)
        if y_proba is not None:
            pred_df["Probability"] = y_proba
            
        pred_csv_path = os.path.join("QSAR/Validation", "Actual_vs_Predicted.csv")
        pred_df.to_csv(pred_csv_path, index=False)

        # 평가 지표 저장
        eval_data = {
            "Metric": ["Accuracy", "F1-Score", "Precision", "Recall (Sensitivity)", "Specificity"],
            "Value": [test_accuracy, test_f1, test_roc_auc, test_precision, test_recall, test_specificity]
        }
            
        eval_df = pd.DataFrame(eval_data)
        eval_csv_path = os.path.join("QSAR/Validation", "Evaluation_Results_TestSet.csv")
        eval_df.to_csv(eval_csv_path, index=False)

        # 로그 메시지 생성
        text_container = create_text_container(
            "🔹 Classification Model Validation Done! 🔹",
            f"📁 Model: {model_path}",
            f"📊 Accuracy: {test_accuracy:.4f}",
            f"📊 F1 Score: {test_f1:.4f}",
            f"📊 ROC-AUC: {test_roc_auc:.4f}" if test_roc_auc is not None else "📊 ROC-AUC: Not Available",
            f"📊 Precision: {test_precision:.4f}",
            f"📊 Recall (Sensitivity): {test_recall:.4f}",
            f"📊 Specificity: {test_specificity:.4f}" if test_specificity is not None else "📊 Specificity: Not Available",
        )

        return {
            "ui": {"text": text_container},
            "result": (str(eval_csv_path), str(pred_csv_path),)
        }

# 노드 등록
NODE_CLASS_MAPPINGS = {
    "Model_Validation_Classification": Model_Validation_Classification
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Model_Validation_Classification": "Model Validation(Classification)"
} 