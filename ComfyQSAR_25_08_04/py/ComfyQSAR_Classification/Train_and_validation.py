import os
import pandas as pd
import joblib
import numpy as np # Import numpy for NaN handling
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score, confusion_matrix) # Added confusion_matrix
# from .Data_Loader import create_text_container # 이제 progress_utils에서 가져옴

# --- 공통 유틸리티 임포트 ---
try:
    from .Data_Loader import send_progress, create_qsar_pages, create_text_container
except ImportError:
    print("[ComfyQSAR Model Validation] Warning: Could not import progress_utils. Progress updates might not work.")
    # 대체 함수 정의
    def send_progress(message, progress=None, node_id=None):
        print(f"[Progress Fallback] {message}" + (f" ({progress}%)" if progress is not None else ""))
    
    # QSAR 페이지 시스템 헬퍼 함수들
    def create_qsar_pages(title, pages_data=None, simple_text=None):
        """
        QSAR 페이지 시스템에 맞는 데이터 생성 함수
        """
        import json
        
        if simple_text is not None:
            # 단순 텍스트 모드 (오류 메시지 등)
            if isinstance(simple_text, list):
                text_content = "\n".join(str(line) for line in simple_text)
            else:
                text_content = str(simple_text)
            
            # 오류 메시지는 간단한 페이지 형태로 감싸기
            error_pages = {"error": text_content}
            encoded_title = f"QSAR_TITLE:{json.dumps(title)}"
            encoded_pages = f"QSAR_PAGES:{json.dumps(error_pages)}"
            
        elif pages_data is not None:
            # 페이지 시스템 모드
            encoded_title = f"QSAR_TITLE:{json.dumps(title)}"
            encoded_pages = f"QSAR_PAGES:{json.dumps(pages_data)}"
            
        else:
            # 기본 오류 처리
            error_pages = {"error": "No data provided"}
            encoded_title = f"QSAR_TITLE:{json.dumps('Error')}"
            encoded_pages = f"QSAR_PAGES:{json.dumps(error_pages)}"
        
        return encoded_title, encoded_pages
    
    # 대체 create_text_container 정의 (레거시 지원)
    def create_text_container(*lines):
        """
        레거시 지원을 위한 함수 - 이제 페이지 시스템 사용을 권장
        단순 텍스트 연결 또는 오류 메시지용
        """
        if len(lines) == 1 and isinstance(lines[0], str):
            # 단일 오류 메시지의 경우 페이지 시스템으로 변환
            title = "❌ Error" if "Error" in lines[0] or "❌" in lines[0] else "📄 Information"
            return create_qsar_pages(title, simple_text=lines[0])
        else:
            # 다중 라인의 경우 단순 연결
            result = []
            for line in lines:
                result.append(str(line))
            return "\n".join(result)

class Model_Validation_Classification:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"tooltip": "Path to the trained model file"}),
                "selected_descriptors_path": ("STRING", {"tooltip": "Path to the selected descriptors file"}),
                "X_test_path": ("STRING", {"tooltip": "Path to the test data file (X_test)"}),
                "y_test_path": ("STRING", {"tooltip": "Path to the test data file (y_test)"}),
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
        send_progress("🚀 Starting Model Validation...", 0)
        output_dir = "QSAR/Validation"
        try:
            os.makedirs(output_dir, exist_ok=True)
            send_progress(f"📂 Output directory created/checked: {output_dir}", 5)
        except Exception as e:
            error_msg = f"❌ Error creating output directory: {str(e)}"
            send_progress(error_msg)
            title, pages = create_qsar_pages("❌ Model Validation Error", simple_text=error_msg)
            return {"ui": {"text": title, "text2": pages}, "result": ("", "")}

        try:
            # Load model
            send_progress(f"⏳ Loading model from: {model_path}", 10)
            model = joblib.load(model_path)
            send_progress("   Model loaded successfully.", 15)

            # Load test data
            send_progress(f"⏳ Loading test data (X_test): {X_test_path}", 20)
            X_test_df = pd.read_csv(X_test_path)
            send_progress(f"⏳ Loading test data (y_test): {y_test_path}", 25)
            y_test_df = pd.read_csv(y_test_path)
            y_test = y_test_df.iloc[:, 0] # Assume target is the first column
            send_progress("   Test data loaded.", 28)

            # Load selected descriptors
            send_progress(f"⏳ Loading selected features from: {selected_descriptors_path}", 30)
            with open(selected_descriptors_path, "r") as f:
                selected_features = [line.strip() for line in f.readlines() if line.strip()] # Avoid empty lines
            send_progress(f"   {len(selected_features)} features loaded.", 35)

            # Ensure all selected features are in X_test
            missing_features = [f for f in selected_features if f not in X_test_df.columns]
            if missing_features:
                raise ValueError(f"Missing features in X_test: {', '.join(missing_features)}")

            # Filter X_test using selected features
            send_progress("⚙️ Filtering test features...", 40)
            X_test_filtered = X_test_df[selected_features]
            send_progress("   Test features filtered.", 45)

            # Make predictions
            send_progress("🧠 Making predictions on test set...", 50)
            y_pred = model.predict(X_test_filtered)
            send_progress("   Predictions complete.", 60)

            # Predict probabilities (attempt)
            send_progress("📊 Calculating prediction probabilities (if available)...", 65)
            y_proba = None
            test_roc_auc = np.nan # Default to NaN
            try:
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test_filtered)[:, 1] # Probability of positive class
                    test_roc_auc = roc_auc_score(y_test, y_proba)
                    send_progress("   Probabilities calculated.", 68)
                else:
                    send_progress("   predict_proba not available for this model.", 68)
            except Exception as proba_e:
                print(f"Warning: Could not calculate probabilities or ROC AUC: {str(proba_e)}")
                send_progress(f"   Warning: Could not calculate probabilities: {str(proba_e)}", 68)


            # Calculate classification metrics
            send_progress("📊 Calculating evaluation metrics...", 70)
            test_accuracy = accuracy_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred, zero_division=0)
            test_precision = precision_score(y_test, y_pred, zero_division=0)
            test_recall = recall_score(y_test, y_pred, zero_division=0) # Sensitivity

            # Calculate Specificity from confusion matrix
            test_specificity = np.nan # Default to NaN
            try:
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                if (tn + fp) > 0:
                    test_specificity = tn / (tn + fp)
                else:
                    test_specificity = 0.0 # Or handle as appropriate if tn+fp is 0
            except ValueError: # Handle cases where confusion matrix is not 2x2 (e.g., only one class predicted)
                print("Warning: Could not calculate specificity (likely due to predictions being all one class).")
                send_progress("   Warning: Could not calculate specificity.", 75)


            send_progress("   Metrics calculated.", 78)

            # Save prediction results
            send_progress("💾 Saving actual vs predicted results...", 80)
            pred_df = pd.DataFrame({
                "Actual": y_test.values,
                "Predicted": y_pred
            })
            if y_proba is not None:
                pred_df["Probability_Class1"] = y_proba
            pred_csv_path = os.path.join(output_dir, "Actual_vs_Predicted_Validation.csv")
            pred_df.to_csv(pred_csv_path, index=False)
            send_progress(f"   Predictions saved to: {pred_csv_path}", 85)

            # Save evaluation metrics
            send_progress("💾 Saving evaluation metrics...", 88)
            eval_data = {
                "Metric": ["Accuracy", "F1-Score", "ROC-AUC", "Precision", "Recall (Sensitivity)", "Specificity"],
                "Value": [test_accuracy, test_f1, test_roc_auc, test_precision, test_recall, test_specificity]
            }
            eval_df = pd.DataFrame(eval_data)
            eval_csv_path = os.path.join(output_dir, "Evaluation_Results_Validation.csv")
            eval_df.to_csv(eval_csv_path, index=False)
            send_progress(f"   Evaluation metrics saved to: {eval_csv_path}", 92)

            # Generate final report text
            send_progress("📝 Generating final report...", 95)
            pages_title = "🔹 Classification Model Validation Complete! 🔹"
            
            # Build evaluation metrics text with proper formatting
            eval_metrics = [
                f"Accuracy: {test_accuracy:.4f}",
                f"F1 Score: {test_f1:.4f}",
                f"Precision: {test_precision:.4f}",
                f"Recall (Sensitivity): {test_recall:.4f}"
            ]
            
            if not pd.isna(test_roc_auc):
                eval_metrics.append(f"ROC AUC: {test_roc_auc:.4f}")
            else:
                eval_metrics.append("ROC AUC: N/A")
                
            if not pd.isna(test_specificity):
                eval_metrics.append(f"Specificity: {test_specificity:.4f}")
            else:
                eval_metrics.append("Specificity: N/A")
            
            pages_data = {
                "summary": f"Model Used: {os.path.basename(model_path)}\nFeatures Used: From {os.path.basename(selected_descriptors_path)}\nTest Set Size: {len(y_test)} samples",
                "test_set_evaluation": f"📊 Performance Metrics:\n" + "\n".join([f"  - {metric}" for metric in eval_metrics]),
                "saved_files": f"💾 Output Files:\n  - Evaluation Metrics: {eval_csv_path}\n  - Predictions: {pred_csv_path}"
            }
            
            title, pages = create_qsar_pages(pages_title, pages_data)
            send_progress("🎉 Validation finished successfully!", 100)

            return {
                "ui": {"text": title, "text2": pages},
                "result": (str(eval_csv_path), str(pred_csv_path),)
            }

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}. Please check input file paths."
            send_progress(error_msg)
            title, pages = create_qsar_pages("❌ Model Validation Error", simple_text=error_msg)
            return {"ui": {"text": title, "text2": pages}, "result": ("", "")}
        except ValueError as val_e:
            error_msg = f"❌ Value Error: {str(val_e)}."
            send_progress(error_msg)
            title, pages = create_qsar_pages("❌ Model Validation Error", simple_text=error_msg)
            return {"ui": {"text": title, "text2": pages}, "result": ("", "")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred during validation: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            send_progress(error_msg)
            title, pages = create_qsar_pages("❌ Model Validation Error", simple_text=error_msg)
            return {"ui": {"text": title, "text2": pages}, "result": ("", "")}


# Node registration
NODE_CLASS_MAPPINGS = {
    "Model_Validation_Classification": Model_Validation_Classification
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Model_Validation_Classification": "Model Validation (Classification)" # Updated display name
} 