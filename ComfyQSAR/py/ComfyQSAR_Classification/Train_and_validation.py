import os
import pandas as pd
import joblib
import numpy as np # Import numpy for NaN handling
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score, confusion_matrix) # Added confusion_matrix
# from .Data_Loader import create_text_container # 이제 progress_utils에서 가져옴

# --- 공통 유틸리티 임포트 ---
try:
    from .Data_Loader import send_progress, create_text_container
except ImportError:
    print("[ComfyQSAR Model Validation] Warning: Could not import progress_utils. Progress updates might not work.")
    # 대체 함수 정의
    def send_progress(message, progress=None, node_id=None):
        print(f"[Progress Fallback] {message}" + (f" ({progress}%)" if progress is not None else ""))
    # 대체 create_text_container 정의
    def create_text_container(*lines):
        return "\n".join(lines)

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
        send_progress("🚀 Starting Model Validation...", 0)
        output_dir = "QSAR/Validation"
        try:
            os.makedirs(output_dir, exist_ok=True)
            send_progress(f"📂 Output directory created/checked: {output_dir}", 5)
        except Exception as e:
            error_msg = f"❌ Error creating output directory: {str(e)}"
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": ("", "")}

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
            result_text = create_text_container(
                f"🔹 **Classification Model Validation Complete** 🔹",
                f"Model Used: {os.path.basename(model_path)}",
                f"Features Used: From {os.path.basename(selected_descriptors_path)}",
                "--- Test Set Evaluation Metrics ---",
                f"Accuracy:            {test_accuracy:.4f}",
                f"F1 Score:            {test_f1:.4f}",
                f"ROC AUC:             {test_roc_auc:.4f}" if not pd.isna(test_roc_auc) else "ROC AUC:             N/A",
                f"Precision:           {test_precision:.4f}",
                f"Recall (Sensitivity):{test_recall:.4f}",
                f"Specificity:         {test_specificity:.4f}" if not pd.isna(test_specificity) else "Specificity:         N/A",
                "--- Saved Files ---",
                f"Evaluation Metrics:  {eval_csv_path}",
                f"Predictions:         {pred_csv_path}"
            )
            send_progress("🎉 Validation finished successfully!", 100)

            return {
                "ui": {"text": result_text},
                "result": (str(eval_csv_path), str(pred_csv_path),)
            }

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}. Please check input file paths."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": ("", "")}
        except ValueError as val_e:
            error_msg = f"❌ Value Error: {str(val_e)}."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": ("", "")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred during validation: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": ("", "")}


# Node registration
NODE_CLASS_MAPPINGS = {
    "Model_Validation_Classification": Model_Validation_Classification
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Model_Validation_Classification": "Model Validation (Classification)" # Updated display name
} 