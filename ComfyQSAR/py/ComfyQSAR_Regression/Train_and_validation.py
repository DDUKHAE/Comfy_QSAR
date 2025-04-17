import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# from .Data_Loader import create_text_container # Now imported below

# --- Common Utility Import ---
try:
    from .Data_Loader import send_progress, create_text_container
except ImportError:
    print("[ComfyQSAR Regression Validation] Warning: Could not import progress_utils. Progress updates might not work.")
    # Fallback functions
    def send_progress(message, progress=None, node_id=None):
        print(f"[Progress Fallback] {message}" + (f" ({progress}%)" if progress is not None else ""))
    def create_text_container(*lines):
        return "\n".join(lines)

class Model_Validation_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        # Keep existing INPUT_TYPES, but use updated names from Grid Search node for clarity
        return {
            "required": {
                "model_path": ("STRING",),
                "selected_descriptors_path": ("STRING",), # Renamed for clarity
                "X_test_path": ("STRING",),           # Renamed for clarity
                "y_test_path": ("STRING",),           # Renamed for clarity
            },
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    # Updated names for consistency
    RETURN_NAMES = ("EVALUATION_RESULTS_PATH", "PREDICTIONS_VS_ACTUAL_PATH",)
    FUNCTION = "validate_regression_model" # Renamed function
    CATEGORY = "QSAR/REGRESSION/VALIDATION"
    OUTPUT_NODE = True # Keep OUTPUT_NODE = True if it should appear in the UI list
    
    def validate_regression_model(self, model_path, X_test_path, y_test_path, selected_descriptors_path):
        send_progress("🚀 Starting Regression Model Validation...", 0)
        output_dir = "QSAR/Train_and_Validation"
        eval_csv_path, predictions_csv_path = "", "" # Default empty paths

        try:
            os.makedirs(output_dir, exist_ok=True)
            send_progress(f"📂 Output directory checked/created: {output_dir}", 5)

            send_progress(f"⏳ Loading trained model from: {model_path}", 10)
            best_model = joblib.load(model_path)
            send_progress("   Model loaded successfully.", 15)

            send_progress(f"⏳ Loading test set features from: {X_test_path}", 20)
            X_test = pd.read_csv(X_test_path)
            send_progress(f"   Test features loaded ({X_test.shape[0]} samples, {X_test.shape[1]} features).", 25)

            send_progress(f"⏳ Loading test set target from: {y_test_path}", 30)
            # Make sure target column name is extracted correctly if y_test.csv has header
            y_test_df = pd.read_csv(y_test_path)
            if y_test_df.shape[1] != 1:
                 raise ValueError(f"y_test file ({y_test_path}) should contain exactly one column (the target variable). Found {y_test_df.shape[1]} columns.")
            y_test = y_test_df.iloc[:, 0].values # Get the first column as a 1D numpy array
            target_column_name = y_test_df.columns[0] # Get the actual target column name
            send_progress(f"   Test target loaded ('{target_column_name}', {len(y_test)} samples).", 35)

            send_progress(f"⏳ Loading selected descriptor list from: {selected_descriptors_path}", 40)
            with open(selected_descriptors_path, "r") as f:
                # Read lines, strip whitespace/newlines, filter empty lines
                selected_features = [line.strip() for line in f if line.strip()]
            send_progress(f"   Loaded {len(selected_features)} selected descriptors.", 45)

            send_progress("⚙️ Filtering test set features to match selected descriptors...", 50)
            missing_cols = [col for col in selected_features if col not in X_test.columns]
            if missing_cols:
                raise ValueError(f"The following selected descriptors are missing from the X_test data: {missing_cols}")

            X_test_filtered = X_test[selected_features]
            send_progress(f"   Test set filtered to {X_test_filtered.shape[1]} features.", 55)

            send_progress("🤖 Making predictions on the test set...", 60)
            predictions = best_model.predict(X_test_filtered)
            send_progress("   Predictions complete.", 65)

            send_progress("💾 Saving actual vs. predicted values...", 70)
            results_df = pd.DataFrame({
                f'Actual_{target_column_name}': y_test.ravel(), # Use actual target name
                'Predicted': predictions.ravel()
            })
            predictions_csv_path = os.path.join(output_dir, "Regression_Actual_vs_Predicted.csv")
            results_df.to_csv(predictions_csv_path, index=False)
            send_progress(f"   Predictions saved to: {predictions_csv_path}", 75)

            send_progress("📊 Calculating evaluation metrics...", 80)
            test_r2 = r2_score(y_test, predictions)
            test_mse = mean_squared_error(y_test, predictions)
            test_rmse = np.sqrt(test_mse)
            test_mae = mean_absolute_error(y_test, predictions)

            # Calculate MAPE, handling potential division by zero
            # Avoid division by zero or near-zero actual values
            non_zero_mask = np.abs(y_test) > 1e-8
            if np.any(non_zero_mask):
                 test_mape = np.mean(np.abs((y_test[non_zero_mask] - predictions[non_zero_mask]) / y_test[non_zero_mask])) * 100
            else:
                 test_mape = np.inf # Or np.nan, or some other indicator
                 send_progress("   ⚠️ Warning: Cannot calculate MAPE as all actual values are zero or near-zero.", 82)

            send_progress("   Metrics calculated.", 85)

            send_progress("💾 Saving evaluation metrics...", 86)
            eval_results = {
                "Metric": ["R2 Score", "MSE", "RMSE", "MAE", "MAPE (%)"],
                "Value": [test_r2, test_mse, test_rmse, test_mae, test_mape]
            }
            eval_df = pd.DataFrame(eval_results)
            eval_csv_path = os.path.join(output_dir, "Regression_Evaluation_Metrics.csv")
            eval_df.to_csv(eval_csv_path, index=False)
            send_progress(f"   Evaluation metrics saved to: {eval_csv_path}", 90)

            send_progress("📝 Generating final summary...", 95)
            summary_lines = [
                "🔹 **Regression Model Validation Completed!** 🔹",
                f"Model File: {os.path.basename(model_path)}",
                f"Test Features File: {os.path.basename(X_test_path)}",
                f"Test Target File: {os.path.basename(y_test_path)}",
                f"Selected Descriptors File: {os.path.basename(selected_descriptors_path)}",
                "--- Evaluation Metrics (Test Set) ---",
                f"R² Score: {test_r2:.4f}",
                f"Mean Squared Error (MSE): {test_mse:.4f}",
                f"Root Mean Squared Error (RMSE): {test_rmse:.4f}",
                f"Mean Absolute Error (MAE): {test_mae:.4f}",
                f"Mean Absolute Percentage Error (MAPE): {test_mape:.2f}%" if np.isfinite(test_mape) else "MAPE: N/A (zero values in target)",
                "--- Saved Files ---",
                f"Evaluation Metrics: {eval_csv_path}",
                f"Actual vs. Predicted: {predictions_csv_path}",
            ]
            text_container_content = create_text_container(*summary_lines)
            send_progress("🎉 Validation process finished successfully!", 100)

            return {"ui": {"text": text_container_content},
                    "result": (str(eval_csv_path), str(predictions_csv_path),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": ("", "")}
        except ValueError as ve:
             error_msg = f"❌ Value Error: {str(ve)}"
             send_progress(error_msg)
             return {"ui": {"text": create_text_container(error_msg)}, "result": ("", "")}
        except KeyError as ke:
             error_msg = f"❌ Key Error: A required column/descriptor might be missing. Details: {str(ke)}"
             send_progress(error_msg)
             return {"ui": {"text": create_text_container(error_msg)}, "result": ("", "")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred during validation: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": ("", "")}


NODE_CLASS_MAPPINGS = {
    "Model_Validation_Regression": Model_Validation_Regression
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Model_Validation_Regression": "Model Validation (Regression)" # Updated
}
