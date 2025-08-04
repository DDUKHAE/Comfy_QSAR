import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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
        output_dir = "QSAR/Train_and_Validation"
        eval_csv_path, predictions_csv_path = "", "" # Default empty paths

        try:
            os.makedirs(output_dir, exist_ok=True)

            best_model = joblib.load(model_path)

            X_test = pd.read_csv(X_test_path)

            # Make sure target column name is extracted correctly if y_test.csv has header
            y_test_df = pd.read_csv(y_test_path)
            if y_test_df.shape[1] != 1:
                 raise ValueError(f"y_test file ({y_test_path}) should contain exactly one column (the target variable). Found {y_test_df.shape[1]} columns.")
            y_test = y_test_df.iloc[:, 0].values # Get the first column as a 1D numpy array
            target_column_name = y_test_df.columns[0] # Get the actual target column name

            with open(selected_descriptors_path, "r") as f:
                # Read lines, strip whitespace/newlines, filter empty lines
                selected_features = [line.strip() for line in f if line.strip()]

            missing_cols = [col for col in selected_features if col not in X_test.columns]
            if missing_cols:
                raise ValueError(f"The following selected descriptors are missing from the X_test data: {missing_cols}")

            X_test_filtered = X_test[selected_features]

            predictions = best_model.predict(X_test_filtered)

            results_df = pd.DataFrame({
                f'Actual_{target_column_name}': y_test.ravel(), # Use actual target name
                'Predicted': predictions.ravel()
            })
            predictions_csv_path = os.path.join(output_dir, "Regression_Actual_vs_Predicted.csv")
            results_df.to_csv(predictions_csv_path, index=False)

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

            eval_results = {
                "Metric": ["R2 Score", "MSE", "RMSE", "MAE", "MAPE (%)"],
                "Value": [test_r2, test_mse, test_rmse, test_mae, test_mape]
            }
            eval_df = pd.DataFrame(eval_results)
            eval_csv_path = os.path.join(output_dir, "Regression_Evaluation_Metrics.csv")
            eval_df.to_csv(eval_csv_path, index=False)

            log_message = f"Model: {os.path.basename(model_path)}\nTest Features: {os.path.basename(X_test_path)}\nTest Target: {os.path.basename(y_test_path)}\nSelected Descriptors: {os.path.basename(selected_descriptors_path)}\nR² Score: {test_r2:.4f}\nMean Squared Error (MSE): {test_mse:.4f}\nRoot Mean Squared Error (RMSE): {test_rmse:.4f}\nMean Absolute Error (MAE): {test_mae:.4f}\nMean Absolute Percentage Error (MAPE): {test_mape:.2f}%" if np.isfinite(test_mape) else f"R² Score: {test_r2:.4f}\nMean Squared Error (MSE): {test_mse:.4f}\nRoot Mean Squared Error (RMSE): {test_rmse:.4f}\nMean Absolute Error (MAE): {test_mae:.4f}\nMAPE: N/A (zero values in target)"

            return {"ui": {"text": log_message},
                    "result": (str(eval_csv_path), str(predictions_csv_path),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}."
            return {"ui": {"text": error_msg}, "result": ("", "")}
        except ValueError as ve:
             error_msg = f"❌ Value Error: {str(ve)}"
             return {"ui": {"text": error_msg}, "result": ("", "")}
        except KeyError as ke:
             error_msg = f"❌ Key Error: A required column/descriptor might be missing. Details: {str(ke)}"
             return {"ui": {"text": error_msg}, "result": ("", "")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred during validation: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": error_msg}, "result": ("", "")}


NODE_CLASS_MAPPINGS = {
    "Model_Validation_Regression": Model_Validation_Regression
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Model_Validation_Regression": "Model Validation (Regression)" # Updated
}
