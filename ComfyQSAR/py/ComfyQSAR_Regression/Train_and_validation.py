import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from .Data_Loader import create_text_container
class Model_Validation_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING",),
                "selected_descriptors": ("STRING",),
                "X_test": ("STRING",),
                "y_test": ("STRING",),
            },
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("EVALUATION_PATH", "PREDICTION_PATH",)
    FUNCTION = "train_and_validate_regression"
    CATEGORY = "QSAR/REGRESSION/VALIDATION"
    
    def train_and_validate_regression(self, model_path, X_test_path, y_test_path, selected_descriptors_path):
        os.makedirs("QSAR/Train_and_Validation", exist_ok=True)

        best_model = joblib.load(model_path)

        # Load test set
        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path).values.ravel()  # Convert to 1D array if needed
        
        # Load selected descriptors (important!)
        with open(selected_descriptors_path, "r") as f:
            selected_features = [line.strip() for line in f.readlines()]

        # Ensure that only selected features are used for testing
        X_test = X_test[selected_features]  # Retain only selected features

        # Perform predictions
        predictions = best_model.predict(X_test)
        
        # Dataframe for comparison of actual and predicted values
        results_df = pd.DataFrame({
            'Actual': y_test.ravel(),
            'Predicted': predictions.ravel() 
        })

        # Save to dataframe
        predictions_csv_path = os.path.join("QSAR/Train_and_Validation", "Actual_vs_Predicted.csv")
        results_df.to_csv(predictions_csv_path, index=False)

        # Compute evaluation metrics
        test_r2 = r2_score(y_test, predictions)  # R² Score
        test_mse = mean_squared_error(y_test, predictions)  # Mean Squared Error
        test_rmse = np.sqrt(test_mse)  # Root Mean Squared Error
        test_mae = mean_absolute_error(y_test, predictions)  # Mean Absolute Error
        test_mape = (np.abs(y_test - predictions) / y_test).mean() * 100  # Mean Absolute Percentage Error (MAPE)

        # Save evaluation results to CSV
        eval_results = {
            "Metric": ["R² Score", "MSE", "RMSE", "MAE", "MAPE"],
            "Value": [test_r2, test_mse, test_rmse, test_mae, test_mape]
        }
        eval_df = pd.DataFrame(eval_results)
        eval_csv_path = os.path.join("QSAR/Train_and_Validation", "Evaluation_Results_TestSet.csv")
        eval_df.to_csv(eval_csv_path, index=False)

        text_container = create_text_container(
            "🔹 Regression Model Validation Done! 🔹",
            f"📁 Model: {model_path}",
            f"📊 R² Score: {test_r2:.4f}",
            f"📊 MSE: {test_mse:.4f}",
            f"📊 RMSE: {test_rmse:.4f}",
            f"📊 MAE: {test_mae:.4f}",
            f"📊 MAPE: {test_mape:.4f}",
        )
        
        return {"ui": {"text": text_container},
                "result": (str(eval_csv_path), str(predictions_csv_path),)}

NODE_CLASS_MAPPINGS = {
    "Model_Validation_Regression": Model_Validation_Regression
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Model_Validation_Regression": "Model Validation(Regression)"
}
