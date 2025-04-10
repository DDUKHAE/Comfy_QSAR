import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.feature_selection import RFE, SelectFromModel
from .Data_Loader import create_text_container
class Feature_Selection_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING",),
                "method": (["Lasso", "RandomForest", "DecisionTree", "XGBoost", "LightGBM", "RFE", "SelectFromModel"], {"default": "Lasso"}),
                "advanced": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                # Model selection and basic parameters
                "model": (["Lasso", "RandomForest", "DecisionTree", "XGBoost", "LightGBM"], {"default": "None", "description": "Model to use with RFE or SelectFromModel methods"}),
                "target_column": ("STRING", {"default": "value", "description": "Target column name in the dataset"}),
                "n_features": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1, "description": "Number of features to select"}),
                "threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number", "description": "Threshold for feature selection"}),
                
                # Lasso related parameters
                "alpha": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number", "description": "Regularization strength for Lasso"}),
                "max_iter": ("INT", {"default": 1000, "min": 1, "max": 10000, "step": 1, "description": "Maximum number of iterations"}),
                
                # Tree-based model parameters
                "n_estimators": ("INT", {"default": 100, "min": 1, "max": 1000, "step": 1, "description": "Number of trees in ensemble models"}),
                "max_depth": ("INT", {"default": None, "min": 1, "max": 10, "step": 1, "description": "Maximum depth of trees"}),
                "min_samples_split": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1, "description": "Minimum samples required to split a node"}),
                "criterion": (["squared_error", "absolute_error", "poisson"], {"default": "squared_error", "description": "Function to measure quality of split"}),
                "learning_rate": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number", "description": "Learning rate for boosting models"}),
                
                # Additional parameters
                "n_iterations": ("INT", {"default": 100, "min": 1, "max": 1000, "step": 1, "description": "Number of iterations for stability analysis"}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("SELECTED_DESCRIPTORS",)
    FUNCTION = "select_descriptors"
    CATEGORY = "QSAR/REGRESSION/SELECTION"
    OUTPUT_NODE = True
    
    def select_descriptors(self, input_file, method, target_column, n_features, model, alpha, max_iter, n_estimators, max_depth, min_samples_split, criterion, learning_rate, n_iterations, threshold, advanced):
        os.makedirs("QSAR/Descriptor_Selection", exist_ok=True)
        
        data = pd.read_csv(input_file)
        
        # Ensure target column exists
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")

        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]

        initial_feature_count = X.shape[1]  # Initial feature count before selection

        print(f"Running feature selection using method: {method}, model: {model}...")

        # Initialize model based on selection
        model_abbr = None  # Ensure model_abbr is always defined
        if isinstance(model, str):
            if model == "random_forest":
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                              min_samples_split=min_samples_split, criterion=criterion, random_state=42)
                model_abbr = "RF"

            elif model == "decision_tree":
                model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split,
                                              criterion=criterion, random_state=42)
                model_abbr = "DT"

            elif model == "xgboost":
                model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                     learning_rate=learning_rate, random_state=42)
                model_abbr = "XGB"

            elif model == "lightgbm":
                model = LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      learning_rate=learning_rate, random_state=42)
                model_abbr = "LGBM"

            elif model == "lasso":
                model = Lasso(alpha=alpha, max_iter=max_iter, random_state=42)
                model_abbr = "LASSO"

            else:
                raise ValueError(f"Invalid model name '{model}'. Choose from 'random_forest', 'decision_tree', 'xgboost', 'lightgbm', 'lasso'.")

        # Perform feature selection
        if method == "lasso":
            model.fit(X, y)
            selected_columns = X.columns[model.coef_ != 0]
            X_new = X[selected_columns]

        elif method == "rfe":
            selector = RFE(estimator=model, n_features_to_select=n_features)
            selector.fit(X, y)
            selected_columns = X.columns[selector.get_support()]
            X_new = X[selected_columns]

        elif method == "select_from_model":
            model.fit(X, y)

            # Convert percentile threshold to actual value
            if isinstance(threshold, str) and "percentile" in threshold:
                if hasattr(model, "feature_importances_"):  # Tree-based models (RandomForest, XGBoost, LightGBM)
                    feature_importances = model.feature_importances_
                elif hasattr(model, "coef_"):  # Linear models (Lasso, Ridge)
                    feature_importances = np.abs(model.coef_)
                else:
                    raise ValueError(f"❌ The model {model} does not provide feature importances!")

                percentile_value = int(threshold.replace("percentile(", "").replace(")", ""))
                threshold = np.percentile(feature_importances, percentile_value)

            selector = SelectFromModel(estimator=model, threshold=threshold, prefit=True)
            selected_columns = X.columns[selector.get_support()]
            X_new = X[selected_columns]

        elif method in ["random_forest", "decision_tree"]:
            model_abbr = "RF" if method == "random_forest" else "DT"
            feature_importance_matrix = np.zeros((n_iterations, X.shape[1]))

            for i in range(n_iterations):
                if method == "random_forest":
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                                  min_samples_split=min_samples_split, criterion=criterion)
                elif method == "decision_tree":
                    model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split,
                                                  criterion=criterion)

                model.fit(X, y)
                feature_importance_matrix[i] = model.feature_importances_

            # Calculate feature_importances
            feature_importances = np.mean(feature_importance_matrix, axis=0)

        elif method in ["xgboost", "lightgbm"]:
            model.fit(X, y)
            feature_importances = model.feature_importances_

        # Convert percentile threshold if needed
        if isinstance(threshold, str) and "percentile" in threshold:
            percentile_value = int(threshold.replace("percentile(", "").replace(")", ""))
            threshold = np.percentile(feature_importances, percentile_value)

        important_indices = np.where(feature_importances >= threshold)[0]
        selected_columns = X.columns[important_indices]
        X_new = X[selected_columns]

        # Calculate feature reduction
        final_feature_count = len(selected_columns)
        removed_features = initial_feature_count - final_feature_count

        # Create final dataset
        selected_features = X_new.copy()
        selected_features[target_column] = y.reset_index(drop=True)

        # Generate optimized filename
        if model_abbr is None:
            model_abbr = method  # Ensure filename is not empty if model_abbr wasn't set earlier
        filename = f"features_{method}_{model_abbr}_{initial_feature_count}_{final_feature_count}.csv"
        output_file = os.path.join("QSAR/Descriptor_Selection", filename)

        # Save dataset
        selected_features.to_csv(output_file, index=False)

        # Print feature reduction summary
        text_container = create_text_container(
            "🔹 Feature Selection Completed! 🔹",
            f"📌 Method: {method}",
            f"📊 Initial Features: {initial_feature_count}",
            f"📉 Selected Features: {final_feature_count}",
            f"🗑️ Removed: {removed_features}",
        )

        return {"ui": {"text": text_container},
                "result": (str(output_file))}

NODE_CLASS_MAPPINGS = {
    "Feature_Selection_Regression": Feature_Selection_Regression
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Feature_Selection_Regression": "Feature Selection(Regression)"
}