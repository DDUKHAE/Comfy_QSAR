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

class Feature_Selection_Classification:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_file": ("STRING", {"tooltip": "Path to the input file"}),
                "method": (["Lasso", "RandomForest", "DecisionTree", "XGBoost", "LightGBM", "RFE", "SelectFromModel"],
                           {"tooltip": "Feature selection method"}),
                "advanced": ("BOOLEAN", {"default": False, "tooltip": "Advanced mode"}),
            },
            "optional": {
                # Model selection and basic parameters
                "model": (["Lasso", "RandomForest", "DecisionTree", "XGBoost", "LightGBM"]),
                "target_column": ("STRING", {"default": "Label", "tooltip": "Target column name"}),
                "n_features": ("INT", {"default": 10, "min": 1, "max": 1000, "step": 1,
                                        "tooltip": "Number of features to select"}),
                "threshold": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Threshold for feature selection"}),
                
                # Lasso related parameters
                "alpha": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 10.0, "step": 0.001,
                                    "tooltip": "Regularization parameter"}),
                "max_iter": ("INT", {"default": 1000, "min": 100, "max": 10000, "step": 100,
                                    "tooltip": "Maximum number of iterations"}),
                
                # Tree-based model parameters
                "n_estimators": ("INT", {"default": 100, "min": 10, "max": 1000, "step": 10,
                                         "tooltip": "Number of trees in the forest"}),
                "max_depth": ("INT", {"default": 5, "min": 1, "max": 50, "step": 1,
                                      "tooltip": "Maximum depth of the trees"}),
                "min_samples_split": ("INT", {"default": 2, "min": 2, "max": 20, "step": 1,
                                              "tooltip": "Minimum number of samples required to split an internal node"}),
                "criterion": (["gini", "entropy"], {"tooltip": "Criterion for splitting"}),
                "learning_rate": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 1.0, "step": 0.01,
                                            "tooltip": "Learning rate for boosting"}),
                
                # Additional parameters
                "n_iterations": ("INT", {"default": 100, "min": 10, "max": 1000, "step": 10,
                                          "tooltip": "Number of iterations for RFE"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("OUTPUT_FILE",)
    FUNCTION = "select_features"
    CATEGORY = "QSAR/CLASSIFICATION/OPTIMIZATION(Model-based)"
    OUTPUT_NODE = True
    
    def select_features(self, input_file, advanced, target_column="Label", method="lasso", n_features=10, threshold=0.9,
                      model=None, alpha=0.01, max_iter=1000, n_estimators=100, max_depth=5, min_samples_split=2, 
                      criterion="gini", learning_rate=0.1, n_iterations=100):
        output_dir = "QSAR/Selection"
        os.makedirs(output_dir, exist_ok=True)

        try:
            df = pd.read_csv(input_file)
        except Exception as e:
            error_msg = f"❌ Error loading input file: {str(e)}"
            return {"ui": {"text": error_msg}, "result": ("",)}

        if target_column not in df.columns:
            error_msg = f"❌ Target column '{target_column}' not found in the dataset."
            return {"ui": {"text": error_msg}, "result": ("",)}

        X = df.drop(columns=[target_column])
        y = df[target_column]

        initial_feature_count = X.shape[1]

        model_abbr = None
        feature_importances = None

        def get_model(model_name):
            if model_name == "RandomForest":
                return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                          min_samples_split=min_samples_split, criterion=criterion, random_state=42), "RF"
            elif model_name == "DecisionTree":
                return DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                          criterion=criterion, random_state=42, n_jobs=-1), "DT"
            elif model_name == "XGBoost":
                return XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                 learning_rate=learning_rate, random_state=42, eval_metric="logloss", use_label_encoder=False), "XGB"
            elif model_name == "LightGBM":
                return LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth if max_depth is not None and max_depth > 0 else -1,
                                  learning_rate=learning_rate, random_state=42, n_jobs=-1), "LGBM"
            elif model_name == "Lasso":
                return LogisticRegression(penalty='l1', solver='saga', C=1/alpha if alpha > 0 else 1e9,
                                        max_iter=max_iter, random_state=42, n_jobs=-1), "Lasso"
            else:
                raise ValueError(f"Invalid model name '{model_name}'.")

        try:
            if method == "Lasso":
                model, model_abbr = get_model("Lasso")
                model.fit(X, y)
                selected_columns = X.columns[model.coef_[0] != 0]
                X_new = X[selected_columns]

            elif method == "RFE":
                if model is None or model == "None":
                    raise ValueError("'RFE' method requires a model.")
                model_instance, model_abbr = get_model(model)
                selector = RFE(estimator=model, n_features_to_select=n_features, step=0.1)
                selector.fit(X, y)
                selected_columns = X.columns[selector.get_support()]
                X_new = X[selected_columns]

            elif method == "SelectFromModel":
                if model is None or model == "None":
                    raise ValueError("'SelectFromModel' method requires a model.")
                model_instance, model_abbr = get_model(model)
                model_instance.fit(X, y)

                if hasattr(model_instance, "feature_importances_"):
                    importances = model_instance.feature_importances_
                elif hasattr(model_instance, "coef_"):
                    importances = np.abs(model_instance.coef_[0])
                else:
                    raise ValueError(f"Model {model_abbr} does not provide feature importances or coefficients.")

                if isinstance(threshold, (str)) and "percentile" in threshold:
                    percentile_value = int(threshold.replace("percentile(", "").replace(")", ""))
                    threshold = np.percentile(importances, percentile_value)

                selector = SelectFromModel(estimator=model_instance, threshold=threshold, prefit=True, max_features=n_features if threshold <=0 else None)
                selected_columns = X.columns[selector.get_support()]
                X_new = X[selected_columns]

            elif method in ["RandomForest", "DecisionTree"]:
                feature_importance_matrix = np.zeros((n_iterations, X.shape[1]))

                for i in range(n_iterations):
                    model_iter, model_abbr = get_model(method)
                    model_iter.fit(X, y)
                    feature_importance_matrix[i] = model_iter.feature_importances_

                feature_importances = np.mean(feature_importance_matrix, axis=0)

                if isinstance(threshold, (str)) and "percentile" in threshold:
                    percentile_value = int(threshold.replace("percentile(", "").replace(")", ""))
                    threshold_value = np.percentile(feature_importances, percentile_value)
                else:
                    threshold_value = float(threshold)

                important_indices = np.where(feature_importances >= threshold_value)[0]
                selected_columns = X.columns[important_indices]
                X_new = X[selected_columns]

            elif method in ["XGBoost", "LightGBM"]:
                model, model_abbr = get_model(method)
                model.fit(X, y)
                importances = model.feature_importances_

                if isinstance(threshold, (str)) and "percentile" in threshold:
                    percentile_value = int(threshold.replace("percentile(", "").replace(")", ""))
                    threshold_value = np.percentile(importances, percentile_value)
                else:
                    threshold_value = float(threshold)

                important_indices = np.where(importances >= threshold_value)[0]
                selected_columns = X.columns[important_indices]
                X_new = X[selected_columns]

            else:
                raise ValueError(f"Unsupported method '{method}'")

            final_feature_count = X_new.shape[1]
            if final_feature_count == 0:
                X_new = X
                selected_columns = X.columns
                final_feature_count = initial_feature_count

            final_df = pd.concat([y, X_new], axis=1)

            output_filename = f"selected_{method}_{model_abbr if model_abbr else 'features'}.csv"
            output_path = os.path.join(output_dir, output_filename)
            final_df.to_csv(output_path, index=False)

            log_message = (
                "========================================\n"
                "🔹 Feature Selection Completed! 🔹\n"
                "========================================\n"
                f"📌 Method: {method} ({model_abbr})\n"
                f"📊 Initial Features: {initial_feature_count}\n"
                f"📉 Selected Features: {final_feature_count}\n"
                f"🗑️ Removed: {initial_feature_count - final_feature_count}\n"
                f"💾 File saved at: {output_path}\n"
                "========================================"
            )
            
            return {"ui": {"text": log_message}, "result": (output_path,)}

        except Exception as e:
            error_msg = f"❌ Feature Selection Failed!\nMethod: {method}\nError: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": error_msg}, "result": ("",)}

# Node registration
NODE_CLASS_MAPPINGS = {
    "Feature_Selection_Classification": Feature_Selection_Classification
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Feature_Selection_Classification": "Feature Selection (Classification)"
} 