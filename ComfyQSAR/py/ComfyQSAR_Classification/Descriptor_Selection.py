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

try:
    from .Data_Loader import send_progress, create_text_container
except ImportError:
    print("[ComfyQSAR Descriptor Selection] Warning: Could not import progress_utils. Progress updates might not work.")
    # 대체 함수 정의
    def send_progress(message, progress=None, node_id=None):
        print(f"[Progress Fallback] {message}" + (f" ({progress}%)" if progress is not None else ""))
    # 대체 create_text_container 정의
    def create_text_container(*lines):
        return "\n".join(lines)

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
    CATEGORY = "QSAR/CLASSIFICATION/SELECTION"
    OUTPUT_NODE = True
    
    def select_features(self, input_file, advanced, target_column="Label", method="lasso", n_features=10, threshold=0.9,
                      model=None, alpha=0.01, max_iter=1000, n_estimators=100, max_depth=5, min_samples_split=2, 
                      criterion="gini", learning_rate=0.1, n_iterations=100):
        """
        Feature selection supporting strategy-method and model-based approaches.
        """
        send_progress("🚀 Starting Feature Selection...", 0)

        output_dir = "QSAR/Selection"
        os.makedirs(output_dir, exist_ok=True)
        send_progress(f"📂 Output directory created: {output_dir}", 5)

        try:
            send_progress(f"⏳ Loading data from: {input_file}", 10)
            df = pd.read_csv(input_file)
        except Exception as e:
            error_msg = f"❌ Error loading input file: {str(e)}"
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": ("",)}

        if target_column not in df.columns:
            error_msg = f"❌ Target column '{target_column}' not found in the dataset."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": ("",)}

        X_raw = df.drop(columns=[target_column])
        y = df[target_column]

        send_progress("⚙️ Preprocessing data (handling inf/NaN, large values)...", 15)
        X = X_raw.replace([np.inf, -np.inf], np.nan)

        nan_cols = []
        for col in X.columns:
            if X[col].isnull().any():
                median_val = X[col].median()
                if pd.isna(median_val): # Use pd.isna for better NaN checking
                    median_val = 0 # If entire column was NaN
                    nan_cols.append(f"{col} (filled with 0)")
                else:
                    nan_cols.append(f"{col} (filled with {median_val:.2f})")
                X[col] = X[col].fillna(median_val)
        if nan_cols:
            send_progress(f"   NaN values handled in columns: {', '.join(nan_cols)}", 20)

        large_val_cols = []
        for col in X.columns:
            if not np.issubdtype(X[col].dtype, np.number): continue # Skip non-numeric columns
            # Check for values exceeding a large threshold or potentially non-finite
            if (X[col].abs() > 1e30).any() or not np.isfinite(X[col]).all():
                 X[col] = np.nan_to_num(X[col].astype(np.float64), nan=0.0, posinf=1e30, neginf=-1e30) # Convert non-finite to finite
                 X[col] = np.clip(X[col], -1e30, 1e30) # Clip extremes
                 large_val_cols.append(col)
        if large_val_cols:
            send_progress(f"   Clipped large/non-finite values in columns: {', '.join(large_val_cols)}", 25)

        initial_feature_count = X.shape[1]
        send_progress(f"🔢 Initial number of features: {initial_feature_count}", 30)

        model_abbr = None
        selected_columns = None
        feature_importances = None # Initialize

        def get_model(model_name):
            if model_name == "RandomForest": # Match INPUT_TYPES casing
                return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                          min_samples_split=min_samples_split, criterion=criterion, random_state=42, n_jobs=-1), "RF"
            elif model_name == "DecisionTree":
                return DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                          criterion=criterion, random_state=42), "DT"
            elif model_name == "XGBoost":
                return XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                 learning_rate=learning_rate, random_state=42, eval_metric="logloss", use_label_encoder=False), "XGB" # use_label_encoder=False recommended
            elif model_name == "LightGBM":
                return LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth if max_depth is not None and max_depth > 0 else -1, # Handle default/None for LGBM
                                  learning_rate=learning_rate, random_state=42, n_jobs=-1), "LGBM"
            elif model_name == "Lasso": # Match INPUT_TYPES casing
                # Use LogisticRegression with L1 penalty for classification Lasso
                return LogisticRegression(penalty='l1', solver='saga', C=1/alpha if alpha > 0 else 1e9, # Prevent division by zero
                                        max_iter=max_iter, random_state=42, n_jobs=-1), "Lasso"
            else:
                raise ValueError(f"Invalid model name '{model_name}'.")

        # --- Feature Selection Logic ---
        try:
            # Apply scaling before model fitting
            send_progress("⚖️ Scaling features...", 35)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            send_progress("   Scaling complete.", 40)

            # --- Method Implementations ---
            if method == "Lasso":
                model_instance, model_abbr = get_model("Lasso")
                send_progress(f"🧬 Applying {method} (Logistic Regression with L1)...", 45)
                model_instance.fit(X_scaled, y)
                # Select features where coefficient is non-zero
                importances = np.abs(model_instance.coef_[0])
                selected_mask = importances > 1e-5 # Use a small threshold instead of exact zero
                selected_columns = X.columns[selected_mask]
                if len(selected_columns) == 0: # Handle case where Lasso removes all features
                     send_progress("   Warning: Lasso removed all features. Selecting top N based on coefficient magnitude.", 75)
                     indices = np.argsort(importances)[::-1][:min(n_features, len(importances))]
                     selected_columns = X.columns[indices]
                X_new = X[selected_columns]
                send_progress(f"   {method} selection complete. {len(selected_columns)} features selected.", 80)

            elif method == "RFE":
                if model is None or model == "None": # Check for None string as well
                    raise ValueError("'RFE' method requires a model selection.")
                model_instance, model_abbr = get_model(model)
                send_progress(f"🧬 Applying {method} with {model_abbr} (target: {n_features} features)...", 45)
                selector = RFE(estimator=model_instance, n_features_to_select=n_features, step=0.1) # step=0.1 for faster removal
                selector.fit(X_scaled, y)
                selected_columns = X.columns[selector.get_support()]
                X_new = X[selected_columns]
                send_progress(f"   {method} selection complete. {len(selected_columns)} features selected.", 80)

            elif method == "SelectFromModel":
                if model is None or model == "None":
                    raise ValueError("'SelectFromModel' method requires a model selection.")
                model_instance, model_abbr = get_model(model)
                send_progress(f"🧬 Applying {method} with {model_abbr}...", 45)
                model_instance.fit(X_scaled, y)

                # Get feature importances or coefficients
                if hasattr(model_instance, "feature_importances_"):
                    importances = model_instance.feature_importances_
                elif hasattr(model_instance, "coef_"):
                    importances = np.abs(model_instance.coef_[0])
                else:
                    raise ValueError(f"Model {model_abbr} does not provide feature importances or coefficients.")

                # Determine threshold dynamically based on n_features if threshold is not set or invalid
                current_threshold = threshold
                if not isinstance(threshold, (int, float)) or threshold <= 0 or threshold > 1 : # Adjust threshold logic if needed
                     # If invalid threshold, use n_features to determine threshold
                     if n_features > 0 and n_features < len(importances):
                           sorted_importances = np.sort(importances)[::-1] # Sort descending
                           current_threshold = sorted_importances[n_features -1] # Threshold is the importance of the Nth feature
                           send_progress(f"   Using importance threshold based on n_features: {current_threshold:.4f}", 65)
                     else: # Default to median if n_features is also invalid
                           current_threshold = "median"
                           send_progress(f"   Using median importance threshold.", 65)

                selector = SelectFromModel(estimator=model_instance, threshold=current_threshold, prefit=True, max_features=n_features if threshold <=0 else None)
                selected_columns = X.columns[selector.get_support()]
                X_new = X[selected_columns]
                send_progress(f"   {method} selection complete. {len(selected_columns)} features selected.", 80)


            elif method in ["RandomForest", "DecisionTree", "XGBoost", "LightGBM"]: # Model itself provides importances
                 model_instance, model_abbr = get_model(method)
                 send_progress(f"🧬 Calculating feature importances using {model_abbr}...", 45)
                 model_instance.fit(X_scaled, y)
                 importances = model_instance.feature_importances_
                 send_progress("   Feature importances calculated.", 65)

                 # Select top N features based on importance
                 indices = np.argsort(importances)[::-1][:min(n_features, len(importances))] # Ensure n_features is not too large
                 selected_columns = X.columns[indices]
                 X_new = X[selected_columns]
                 send_progress(f"   Selected top {len(selected_columns)} features based on importance.", 80)

            else:
                raise ValueError(f"Unsupported method '{method}'")

            # --- Post-selection ---
            final_feature_count = X_new.shape[1]
            if final_feature_count == 0:
                send_progress("⚠️ Warning: No features were selected. Returning original features.", 85)
                X_new = X # Return original if none selected
                selected_columns = X.columns
                final_feature_count = initial_feature_count

            # Combine selected features with the target column
            final_df = pd.concat([y, X_new], axis=1)

            # Save the result
            output_filename = f"selected_{method}_{model_abbr if model_abbr else 'features'}.csv"
            output_path = os.path.join(output_dir, output_filename)
            send_progress(f"💾 Saving selected features to: {output_path}", 90)
            final_df.to_csv(output_path, index=False)
            send_progress("   Saving complete.", 95)

            # Prepare result text
            result_text = create_text_container(
                f"🔹 **Feature Selection Done!** 🔹",
                f"Method: {method}" + (f" with {model_abbr}" if model_abbr else ""),
                f"Initial Features: {initial_feature_count}",
                f"Selected Features: {final_feature_count}",
                f"Output File: {output_path}"
            )
            send_progress("🎉 Feature selection finished successfully!", 100)
            return {"ui": {"text": result_text}, "result": (output_path,)}

        except Exception as e:
            error_msg = f"❌ Feature Selection Failed!\nMethod: {method}\nError: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            send_progress(error_msg) # Send error without progress
            return {"ui": {"text": create_text_container(error_msg)}, "result": ("",)}

# Node registration
NODE_CLASS_MAPPINGS = {
    "Feature_Selection_Classification": Feature_Selection_Classification
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Feature_Selection_Classification": "Feature Selection (Classification)"
} 