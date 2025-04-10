import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from .Descriptor_Preprocessing import create_text_container

class Remove_Low_Variance_Descriptors_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING",),
                "threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("DATA",)
    FUNCTION = "remove_low_variance_descriptors"
    CATEGORY = "QSAR/REGRESSION/OPTIMIZATION"
    OUTPUT_NODE = True

    @staticmethod
    def remove_low_variance_descriptors(input_file, threshold):
        
        os.makedirs("QSAR/Descriptor_Optimization", exist_ok=True)
        
        data = pd.read_csv(input_file)
        
        if "value" not in data.columns:
            raise ValueError ("The dataset must contain a 'value' column.")
        
        target_column = data["value"]

        feature_columns = data.drop(columns = ["value"] + (["SMILES"] if "SMILES" in data.columns else []))

        selector = VarianceThreshold(threshold = threshold)
        selected_features = selector.fit_transform(feature_columns)

        retained_columns = feature_columns.columns[selector.get_support()]

        df_retained = pd.DataFrame(selected_features, columns = retained_columns)
        df_retained["value"] = target_column

        initial_feature_count = feature_columns.shape[1]
        final_feature_count = len(retained_columns)

        output_file = os.path.join("QSAR/Descriptor_Optimization", f"low_variance_result_({initial_feature_count}_{final_feature_count}).csv")
        df_retained.to_csv(output_file, index = False)

        text_container = create_text_container(
            "🔹 Low Variance Feature Removal Complete! 🔹",
            f"📊 Initial number of features: {initial_feature_count}",
            f"📉 Remaining number of features: {final_feature_count}",
            f"🗑️ Features removed: {initial_feature_count - final_feature_count}",
        )

        return {"ui": {"text": text_container},
                "result": (str(output_file),)}

class Remove_High_Correlation_Features_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING",),
                "threshold": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mode": (["target_based","upper", "lower"],),
                "importance_model": (["lasso", "random_forest",],)
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("DATA",)
    FUNCTION = "remove_high_correlation_features"
    CATEGORY = "QSAR/REGRESSION/OPTIMIZATION"
    OUTPUT_NODE = True
    
    @staticmethod
    def remove_high_correlation_features(input_file, threshold, mode, importance_model):
        os.makedirs("QSAR/Descriptor_Optimization", exist_ok=True)

        data = pd.read_csv(input_file)

        if "value" not in data.columns:
            raise ValueError("❌ The dataset must contain a 'value' column.")

        # Separate 'value' column (target variable)
        target_column = data["value"]
        feature_columns = data.drop(columns=["value"] + (["SMILES"] if "SMILES" in data.columns else []))

        correlation_matrix = feature_columns.corr()

        to_remove = set()

        if mode == "target_based":
            # Compute correlation with target variable
            feature_target_corr = feature_columns.corrwith(target_column).abs()

            # Compute feature importance if enabled
            feature_importance = {}
            if importance_model in ["random_forest", "lasso"]:
                X, y = feature_columns, target_column

                if importance_model == "random_forest":
                    model = RandomForestRegressor(n_estimators=200, random_state=42)
                elif importance_model == "lasso":
                    model = Lasso(alpha=0.01, max_iter=1000, random_state=42)

                model.fit(X, y)
                importance_values = np.abs(model.coef_) if importance_model == "lasso" else model.feature_importances_
                feature_importance = dict(zip(feature_columns.columns, importance_values))

            # Find highly correlated pairs
            rows, cols = np.where(np.abs(np.triu(correlation_matrix, k=1)) > threshold)
            for row, col in zip(rows, cols):
                feature1 = correlation_matrix.columns[row]
                feature2 = correlation_matrix.columns[col]

                # Compare correlation with target variable
                if feature_target_corr[feature1] > feature_target_corr[feature2]:
                    weaker_feature = feature2
                elif feature_target_corr[feature1] < feature_target_corr[feature2]:
                    weaker_feature = feature1
                else:
                    # If equal, use feature importance (if available)
                    weaker_feature = feature2 if feature_importance.get(feature1, 0) > feature_importance.get(feature2, 0) else feature1

                to_remove.add(weaker_feature)

        else:
            # Use "upper" or "lower" mode
            tri_matrix = np.triu(correlation_matrix, k=1) if mode == "upper" else np.tril(correlation_matrix, k=-1)
            rows, cols = np.where(np.abs(tri_matrix) > threshold)
            for row, col in zip(rows, cols):
                feature1 = correlation_matrix.columns[row]
                feature2 = correlation_matrix.columns[col]
                to_remove.add(feature2 if mode == "upper" else feature1)

        # Retain only non-removed columns
        retained_columns = [col for col in feature_columns.columns if col not in to_remove]
        df_retained = feature_columns[retained_columns]
        df_retained["value"] = target_column

        # Generate dynamic filename
        initial_feature_count = feature_columns.shape[1]
        final_feature_count = len(retained_columns)
        output_file = os.path.join("QSAR/Descriptor_Optimization", f"high_correlation_results_({initial_feature_count}_{final_feature_count}).csv")

        # Always save the dataset (even if unchanged)
        df_retained.to_csv(output_file, index=False)

        text_container = create_text_container(
            "🔹 High Correlation Feature Removal Complete! 🔹",
            f"📊 Initial number of features: {initial_feature_count}",
            f"📉 Remaining number of features: {final_feature_count}",
            f"🗑️ Features removed: {initial_feature_count - final_feature_count}",
        )

        return {"ui": {"text": text_container},
                "result": (str(output_file),)}

class Descriptor_Optimization_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING",),
                "variance_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "correlation_threshold": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "correlation_mode": (["target_based", "upper", "lower"],),
                "importance_model": (["lasso", "random_forest",],),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("OPTIMIZED_DATA",)
    FUNCTION = "optimize_descriptors"
    CATEGORY = "QSAR/REGRESSION/OPTIMIZATION"
    OUTPUT_NODE = True
    
    @staticmethod
    def optimize_descriptors(input_file, variance_threshold, 
                             correlation_threshold, correlation_mode, importance_model=None):        
        os.makedirs("QSAR/Descriptor_Optimization", exist_ok=True)
        
        data = pd.read_csv(input_file)
        original_shape = data.shape
        
        log_messages = []
        log_messages.append(f"Original data shape: {original_shape[0]} compounds, {original_shape[1]} features")
        
        # Check target and SMILES columns
        if "value" not in data.columns:
            raise ValueError("Dataset does not contain 'value' column")
        
        target_column = data["value"]
        smiles_column = data["SMILES"] if "SMILES" in data.columns else None
        
        # Separate feature columns
        feature_columns = data.drop(columns=["value"] + (["SMILES"] if smiles_column is not None else []))
        
        # Step 1: Remove low variance features
        selector = VarianceThreshold(threshold=variance_threshold)
        selected_features = selector.fit_transform(feature_columns)
        retained_columns = feature_columns.columns[selector.get_support()]
        
        # Update feature dataframe
        feature_columns = pd.DataFrame(selected_features, columns=retained_columns)
        
        # Record removed features count
        removed_count = original_shape[1] - len(retained_columns) - (1 + (1 if smiles_column is not None else 0))
        log_messages.append(f"Low variance descriptors removed: {removed_count}")
        log_messages.append(f"Remaining descriptors: {feature_columns.shape[1]}")
        
        # Save intermediate results
        variance_output = os.path.join("QSAR/Descriptor_Optimization", "variance_filtered_descriptors.csv")
        variance_df = feature_columns.copy()
        variance_df["value"] = target_column.reset_index(drop=True)
        if smiles_column is not None:
            variance_df["SMILES"] = smiles_column.reset_index(drop=True)
        variance_df.to_csv(variance_output, index=False)
        log_messages.append(f"Variance filtering results saved: {variance_output}")
        
        # Step 2: Remove high correlation features
        correlation_matrix = feature_columns.corr()
        to_remove = set()
        
        if correlation_mode == "target_based":
            # Calculate correlation with target variable
            feature_target_corr = feature_columns.corrwith(target_column).abs()
            
            # Calculate feature importance (optional)
            feature_importance = {}
            if importance_model:
                X, y = feature_columns, target_column
                
                if importance_model == "random_forest":
                    model = RandomForestRegressor(n_estimators=200, random_state=42)
                elif importance_model == "lasso":
                    model = Lasso(alpha=0.01, max_iter=1000, random_state=42)
                
                model.fit(X, y)
                importance_values = np.abs(model.coef_) if importance_model == "lasso" else model.feature_importances_
                feature_importance = dict(zip(feature_columns.columns, importance_values))
            
            # Find high correlation pairs
            rows, cols = np.where(np.abs(np.triu(correlation_matrix, k=1)) > correlation_threshold)
            for row, col in zip(rows, cols):
                feature1 = correlation_matrix.columns[row]
                feature2 = correlation_matrix.columns[col]
                
                # Compare correlation with target variable
                if feature_target_corr[feature1] > feature_target_corr[feature2]:
                    weaker_feature = feature2
                elif feature_target_corr[feature1] < feature_target_corr[feature2]:
                    weaker_feature = feature1
                else:
                    # If equal correlation, use feature importance (if available)
                    weaker_feature = feature2 if feature_importance.get(feature1, 0) > feature_importance.get(feature2, 0) else feature1
                
                to_remove.add(weaker_feature)
        
        else:
            # Use "upper" or "lower" mode
            tri_matrix = np.triu(correlation_matrix, k=1) if correlation_mode == "upper" else np.tril(correlation_matrix, k=-1)
            rows, cols = np.where(np.abs(tri_matrix) > correlation_threshold)
            for row, col in zip(rows, cols):
                feature1 = correlation_matrix.columns[row]
                feature2 = correlation_matrix.columns[col]
                to_remove.add(feature2 if correlation_mode == "upper" else feature1)
        
        # Keep only non-removed columns
        pre_corr_count = feature_columns.shape[1]
        retained_columns = [col for col in feature_columns.columns if col not in to_remove]
        feature_columns = feature_columns[retained_columns]
        
        # Record removed features count
        log_messages.append(f"High correlation descriptors removed: {pre_corr_count - feature_columns.shape[1]}")
        log_messages.append(f"Remaining descriptors: {feature_columns.shape[1]}")
        
        # Generate final results
        final_data = feature_columns.copy()
        final_data["value"] = target_column.reset_index(drop=True)
        if smiles_column is not None:
            final_data["SMILES"] = smiles_column.reset_index(drop=True)
        
        # Save final results
        output_file = os.path.join("QSAR/Descriptor_Optimization", "integrated_optimized_descriptors.csv")
        final_data.to_csv(output_file, index=False)
        
        # Generate final summary
        final_shape = final_data.shape
        total_removed = original_shape[1] - final_shape[1]
        log_messages.append(f"Final descriptor optimization results: {total_removed} removed")
        log_messages.append(f"Final data shape: {final_shape[0]} compounds, {final_shape[1]} features")
        log_messages.append(f"Optimized data saved at: {output_file}")
        
        text_container = create_text_container(
            "🔍 Integrated Descriptor Optimization Complete",
            f"📊 Initial number of features: {original_shape[1]}",
            f"📉 Remaining number of features: {final_shape[1]}",
            f"🗑️ Features removed: {total_removed}",
        )
        
        return {"ui": {"text": text_container},
                "result": (str(output_file),)}

# 노드 등록 업데이트
NODE_CLASS_MAPPINGS = {
    "Remove_Low_Variance_Features_Regression": Remove_Low_Variance_Descriptors_Regression,
    "Remove_High_Correlation_Features_Regression": Remove_High_Correlation_Features_Regression,
    "Descriptor_Optimization_Regression": Descriptor_Optimization_Regression
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Remove_Low_Variance_Features_Regression": "Remove Low Variance Features(Regression)",
    "Remove_High_Correlation_Features_Regression": "Remove High Correlation Features(Regression)",
    "Descriptor_Optimization_Regression": "Descriptor Optimization(Regression)"
}