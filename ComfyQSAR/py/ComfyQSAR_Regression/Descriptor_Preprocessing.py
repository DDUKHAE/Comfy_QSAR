import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from .Data_Loader import create_text_container

class Replace_inf_with_nan_Regression():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING",),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PREPROCESSED_DATA",)
    FUNCTION = "replace_inf_with_nan"
    CATEGORY = "QSAR/REGRESSION/PREPROCESSING"
    OUTPUT_NODE = True

    @staticmethod
    def replace_inf_with_nan(input_file):
        
        os.makedirs("QSAR/Descriptor_Preprocessing", exist_ok=True)

        data = pd.read_csv(input_file)
        
        numeric_df = data.select_dtypes(include=[np.number])

        inf_columns = numeric_df.columns[numeric_df.isin([np.inf, -np.inf]).any(axis = 0)].tolist()

        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        inf_file = None

        if inf_columns:
            inf_counts = numeric_df[inf_columns].isin([np.inf, -np.inf]).sum().reset_index()
            inf_counts.columns = ["Feature", "Inf_Count"]

            inf_file = os.path.join("QSAR/Descriptor_Preprocessing", "inf_features.csv")
            inf_counts.to_csv(inf_file, index=False)
        

        output_file = os.path.join("QSAR/Descriptor_Preprocessing", "cleaned_data.csv")
        data.to_csv(output_file, index=False)

        text_container = create_text_container(
            "🔹 **Inf Replacement Completed!** 🔹",
            f"📌 Inf columns detected: {len(inf_columns)}",
            f"📄 Inf report: {inf_file}" if inf_file else "✅ No Inf values detected.",
        )
        
        return {"ui": {"text": text_container},
                "result": (str(output_file),)}

class Remove_high_nan_compounds_Regression():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PREPROCESSED_DATA",)
    FUNCTION = "remove_high_nan_compounds"
    CATEGORY = "QSAR/REGRESSION/PREPROCESSING" 
    OUTPUT_NODE = True

    @staticmethod
    def remove_high_nan_compounds(input_file, threshold):

        os.makedirs("QSAR/Descriptor_Preprocessing", exist_ok=True)

        data = pd.read_csv(input_file)

        nan_counts = data.isna().sum(axis = 1)
        total_columns = data.shape[1]
        nan_percentage = nan_counts / total_columns

        filtered_data = data[nan_percentage <= threshold]
        removed_count = data.shape[0] - filtered_data.shape[0]

        output_file = os.path.join("QSAR/Descriptor_Preprocessing", f"filtered_high_nan_compounds_({data.shape[0]}_{filtered_data.shape[0]}).csv")

        filtered_data.to_csv(output_file, index=False)

        text_container = create_text_container(
            "🔹 **Compounds with High NaN Percentages Removed!** 🔹",
            f"📌 Retained Compounds: {filtered_data.shape[0]}/{data.shape[0]} ✅",
            f"🗑️ Removed Compounds: {removed_count} rows with > {threshold*100:.0f}% NaN"
        )

        return {"ui": {"text": text_container},
                "result": (str(output_file),)}
        
class Remove_high_nan_descriptors_Regression():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PREPROCESSED_DATA",)
    FUNCTION = "remove_high_nan_descriptors"
    CATEGORY = "QSAR/REGRESSION/PREPROCESSING" 
    OUTPUT_NODE = True

    @staticmethod
    def remove_high_nan_descriptors(input_file, threshold):

        os.makedirs("QSAR/Descriptor_Preprocessing", exist_ok=True)
        
        data = pd.read_csv(input_file)

        nan_percentage = data.isna().mean()
        retained_columns = nan_percentage[nan_percentage <= threshold].index.tolist()

        if "SMILES" not in retained_columns:
            retained_columns.append("SMILES")
        if "value" not in retained_columns:
            retained_columns.append("value")

        filtered_data = data[retained_columns]
        removed_count = data.shape[1] - len(retained_columns)

        output_file = os.path.join("QSAR/Descriptor_Preprocessing", f"filtered_high_nan_descriptors_({data.shape[1]}_{len(retained_columns)}).csv")

        filtered_data.to_csv(output_file, index=False)

        text_container = create_text_container(
            "🔹 **Descriptors with High NaN Percentages Removed!** 🔹",
            f"📌 Retained Compounds: {len(retained_columns)} / {data.shape[1]} descriptors",
            f"🗑️ Removed Compounds: {removed_count} columns with > {threshold*100:.0f}% NaN",
        )

        return {"ui": {"text": text_container},
                "result": (str(output_file),)}

class Impute_missing_values_Regression():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING",),
                "method": (["mean", "median", "most_frequent"], {"default": "mean"}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PREPROCESSED_DATA",)
    FUNCTION = "impute_missing_values"
    CATEGORY = "QSAR/REGRESSION/PREPROCESSING" 
    OUTPUT_NODE = True

    @staticmethod
    def impute_missing_values(input_file, method):

        os.makedirs("QSAR/Descriptor_Preprocessing", exist_ok=True)

        data = pd.read_csv(input_file)

        if "Name" in data.columns:
            data = data.drop(columns = ["Name"])

        smiles_values = data[["SMILES", "value"]]
        descriptors = data.drop(columns = ["SMILES", "value"])

        imputer = SimpleImputer(strategy = method)
        imputed_descriptors = pd.DataFrame(imputer.fit_transform(descriptors), columns = descriptors.columns)

        final_data = pd.concat([smiles_values.reset_index(drop = True), imputed_descriptors.reset_index(drop = True)], axis = 1)

        output_file = os.path.join("QSAR/Descriptor_Preprocessing", "preprocessed_data.csv")

        final_data.to_csv(output_file, index = False)

        text_container = create_text_container(
            "🔹 **Imputation Completed!** 🔹",
            f"🛠 Imputation Method: {method}",
        )

        return {"ui": {"text": text_container},
                "result": (str(output_file),)}

class Descriptor_preprocessing_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING",),
                "compounds_nan_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"}),
                "descriptors_nan_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"}),
                "imputation_method": (["mean", "median", "most_frequent"], {"default": "mean"}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("INTEGRATED_PREPROCESSED_DATA",)
    FUNCTION = "preprocess"
    CATEGORY = "QSAR/REGRESSION/PREPROCESSING" 
    OUTPUT_NODE = True

    def preprocess(self, input_file, compounds_nan_threshold, descriptors_nan_threshold, imputation_method):
        
        log_messages = []
        os.makedirs("QSAR/Descriptor_Preprocessing", exist_ok=True)
        
        # Load original data
        data = pd.read_csv(input_file)
        original_shape = data.shape
        log_messages.append(f"Original data shape: {original_shape[0]} compounds, {original_shape[1]} features")
        
        # Step 1: Replace infinite values with NaN
        numeric_df = data.select_dtypes(include=[np.number])
        inf_columns = numeric_df.columns[numeric_df.isin([np.inf, -np.inf]).any(axis=0)].tolist()
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        inf_file = None
        if inf_columns:
            inf_counts = numeric_df[inf_columns].isin([np.inf, -np.inf]).sum().reset_index()
            inf_counts.columns = ["Feature", "Inf_Count"]
            inf_file = os.path.join("QSAR/Descriptor_Preprocessing", "inf_features.csv")
            inf_counts.to_csv(inf_file, index=False)
            log_messages.append(f"Detected and converted {len(inf_columns)} features containing infinite values to NaN")

        # Step 2: Remove compounds with high NaN ratio
        nan_counts = data.isna().sum(axis=1)
        total_columns = data.shape[1]
        nan_percentage = nan_counts / total_columns
            
        filtered_data = data[nan_percentage <= compounds_nan_threshold]
        removed_count = data.shape[0] - filtered_data.shape[0]
        data = filtered_data
        log_messages.append(f"Removed {removed_count} compounds with NaN ratio higher than {compounds_nan_threshold}")
        log_messages.append(f"Remaining compounds: {data.shape[0]}/{original_shape[0]}")
        
        # Step 3: Remove descriptors with high NaN ratio
        nan_percentage = data.isna().mean()
        retained_columns = nan_percentage[nan_percentage <= descriptors_nan_threshold].index.tolist()
        
        # Always retain SMILES and value columns
        if "SMILES" not in retained_columns and "SMILES" in data.columns:
            retained_columns.append("SMILES")
        if "value" not in retained_columns and "value" in data.columns:
            retained_columns.append("value")
            
        filtered_data = data[retained_columns]
        removed_count = data.shape[1] - len(retained_columns)
        data = filtered_data
        log_messages.append(f"Removed {removed_count} descriptors with NaN ratio higher than {descriptors_nan_threshold}")
        log_messages.append(f"Remaining descriptors: {data.shape[1]}/{original_shape[1]}")
        
        # Step 4: Impute missing values
        # Only impute descriptors (exclude SMILES and value)
        critical_cols = []
        if "SMILES" in data.columns:
            critical_cols.append("SMILES")
        if "value" in data.columns:
            critical_cols.append("value")
        if "Name" in data.columns:
            critical_cols.append("Name")
                
        critical_data = data[critical_cols] if critical_cols else pd.DataFrame()
        descriptors = data.drop(columns=critical_cols, errors='ignore')
            
        if not descriptors.empty:
            imputer = SimpleImputer(strategy=imputation_method)
            imputed_descriptors = pd.DataFrame(imputer.fit_transform(descriptors), columns=descriptors.columns)
                
            # Create result dataframe
            if not critical_data.empty:
                data = pd.concat([critical_data.reset_index(drop=True), 
                                imputed_descriptors.reset_index(drop=True)], axis=1)
            else:
                data = imputed_descriptors
                    
            log_messages.append(f"Missing values imputed using '{imputation_method}' method")
        
        # Save final preprocessed data
        output_file = os.path.join("QSAR/Descriptor_Preprocessing", "integrated_preprocessed_data.csv")
        data.to_csv(output_file, index=False)
        
        # Generate final summary
        final_shape = data.shape
        log_messages.append(f"Final data shape: {final_shape[0]} compounds, {final_shape[1]} features")
        log_messages.append(f"Preprocessed data saved to: {output_file}")
        
        text_container = create_text_container(
            "🔍 Descriptor Preprocessing Complete",
            f"📌 Inf columns detected: {len(inf_columns)}",
            f"📄 Inf report: {inf_file}" if inf_file else "✅ No Inf values detected.", 
            f"📌 Retained Compounds: {data.shape[0]}/{original_shape[0]} ✅",
            f"🗑️ Removed Compounds: {removed_count} rows with > {compounds_nan_threshold*100:.0f}% NaN",
            f"📌 Retained Compounds: {len(retained_columns)} / {data.shape[1]} descriptors",
            f"🗑️ Removed Compounds: {removed_count} columns with > {descriptors_nan_threshold*100:.0f}% NaN",
            f"🛠 Imputation Method: {imputation_method}",
        )
        
        return {"ui": {"text": text_container},
                "result": (str(output_file),)}
    
NODE_CLASS_MAPPINGS = {
    "Replace_inf_with_nan_Regression": Replace_inf_with_nan_Regression,
    "Remove_high_nan_compounds_Regression": Remove_high_nan_compounds_Regression,
    "Remove_high_nan_descriptors_Regression": Remove_high_nan_descriptors_Regression,
    "Impute_missing_values_Regression": Impute_missing_values_Regression,
    "Descriptor_preprocessing_Regression": Descriptor_preprocessing_Regression
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Replace_inf_with_nan_Regression": "Replace Infinite Values with NaN(Regression)",
    "Remove_high_nan_compounds_Regression": "Remove Compounds with High NaN Ratio(Regression)",
    "Remove_high_nan_descriptors_Regression": "Remove Descriptors with High NaN Ratio(Regression)",
    "Impute_missing_values_Regression": "Impute Missing Values(Regression)",
    "Descriptor_preprocessing_Regression": "Descriptor Preprocessing(Regression)"
}