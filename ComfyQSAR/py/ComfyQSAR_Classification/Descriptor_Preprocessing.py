import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from .Data_Loader import create_text_container

class Replace_inf_with_nan_Classification():

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
    CATEGORY = "QSAR/CLASSIFICATION/PREPROCESSING"
    OUTPUT_NODE = True

    @staticmethod
    def replace_inf_with_nan(input_file):
        
        os.makedirs("QSAR/Descriptor_Preprocessing", exist_ok=True)

        data = pd.read_csv(input_file)
        
        numeric_df = data.select_dtypes(include=[np.number])

        inf_columns = numeric_df.columns[numeric_df.isin([np.inf, -np.inf]).any(axis = 0)].tolist()

        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        if inf_columns:
            inf_counts = numeric_df[inf_columns].isin([np.inf, -np.inf]).sum().reset_index()
            inf_counts.columns = ["Feature", "Inf_Count"]

            inf_file = os.path.join("QSAR/Descriptor_Preprocessing", "inf_features.csv")
            inf_counts.to_csv(inf_file, index=False)

        else:
            inf_file = None
        
        output_file = os.path.join("QSAR/Descriptor_Preprocessing", "cleaned_data.csv")
        data.to_csv(output_file, index=False)

        text_container = create_text_container(
            "🔹 **Inf Replacement Completed!** 🔹",
            f"📌 Inf columns detected: {len(inf_columns)}",
            f"📄 Inf report: {inf_file}" if inf_file else "✅ No Inf values detected.",
        )
        
        return {"ui": {"text": text_container},
                "result": (str(output_file),)}

class Remove_high_nan_compounds_Classification():
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
    CATEGORY = "QSAR/CLASSIFICATION/PREPROCESSING" 
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
        
class Remove_high_nan_descriptors_Classification():
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
    CATEGORY = "QSAR/CLASSIFICATION/PREPROCESSING" 
    OUTPUT_NODE = True

    @staticmethod
    def remove_high_nan_descriptors(input_file, threshold):

        os.makedirs("QSAR/Descriptor_Preprocessing", exist_ok=True)
        
        data = pd.read_csv(input_file)

        nan_percentage = data.isna().mean()
        retained_columns = nan_percentage[nan_percentage <= threshold].index.tolist()

        if "Label" not in retained_columns:
            retained_columns.append("Label")

        filtered_data = data[retained_columns]
        removed_count = data.shape[1] - len(retained_columns)

        output_file = os.path.join("QSAR/Descriptor_Preprocessing", f"filtered_high_nan_descriptors_({data.shape[1]}_{len(retained_columns)}).csv")

        filtered_data.to_csv(output_file, index=False)

        text_container = create_text_container(
            "🔹 **Descriptors with High NaN Percentages Removed!** 🔹",
            f"📌 Retained Compounds: {len(retained_columns)} / {data.shape[1]} descriptors",
            f"🗑️ Removed Compounds: {removed_count} columns with > {threshold*100:.0f}% NaN"
        )

        return {"ui": {"text": text_container},
                "result": (str(output_file),)}

class Impute_missing_values_Classification():
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
    CATEGORY = "QSAR/CLASSIFICATION/PREPROCESSING" 
    OUTPUT_NODE = True

    @staticmethod
    def impute_missing_values(input_file, method):

        os.makedirs("QSAR/Descriptor_Preprocessing", exist_ok=True)

        data = pd.read_csv(input_file)

        if "Name" in data.columns:
            data = data.drop(columns = ["Name"])

        # SMILES과 Label 열 분리
        label_column = data[["Label"]]
        descriptors = data.drop(columns = ["Label"])

        imputer = SimpleImputer(strategy = method)
        imputed_descriptors = pd.DataFrame(imputer.fit_transform(descriptors), columns = descriptors.columns)

        final_data = pd.concat([imputed_descriptors, label_column.reset_index(drop = True)], axis = 1)

        output_file = os.path.join("QSAR/Descriptor_Preprocessing", "preprocessed_data.csv")

        final_data.to_csv(output_file, index = False)

        text_container = create_text_container(
            "🔹 **Imputation Completed!** 🔹",
            f"🛠 Imputation Method: {method}",
        )

        return {"ui": {"text": text_container},
                "result": (str(output_file),)}

class Descriptor_preprocessing_Classification:
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
    CATEGORY = "QSAR/CLASSIFICATION/PREPROCESSING" 
    OUTPUT_NODE = True

    def preprocess(self, input_file, compounds_nan_threshold, descriptors_nan_threshold, imputation_method):
        
        data = pd.read_csv(input_file)
        
        #replace inf with nan
        numeric_df = data.select_dtypes(include=[np.number])

        inf_columns = numeric_df.columns[numeric_df.isin([np.inf, -np.inf]).any(axis = 0)].tolist()

        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        if inf_columns:
            inf_counts = numeric_df[inf_columns].isin([np.inf, -np.inf]).sum().reset_index()
            inf_counts.columns = ["Feature", "Inf_Count"]

            inf_file = os.path.join("QSAR/Descriptor_Preprocessing", "inf_features.csv")
            inf_counts.to_csv(inf_file, index=False)

        else:
            inf_file = None

        #remove compounds with high nan percentage
        nan_counts = data.isna().sum(axis = 1)
        total_columns = data.shape[1]
        nan_percentage = nan_counts / total_columns

        filtered_data = data[nan_percentage <= compounds_nan_threshold]
        removed_count = data.shape[0] - filtered_data.shape[0]

        #remove descriptors with high nan percentage
        nan_percentage = filtered_data.isna().mean()
        retained_columns = nan_percentage[nan_percentage <= descriptors_nan_threshold].index.tolist()

        if "Label" not in retained_columns:
            retained_columns.append("Label")

        filtered_data = filtered_data[retained_columns]
        removed_count = filtered_data.shape[1] - len(retained_columns)

        #impute missing values
        imputer = SimpleImputer(strategy = imputation_method)
        imputed_descriptors = pd.DataFrame(imputer.fit_transform(filtered_data.drop(columns = ["Label"])), columns = filtered_data.drop(columns = ["Label"]).columns)

        final_data = pd.concat([imputed_descriptors, filtered_data[["Label"]]], axis = 1)

        output_file = os.path.join("QSAR/Descriptor_Preprocessing", "preprocessed_data.csv")

        final_data.to_csv(output_file, index = False)

        text_container = create_text_container(
            "🔹 **Preprocessing Completed!** 🔹",
            f"📌 Inf columns detected: {len(inf_columns)}",
            f"📄 Inf report: {inf_file}" if inf_file else "✅ No Inf values detected.",
            f"📌 Retained Compounds: {filtered_data.shape[0]}/{data.shape[0]} ✅",
            f"🗑️ Removed Compounds: {removed_count} rows with > {compounds_nan_threshold*100:.0f}% NaN",
            f"📌 Retained Compounds: {len(retained_columns)} / {data.shape[1]} descriptors",
            f"🗑️ Removed Compounds: {removed_count} columns with > {descriptors_nan_threshold*100:.0f}% NaN",
            f"🛠 Imputation Method: {imputation_method}",
        )

        return {"ui": {"text": text_container},
                "result": (str(output_file),)}

NODE_CLASS_MAPPINGS = {
    "Replace_inf_with_nan_Classification": Replace_inf_with_nan_Classification,
    "Remove_high_nan_compounds_Classification": Remove_high_nan_compounds_Classification,
    "Remove_high_nan_descriptors_Classification": Remove_high_nan_descriptors_Classification,
    "Impute_missing_values_Classification": Impute_missing_values_Classification,
    "Descriptor_preprocessing_Classification": Descriptor_preprocessing_Classification
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Replace_inf_with_nan_Classification": "Replace Infinite Values with NaN(Classification)",
    "Remove_high_nan_compounds_Classification": "Remove Compounds with High NaN Ratio(Classification)",
    "Remove_high_nan_descriptors_Classification": "Remove Descriptors with High NaN Ratio(Classification)",
    "Impute_missing_values_Classification": "Impute Missing Values(Classification)",
    "Descriptor_preprocessing_Classification": "Descriptor Preprocessing(Classification)"
} 