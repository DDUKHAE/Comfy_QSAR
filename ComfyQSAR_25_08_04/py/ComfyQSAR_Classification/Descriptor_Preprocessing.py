import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

class Replace_inf_with_nan_Classification():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "descriptors": ("STRING", {"tooltip": "Path to the input file"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PREPROCESSED_DESCRIPTORS",) # More specific name
    FUNCTION = "replace_inf_with_nan"
    CATEGORY = "QSAR/CLASSIFICATION/PREPROCESSING"
    OUTPUT_NODE = True

    @staticmethod
    def replace_inf_with_nan(descriptors):
        output_dir = "QSAR/Descriptor_Preprocessing"
        inf_file = None
        output_file = ""

        try:
            os.makedirs(output_dir, exist_ok=True)

            data = pd.read_csv(descriptors)

            numeric_df = data.select_dtypes(include=[np.number])
            inf_mask = numeric_df.isin([np.inf, -np.inf])
            inf_columns = numeric_df.columns[inf_mask.any(axis=0)].tolist()
            total_inf_count = inf_mask.sum().sum()

            if total_inf_count > 0:
                data.replace([np.inf, -np.inf], np.nan, inplace=True)

                if inf_columns:
                    inf_counts = numeric_df[inf_columns].isin([np.inf, -np.inf]).sum().reset_index()
                    inf_counts.columns = ["Feature", "Original_Inf_Count"]
                    inf_counts = inf_counts[inf_counts["Original_Inf_Count"] > 0] # Only report columns that had inf
                    inf_file = os.path.join(output_dir, "inf_features_report.csv")
                    inf_counts.to_csv(inf_file, index=False)
            else:
                output_file = os.path.join(output_dir, "inf_replaced_data.csv")

            output_file = os.path.join(output_dir, "inf_replaced_data.csv")
            data.to_csv(output_file, index=False)

            log_message = (
                "========================================\n"
                "🔹 Inf Value Replacement Completed! 🔹\n"
                "========================================\n"
                f"✅ Input File: {os.path.basename(descriptors)}\n"
                f"✅ Columns with Inf: {len(inf_columns)}\n"
                f"✅ Total Inf Values Replaced: {total_inf_count}\n"
                f"✅ Inf Report Saved: {inf_file if inf_file else 'Not generated (no Inf values found)'}\n"
                f"✅ Output File: {output_file}\n"
                "========================================"
            )

            return {"ui": {"text": log_message},
                    "result": (str(output_file),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}. Please check input file path."
            log_message = (
                "========================================\n"
                "❌ **Inf Value Replacement Error!** ❌\n"
                "========================================\n"
                f"Error: {error_msg}\n"
                "Please check the file path.\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred during Inf replacement: {str(e)}"
            log_message = (
                "========================================\n"
                "❌ **Inf Value Replacement Error!** ❌\n"
                "========================================\n"
                f"Error: {error_msg}\n"
                "Please check the file path.\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (",")}


class Remove_high_nan_compounds_Classification():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preprocessed_descriptors": ("STRING", {"tooltip": "Path to the input file"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, 
                                        "tooltip": "Threshold for compound filtering"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PREPROCESSED_DESCRIPTORS",) # More specific name
    FUNCTION = "remove_high_nan_compounds"
    CATEGORY = "QSAR/CLASSIFICATION/PREPROCESSING"
    OUTPUT_NODE = True

    @staticmethod
    def remove_high_nan_compounds(preprocessed_descriptors, threshold):
        output_dir = "QSAR/Descriptor_Preprocessing"
        output_file = ""

        try:
            os.makedirs(output_dir, exist_ok=True)

            data = pd.read_csv(preprocessed_descriptors)
            original_rows, original_cols = data.shape

            nan_counts = data.isna().sum(axis=1)
            nan_percentage = nan_counts / original_cols # Use original_cols for percentage calculation

            filtered_data = data[nan_percentage <= threshold].copy() # Use copy()
            filtered_rows = filtered_data.shape[0]
            removed_count = original_rows - filtered_rows

            output_file = os.path.join(output_dir, f"filtered_compounds_nan_{original_rows}_to_{filtered_rows}.csv")
            filtered_data.to_csv(output_file, index=False)

            log_message = (
                "========================================\n"
                "🔹 High NaN Compound Removal Completed! 🔹\n"
                "========================================\n"
                f"✅ Input File: {os.path.basename(preprocessed_descriptors)}\n"
                f"✅ NaN Threshold: {threshold*100:.0f}% per compound\n"
                f"✅ Compounds Retained: {filtered_rows}\n"
                f"✅ Compounds Removed: {removed_count}\n"
                f"✅ Output File: {output_file}\n"
                "========================================"
            )

            return {"ui": {"text": log_message},
                    "result": (str(output_file),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}. Please check input file path."
            log_message = (
                "========================================\n"
                "❌ **High NaN Compound Removal Error!** ❌\n"
                "========================================\n"
                f"Error: {error_msg}\n"
                "Please check the file path.\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred during high NaN compound removal: {str(e)}"
            log_message = (
                "========================================\n"
                "❌ **High NaN Compound Removal Error!** ❌\n"
                "========================================\n"
                f"Error: {error_msg}\n"
                "Please check the file path.\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (",")}


class Remove_high_nan_descriptors_Classification():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preprocessed_descriptors": ("STRING", {"tooltip": "Path to the input file"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, 
                                        "tooltip": "Threshold for descriptor filtering"}),                       
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PREPROCESSED_DESCRIPTORS",) # More specific name
    FUNCTION = "remove_high_nan_descriptors"
    CATEGORY = "QSAR/CLASSIFICATION/PREPROCESSING"
    OUTPUT_NODE = True

    @staticmethod
    def remove_high_nan_descriptors(preprocessed_descriptors, threshold):
        output_dir = "QSAR/Descriptor_Preprocessing"
        output_file = ""

        try:
            os.makedirs(output_dir, exist_ok=True)

            data = pd.read_csv(preprocessed_descriptors)
            original_rows, original_cols = data.shape

            nan_percentage = data.isna().mean() # Calculates mean (which is percentage for boolean mask)

            retained_columns = nan_percentage[nan_percentage <= threshold].index.tolist()

            # Ensure essential columns like 'Label' (or maybe 'SMILES', 'Name' if present) are kept
            essential_cols = ["Label", "SMILES", "Name"] # Add other potential identifiers
            for col in essential_cols:
                 if col in data.columns and col not in retained_columns:
                      retained_columns.append(col)

            filtered_data = data[retained_columns].copy() # Use .copy()
            filtered_cols_count = filtered_data.shape[1]
            removed_count = original_cols - filtered_cols_count

            output_file = os.path.join(output_dir, f"filtered_descriptors_nan_{original_cols}_to_{filtered_cols_count}.csv")
            filtered_data.to_csv(output_file, index=False)

            log_message = (
                "========================================\n"
                "🔹 High NaN Descriptor Removal Completed! 🔹\n"
                "========================================\n"
                f"✅ Input File: {os.path.basename(preprocessed_descriptors)}\n"
                f"✅ NaN Threshold: {threshold*100:.0f}% per descriptor\n"
                f"✅ Descriptors Retained: {filtered_cols_count}\n"
                f"✅ Descriptors Removed: {removed_count}\n"
                f"✅ Output File: {output_file}\n"
                "========================================"
            )

            return {"ui": {"text": log_message},
                    "result": (str(output_file),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}. Please check input file path."
            log_message = (
                "========================================\n"
                "❌ **High NaN Descriptor Removal Error!** ❌\n"
                "========================================\n"
                f"Error: {error_msg}\n"
                "Please check the file path.\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred during high NaN descriptor removal: {str(e)}"
            log_message = (
                "========================================\n"
                "❌ **High NaN Descriptor Removal Error!** ❌\n"
                "========================================\n"
                f"Error: {error_msg}\n"
                "Please check the file path.\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (",")}

class Impute_missing_values_Classification():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preprocessed_descriptors": ("STRING", {"tooltip": "Path to the input file"}),
                "method": (["mean", "median", "most_frequent"], 
                           {"tooltip": "Imputation method"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PREPROCESSED_DESCRIPTORS",) # More specific name
    FUNCTION = "impute_missing_values"
    CATEGORY = "QSAR/CLASSIFICATION/PREPROCESSING"
    OUTPUT_NODE = True

    @staticmethod
    def impute_missing_values(preprocessed_descriptors, method):
        output_dir = "QSAR/Descriptor_Preprocessing"
        output_file = ""

        try:
            os.makedirs(output_dir, exist_ok=True)

            data = pd.read_csv(preprocessed_descriptors)
            original_rows, original_cols = data.shape

            
            # Remove 'Name' column if it exists (following notebook approach)
            if "Name" in data.columns:
                data = data.drop(columns=["Name"])
                
            # Separate Label column (must exist for QSAR)
            if "Label" not in data.columns:
                raise ValueError("'Label' column not found in the dataset. Required for QSAR classification.")
                
            label_col = data["Label"]
            descriptors = data.drop(columns=["Label"])
            initial_nan_count = descriptors.isna().sum().sum()

            # Apply imputation to descriptor columns only
            imputer = SimpleImputer(strategy=method)
            imputed_descriptors = pd.DataFrame(
                imputer.fit_transform(descriptors), 
                columns=descriptors.columns,
                index=descriptors.index
            )
            final_nan_count = imputed_descriptors.isna().sum().sum() # Should be 0

            # Recombine imputed descriptors with Label column
            final_data = pd.concat([imputed_descriptors, label_col.reset_index(drop=True)], axis=1)

            output_file = os.path.join(output_dir, "preprocessed_data.csv") # Following notebook naming
            final_data.to_csv(output_file, index=False)

            
            # 페이지 시스템을 위한 데이터 구조 생성
            log_message = (
                "========================================\n"
                "🔹 Missing Value Imputation Completed! 🔹\n"
                "========================================\n"
                f"✅ Input File: {os.path.basename(preprocessed_descriptors)}\n"
                f"✅ Imputation Strategy: '{method}'\n"
                f"✅ Descriptor Columns Processed: {descriptors.shape[1]}\n"
                f"✅ Output File: {output_file}\n"
                "========================================"
            )
            
            return {"ui": {"text": log_message},
                    "result": (str(output_file),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}. Please check input file path."
            log_message = (
                "========================================\n"
                "❌ **Missing Value Imputation Error!** ❌\n"
                "========================================\n"
                f"Error: {error_msg}\n"
                "Please check the file path.\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (",")}
        except ValueError as ve: # Catch specific errors like no numeric columns
            error_msg = f"❌ Value Error during imputation: {str(ve)}"
            log_message = (
                "========================================\n"
                "❌ **Missing Value Imputation Error!** ❌\n"
                "========================================\n"
                f"Error: {error_msg}\n"
                "Please check the file path.\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred during imputation: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            log_message = (
                "========================================\n"
                "❌ **Missing Value Imputation Error!** ❌\n"
                "========================================\n"
                f"Error: {error_msg}\n"
                "Please check the file path.\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (",")}


class Descriptor_preprocessing_Classification:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "descriptors": ("STRING", {"tooltip": "Path to the input file"}),
                "compounds_nan_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, 
                                                     "tooltip": "Threshold for compound filtering"}),
                "descriptors_nan_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, 
                                                       "tooltip": "Threshold for descriptor filtering"}),
                "imputation_method": (["mean", "median", "most_frequent"], 
                                       {"tooltip": "Imputation method"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PREPROCESSED_DESCRIPTORS",)
    FUNCTION = "preprocess"
    CATEGORY = "QSAR/CLASSIFICATION/PREPROCESSING"
    OUTPUT_NODE = True

    def preprocess(self, descriptors, compounds_nan_threshold, descriptors_nan_threshold, imputation_method):
        output_dir = "QSAR/Descriptor_Preprocessing"
        inf_file = None
        output_file = ""
        initial_rows, initial_cols = 0, 0
        rows_after_compound_filter, cols_after_descriptor_filter = 0, 0
        final_rows, final_cols = 0, 0
        removed_compounds_count = 0
        removed_descriptors_count = 0
        total_inf_count = 0
        initial_nan_count_impute = 0
        final_nan_count_impute = -1 # Flag for not imputed yet

        try:
            os.makedirs(output_dir, exist_ok=True)

            data = pd.read_csv(descriptors)
            initial_rows, initial_cols = data.shape

            # --- 1. Replace inf with nan ---
            numeric_df = data.select_dtypes(include=[np.number])
            inf_mask = numeric_df.isin([np.inf, -np.inf])
            inf_columns = numeric_df.columns[inf_mask.any(axis=0)].tolist()
            total_inf_count = inf_mask.sum().sum()

            if total_inf_count > 0:
                data.replace([np.inf, -np.inf], np.nan, inplace=True)
                # Save Inf report (optional, could be a parameter)
                try:
                    inf_counts = numeric_df[inf_columns].isin([np.inf, -np.inf]).sum().reset_index()
                    inf_counts.columns = ["Feature", "Original_Inf_Count"]
                    inf_counts = inf_counts[inf_counts["Original_Inf_Count"] > 0]
                    inf_file = os.path.join(output_dir, "integrated_inf_features_report.csv")
                    inf_counts.to_csv(inf_file, index=False)
                except Exception as report_e:
                    inf_file = "Error saving report"
            else:
                 inf_file = "Error saving report"


            # --- 2. Remove compounds with high nan percentage ---
            current_cols = data.shape[1] # Use current number of columns
            nan_counts_comp = data.isna().sum(axis=1)
            nan_percentage_comp = nan_counts_comp / current_cols
            data = data[nan_percentage_comp <= compounds_nan_threshold].copy() # Apply filter and copy
            rows_after_compound_filter = data.shape[0]
            removed_compounds_count = initial_rows - rows_after_compound_filter

            # --- 3. Remove descriptors with high nan percentage ---
            nan_percentage_desc = data.isna().mean()
            retained_columns = nan_percentage_desc[nan_percentage_desc <= descriptors_nan_threshold].index.tolist()
            # Ensure essential columns
            essential_cols = ["Label", "SMILES", "Name"]
            for col in essential_cols:
                 if col in data.columns and col not in retained_columns:
                      retained_columns.append(col)
            data = data[retained_columns].copy() # Apply filter and copy
            cols_after_descriptor_filter = data.shape[1]
            removed_descriptors_count = current_cols - cols_after_descriptor_filter # Compare to cols *before* this step

            # --- 4. Impute remaining missing values (following notebook approach) ---
            
            # Remove 'Name' column if it exists (following notebook approach)
            if "Name" in data.columns:
                data = data.drop(columns=["Name"])
                # Update retained_columns list
                if "Name" in retained_columns:
                    retained_columns.remove("Name")
                    
            # Separate Label column (must exist for QSAR)
            if "Label" not in data.columns:
                # Fall back to general numeric/non-numeric approach
                non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                
                if numeric_cols:
                    non_numeric_data = data[non_numeric_cols]
                    numeric_data = data[numeric_cols]
                    initial_nan_count_impute = numeric_data.isna().sum().sum()

                    if initial_nan_count_impute > 0:
                        imputer = SimpleImputer(strategy=imputation_method)
                        imputed_numeric_data = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_cols, index=data.index)
                        final_nan_count_impute = imputed_numeric_data.isna().sum().sum()
                        # Recombine
                        data = pd.concat([non_numeric_data, imputed_numeric_data], axis=1)
                        data = data[retained_columns] # Keep column order
                    else:
                        final_nan_count_impute = 0
                else:
                     final_nan_count_impute = 0
            else:
                # Label-based approach (notebook method)
                label_col = data["Label"]
                descriptors = data.drop(columns=["Label"])
                initial_nan_count_impute = descriptors.isna().sum().sum()
                
                if initial_nan_count_impute > 0:
                    imputer = SimpleImputer(strategy=imputation_method)
                    imputed_descriptors = pd.DataFrame(
                        imputer.fit_transform(descriptors), 
                        columns=descriptors.columns,
                        index=descriptors.index
                    )
                    final_nan_count_impute = imputed_descriptors.isna().sum().sum()
                    
                    # Recombine with Label
                    data = pd.concat([imputed_descriptors, label_col.reset_index(drop=True)], axis=1)
                else:
                    final_nan_count_impute = 0


            # Final shape
            final_rows, final_cols = data.shape

            # --- 5. Save final data ---
            output_file = os.path.join(output_dir, "integrated_preprocessed_data.csv")
            data.to_csv(output_file, index=False)

            # --- 6. Generate Summary ---
            
            # 페이지 시스템을 위한 데이터 구조 생성
            log_message = (
                f"[Preprocessing Done]\n"
                f"Input: {os.path.basename(descriptors)}\n"
                f"Original: {initial_rows}x{initial_cols}\n"
                f"Inf→NaN: {total_inf_count} ({len(inf_columns) if 'inf_columns' in locals() else 0} cols)\n"
                f"Compound filter: {removed_compounds_count} removed, {rows_after_compound_filter} left\n"
                f"Descriptor filter: {removed_descriptors_count} removed, {cols_after_descriptor_filter} left\n"
                f"Impute({imputation_method}): {initial_nan_count_impute}→{final_nan_count_impute}\n"
                f"Final: {final_rows}x{final_cols}, File: {output_file}\n"
            )
            
            return {"ui": {"text": log_message},
                    "result": (str(output_file),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}. Please check input file path."
            log_message = (
                "========================================\n"
                "❌ **Descriptor Preprocessing Error!** ❌\n"
                "========================================\n"
                f"Error: {error_msg}\n"
                "Please check the file path.\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred during integrated preprocessing: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            log_message = (
                "========================================\n"
                "❌ **Descriptor Preprocessing Error!** ❌\n"
                "========================================\n"
                f"Error: {error_msg}\n"
                "Please check the file path.\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (",")}


NODE_CLASS_MAPPINGS = {
    "Replace_inf_with_nan_Classification": Replace_inf_with_nan_Classification,
    "Remove_high_nan_compounds_Classification": Remove_high_nan_compounds_Classification,
    "Remove_high_nan_descriptors_Classification": Remove_high_nan_descriptors_Classification,
    "Impute_missing_values_Classification": Impute_missing_values_Classification,
    "Descriptor_preprocessing_Classification": Descriptor_preprocessing_Classification
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Replace_inf_with_nan_Classification": "Replace Inf with NaN (Classification)", # Updated
    "Remove_high_nan_compounds_Classification": "Remove High NaN Compounds (Classification)", # Updated
    "Remove_high_nan_descriptors_Classification": "Remove High NaN Descriptors (Classification)", # Updated
    "Impute_missing_values_Classification": "Impute Missing Values (Classification)", # Updated
    "Descriptor_preprocessing_Classification": "Descriptor Preprocessing (Integrated) (Classification)" # Updated
} 