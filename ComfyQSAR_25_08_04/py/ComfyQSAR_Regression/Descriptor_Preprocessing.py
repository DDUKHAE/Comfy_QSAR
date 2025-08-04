import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

class Replace_inf_with_nan_Regression():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING",),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("INF_REPLACED_PATH",)
    FUNCTION = "replace_inf_with_nan"
    CATEGORY = "QSAR/REGRESSION/PREPROCESSING"
    OUTPUT_NODE = True

    @staticmethod
    def replace_inf_with_nan(input_file):
        output_dir = "QSAR/Descriptor_Preprocessing"
        output_file = ""
        inf_report_file = None
        inf_columns_count = 0

        try:
            os.makedirs(output_dir, exist_ok=True)

            data = pd.read_csv(input_file)
            initial_rows, initial_cols = data.shape

            numeric_df = data.select_dtypes(include=[np.number])
            inf_columns = numeric_df.columns[numeric_df.isin([np.inf, -np.inf]).any(axis = 0)].tolist()
            inf_columns_count = len(inf_columns)

            if inf_columns_count > 0:
                data.replace([np.inf, -np.inf], np.nan, inplace=True)

                inf_counts = numeric_df[inf_columns].isin([np.inf, -np.inf]).sum().reset_index()
                inf_counts.columns = ["Feature", "Inf_Count"]
                inf_report_file = os.path.join(output_dir, "regression_inf_features_report.csv")
                inf_counts.to_csv(inf_report_file, index=False)


            output_file = os.path.join(output_dir, f"regression_inf_replaced_{initial_cols}.csv")
            data.to_csv(output_file, index=False)

            log_message = (
                "========================================\n"
                "🔹 **Inf Replacement Completed!** 🔹\n"
                "========================================\n"
                f"✅ Input File: {os.path.basename(input_file)}\n"
                f"✅ Columns with Inf values: {inf_columns_count}\n"
                f"💾 Inf Report File: {inf_report_file if inf_report_file else 'N/A (No Inf found)'}\n"
                f"💾 Output File: {output_file}\n"
                "========================================"
            )
            return {"ui": {"text": log_message},
                    "result": (str(output_file),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}."
            return {"ui": {"text": error_msg}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": error_msg}, "result": (",")}


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
    RETURN_NAMES = ("COMPOUND_FILTERED_PATH",)
    FUNCTION = "remove_high_nan_compounds"
    CATEGORY = "QSAR/REGRESSION/PREPROCESSING" 
    OUTPUT_NODE = True

    @staticmethod
    def remove_high_nan_compounds(input_file, threshold):
        output_dir = "QSAR/Descriptor_Preprocessing"
        output_file = ""
        initial_rows, final_rows = 0, 0

        try:
            os.makedirs(output_dir, exist_ok=True)

            data = pd.read_csv(input_file)
            initial_rows, initial_cols = data.shape

            nan_counts = data.isna().sum(axis = 1)
            total_columns = data.shape[1]
            # Avoid division by zero if no columns
            nan_percentage = nan_counts / total_columns if total_columns > 0 else pd.Series([0.0] * initial_rows, index=data.index)

            filtered_data = data[nan_percentage <= threshold]
            final_rows = filtered_data.shape[0]
            removed_count = initial_rows - final_rows

            output_file = os.path.join(output_dir, f"regression_compound_filtered_{initial_rows}_to_{final_rows}.csv")
            filtered_data.to_csv(output_file, index=False)

            log_message = (
                "========================================\n"
                "🔹 **High NaN Compound Removal Completed!** 🔹\n"
                "========================================\n"
                f"✅ Input File: {os.path.basename(input_file)}\n"
                f"✅ NaN Threshold: > {threshold*100:.0f}% per compound\n"
                f"✅ Initial Compounds: {initial_rows}\n"
                f"✅ Compounds Removed: {removed_count}\n"
                f"✅ Remaining Compounds: {final_rows}\n"
                f"💾 Output File: {output_file}\n"
                "========================================"
            )
            return {"ui": {"text": log_message},
                    "result": (str(output_file),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}."
            return {"ui": {"text": error_msg}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": error_msg}, "result": (",")}
        

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
    RETURN_NAMES = ("DESCRIPTOR_FILTERED_PATH",)
    FUNCTION = "remove_high_nan_descriptors"
    CATEGORY = "QSAR/REGRESSION/PREPROCESSING" 
    OUTPUT_NODE = True

    @staticmethod
    def remove_high_nan_descriptors(input_file, threshold):
        output_dir = "QSAR/Descriptor_Preprocessing"
        output_file = ""
        initial_cols, final_cols = 0, 0

        try:
            os.makedirs(output_dir, exist_ok=True)

            data = pd.read_csv(input_file)
            initial_rows, initial_cols = data.shape

            nan_percentage = data.isna().mean()

            retained_columns_initial = nan_percentage[nan_percentage <= threshold].index.tolist()

            # Ensure SMILES and value are always kept if they exist
            retained_columns = retained_columns_initial.copy()
            if "SMILES" in data.columns and "SMILES" not in retained_columns:
                retained_columns.append("SMILES")
            if "value" in data.columns and "value" not in retained_columns:
                retained_columns.append("value")

            # Reorder to keep SMILES first, value last if present
            ordered_cols = []
            if "SMILES" in retained_columns: ordered_cols.append("SMILES")
            ordered_cols.extend([col for col in retained_columns if col not in ["SMILES", "value"]])
            if "value" in retained_columns: ordered_cols.append("value")

            filtered_data = data[ordered_cols]
            final_cols = filtered_data.shape[1]
            removed_count = initial_cols - final_cols

            output_file = os.path.join(output_dir, f"regression_descriptor_filtered_{initial_cols}_to_{final_cols}.csv")
            filtered_data.to_csv(output_file, index=False)

            log_message = (
                "========================================\n"
                "🔹 **High NaN Descriptor Removal Completed!** 🔹\n"
                "========================================\n"
                f"✅ Input File: {os.path.basename(input_file)}\n"
                f"✅ NaN Threshold: > {threshold*100:.0f}% per descriptor\n"
                f"✅ Initial Descriptors: {initial_cols}\n"
                f"✅ Descriptors Removed: {removed_count}\n"
                f"✅ Remaining Descriptors: {final_cols}\n"
                f"💾 Output File: {output_file}\n"
                "========================================"
            )

            return {"ui": {"text": log_message},
                    "result": (str(output_file),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}."
            return {"ui": {"text": error_msg}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": error_msg}, "result": (",")}


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
    RETURN_NAMES = ("IMPUTED_DATA_PATH",)
    FUNCTION = "impute_missing_values"
    CATEGORY = "QSAR/REGRESSION/PREPROCESSING" 
    OUTPUT_NODE = True

    @staticmethod
    def impute_missing_values(input_file, method):
        output_dir = "QSAR/Descriptor_Preprocessing"
        output_file = ""
        imputed_cols_count = 0

        try:
            os.makedirs(output_dir, exist_ok=True)

            data = pd.read_csv(input_file)
            initial_rows, initial_cols = data.shape

            # Drop 'Name' column if it exists (often redundant with SMILES)
            if "Name" in data.columns:
                data = data.drop(columns = ["Name"])
                initial_cols = data.shape[1] # Update initial_cols

            # Separate non-descriptor columns
            critical_cols = []
            if "SMILES" in data.columns: critical_cols.append("SMILES")
            if "value" in data.columns: critical_cols.append("value")

            critical_data = data[critical_cols].copy() if critical_cols else pd.DataFrame()
            descriptors = data.drop(columns=critical_cols, errors='ignore')

            if descriptors.empty:
                final_data = critical_data # If only critical cols existed
            else:
                imputer = SimpleImputer(strategy = method)
                # Check for NaN before imputation
                nan_before = descriptors.isnull().sum().sum()

                imputed_descriptors_array = imputer.fit_transform(descriptors)
                imputed_descriptors = pd.DataFrame(imputed_descriptors_array, columns = descriptors.columns, index=descriptors.index)

                # Check for NaN after imputation (should be 0)
                nan_after = imputed_descriptors.isnull().sum().sum()
                if nan_after > 0:
                     log_message += "⚠️ Warning: NaN values remain after imputation. Check input data or imputation strategy."

                imputed_cols_count = descriptors.shape[1]

                if not critical_data.empty:
                     final_data = pd.concat([critical_data.reset_index(drop=True), imputed_descriptors.reset_index(drop=True)], axis = 1)
                else:
                     final_data = imputed_descriptors # Only descriptors existed

                # Ensure correct column order
                final_cols_order = []
                if "SMILES" in critical_cols: final_cols_order.append("SMILES")
                final_cols_order.extend(imputed_descriptors.columns)
                if "value" in critical_cols: final_cols_order.append("value")
                final_data = final_data[final_cols_order]


            output_file = os.path.join(output_dir, f"regression_imputed_{method}.csv")
            final_data.to_csv(output_file, index = False)

            log_message = (
                "========================================\n"
                "🔹 **Missing Value Imputation Completed!** 🔹\n"
                "========================================\n"
                f"✅ Input File: {os.path.basename(input_file)}\n"
                f"✅ Imputation Method: {method}\n"
                f"✅ Descriptor Columns Imputed: {imputed_cols_count}\n"
                f"💾 Output File: {output_file}\n"
                f"✅ Final Data Shape: {final_data.shape[0]} rows, {final_data.shape[1]} columns"
                "========================================"
            )
            
            return {"ui": {"text": log_message},
                    "result": (str(output_file),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}."
            return {"ui": {"text": error_msg}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": error_msg}, "result": (",")}


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
    RETURN_NAMES = ("INTEGRATED_PREPROCESSED_PATH",)
    FUNCTION = "preprocess"
    CATEGORY = "QSAR/REGRESSION/PREPROCESSING" 
    OUTPUT_NODE = True

    def preprocess(self, input_file, compounds_nan_threshold, descriptors_nan_threshold, imputation_method):
        output_dir = "QSAR/Descriptor_Preprocessing"
        output_file = ""
        inf_report_file = None
        original_shape, shape_after_inf, shape_after_compound, shape_after_descriptor = (0, 0), (0, 0), (0, 0), (0, 0)
        inf_columns_count, compound_removed_count, descriptor_removed_count = 0, 0, 0

        try:
            os.makedirs(output_dir, exist_ok=True)

            data = pd.read_csv(input_file)
            original_shape = data.shape

            # --- Step 1: Replace infinite values with NaN ---
            numeric_df = data.select_dtypes(include=[np.number])
            inf_columns = numeric_df.columns[numeric_df.isin([np.inf, -np.inf]).any(axis=0)].tolist()
            inf_columns_count = len(inf_columns)
            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            shape_after_inf = data.shape

            if inf_columns_count > 0:
                inf_counts = numeric_df[inf_columns].isin([np.inf, -np.inf]).sum().reset_index()
                inf_counts.columns = ["Feature", "Inf_Count"]
                inf_report_file = os.path.join(output_dir, "regression_integrated_inf_report.csv")
                inf_counts.to_csv(inf_report_file, index=False)

            # --- Step 2: Remove compounds with high NaN ratio ---
            nan_counts = data.isna().sum(axis=1)
            total_columns_s2 = data.shape[1]
            nan_percentage_s2 = nan_counts / total_columns_s2 if total_columns_s2 > 0 else pd.Series([0.0] * data.shape[0], index=data.index)
            filtered_data_s2 = data[nan_percentage_s2 <= compounds_nan_threshold]
            shape_after_compound = filtered_data_s2.shape
            compound_removed_count = shape_after_inf[0] - shape_after_compound[0]
            data = filtered_data_s2 # Update data

            # --- Step 3: Remove descriptors with high NaN ratio ---
            nan_percentage_s3 = data.isna().mean()
            retained_columns_s3_initial = nan_percentage_s3[nan_percentage_s3 <= descriptors_nan_threshold].index.tolist()

            # Ensure SMILES and value are kept
            retained_columns_s3 = retained_columns_s3_initial.copy()
            critical_cols_s3 = []
            if "SMILES" in data.columns: critical_cols_s3.append("SMILES")
            if "value" in data.columns: critical_cols_s3.append("value")
            for col in critical_cols_s3:
                 if col not in retained_columns_s3:
                     retained_columns_s3.append(col)

            filtered_data_s3 = data[retained_columns_s3]
            shape_after_descriptor = filtered_data_s3.shape
            descriptor_removed_count = shape_after_compound[1] - shape_after_descriptor[1]
            data = filtered_data_s3 # Update data

            # --- Step 4: Impute remaining missing values ---
            # Separate non-descriptor columns for imputation
            critical_cols_s4 = []
            if "SMILES" in data.columns: critical_cols_s4.append("SMILES")
            if "value" in data.columns: critical_cols_s4.append("value")
            # Check for Name column and add if present, as it might be non-numeric
            if "Name" in data.columns: critical_cols_s4.append("Name")

            critical_data_s4 = data[critical_cols_s4].copy() if critical_cols_s4 else pd.DataFrame()
            descriptors_s4 = data.drop(columns=critical_cols_s4, errors='ignore')

            if descriptors_s4.empty:
                final_data = critical_data_s4 # Only critical columns remained
            else:
                nan_before_s4 = descriptors_s4.isnull().sum().sum()
                if nan_before_s4 == 0:
                    final_data = data # Data is already clean
                else:
                    imputer_s4 = SimpleImputer(strategy=imputation_method)
                    imputed_descriptors_array_s4 = imputer_s4.fit_transform(descriptors_s4)
                    imputed_descriptors_s4 = pd.DataFrame(imputed_descriptors_array_s4, columns = descriptors_s4.columns, index=descriptors_s4.index)
                    nan_after_s4 = imputed_descriptors_s4.isnull().sum().sum()

                    # Recombine
                    if not critical_data_s4.empty:
                        final_data = pd.concat([critical_data_s4.reset_index(drop=True), imputed_descriptors_s4.reset_index(drop=True)], axis = 1)
                    else:
                        final_data = imputed_descriptors_s4

                    # Ensure correct column order
                    final_cols_order = []
                    if "SMILES" in critical_cols_s4: final_cols_order.append("SMILES")
                    if "Name" in critical_cols_s4: final_cols_order.append("Name") # Keep Name if it existed
                    final_cols_order.extend(imputed_descriptors_s4.columns)
                    if "value" in critical_cols_s4: final_cols_order.append("value")
                    final_data = final_data[final_cols_order]

            # --- Save final preprocessed data ---
            final_shape = final_data.shape
            output_file = os.path.join(output_dir, f"regression_integrated_preprocessed_{original_shape[0]}x{original_shape[1]}_to_{final_shape[0]}x{final_shape[1]}.csv")
            final_data.to_csv(output_file, index=False)

            # --- Generate final summary ---
            log_message = (
                "========================================\n"
                "🔹 **Integrated Preprocessing Completed!** 🔹\n"
                "========================================\n"
                f"✅ Input File: {os.path.basename(input_file)} ({original_shape[0]}x{original_shape[1]})\n"
                f"✅ Replace Inf: {inf_columns_count} columns affected. {'Report: ' + inf_report_file if inf_report_file else ''}\n"
                f"✅ Remove Compounds (> {compounds_nan_threshold*100:.0f}% NaN): Removed {compound_removed_count} compounds.\n"
                f"✅ Remove Descriptors (> {descriptors_nan_threshold*100:.0f}% NaN): Removed {descriptor_removed_count} descriptors.\n"
                f"✅ Impute Missing Values: Method '{imputation_method}'.\n"
                f"✅ Final Data Shape: {final_shape[0]} rows, {final_shape[1]} columns\n"
                f"💾 Output File: {output_file}\n"
                "========================================"
            )

            return {"ui": {"text": log_message},
                    "result": (str(output_file),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}."
            return {"ui": {"text": error_msg}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": error_msg}, "result": (",")}
    

# Node Registration (Updated)
NODE_CLASS_MAPPINGS = {
    "Replace_inf_with_nan_Regression": Replace_inf_with_nan_Regression,
    "Remove_high_nan_compounds_Regression": Remove_high_nan_compounds_Regression,
    "Remove_high_nan_descriptors_Regression": Remove_high_nan_descriptors_Regression,
    "Impute_missing_values_Regression": Impute_missing_values_Regression,
    "Descriptor_preprocessing_Regression": Descriptor_preprocessing_Regression
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Replace_inf_with_nan_Regression": "Replace Inf with NaN (Regression)",
    "Remove_high_nan_compounds_Regression": "Remove High NaN Compounds (Regression)",
    "Remove_high_nan_descriptors_Regression": "Remove High NaN Descriptors (Regression)",
    "Impute_missing_values_Regression": "Impute Missing Values (Regression)",
    "Descriptor_preprocessing_Regression": "Descriptor Preprocessing (Integrated) (Regression)"
}