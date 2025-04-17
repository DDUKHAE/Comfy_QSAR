import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
# from .Data_Loader import create_text_container # Now imported from progress_utils

# --- Common Utility Import ---
from server import PromptServer
import time # time 모듈 추가

# WebSocket 이벤트 이름 정의 (모든 QSAR 노드에서 공통 사용)
QSAR_PROGRESS_EVENT = "qsar-desc-calc-progress" # 이름을 좀 더 범용적으로 변경 (선택적)
# 또는 기존 이름 유지: QSAR_DESC_CALC_PROGRESS_EVENT = "qsar-desc-calc-progress"

def send_progress(message, progress=None, node_id=None):
    """
    지정된 메시지와 진행률(0-100)을 WebSocket을 통해 프론트엔드로 전송하고,
    중간 단계 업데이트 시 짧은 지연 시간을 추가하여 UI에서 볼 수 있도록 합니다.
    Args:
        message (str): 표시할 상태 메시지.
        progress (Optional[float]): 0부터 100 사이의 진행률 값.
        node_id (Optional[str]): 특정 노드를 대상으로 할 경우 노드 ID.
    """
    payload = {"text": [message]}
    is_intermediate_update = False # 중간 업데이트 여부 플래그

    if progress is not None:
        # 진행률 값을 0과 100 사이로 제한하고 소수점 첫째 자리까지 반올림 (선택적)
        clamped_progress = max(0.0, min(100.0, float(progress)))
        payload['progress'] = round(clamped_progress, 1)
        # 100%가 아닌 진행률 업데이트인지 확인
        if clamped_progress < 100:
            is_intermediate_update = True

    # node ID 추가 (프론트엔드에서 필터링 시 사용 가능)
    # if node_id: payload['node'] = node_id

    try:
        # PromptServer 인스턴스를 통해 동기적으로 메시지 전송
        PromptServer.instance.send_sync(QSAR_PROGRESS_EVENT, payload)

        # 중간 진행률 업데이트 후 짧은 지연 시간 추가 (0.2초)
        # 최종(100%) 업데이트 시에는 지연 없음
        if is_intermediate_update:
            time.sleep(0.2) # 0.2초 대기

    except Exception as e:
        print(f"[ComfyQSAR Progress Util] WebSocket 전송 오류: {e}")

# 필요에 따라 다른 유틸리티 함수 추가 가능 (예: 시간 포맷팅 등) 

# 텍스트 컨테이너 생성 헬퍼 함수
def create_text_container(*lines):
    # 가장 긴 라인을 기준으로 구분선 길이 결정
    max_length = max(len(line) for line in lines)
    separator = "=" * max_length
    
    # 첫 구분선 추가
    result = [separator]
    
    # 각 라인 추가
    for line in lines:
        result.append(line)
    
    # 마지막 구분선 추가
    result.append(separator)
    
    # 줄바꿈으로 조인
    return "\n".join(result)

class Replace_inf_with_nan_Classification():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("CLEANED_DATA_PATH",) # More specific name
    FUNCTION = "replace_inf_with_nan"
    CATEGORY = "QSAR/CLASSIFICATION/PREPROCESSING"
    OUTPUT_NODE = True

    @staticmethod
    def replace_inf_with_nan(input_file):
        send_progress("🚀 Starting Inf value replacement...", 0)
        output_dir = "QSAR/Descriptor_Preprocessing"
        inf_file = None # Initialize inf_file path
        output_file = "" # Initialize output_file path

        try:
            os.makedirs(output_dir, exist_ok=True)
            send_progress(f"📂 Output directory checked/created: {output_dir}", 5)

            send_progress(f"⏳ Loading data from: {input_file}", 10)
            data = pd.read_csv(input_file)
            send_progress("   Data loaded.", 15)

            send_progress("⚙️ Identifying numeric columns and checking for Inf values...", 20)
            numeric_df = data.select_dtypes(include=[np.number])
            inf_mask = numeric_df.isin([np.inf, -np.inf])
            inf_columns = numeric_df.columns[inf_mask.any(axis=0)].tolist()
            total_inf_count = inf_mask.sum().sum()
            send_progress(f"   Found {len(inf_columns)} columns with Inf values. Total Inf count: {total_inf_count}", 30)

            if total_inf_count > 0:
                send_progress("🛠️ Replacing Inf values with NaN...", 40)
                data.replace([np.inf, -np.inf], np.nan, inplace=True)
                send_progress("   Inf replacement complete.", 50)

                if inf_columns:
                    send_progress("📊 Generating Inf report...", 60)
                    inf_counts = numeric_df[inf_columns].isin([np.inf, -np.inf]).sum().reset_index()
                    inf_counts.columns = ["Feature", "Original_Inf_Count"]
                    inf_counts = inf_counts[inf_counts["Original_Inf_Count"] > 0] # Only report columns that had inf
                    inf_file = os.path.join(output_dir, "inf_features_report.csv")
                    inf_counts.to_csv(inf_file, index=False)
                    send_progress(f"   Inf report saved to: {inf_file}", 70)
            else:
                 send_progress("✅ No Inf values detected.", 50)


            send_progress("💾 Saving cleaned data...", 80)
            output_file = os.path.join(output_dir, "inf_replaced_data.csv") # More specific name
            data.to_csv(output_file, index=False)
            send_progress(f"   Cleaned data saved to: {output_file}", 85)

            send_progress("📝 Generating summary...", 95)
            text_container_content = create_text_container(
                "🔹 **Inf Value Replacement Completed!** 🔹",
                f"Input File: {os.path.basename(input_file)}",
                f"Columns with Inf: {len(inf_columns)}",
                f"Total Inf Values Replaced: {total_inf_count}",
                f"Inf Report Saved: {inf_file}" if inf_file else "Inf Report: Not generated (no Inf values found).",
                f"Output File: {output_file}"
            )
            send_progress("🎉 Inf replacement process finished.", 100)

            return {"ui": {"text": text_container_content},
                    "result": (str(output_file),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}. Please check input file path."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred during Inf replacement: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}


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
    RETURN_NAMES = ("COMPOUND_FILTERED_PATH",) # More specific name
    FUNCTION = "remove_high_nan_compounds"
    CATEGORY = "QSAR/CLASSIFICATION/PREPROCESSING"
    OUTPUT_NODE = True

    @staticmethod
    def remove_high_nan_compounds(input_file, threshold):
        send_progress("🚀 Starting removal of compounds with high NaN ratio...", 0)
        output_dir = "QSAR/Descriptor_Preprocessing"
        output_file = ""

        try:
            os.makedirs(output_dir, exist_ok=True)
            send_progress(f"📂 Output directory checked/created: {output_dir}", 5)

            send_progress(f"⏳ Loading data from: {input_file}", 10)
            data = pd.read_csv(input_file)
            original_rows, original_cols = data.shape
            send_progress(f"   Data loaded ({original_rows} rows, {original_cols} columns).", 15)

            send_progress(f"⚙️ Calculating NaN percentage per compound (row)... Threshold: {threshold:.2f}", 20)
            nan_counts = data.isna().sum(axis=1)
            nan_percentage = nan_counts / original_cols # Use original_cols for percentage calculation
            send_progress("   NaN percentage calculation complete.", 30)

            send_progress(f"✂️ Filtering compounds with NaN ratio <= {threshold:.2f}...", 40)
            filtered_data = data[nan_percentage <= threshold].copy() # Use copy()
            filtered_rows = filtered_data.shape[0]
            removed_count = original_rows - filtered_rows
            send_progress(f"   Filtering complete. Kept {filtered_rows} compounds, removed {removed_count}.", 60)

            send_progress("💾 Saving filtered data...", 80)
            # Include original and filtered counts in filename
            output_file = os.path.join(output_dir, f"filtered_compounds_nan_{original_rows}_to_{filtered_rows}.csv")
            filtered_data.to_csv(output_file, index=False)
            send_progress(f"   Filtered data saved to: {output_file}", 85)

            send_progress("📝 Generating summary...", 95)
            text_container_content = create_text_container(
                "🔹 **High NaN Compound Removal Completed!** 🔹",
                f"Input File: {os.path.basename(input_file)} ({original_rows} rows)",
                f"NaN Threshold: {threshold*100:.0f}% per compound",
                f"Compounds Retained: {filtered_rows}",
                f"Compounds Removed: {removed_count}",
                f"Output File: {output_file}"
            )
            send_progress("🎉 High NaN compound removal process finished.", 100)

            return {"ui": {"text": text_container_content},
                    "result": (str(output_file),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}. Please check input file path."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred during high NaN compound removal: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}


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
    RETURN_NAMES = ("DESCRIPTOR_FILTERED_PATH",) # More specific name
    FUNCTION = "remove_high_nan_descriptors"
    CATEGORY = "QSAR/CLASSIFICATION/PREPROCESSING"
    OUTPUT_NODE = True

    @staticmethod
    def remove_high_nan_descriptors(input_file, threshold):
        send_progress("🚀 Starting removal of descriptors with high NaN ratio...", 0)
        output_dir = "QSAR/Descriptor_Preprocessing"
        output_file = ""

        try:
            os.makedirs(output_dir, exist_ok=True)
            send_progress(f"📂 Output directory checked/created: {output_dir}", 5)

            send_progress(f"⏳ Loading data from: {input_file}", 10)
            data = pd.read_csv(input_file)
            original_rows, original_cols = data.shape
            send_progress(f"   Data loaded ({original_rows} rows, {original_cols} columns).", 15)

            send_progress(f"⚙️ Calculating NaN percentage per descriptor (column)... Threshold: {threshold:.2f}", 20)
            nan_percentage = data.isna().mean() # Calculates mean (which is percentage for boolean mask)
            send_progress("   NaN percentage calculation complete.", 30)

            send_progress(f"✂️ Filtering descriptors with NaN ratio <= {threshold:.2f}...", 40)
            retained_columns = nan_percentage[nan_percentage <= threshold].index.tolist()

            # Ensure essential columns like 'Label' (or maybe 'SMILES', 'Name' if present) are kept
            essential_cols = ["Label", "SMILES", "Name"] # Add other potential identifiers
            for col in essential_cols:
                 if col in data.columns and col not in retained_columns:
                      retained_columns.append(col)
                      send_progress(f"   Ensured essential column '{col}' is retained.", 45)


            filtered_data = data[retained_columns].copy() # Use .copy()
            filtered_cols_count = filtered_data.shape[1]
            removed_count = original_cols - filtered_cols_count
            send_progress(f"   Filtering complete. Kept {filtered_cols_count} descriptors, removed {removed_count}.", 60)

            send_progress("💾 Saving filtered data...", 80)
            # Include original and filtered counts in filename
            output_file = os.path.join(output_dir, f"filtered_descriptors_nan_{original_cols}_to_{filtered_cols_count}.csv")
            filtered_data.to_csv(output_file, index=False)
            send_progress(f"   Filtered data saved to: {output_file}", 85)

            send_progress("📝 Generating summary...", 95)
            text_container_content = create_text_container(
                "🔹 **High NaN Descriptor Removal Completed!** 🔹",
                f"Input File: {os.path.basename(input_file)} ({original_cols} columns)",
                f"NaN Threshold: {threshold*100:.0f}% per descriptor",
                f"Descriptors Retained: {filtered_cols_count}",
                f"Descriptors Removed: {removed_count}",
                f"Output File: {output_file}"
            )
            send_progress("🎉 High NaN descriptor removal process finished.", 100)

            return {"ui": {"text": text_container_content},
                    "result": (str(output_file),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}. Please check input file path."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred during high NaN descriptor removal: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}


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
    RETURN_NAMES = ("IMPUTED_DATA_PATH",) # More specific name
    FUNCTION = "impute_missing_values"
    CATEGORY = "QSAR/CLASSIFICATION/PREPROCESSING"
    OUTPUT_NODE = True

    @staticmethod
    def impute_missing_values(input_file, method):
        send_progress(f"🚀 Starting missing value imputation using '{method}' strategy...", 0)
        output_dir = "QSAR/Descriptor_Preprocessing"
        output_file = ""

        try:
            os.makedirs(output_dir, exist_ok=True)
            send_progress(f"📂 Output directory checked/created: {output_dir}", 5)

            send_progress(f"⏳ Loading data from: {input_file}", 10)
            data = pd.read_csv(input_file)
            original_rows, original_cols = data.shape
            send_progress(f"   Data loaded ({original_rows} rows, {original_cols} columns).", 15)

            # Store non-numeric columns (like Name, SMILES, Label) to add back later
            send_progress("⚙️ Separating non-numeric columns...", 20)
            non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

            if not numeric_cols:
                 raise ValueError("No numeric columns found in the input file to impute.")

            non_numeric_data = data[non_numeric_cols]
            numeric_data = data[numeric_cols]
            initial_nan_count = numeric_data.isna().sum().sum()
            send_progress(f"   Identified {len(numeric_cols)} numeric columns with {initial_nan_count} total missing values.", 25)


            send_progress(f"🛠️ Applying '{method}' imputation...", 30)
            imputer = SimpleImputer(strategy=method)
            imputed_numeric_data = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_cols, index=data.index) # Preserve index
            final_nan_count = imputed_numeric_data.isna().sum().sum() # Should be 0
            send_progress(f"   Imputation complete. Remaining NaNs in numeric columns: {final_nan_count}", 70)


            send_progress("⚙️ Recombining imputed numeric data with non-numeric columns...", 75)
            # Concatenate based on index
            final_data = pd.concat([non_numeric_data, imputed_numeric_data], axis=1)
            # Ensure original column order if important (optional)
            final_data = final_data[data.columns]
            send_progress("   Data recombined.", 80)


            send_progress("💾 Saving imputed data...", 85)
            output_file = os.path.join(output_dir, f"imputed_{method}_data.csv") # Include method in name
            final_data.to_csv(output_file, index=False)
            send_progress(f"   Imputed data saved to: {output_file}", 90)

            send_progress("📝 Generating summary...", 95)
            text_container_content = create_text_container(
                "🔹 **Missing Value Imputation Completed!** 🔹",
                f"Input File: {os.path.basename(input_file)} ({original_rows} rows, {original_cols} columns)",
                f"Imputation Strategy: '{method}'",
                f"Numeric Columns Imputed: {len(numeric_cols)}",
                f"Original Missing Values (Numeric): {initial_nan_count}",
                f"Remaining Missing Values (Numeric): {final_nan_count}",
                f"Output File: {output_file}"
            )
            send_progress("🎉 Imputation process finished.", 100)

            return {"ui": {"text": text_container_content},
                    "result": (str(output_file),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}. Please check input file path."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except ValueError as ve: # Catch specific errors like no numeric columns
            error_msg = f"❌ Value Error during imputation: {str(ve)}"
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred during imputation: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}


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
        send_progress("🚀 Starting Integrated Descriptor Preprocessing...", 0)
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
            send_progress(f"📂 Output directory checked/created: {output_dir}", 2)

            send_progress(f"⏳ Loading data from: {input_file}", 5)
            data = pd.read_csv(input_file)
            initial_rows, initial_cols = data.shape
            send_progress(f"   Data loaded ({initial_rows} rows, {initial_cols} columns).", 8)

            # --- 1. Replace inf with nan ---
            send_progress("➡️ Step 1: Replacing Inf values with NaN...", 10)
            numeric_df = data.select_dtypes(include=[np.number])
            inf_mask = numeric_df.isin([np.inf, -np.inf])
            inf_columns = numeric_df.columns[inf_mask.any(axis=0)].tolist()
            total_inf_count = inf_mask.sum().sum()

            if total_inf_count > 0:
                data.replace([np.inf, -np.inf], np.nan, inplace=True)
                send_progress(f"   Replaced {total_inf_count} Inf values in {len(inf_columns)} columns.", 15)
                # Save Inf report (optional, could be a parameter)
                try:
                    inf_counts = numeric_df[inf_columns].isin([np.inf, -np.inf]).sum().reset_index()
                    inf_counts.columns = ["Feature", "Original_Inf_Count"]
                    inf_counts = inf_counts[inf_counts["Original_Inf_Count"] > 0]
                    inf_file = os.path.join(output_dir, "integrated_inf_features_report.csv")
                    inf_counts.to_csv(inf_file, index=False)
                    send_progress(f"   Inf report saved: {inf_file}", 20)
                except Exception as report_e:
                    send_progress(f"   Warning: Could not save Inf report: {report_e}", 20)
                    inf_file = "Error saving report"
            else:
                 send_progress("   ✅ No Inf values found.", 20)
            send_progress("   Step 1 Complete.", 25)


            # --- 2. Remove compounds with high nan percentage ---
            send_progress(f"➡️ Step 2: Removing compounds with > {compounds_nan_threshold*100:.0f}% NaN...", 30)
            current_cols = data.shape[1] # Use current number of columns
            nan_counts_comp = data.isna().sum(axis=1)
            nan_percentage_comp = nan_counts_comp / current_cols
            data = data[nan_percentage_comp <= compounds_nan_threshold].copy() # Apply filter and copy
            rows_after_compound_filter = data.shape[0]
            removed_compounds_count = initial_rows - rows_after_compound_filter
            send_progress(f"   Removed {removed_compounds_count} compounds. Kept {rows_after_compound_filter}.", 45)
            send_progress("   Step 2 Complete.", 50)

            # --- 3. Remove descriptors with high nan percentage ---
            send_progress(f"➡️ Step 3: Removing descriptors with > {descriptors_nan_threshold*100:.0f}% NaN...", 55)
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
            send_progress(f"   Removed {removed_descriptors_count} descriptors. Kept {cols_after_descriptor_filter}.", 70)
            send_progress("   Step 3 Complete.", 75)

            # --- 4. Impute remaining missing values ---
            send_progress(f"➡️ Step 4: Imputing remaining NaNs using '{imputation_method}'...", 80)
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
                    send_progress(f"   Imputed {initial_nan_count_impute} values. Remaining NaNs (numeric): {final_nan_count_impute}.", 88)
                else:
                    send_progress("   ✅ No missing values to impute in numeric columns.", 88)
                    final_nan_count_impute = 0 # Set to 0 if none needed
            else:
                 send_progress("   No numeric columns found for imputation.", 88)
                 final_nan_count_impute = 0 # Set to 0 if none needed

            send_progress("   Step 4 Complete.", 90)

            # Final shape
            final_rows, final_cols = data.shape

            # --- 5. Save final data ---
            send_progress("💾 Saving final preprocessed data...", 92)
            output_file = os.path.join(output_dir, "integrated_preprocessed_data.csv")
            data.to_csv(output_file, index=False)
            send_progress(f"   Preprocessed data saved to: {output_file}", 94)

            # --- 6. Generate Summary ---
            send_progress("📝 Generating final summary...", 95)
            text_container_content = create_text_container(
                "🔹 **Integrated Descriptor Preprocessing Completed!** 🔹",
                f"Input File: {os.path.basename(input_file)} ({initial_rows} rows, {initial_cols} columns)",
                "--- Processing Steps ---",
                f"1. Inf Values Replaced: {total_inf_count} (Report: {inf_file if inf_file else 'N/A'})",
                f"2. Compounds Removed (> {compounds_nan_threshold*100:.0f}% NaN): {removed_compounds_count} (Kept: {rows_after_compound_filter})",
                f"3. Descriptors Removed (> {descriptors_nan_threshold*100:.0f}% NaN): {removed_descriptors_count} (Kept: {cols_after_descriptor_filter})",
                f"4. Imputation ('{imputation_method}'): {initial_nan_count_impute} values imputed. Remaining NaNs: {final_nan_count_impute}",
                "--- Final Output ---",
                f"Final Data Shape: {final_rows} rows, {final_cols} columns",
                f"Output File: {output_file}"
            )
            send_progress("🎉 Integrated preprocessing finished successfully!", 100)

            return {"ui": {"text": text_container_content},
                    "result": (str(output_file),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}. Please check input file path."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred during integrated preprocessing: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}


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