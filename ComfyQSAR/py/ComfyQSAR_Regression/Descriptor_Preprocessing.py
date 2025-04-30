import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
# from .Data_Loader import create_text_container # Now imported below

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
    # node_id가 None이면 환경변수에서 가져오기
    if node_id is None:
        node_id = os.environ.get("NODE_ID")
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
    if node_id:
        payload['node'] = node_id

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
        send_progress("🚀 Starting Inf Value Replacement (Regression)...", 0)
        output_dir = "QSAR/Descriptor_Preprocessing"
        output_file = ""
        inf_report_file = None
        inf_columns_count = 0

        try:
            os.makedirs(output_dir, exist_ok=True)
            send_progress(f"📂 Output directory checked/created: {output_dir}", 5)

            send_progress(f"⏳ Loading data from: {input_file}", 10)
            data = pd.read_csv(input_file)
            initial_rows, initial_cols = data.shape
            send_progress(f"   Data loaded ({initial_rows} rows, {initial_cols} columns).", 20)

            send_progress("🔎 Identifying numeric columns and checking for Inf values...", 30)
            numeric_df = data.select_dtypes(include=[np.number])
            inf_columns = numeric_df.columns[numeric_df.isin([np.inf, -np.inf]).any(axis = 0)].tolist()
            inf_columns_count = len(inf_columns)
            send_progress(f"   Found {inf_columns_count} columns with Inf values.", 40)

            if inf_columns_count > 0:
                send_progress(f"🔄 Replacing Inf values with NaN in {inf_columns_count} columns...", 50)
                data.replace([np.inf, -np.inf], np.nan, inplace=True)
                send_progress("   Replacement complete.", 60)

                send_progress("📝 Generating Inf report...", 65)
                inf_counts = numeric_df[inf_columns].isin([np.inf, -np.inf]).sum().reset_index()
                inf_counts.columns = ["Feature", "Inf_Count"]
                inf_report_file = os.path.join(output_dir, "regression_inf_features_report.csv")
                inf_counts.to_csv(inf_report_file, index=False)
                send_progress(f"   Inf report saved to: {inf_report_file}", 75)
            else:
                 send_progress("✅ No Inf values found to replace or report.", 75)


            send_progress("💾 Saving processed data...", 80)
            output_file = os.path.join(output_dir, f"regression_inf_replaced_{initial_cols}.csv")
            data.to_csv(output_file, index=False)
            send_progress(f"   Processed data saved to: {output_file}", 90)

            send_progress("📝 Generating final summary...", 95)
            text_container_content = create_text_container(
                "🔹 **Inf Replacement Completed (Regression)!** 🔹",
                f"Input File: {os.path.basename(input_file)}",
                f"Columns with Inf values: {inf_columns_count}",
                f"Inf Report File: {inf_report_file if inf_report_file else 'N/A (No Inf found)'}",
                f"Output File: {output_file}"
            )
            send_progress("🎉 Inf replacement process finished.", 100)

            return {"ui": {"text": text_container_content},
                    "result": (str(output_file),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}


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
        send_progress("🚀 Starting High NaN Compound Removal (Regression)...", 0)
        output_dir = "QSAR/Descriptor_Preprocessing"
        output_file = ""
        initial_rows, final_rows = 0, 0

        try:
            os.makedirs(output_dir, exist_ok=True)
            send_progress(f"📂 Output directory checked/created: {output_dir}", 5)

            send_progress(f"⏳ Loading data from: {input_file}", 10)
            data = pd.read_csv(input_file)
            initial_rows, initial_cols = data.shape
            send_progress(f"   Data loaded ({initial_rows} rows, {initial_cols} columns).", 20)

            send_progress(f"📊 Calculating NaN percentage per compound (threshold = {threshold*100:.0f}%)...", 30)
            nan_counts = data.isna().sum(axis = 1)
            total_columns = data.shape[1]
            # Avoid division by zero if no columns
            nan_percentage = nan_counts / total_columns if total_columns > 0 else pd.Series([0.0] * initial_rows, index=data.index)
            send_progress("   NaN percentages calculated.", 50)

            send_progress("✂️ Filtering compounds based on threshold...", 60)
            filtered_data = data[nan_percentage <= threshold]
            final_rows = filtered_data.shape[0]
            removed_count = initial_rows - final_rows
            send_progress(f"   Filtering complete. Kept {final_rows} compounds, removed {removed_count}.", 75)

            send_progress("💾 Saving filtered data...", 80)
            output_file = os.path.join(output_dir, f"regression_compound_filtered_{initial_rows}_to_{final_rows}.csv")
            filtered_data.to_csv(output_file, index=False)
            send_progress(f"   Filtered data saved to: {output_file}", 90)

            send_progress("📝 Generating summary...", 95)
            text_container_content = create_text_container(
                "🔹 **High NaN Compound Removal Completed (Regression)!** 🔹",
                f"Input File: {os.path.basename(input_file)}",
                f"NaN Threshold: > {threshold*100:.0f}% per compound",
                f"Initial Compounds: {initial_rows}",
                f"Compounds Removed: {removed_count}",
                f"Remaining Compounds: {final_rows}",
                f"Output File: {output_file}"
            )
            send_progress("🎉 Compound filtering process finished.", 100)

            return {"ui": {"text": text_container_content},
                    "result": (str(output_file),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        

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
        send_progress("🚀 Starting High NaN Descriptor Removal (Regression)...", 0)
        output_dir = "QSAR/Descriptor_Preprocessing"
        output_file = ""
        initial_cols, final_cols = 0, 0

        try:
            os.makedirs(output_dir, exist_ok=True)
            send_progress(f"📂 Output directory checked/created: {output_dir}", 5)

            send_progress(f"⏳ Loading data from: {input_file}", 10)
            data = pd.read_csv(input_file)
            initial_rows, initial_cols = data.shape
            send_progress(f"   Data loaded ({initial_rows} rows, {initial_cols} columns).", 20)

            send_progress(f"📊 Calculating NaN percentage per descriptor (threshold = {threshold*100:.0f}%)...", 30)
            nan_percentage = data.isna().mean()
            send_progress("   NaN percentages calculated.", 50)

            send_progress("✂️ Filtering descriptors based on threshold...", 60)
            retained_columns_initial = nan_percentage[nan_percentage <= threshold].index.tolist()

            # Ensure SMILES and value are always kept if they exist
            retained_columns = retained_columns_initial.copy()
            if "SMILES" in data.columns and "SMILES" not in retained_columns:
                retained_columns.append("SMILES")
                send_progress("   Forcibly kept 'SMILES' column.", 65)
            if "value" in data.columns and "value" not in retained_columns:
                retained_columns.append("value")
                send_progress("   Forcibly kept 'value' column.", 70)

            # Reorder to keep SMILES first, value last if present
            ordered_cols = []
            if "SMILES" in retained_columns: ordered_cols.append("SMILES")
            ordered_cols.extend([col for col in retained_columns if col not in ["SMILES", "value"]])
            if "value" in retained_columns: ordered_cols.append("value")

            filtered_data = data[ordered_cols]
            final_cols = filtered_data.shape[1]
            removed_count = initial_cols - final_cols
            send_progress(f"   Filtering complete. Kept {final_cols} descriptors, removed {removed_count}.", 75)

            send_progress("💾 Saving filtered data...", 80)
            output_file = os.path.join(output_dir, f"regression_descriptor_filtered_{initial_cols}_to_{final_cols}.csv")
            filtered_data.to_csv(output_file, index=False)
            send_progress(f"   Filtered data saved to: {output_file}", 90)

            send_progress("📝 Generating summary...", 95)
            text_container_content = create_text_container(
                "🔹 **High NaN Descriptor Removal Completed (Regression)!** 🔹",
                f"Input File: {os.path.basename(input_file)}",
                f"NaN Threshold: > {threshold*100:.0f}% per descriptor",
                f"Initial Descriptors: {initial_cols}",
                f"Descriptors Removed: {removed_count}",
                f"Remaining Descriptors: {final_cols}",
                f"Output File: {output_file}"
            )
            send_progress("🎉 Descriptor filtering process finished.", 100)

            return {"ui": {"text": text_container_content},
                    "result": (str(output_file),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}


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
        send_progress("🚀 Starting Missing Value Imputation (Regression)...", 0)
        output_dir = "QSAR/Descriptor_Preprocessing"
        output_file = ""
        imputed_cols_count = 0

        try:
            os.makedirs(output_dir, exist_ok=True)
            send_progress(f"📂 Output directory checked/created: {output_dir}", 5)

            send_progress(f"⏳ Loading data from: {input_file}", 10)
            data = pd.read_csv(input_file)
            initial_rows, initial_cols = data.shape
            send_progress(f"   Data loaded ({initial_rows} rows, {initial_cols} columns).", 20)

            # Drop 'Name' column if it exists (often redundant with SMILES)
            if "Name" in data.columns:
                 send_progress("   Dropping 'Name' column.", 25)
                 data = data.drop(columns = ["Name"])
                 initial_cols = data.shape[1] # Update initial_cols

            # Separate non-descriptor columns
            send_progress("📊 Separating non-descriptor columns (SMILES, value)...", 30)
            critical_cols = []
            if "SMILES" in data.columns: critical_cols.append("SMILES")
            if "value" in data.columns: critical_cols.append("value")

            critical_data = data[critical_cols].copy() if critical_cols else pd.DataFrame()
            descriptors = data.drop(columns=critical_cols, errors='ignore')
            send_progress(f"   Identified {descriptors.shape[1]} descriptor columns.", 35)

            if descriptors.empty:
                send_progress("⚠️ No descriptor columns found to impute. Saving data as is.", 40)
                final_data = critical_data # If only critical cols existed
            else:
                send_progress(f"⚙️ Applying '{method}' imputation to descriptor columns...", 40)
                imputer = SimpleImputer(strategy = method)
                # Check for NaN before imputation
                nan_before = descriptors.isnull().sum().sum()
                send_progress(f"   Total NaN values before imputation: {nan_before}", 45)

                imputed_descriptors_array = imputer.fit_transform(descriptors)
                imputed_descriptors = pd.DataFrame(imputed_descriptors_array, columns = descriptors.columns, index=descriptors.index)

                # Check for NaN after imputation (should be 0)
                nan_after = imputed_descriptors.isnull().sum().sum()
                send_progress(f"   Total NaN values after imputation: {nan_after}", 60)
                if nan_after > 0:
                     send_progress("   ⚠️ Warning: NaN values remain after imputation. Check input data or imputation strategy.", 65)

                imputed_cols_count = descriptors.shape[1]

                send_progress("🔄 Merging imputed descriptors with non-descriptor columns...", 70)
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

                send_progress("   Merging complete.", 75)

            send_progress("💾 Saving imputed data...", 80)
            output_file = os.path.join(output_dir, f"regression_imputed_{method}.csv")
            final_data.to_csv(output_file, index = False)
            send_progress(f"   Imputed data saved to: {output_file}", 90)

            send_progress("📝 Generating summary...", 95)
            text_container_content = create_text_container(
                "🔹 **Missing Value Imputation Completed (Regression)!** 🔹",
                f"Input File: {os.path.basename(input_file)}",
                f"Imputation Method: {method}",
                f"Descriptor Columns Imputed: {imputed_cols_count}",
                f"Output File: {output_file}",
                f"Final Data Shape: {final_data.shape[0]} rows, {final_data.shape[1]} columns"
            )
            send_progress("🎉 Imputation process finished.", 100)

            return {"ui": {"text": text_container_content},
                    "result": (str(output_file),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}


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
        send_progress("🚀 Starting Integrated Descriptor Preprocessing (Regression)...", 0)
        output_dir = "QSAR/Descriptor_Preprocessing"
        output_file = ""
        inf_report_file = None
        original_shape, shape_after_inf, shape_after_compound, shape_after_descriptor = (0, 0), (0, 0), (0, 0), (0, 0)
        inf_columns_count, compound_removed_count, descriptor_removed_count = 0, 0, 0

        try:
            os.makedirs(output_dir, exist_ok=True)
            send_progress(f"📂 Output directory checked/created: {output_dir}", 2)

            send_progress(f"⏳ Loading data from: {input_file}", 5)
            data = pd.read_csv(input_file)
            original_shape = data.shape
            send_progress(f"   Data loaded ({original_shape[0]} rows, {original_shape[1]} columns).", 8)

            # --- Step 1: Replace infinite values with NaN ---
            send_progress("➡️ Step 1: Replacing Infinite Values with NaN...", 10)
            numeric_df = data.select_dtypes(include=[np.number])
            inf_columns = numeric_df.columns[numeric_df.isin([np.inf, -np.inf]).any(axis=0)].tolist()
            inf_columns_count = len(inf_columns)
            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            shape_after_inf = data.shape
            send_progress(f"   Replaced Inf in {inf_columns_count} columns. Shape: {shape_after_inf}", 20)

            if inf_columns_count > 0:
                inf_counts = numeric_df[inf_columns].isin([np.inf, -np.inf]).sum().reset_index()
                inf_counts.columns = ["Feature", "Inf_Count"]
                inf_report_file = os.path.join(output_dir, "regression_integrated_inf_report.csv")
                inf_counts.to_csv(inf_report_file, index=False)
                send_progress(f"   Inf report saved: {inf_report_file}", 25)

            # --- Step 2: Remove compounds with high NaN ratio ---
            send_progress(f"➡️ Step 2: Removing Compounds (NaN% > {compounds_nan_threshold*100:.0f}%)...", 30)
            nan_counts = data.isna().sum(axis=1)
            total_columns_s2 = data.shape[1]
            nan_percentage_s2 = nan_counts / total_columns_s2 if total_columns_s2 > 0 else pd.Series([0.0] * data.shape[0], index=data.index)
            filtered_data_s2 = data[nan_percentage_s2 <= compounds_nan_threshold]
            shape_after_compound = filtered_data_s2.shape
            compound_removed_count = shape_after_inf[0] - shape_after_compound[0]
            data = filtered_data_s2 # Update data
            send_progress(f"   Removed {compound_removed_count} compounds. Shape: {shape_after_compound}", 45)

            # --- Step 3: Remove descriptors with high NaN ratio ---
            send_progress(f"➡️ Step 3: Removing Descriptors (NaN% > {descriptors_nan_threshold*100:.0f}%)...", 50)
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
            send_progress(f"   Removed {descriptor_removed_count} descriptors. Shape: {shape_after_descriptor}", 65)


            # --- Step 4: Impute remaining missing values ---
            send_progress(f"➡️ Step 4: Imputing Remaining Missing Values (Method: {imputation_method})...", 70)
            # Separate non-descriptor columns for imputation
            critical_cols_s4 = []
            if "SMILES" in data.columns: critical_cols_s4.append("SMILES")
            if "value" in data.columns: critical_cols_s4.append("value")
            # Check for Name column and add if present, as it might be non-numeric
            if "Name" in data.columns: critical_cols_s4.append("Name")

            critical_data_s4 = data[critical_cols_s4].copy() if critical_cols_s4 else pd.DataFrame()
            descriptors_s4 = data.drop(columns=critical_cols_s4, errors='ignore')

            if descriptors_s4.empty:
                 send_progress("   No descriptor columns left to impute.", 75)
                 final_data = critical_data_s4 # Only critical columns remained
            else:
                 nan_before_s4 = descriptors_s4.isnull().sum().sum()
                 send_progress(f"   NaN values before imputation: {nan_before_s4}", 75)
                 if nan_before_s4 == 0:
                      send_progress("   No NaNs found, skipping imputation logic.", 80)
                      final_data = data # Data is already clean
                 else:
                      imputer_s4 = SimpleImputer(strategy=imputation_method)
                      imputed_descriptors_array_s4 = imputer_s4.fit_transform(descriptors_s4)
                      imputed_descriptors_s4 = pd.DataFrame(imputed_descriptors_array_s4, columns = descriptors_s4.columns, index=descriptors_s4.index)
                      nan_after_s4 = imputed_descriptors_s4.isnull().sum().sum()
                      send_progress(f"   Imputation applied. NaN values after: {nan_after_s4}", 80)

                      # Recombine
                      if not critical_data_s4.empty:
                          final_data = pd.concat([critical_data_s4.reset_index(drop=True), imputed_descriptors_s4.reset_index(drop=True)], axis = 1)
                      else: # Only descriptors existed
                          final_data = imputed_descriptors_s4

                      # Ensure correct column order
                      final_cols_order = []
                      if "SMILES" in critical_cols_s4: final_cols_order.append("SMILES")
                      if "Name" in critical_cols_s4: final_cols_order.append("Name") # Keep Name if it existed
                      final_cols_order.extend(imputed_descriptors_s4.columns)
                      if "value" in critical_cols_s4: final_cols_order.append("value")
                      final_data = final_data[final_cols_order]

                      send_progress("   Imputation complete.", 85)

            # --- Save final preprocessed data ---
            final_shape = final_data.shape
            send_progress("💾 Saving final preprocessed data...", 90)
            output_file = os.path.join(output_dir, f"regression_integrated_preprocessed_{original_shape[0]}x{original_shape[1]}_to_{final_shape[0]}x{final_shape[1]}.csv")
            final_data.to_csv(output_file, index=False)
            send_progress(f"   Preprocessed data saved to: {output_file}", 94)

            # --- Generate final summary ---
            send_progress("📝 Generating final summary...", 95)
            text_container_content = create_text_container(
                "🔹 **Integrated Preprocessing Completed (Regression)!** 🔹",
                f"Input File: {os.path.basename(input_file)} ({original_shape[0]}x{original_shape[1]})",
                "--- Processing Steps ---",
                f"1. Replace Inf: {inf_columns_count} columns affected. {'Report: ' + inf_report_file if inf_report_file else ''}",
                f"2. Remove Compounds (> {compounds_nan_threshold*100:.0f}% NaN): Removed {compound_removed_count} compounds.",
                f"3. Remove Descriptors (> {descriptors_nan_threshold*100:.0f}% NaN): Removed {descriptor_removed_count} descriptors.",
                f"4. Impute Missing Values: Method '{imputation_method}'.",
                "--- Final Output ---",
                f"Final Data Shape: {final_shape[0]} rows, {final_shape[1]} columns",
                f"Output File: {output_file}"
            )
            send_progress("🎉 Integrated preprocessing finished successfully!", 100)

            return {"ui": {"text": text_container_content},
                    "result": (str(output_file),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
    

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