import pandas as pd
import os
from rdkit import Chem
# from ..utils.progress_utils import create_text_container # Now imported below

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

class Data_Loader_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "smiles_file_path": ("STRING", {"default": "smiles.tsv"}), # Added default examples
                "biological_value_file_path": ("STRING", {"default": "values.tsv"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("COMBINED_DATA_PATH",) # Updated name
    FUNCTION = "load_data"
    CATEGORY = "QSAR/REGRESSION/LOAD & STANDARDIZATION"
    OUTPUT_NODE = True

    def load_data(self, smiles_file_path, biological_value_file_path):
        send_progress("🚀 Starting Regression Data Loading & Merging...", 0)
        output_dir = "QSAR/Load_Data"
        combined_df_path = "" # Initialize path

        try:
            os.makedirs(output_dir, exist_ok=True)
            send_progress(f"📂 Output directory checked/created: {output_dir}", 5)

            # Load SMILES file
            send_progress(f"⏳ Loading SMILES data from: {smiles_file_path}", 10)
            if not os.path.exists(smiles_file_path):
                raise FileNotFoundError(f"SMILES file not found: {smiles_file_path}")
            # Assuming tab-separated, no header
            smiles_df = pd.read_csv(smiles_file_path, sep="\t", header=None, names=["SMILES"])
            send_progress(f"   Loaded {len(smiles_df)} SMILES records.", 25)

            # Load Value file
            send_progress(f"⏳ Loading biological value data from: {biological_value_file_path}", 30)
            if not os.path.exists(biological_value_file_path):
                raise FileNotFoundError(f"Biological value file not found: {biological_value_file_path}")
            # Assuming tab-separated, no header
            value_df = pd.read_csv(biological_value_file_path, sep="\t", header=None, names=["value"])
            send_progress(f"   Loaded {len(value_df)} value records.", 45)

            # Check row count match
            send_progress("⚖️ Checking row count consistency...", 50)
            if len(smiles_df) != len(value_df):
                raise ValueError(f"Mismatched row count between SMILES ({len(smiles_df)}) and values ({len(value_df)})!")
            send_progress("   Row counts match.", 60)

            # Combine data
            send_progress("⚙️ Merging SMILES and value data...", 65)
            combined_df = pd.concat([smiles_df, value_df], axis=1)
            send_progress("   Data merged.", 75)

            # Save combined data
            send_progress("💾 Saving combined data...", 80)
            combined_df_path = os.path.join(output_dir, "regression_combined_input_data.csv")
            combined_df.to_csv(combined_df_path, index=False)
            send_progress(f"   Combined data saved to: {combined_df_path}", 85)

            # Generate summary
            send_progress("📝 Generating summary...", 95)
            text_container_content = create_text_container(
                "🔹 **Regression Data Loading & Merging Completed!** 🔹",
                f"SMILES File: {os.path.basename(smiles_file_path)} ({len(smiles_df)} records)",
                f"Values File: {os.path.basename(biological_value_file_path)} ({len(value_df)} records)",
                f"Total Merged Records: {len(combined_df)}",
                f"Output File: {combined_df_path}"
            )
            send_progress("🎉 Data loading and merging finished successfully!", 100)

            return {"ui": {"text": text_container_content},
                    "result": (str(combined_df_path),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}. Please check input file paths."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except ValueError as ve:
            error_msg = f"❌ Value Error: {str(ve)}."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred during data loading/merging: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}


class Standardization_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_data_path": ("STRING",), # Changed name for clarity
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STANDARDIZED_DATA_PATH",) # Updated name
    FUNCTION = "standardize_data"
    CATEGORY = "QSAR/REGRESSION/LOAD & STANDARDIZATION"
    OUTPUT_NODE = True

    def standardize_data(self, input_data_path):
        send_progress("🚀 Starting Regression Data Standardization...", 0)
        METAL_IONS = { # Define METAL_IONS inside the method
            'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
            'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Th', 'Pa', 'U'
        }
        output_dir = "QSAR/Standardization"
        filtered_data_path = ""

        try:
            os.makedirs(output_dir, exist_ok=True)
            send_progress(f"📂 Output directory checked/created: {output_dir}", 5)

            send_progress(f"⏳ Loading data from: {input_data_path}", 10)
            data = pd.read_csv(input_data_path)
            original_count = len(data)
            send_progress(f"   Data loaded ({original_count} records).", 15)

            send_progress("⚙️ Checking for required columns ('SMILES', 'value')...", 20)
            if "SMILES" not in data.columns or "value" not in data.columns:
                raise ValueError("Required columns 'SMILES' and/or 'value' not found in the input data!")
            send_progress("   Required columns found.", 25)

            # Internal function for filtering
            def filter_molecule(mol):
                if mol is None: return False
                atom_symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
                if not atom_symbols or atom_symbols.issubset(METAL_IONS): return False
                if len(Chem.GetMolFrags(mol)) > 1: return False
                return True

            send_progress("🧪 Standardizing molecules (checking validity, removing metals, fragments)...", 30)
            data["RDKit_Mol"] = data["SMILES"].apply(lambda x: Chem.MolFromSmiles(x) if pd.notna(x) else None)
            filtered_data = data[data["RDKit_Mol"].apply(filter_molecule)].copy() # Filter and copy
            filtered_data.drop(columns=["RDKit_Mol"], inplace=True)
            filtered_count = len(filtered_data)
            removed_count = original_count - filtered_count
            send_progress(f"   Standardization complete. Kept {filtered_count} records, removed {removed_count}.", 75)


            send_progress("💾 Saving standardized data...", 85)
            filtered_data_path = os.path.join(output_dir, f"regression_standardized_{original_count}_to_{filtered_count}.csv")
            filtered_data.to_csv(filtered_data_path, index=False)
            send_progress(f"   Standardized data saved to: {filtered_data_path}", 90)

            send_progress("📝 Generating summary...", 95)
            text_container_content = create_text_container(
                "🔹 **Regression Data Standardization Completed!** 🔹",
                f"Input File: {os.path.basename(input_data_path)} ({original_count} records)",
                f"Records Kept: {filtered_count}",
                f"Records Removed (Invalid SMILES, Metals, Fragments): {removed_count}",
                f"Output File: {filtered_data_path}"
            )
            send_progress("🎉 Standardization finished successfully!", 100)

            return {"ui": {"text": text_container_content},
                    "result": (str(filtered_data_path),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}. Please check input file path."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except ValueError as ve:
            error_msg = f"❌ Value Error: {str(ve)}."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred during standardization: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}


class Load_and_Standardize_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "smiles_file_path": ("STRING", {"default": "smiles.tsv"}),
                "biological_value_file_path": ("STRING", {"default": "values.tsv"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STANDARDIZED_DATA_PATH",) # Updated name
    FUNCTION = "load_and_standardize_data"
    CATEGORY = "QSAR/REGRESSION/LOAD & STANDARDIZATION"
    OUTPUT_NODE = True

    def load_and_standardize_data(self, smiles_file_path, biological_value_file_path):
        send_progress("🚀 Starting Regression Load & Standardization...", 0)
        METAL_IONS = { # Define METAL_IONS inside the method
            'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
            'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Th', 'Pa', 'U'
        }
        output_dir = "QSAR/Load_and_Standardize"
        filtered_data_path = ""
        original_count, filtered_count = 0, 0

        try:
            os.makedirs(output_dir, exist_ok=True)
            send_progress(f"📂 Output directory checked/created: {output_dir}", 5)

            # --- Load Data Step (from Data_Loader_Regression) ---
            send_progress(f"⏳ Loading SMILES data from: {smiles_file_path}", 10)
            if not os.path.exists(smiles_file_path): raise FileNotFoundError(f"SMILES file not found: {smiles_file_path}")
            smiles_df = pd.read_csv(smiles_file_path, sep="\t", header=None, names=["SMILES"])
            send_progress(f"   Loaded {len(smiles_df)} SMILES records.", 15)

            send_progress(f"⏳ Loading biological value data from: {biological_value_file_path}", 20)
            if not os.path.exists(biological_value_file_path): raise FileNotFoundError(f"Value file not found: {biological_value_file_path}")
            value_df = pd.read_csv(biological_value_file_path, sep="\t", header=None, names=["value"])
            send_progress(f"   Loaded {len(value_df)} value records.", 25)

            send_progress("⚖️ Checking row count consistency...", 30)
            if len(smiles_df) != len(value_df): raise ValueError(f"Mismatched rows: SMILES ({len(smiles_df)}) vs values ({len(value_df)})!")
            send_progress("   Row counts match.", 35)

            send_progress("⚙️ Merging SMILES and value data...", 40)
            combined_df = pd.concat([smiles_df, value_df], axis=1)
            original_count = len(combined_df)
            send_progress(f"   Data merged ({original_count} records).", 45)
            # --- End Load Data Step ---

            # --- Standardization Step (from Standardization_Regression) ---
            send_progress("⚙️ Checking for required columns ('SMILES', 'value')...", 50)
            if "SMILES" not in combined_df.columns or "value" not in combined_df.columns:
                 raise ValueError("Internal Error: Required columns 'SMILES'/'value' missing after merge!")
            send_progress("   Required columns present.", 55)

            def filter_molecule(mol): # Internal function
                if mol is None: return False
                atom_symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
                if not atom_symbols or atom_symbols.issubset(METAL_IONS): return False
                if len(Chem.GetMolFrags(mol)) > 1: return False
                return True

            send_progress("🧪 Standardizing molecules...", 60)
            combined_df["RDKit_Mol"] = combined_df["SMILES"].apply(lambda x: Chem.MolFromSmiles(x) if pd.notna(x) else None)
            filtered_data = combined_df[combined_df["RDKit_Mol"].apply(filter_molecule)].copy()
            filtered_data.drop(columns=["RDKit_Mol"], inplace=True)
            filtered_count = len(filtered_data)
            removed_count = original_count - filtered_count
            send_progress(f"   Standardization complete. Kept {filtered_count} records, removed {removed_count}.", 85)
            # --- End Standardization Step ---

            # --- Save Final Data ---
            send_progress("💾 Saving standardized data...", 90)
            filtered_data_path = os.path.join(output_dir, f"regression_loaded_standardized_{original_count}_to_{filtered_count}.csv")
            filtered_data.to_csv(filtered_data_path, index=False)
            send_progress(f"   Standardized data saved to: {filtered_data_path}", 94)

            # --- Generate Summary ---
            send_progress("📝 Generating summary...", 95)
            text_container_content = create_text_container(
                "🔹 **Regression Load & Standardization Completed!** 🔹",
                f"Input SMILES: {os.path.basename(smiles_file_path)}",
                f"Input Values: {os.path.basename(biological_value_file_path)}",
                f"Original Records: {original_count}",
                f"Records Kept After Standardization: {filtered_count}",
                f"Records Removed: {removed_count}",
                f"Output File: {filtered_data_path}"
            )
            send_progress("🎉 Load & Standardization finished successfully!", 100)

            return {"ui": {"text": text_container_content},
                    "result": (str(filtered_data_path),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except ValueError as ve:
            error_msg = f"❌ Value Error: {str(ve)}."
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            send_progress(error_msg)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}


NODE_CLASS_MAPPINGS = {
    "Data_Loader_Regression": Data_Loader_Regression,
    "Standardization_Regression": Standardization_Regression,
    "Load_and_Standardize_Regression": Load_and_Standardize_Regression,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Data_Loader_Regression": "Data Loader (Regression)", # Updated
    "Standardization_Regression": "Standardization (Regression)", # Updated
    "Load_and_Standardize_Regression": "Load & Standardization (Regression)", # Updated
}