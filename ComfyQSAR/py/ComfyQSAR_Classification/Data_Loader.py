import os
import pandas as pd
from rdkit import Chem
# from rdkit.Chem import SDWriter # SDWriter 는 Load_and_Standardize, Standardization 에서만 사용
# from ..utils.progress_utils import create_text_container # 이제 progress_utils에서 가져옴

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
    is_intermediate_update = True # 중간 업데이트 여부 플래그

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

class Data_Loader_Classification:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_file_path": ("STRING", 
                                       {"default": "",
                                        "placeholder": "positive.sdf, .csv or .smi",
                                        "tooltip": "Path to the positive file"}), # Added default examples
                "negative_file_path": ("STRING", 
                                       {"default": "",
                                        "placeholder": "negative.sdf, .csv or .smi",
                                        "tooltip": "Path to the negative file"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("POSITIVE_PATH", "NEGATIVE_PATH",)
    FUNCTION = "load_data"
    CATEGORY = "QSAR/CLASSIFICATION/LOAD &STANDARDIZATION"
    OUTPUT_NODE = True

    def load_data(self, positive_file_path, negative_file_path):
        node_id = os.environ.get("NODE_ID")
        send_progress("🚀 Starting Data Loading...", 0, node_id)
        output_dir = "QSAR/Load_Data" # Keep track of where outputs *might* go if processed later
        try:
            os.makedirs(output_dir, exist_ok=True)
            send_progress(f"📂 Output directory checked/created: {output_dir}", 5, node_id)

            # Check positive file
            send_progress(f"⏳ Checking positive file: {positive_file_path}", 10, node_id)
            if not os.path.exists(positive_file_path):
                raise FileNotFoundError(f"Positive file not found: {positive_file_path}")
            if not (positive_file_path.endswith('.smi') or positive_file_path.endswith('.csv') or positive_file_path.endswith('.sdf')):
                raise ValueError("Unsupported positive file format. Use .smi, .csv, or .sdf.")
            send_progress("   Positive file format OK.", 15, node_id)

            # Check negative file
            send_progress(f"⏳ Checking negative file: {negative_file_path}", 20, node_id)
            if not os.path.exists(negative_file_path):
                raise FileNotFoundError(f"Negative file not found: {negative_file_path}")
            if not (negative_file_path.endswith('.smi') or negative_file_path.endswith('.csv') or negative_file_path.endswith('.sdf')):
                raise ValueError("Unsupported negative file format. Use .smi, .csv, or .sdf.")
            send_progress("   Negative file format OK.", 25, node_id)

        except (FileNotFoundError, ValueError) as e:
            error_msg = f"❌ Error checking input files: {str(e)}"
            send_progress(error_msg, None, node_id)
            return {"ui": {"text": create_text_container(error_msg)}, "result": ("", "")}


        # Function to count molecules (encapsulated)
        def count_molecules(file_path, file_type_label):
            send_progress(f"⏳ Counting molecules in {file_type_label} file...", 30 if file_type_label == "positive" else 60, node_id)
            count = 0
            try:
                if file_path.endswith('.sdf'):
                    suppl = Chem.SDMolSupplier(file_path, removeHs=False, strictParsing=False)
                    count = sum(1 for mol in suppl if mol is not None)
                elif file_path.endswith('.smi'):
                    # Handle potential errors during read_csv
                    df = pd.read_csv(file_path, header=None)
                    count = len(df)
                elif file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    if "SMILES" not in df.columns:
                        raise ValueError(f"CSV file {file_path} must contain a 'SMILES' column")
                    count = len(df["SMILES"].dropna()) # Count non-NA SMILES
                send_progress(f"   Found {count} molecules in {file_type_label} file.", 50 if file_type_label == "positive" else 80, node_id)
                return count
            except Exception as e:
                raise ValueError(f"Error processing {file_type_label} file {file_path}: {str(e)}") from e

        try:
            # Count molecules
            pos_count = count_molecules(positive_file_path, "positive")
            neg_count = count_molecules(negative_file_path, "negative")
            total_count = pos_count + neg_count

            # Log message
            send_progress("📝 Generating summary...", 95, node_id)
            text_container_content = create_text_container(
                "🔹 Classification Data Loaded! 🔹",
                f"✅ Positive Compounds: {pos_count} (from {os.path.basename(positive_file_path)})",
                f"✅ Negative Compounds: {neg_count} (from {os.path.basename(negative_file_path)})",
                f"📊 Total Molecules Loaded: {total_count}",
            )
            send_progress("🎉 Data loading complete!", 100, node_id)

            return {
                "ui": {"text": text_container_content},
                "result": (str(positive_file_path), str(negative_file_path)) # Return original paths
            }

        except Exception as e:
            error_msg = f"❌ Error counting molecules: {str(e)}"
            send_progress(error_msg, None, node_id)
            return {"ui": {"text": create_text_container(error_msg)}, "result": ("", "")}


class Standardization_Classification:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_path": ("STRING", 
                                  {"default": "",
                                   "placeholder": "positive.sdf, .csv or .smi",
                                   "tooltip": "Path to the positive file"}),
                "negative_path": ("STRING", 
                                  {"default": "",
                                   "placeholder": "negative.sdf, .csv or .smi",
                                   "tooltip": "Path to the negative file"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("POSITIVE_STD_PATH", "NEGATIVE_STD_PATH",) # Renamed for clarity
    FUNCTION = "standardize_data"
    CATEGORY = "QSAR/CLASSIFICATION/LOAD & STANDARDIZATION"
    OUTPUT_NODE = True
    
    def standardize_data(self, positive_path, negative_path):
        node_id = os.environ.get("NODE_ID")
        send_progress("🚀 Starting Standardization...", 0, node_id)
        METAL_IONS = { # Define METAL_IONS inside the method or make it a class attribute
            'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
            'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Th', 'Pa', 'U'
        }
        output_dir = "QSAR/Standardization"
        try:
            os.makedirs(output_dir, exist_ok=True)
            send_progress(f"📂 Output directory created/checked: {output_dir}", 5, node_id)
        except Exception as e:
            error_msg = f"❌ Error creating output directory: {str(e)}"
            send_progress(error_msg, None, node_id)
            return {"ui": {"text": create_text_container(error_msg)}, "result": ("", "")}

        # Molecule filtering function (remains internal)
        def filter_molecule(mol):
            if mol is None: return False
            atom_symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
            if not atom_symbols or atom_symbols.issubset(METAL_IONS): return False # Handle empty or metal-only
            if len(Chem.GetMolFrags(mol)) > 1: return False
            return True

        # File processing function (now handles progress and errors better)
        def process_file(file_path, output_name, progress_start, progress_end):
            send_progress(f"⏳ Standardizing {output_name} file: {os.path.basename(file_path)}...", progress_start, node_id)
            output_file = ""
            filtered_count = 0
            try:
                from rdkit.Chem import SDWriter # Import here to avoid global scope if not needed elsewhere

                if file_path.endswith('.sdf'):
                    suppl = Chem.SDMolSupplier(file_path, removeHs=True, strictParsing=False)
                    filtered_mols = [mol for mol in suppl if filter_molecule(mol)]
                    filtered_count = len(filtered_mols)
                    output_file = os.path.join(output_dir, f"{output_name}_standardized.sdf")
                    with SDWriter(output_file) as writer:
                        for mol in filtered_mols: writer.write(mol)

                elif file_path.endswith('.smi') or file_path.endswith('.csv'):
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(file_path)
                        if "SMILES" not in df.columns:
                            raise ValueError(f"CSV file {file_path} must contain a 'SMILES' column")
                        smiles_col = "SMILES"
                    else: # .smi file
                        df = pd.read_csv(file_path, header=None, names=["SMILES"], skip_blank_lines=True)
                        smiles_col = "SMILES"

                    df["RDKit_Mol"] = df[smiles_col].apply(lambda x: Chem.MolFromSmiles(x) if pd.notna(x) else None)
                    filtered_df = df[df["RDKit_Mol"].apply(filter_molecule)].copy() # Use .copy()
                    filtered_df.drop(columns=["RDKit_Mol"], inplace=True)
                    filtered_count = len(filtered_df)

                    output_file = os.path.join(output_dir, f"{output_name}_standardized.csv")
                    filtered_df.to_csv(output_file, index=False)
                else:
                    raise ValueError(f"Unsupported file format: {file_path}")

                send_progress(f"   Finished standardizing {output_name}. Kept {filtered_count} molecules.", progress_end, node_id)
                return output_file, filtered_count

            except Exception as e:
                error_msg = f"❌ Error standardizing {output_name} file ({os.path.basename(file_path)}): {str(e)}"
                # Don't send progress here, raise the exception to be caught outside
                send_progress(f"❌ Error standardizing {output_name} file ({os.path.basename(file_path)}): {str(e)}", None, node_id)
                raise RuntimeError(error_msg) from e


        try:
            # Process files
            positive_output, pos_filtered_count = process_file(positive_path, "positive", 10, 50)
            negative_output, neg_filtered_count = process_file(negative_path, "negative", 55, 90)

            # Log message
            send_progress("📝 Generating summary...", 95, node_id)
            text_container_content = create_text_container(
                "🔹 Standardization Completed! 🔹",
                f"✅ Positive Molecules Standardized: {pos_filtered_count} (saved to {os.path.basename(positive_output)})",
                f"✅ Negative Molecules Standardized: {neg_filtered_count} (saved to {os.path.basename(negative_output)})",
                f"📊 Total Molecules After Standardization: {pos_filtered_count + neg_filtered_count}",
            )
            send_progress("🎉 Standardization finished successfully!", 100, node_id)

            return {
                "ui": {"text": text_container_content},
                "result": (str(positive_output), str(negative_output))
            }

        except Exception as e:
            # Error already contains details from process_file if it failed there
            error_msg = str(e)
            send_progress(error_msg, None, node_id)
            return {"ui": {"text": create_text_container(error_msg)}, "result": ("", "")}


class Load_and_Standardize_Classification:
    @classmethod
    def INPUT_TYPES(cls):
         return {
            "required": {
                "positive_file_path": ("STRING", 
                                       {"default": "",
                                        "placeholder": "positive.sdf .csv or .smi",
                                        "tooltip": "Path to the positive file"}),
                "negative_file_path": ("STRING", 
                                       {"default": "",
                                        "placeholder": "negative.sdf .csv or .smi",
                                        "tooltip": "Path to the negative file"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("POSITIVE_STD_PATH", "NEGATIVE_STD_PATH",) # Renamed for clarity
    FUNCTION = "load_and_standardize"
    CATEGORY = "QSAR/CLASSIFICATION/LOAD & STANDARDIZATION"
    OUTPUT_NODE = True

    def load_and_standardize(self, positive_file_path, negative_file_path):
        node_id = os.environ.get("NODE_ID")
        send_progress("🚀 Starting Load & Standardization...", 0, node_id)
        METAL_IONS = { # Define METAL_IONS inside the method or make it a class attribute
            'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
            'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Th', 'Pa', 'U'
        }
        output_dir = "QSAR/Load_and_Standardize" # Different output dir
        try:
            os.makedirs(output_dir, exist_ok=True)
            send_progress(f"📂 Output directory created/checked: {output_dir}", 5, node_id)

            # --- File Checks (Copied from Data_Loader) ---
            send_progress(f"⏳ Checking positive file: {positive_file_path}", 6, node_id)
            if not os.path.exists(positive_file_path):
                raise FileNotFoundError(f"Positive file not found: {positive_file_path}")
            if not (positive_file_path.endswith('.smi') or positive_file_path.endswith('.csv') or positive_file_path.endswith('.sdf')):
                raise ValueError("Unsupported positive file format. Use .smi, .csv, or .sdf.")
            send_progress("   Positive file format OK.", 7, node_id)

            send_progress(f"⏳ Checking negative file: {negative_file_path}", 8, node_id)
            if not os.path.exists(negative_file_path):
                raise FileNotFoundError(f"Negative file not found: {negative_file_path}")
            if not (negative_file_path.endswith('.smi') or negative_file_path.endswith('.csv') or negative_file_path.endswith('.sdf')):
                raise ValueError("Unsupported negative file format. Use .smi, .csv, or .sdf.")
            send_progress("   Negative file format OK.", 10, node_id)
            # --- End File Checks ---

        except (FileNotFoundError, ValueError) as e:
            error_msg = f"❌ Error checking input files: {str(e)}"
            send_progress(error_msg, None, node_id)
            return {"ui": {"text": create_text_container(error_msg)}, "result": ("", "")}

        # Molecule filtering function (internal)
        def filter_molecule(mol):
            if mol is None: return False
            atom_symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
            if not atom_symbols or atom_symbols.issubset(METAL_IONS): return False # Handle empty or metal-only
            if len(Chem.GetMolFrags(mol)) > 1: return False
            return True

        # File processing function (modified for Load & Standardize)
        def process_file(file_path, output_name, progress_start, progress_end):
            send_progress(f"⏳ Loading & Standardizing {output_name} file: {os.path.basename(file_path)}...", progress_start, node_id)
            output_file = ""
            original_count = 0
            filtered_count = 0
            try:
                from rdkit.Chem import SDWriter # Import here

                if file_path.endswith('.sdf'):
                    suppl = Chem.SDMolSupplier(file_path, removeHs=True, strictParsing=False)
                    valid_molecules = []
                    # Iterate once to count originals and filter
                    for mol in suppl:
                         original_count += 1
                         if filter_molecule(mol):
                             valid_molecules.append(mol)
                    filtered_count = len(valid_molecules)
                    output_file = os.path.join(output_dir, f"{output_name}_standardized.sdf")
                    with SDWriter(output_file) as writer:
                        for mol in valid_molecules: writer.write(mol)

                elif file_path.endswith('.smi') or file_path.endswith('.csv'):
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(file_path)
                        if "SMILES" not in df.columns:
                            raise ValueError(f"CSV file {file_path} must contain a 'SMILES' column")
                        smiles_col = "SMILES"
                    else: # .smi
                        df = pd.read_csv(file_path, header=None, names=["SMILES"], skip_blank_lines=True)
                        smiles_col = "SMILES"

                    original_count = len(df)
                    df["RDKit_Mol"] = df[smiles_col].apply(lambda x: Chem.MolFromSmiles(x) if pd.notna(x) else None)
                    filtered_df = df[df["RDKit_Mol"].apply(filter_molecule)].copy()
                    filtered_df.drop(columns=["RDKit_Mol"], inplace=True)
                    filtered_count = len(filtered_df)

                    output_file = os.path.join(output_dir, f"{output_name}_standardized.csv")
                    filtered_df.to_csv(output_file, index=False)
                else:
                    raise ValueError(f"Unsupported file format: {file_path}")

                send_progress(f"   Finished {output_name}. Original: {original_count}, Kept: {filtered_count}.", progress_end, node_id)
                return output_file, original_count, filtered_count

            except Exception as e:
                error_msg = f"❌ Error processing {output_name} file ({os.path.basename(file_path)}): {str(e)}"
                send_progress(f"❌ Error processing {output_name} file ({os.path.basename(file_path)}): {str(e)}", None, node_id)
                raise RuntimeError(error_msg) from e

        try:
            # Process files
            positive_output, pos_orig_count, pos_filtered_count = process_file(positive_file_path, "positive", 15, 50)
            negative_output, neg_orig_count, neg_filtered_count = process_file(negative_file_path, "negative", 55, 90)

            # Log message
            send_progress("📝 Generating summary...", 95, node_id)
            text_container_content = create_text_container(
                "🔹 Load & Standardization Completed! 🔹",
                f"📊 Original Counts:",
                f"  - Positive: {pos_orig_count}",
                f"  - Negative: {neg_orig_count}",
                f"  - Total: {pos_orig_count + neg_orig_count}",
                f"📊 Counts After Standardization:",
                f"  - Positive: {pos_filtered_count} (saved to {os.path.basename(positive_output)})",
                f"  - Negative: {neg_filtered_count} (saved to {os.path.basename(negative_output)})",
                f"  - Total Kept: {pos_filtered_count + neg_filtered_count}",
            )
            send_progress("🎉 Load & Standardization finished successfully!", 100, node_id)

            return {
                "ui": {"text": text_container_content},
                "result": (str(positive_output), str(negative_output))
            }
        except Exception as e:
            error_msg = str(e) # Already formatted error message
            send_progress(error_msg, None, node_id)
            return {"ui": {"text": create_text_container(error_msg)}, "result": ("", "")}


NODE_CLASS_MAPPINGS = {
    "Data_Loader_Classification": Data_Loader_Classification,
    "Standardization_Classification": Standardization_Classification,
    "Load_and_Standardize_Classification": Load_and_Standardize_Classification,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Data_Loader_Classification": "Data Loader (Classification)",             # Updated display name
    "Standardization_Classification": "Standardization (Classification)",       # Updated display name
    "Load_and_Standardize_Classification": "Load & Standardization (Classification)", # Updated display name
}