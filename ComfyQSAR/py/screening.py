import os
import joblib
import pandas as pd
import numpy as np
import folder_paths # Import ComfyUI folder paths
from sklearn.impute import SimpleImputer

# Attempt to import RDKit, provide helpful error message if missing
try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("[ComfyQSAR Screening] Warning: RDKit is not installed. 'Extract Screened Molecules' node will not be available.")
    print("[ComfyQSAR Screening] Please install RDKit (e.g., 'pip install rdkit') to enable this node.")

# --- PadelPy Import (Not strictly needed for these nodes, but keep for consistency if Custom_Screening is added later) ---
try:
    from padelpy import padeldescriptor
    PADELPY_AVAILABLE = True
except ImportError:
    PADELPY_AVAILABLE = False
    # print("[ComfyQSAR Screening] Warning: padelpy library is not installed. Custom Screening node (if added) will be disabled.")

# --- Common Utility Import ---
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

# --- Screening Databases Definition (Using Tuple format for CSV & SDF) ---
# Define relative paths for CSV descriptor files and SDF molecule files
# These paths should be relative to the 'db_directory' input provided by the user.
SCREENING_DATABASES = {
    # DB_Name: (relative_csv_path, relative_sdf_path)
    "ASINEX": ("ASINEX/Des_ASINEX_10177.csv", "ASINEX/ASINEX_10177.sdf"),
    "IBS_NP": ("IBS_NP/Des_IBS_NP_3678.csv", "IBS_NP/IBS_NP_3678.sdf"),
    "IBS_SP1": ("IBS_SP1/Des_IBS_SP1_5629.csv", "IBS_SP1/IBS_SP1_5629.sdf"),
    "IBS_SP2": ("IBS_SP2/Des_IBS_SP2_3424.csv", "IBS_SP2/IBS_SP2_3424.sdf"),
    "IBS_SP3": ("IBS_SP3/Des_IBS_SP3_9690.csv", "IBS_SP3/IBS_SP3_9690.sdf"),
    "NCI": ("NCI/Des_NCI_10283.csv", "NCI/NCI_10283.sdf"),
    "ZINC_NP": ("ZINC_NP/Des_ZINC_NP_9644.csv", "ZINC_NP/ZINC_NP_9644.sdf")
}

# Base output directory within ComfyUI's output folder
output_node_dir = os.path.join(folder_paths.get_output_directory(), "qsar_screening")
# Temp directory for intermediate files if needed
# temp_node_dir = os.path.join(folder_paths.get_temp_directory(), "qsar_screening_temp")

# --- Virtual Screening Node ---
class VirtualScreening:
    @classmethod
    def INPUT_TYPES(cls):
        # Get database names for the dropdown
        db_names = list(SCREENING_DATABASES.keys())
        return {
            "required": {
                "model_path": ("STRING", {"default": "ComfyUI/models/qsar_models/your_model.pkl", "description": "Path to the trained QSAR model (.pkl)"}),
                "descriptors_path": ("STRING", {"default": "ComfyUI/output/qsar_selection/selected_descriptors.txt", "description": "Path to the selected descriptors list (.txt)"}),
                "selected_db": (db_names, {"default": db_names[0] if db_names else "", "description": "Select the screening database"}),
                "db_directory": ("STRING", {"default": folder_paths.get_input_directory() + "/qsar_screening_db", "description": "Directory containing the screening database folders (ASINEX, IBS_NP, etc.)"}),
                "task_type": (["Regression", "Classification"], {"default": "Regression", "description": "Type of QSAR task the model performs"}),
                "threshold": ("FLOAT", {"default": 10.0, "step": 0.1, "description": "Threshold for selection (Regression: value <= threshold, Classification: prob >= threshold)"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("SELECTED_COMPOUNDS_CSV", "SELECTED_INDICES_TXT",)
    FUNCTION = "screen"
    CATEGORY = "QSAR/Screening"
    OUTPUT_NODE = True # Indicate this node produces outputs on the filesystem

    def screen(self, model_path, descriptors_path, selected_db, db_directory, task_type, threshold):
        node_id = os.environ.get("NODE_ID") # Get node ID for progress if available
        send_progress(f"🚀 Starting Virtual Screening (DB: {selected_db}, Task: {task_type})...", 0, node_id)
        selected_compounds_df_path = ""
        selected_indices_path = ""

        try:
            # Ensure output directory exists
            os.makedirs(output_node_dir, exist_ok=True)
            send_progress(f"📂 Output directory set: {output_node_dir}", 5, node_id)

            # --- Validate Inputs ---
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(descriptors_path):
                raise FileNotFoundError(f"Descriptors file not found: {descriptors_path}")
            if not os.path.isdir(db_directory):
                 raise FileNotFoundError(f"Database directory not found: {db_directory}")

            if selected_db not in SCREENING_DATABASES:
                raise ValueError(f"Invalid database name '{selected_db}'. Choose from: {list(SCREENING_DATABASES.keys())}")

            db_info = SCREENING_DATABASES[selected_db]
            screening_csv_rel_path = db_info[0] # Get CSV relative path
            screening_csv_abs_path = os.path.join(db_directory, screening_csv_rel_path)

            if not os.path.exists(screening_csv_abs_path):
                 raise FileNotFoundError(f"Screening DB CSV file not found: {screening_csv_abs_path}")
            send_progress("   Input paths validated.", 10, node_id)

            # --- Load Data and Model ---
            send_progress(f"⏳ Loading screening dataset: {os.path.basename(screening_csv_abs_path)}...", 15, node_id)
            screening_data = pd.read_csv(screening_csv_abs_path)
            # Store original indices if needed later (though screening_data.iloc works)
            original_indices = screening_data.index
            send_progress(f"   Screening data loaded ({screening_data.shape[0]} compounds, {screening_data.shape[1]} columns).", 25, node_id)

            send_progress(f"⏳ Loading QSAR model from {os.path.basename(model_path)}...", 30, node_id)
            model = joblib.load(model_path)
            send_progress("   Model loaded.", 35, node_id)

            send_progress(f"⏳ Loading selected descriptors from {os.path.basename(descriptors_path)}...", 40, node_id)
            with open(descriptors_path, "r") as f:
                selected_descriptors = [line.strip() for line in f if line.strip()]
            if not selected_descriptors:
                 raise ValueError("No descriptors found in the selected descriptors file.")
            send_progress(f"   Loaded {len(selected_descriptors)} selected descriptors.", 45, node_id)

            # --- Prepare Screening Data ---
            send_progress("⚙️ Preparing screening data (selecting features)...", 50, node_id)
            missing_descriptors = [desc for desc in selected_descriptors if desc not in screening_data.columns]
            if missing_descriptors:
                raise ValueError(f"Missing descriptors in screening dataset '{selected_db}': {missing_descriptors}")

            X_screen = screening_data[selected_descriptors]
            # Handle potential NaN/Inf in screening data - Use median imputation as a default
            X_screen = X_screen.replace([np.inf, -np.inf], np.nan)
            if X_screen.isnull().values.any():
                  send_progress("   Imputing NaN/Inf values in screening data (median)...", 55, node_id)
                  imputer = SimpleImputer(strategy='median')
                  X_screen_imputed = imputer.fit_transform(X_screen)
                  X_screen = pd.DataFrame(X_screen_imputed, columns=selected_descriptors, index=X_screen.index) # Keep original index
            send_progress("   Screening data prepared.", 60, node_id)


            # --- Perform Predictions ---
            send_progress("🤖 Performing predictions...", 65, node_id)
            predictions = None
            selected_df_indices = np.array([], dtype=int) # Indices relative to the DataFrame

            if task_type == "Classification":
                if hasattr(model, "predict_proba"):
                    predictions = model.predict_proba(X_screen)[:, 1] # Probability of class 1
                    selected_df_indices = np.where(predictions >= threshold)[0]
                    send_progress(f"   Classification predictions (probability) completed. Threshold >= {threshold}", 75, node_id)
                else: # Fallback to predict if predict_proba is not available
                    predictions = model.predict(X_screen)
                    # Assuming threshold refers to the positive class label (e.g., 1)
                    # Make sure threshold is comparable to prediction type
                    try: predicted_class = int(threshold)
                    except ValueError: raise ValueError("Threshold for classification 'predict' must be an integer class label (e.g., 1).")
                    selected_df_indices = np.where(predictions == predicted_class)[0]
                    send_progress(f"   Classification predictions (label) completed. Selected class = {predicted_class}", 75, node_id)

            elif task_type == "Regression":
                if hasattr(model, "predict"):
                    predictions = model.predict(X_screen)
                    selected_df_indices = np.where(predictions <= threshold)[0]
                    send_progress(f"   Regression predictions completed. Threshold <= {threshold}", 75, node_id)
                else:
                    raise AttributeError(f"The loaded model (type: {type(model)}) does not have a 'predict' method required for regression.")
            else:
                 raise ValueError(f"Invalid task_type: {task_type}. Choose 'Regression' or 'Classification'.")

            num_selected = len(selected_df_indices)
            send_progress(f"   Found {num_selected} compounds meeting the threshold criteria.", 80, node_id)

            # --- Save Results ---
            send_progress("💾 Saving results...", 85, node_id)

            # Get the original file indices corresponding to the selected DataFrame indices
            # Use the original index stored before potential imputation, or just screening_data.index if no imputation happened
            selected_original_indices = screening_data.index[selected_df_indices].to_numpy()

            # Save selected ORIGINAL indices (important for molecule extraction)
            # Make filename unique
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            indices_filename = f"Screening_{selected_db}_{task_type}_Indices_{num_selected}_{timestamp}.txt"
            selected_indices_path = os.path.join(output_node_dir, indices_filename)
            np.savetxt(selected_indices_path, selected_original_indices, fmt="%d")
            send_progress(f"   Selected original indices saved: {indices_filename}", 88, node_id)

            # Save selected compounds data + predictions
            if num_selected > 0:
                 # Use the DataFrame indices to select rows from the potentially imputed data
                 selected_compounds_df = screening_data.iloc[selected_df_indices].copy()
                 # Add predictions corresponding to the selected rows
                 selected_compounds_df["prediction_value"] = predictions[selected_df_indices]

                 # Attempt to add SMILES if present in the original data
                 if "SMILES" not in selected_compounds_df.columns and "SMILES" in screening_data.columns:
                      # Ensure we select SMILES corresponding to the *selected* rows
                      selected_compounds_df.insert(0, "SMILES", screening_data["SMILES"].iloc[selected_df_indices].values)
                 # Add original index if desired
                 # selected_compounds_df.insert(0, "Original_Index", selected_original_indices)

                 compounds_filename = f"Screening_{selected_db}_{task_type}_Compounds_{num_selected}_{timestamp}.csv"
                 selected_compounds_df_path = os.path.join(output_node_dir, compounds_filename)
                 selected_compounds_df.to_csv(selected_compounds_df_path, index=False)
                 send_progress(f"   Selected compounds data saved: {compounds_filename}", 92, node_id)
            else:
                 selected_compounds_df_path = "N/A (No compounds selected)"
                 send_progress("   No compounds selected, skipping saving compounds CSV.", 92, node_id)


            # --- Generate Summary ---
            send_progress("📝 Generating summary...", 95, node_id)
            summary_lines = [
                "**Virtual Screening Completed!**",
                f"Database: {selected_db}",
                f"Model: {os.path.basename(model_path)}",
                f"Task: {task_type}",
                f"Criteria: {'<=' if task_type=='Regression' else '>='} {threshold}",
                f"Compounds Screened: {screening_data.shape[0]}",
                f"Compounds Selected: {num_selected}",
                "--- Saved Files ---",
                f"Selected Compounds CSV: {os.path.basename(selected_compounds_df_path) if num_selected > 0 else 'N/A'}",
                f"Selected Indices TXT: {os.path.basename(selected_indices_path)}"
            ]
            text_container_content = create_text_container(*summary_lines)
            send_progress("🎉 Screening process finished successfully!", 100, node_id)

            # Return the absolute paths to the created files
            return {"ui": {"text": text_container_content},
                    "result": (str(selected_compounds_df_path), str(selected_indices_path),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File/Directory Not Found Error: {str(fnf_e)}."
            send_progress(error_msg, 100, node_id) # Indicate completion with error
            return {"ui": {"text": create_text_container(error_msg)}, "result": ("", "")}
        except ValueError as ve:
             error_msg = f"❌ Value Error: {str(ve)}"
             send_progress(error_msg, 100, node_id)
             return {"ui": {"text": create_text_container(error_msg)}, "result": ("", "")}
        except AttributeError as ae: # Handle predict/predict_proba errors
             error_msg = f"❌ Model Error: {str(ae)}"
             send_progress(error_msg, 100, node_id)
             return {"ui": {"text": create_text_container(error_msg)}, "result": ("", "")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred during screening: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            send_progress(error_msg, 100, node_id)
            return {"ui": {"text": create_text_container(error_msg)}, "result": ("", "")}


# --- Molecule Extraction Node ---
class ExtractScreenedMolecules:
    @classmethod
    def INPUT_TYPES(cls):
         if not RDKIT_AVAILABLE: # If RDKit not installed, return empty types to hide node
              return {"required": {}}
         db_names = list(SCREENING_DATABASES.keys())
         return {
            "required": {
                "selected_db": (db_names, {"default": db_names[0] if db_names else ""}),
                "db_directory": ("STRING", {"default": folder_paths.get_input_directory() + "/qsar_screening_db", "description": "Directory containing the screening database folders (ASINEX, IBS_NP, etc.)"}),
                "index_file": ("STRING", {"default": "ComfyUI/output/qsar_screening/Screening_Selected_Indices.txt", "description": "Path to the .txt file containing selected indices (one per line)"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("EXTRACTED_SDF_PATH",)
    FUNCTION = "extract"
    CATEGORY = "QSAR/Screening"
    OUTPUT_NODE = True

    def extract(self, selected_db, db_directory, index_file):
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is not installed. This node requires RDKit.")

        node_id = os.environ.get("NODE_ID")
        send_progress(f"🚀 Starting Molecule Extraction (DB: {selected_db})...", 0, node_id)
        output_sdf_path = ""
        num_extracted = 0
        num_requested = 0

        try:
            os.makedirs(output_node_dir, exist_ok=True)
            send_progress(f"📂 Output directory set: {output_node_dir}", 5, node_id)

            # --- Validate Inputs ---
            if not os.path.exists(index_file):
                raise FileNotFoundError(f"Index file not found: {index_file}")
            if not os.path.isdir(db_directory):
                 raise FileNotFoundError(f"Database directory not found: {db_directory}")

            if selected_db not in SCREENING_DATABASES:
                raise ValueError(f"Invalid database name '{selected_db}'. Choose from: {list(SCREENING_DATABASES.keys())}")

            db_info = SCREENING_DATABASES[selected_db]
            sdf_rel_path = db_info[1] # Get SDF relative path from tuple
            sdf_abs_path = os.path.join(db_directory, sdf_rel_path)

            if not os.path.exists(sdf_abs_path):
                 raise FileNotFoundError(f"Screening DB SDF file not found: {sdf_abs_path}")
            send_progress("   Input paths validated.", 15, node_id)

            # --- Load Indices ---
            send_progress(f"⏳ Loading selected indices from: {os.path.basename(index_file)}...", 20, node_id)
            try:
                 # Use np.loadtxt for potentially large index files
                 indices = np.loadtxt(index_file, dtype=int, ndmin=1) # Ensure it's always an array
                 num_requested = len(indices)
                 send_progress(f"   Loaded {num_requested} indices.", 30, node_id)
            except Exception as e:
                 raise ValueError(f"Error reading index file '{index_file}': {e}. Ensure it contains one integer index per line.")

            if num_requested == 0:
                 send_progress("   ⚠️ No indices found in the file. Nothing to extract.", 35, node_id)
                 output_sdf_path = "N/A (No indices provided)"
                 summary_lines = [
                     "**Molecule Extraction Skipped**",
                     "Reason: No indices found in the provided file.",
                     f"Index File: {os.path.basename(index_file)}",
                 ]
                 text_container_content = create_text_container(*summary_lines)
                 return {"ui": {"text": text_container_content}, "result": (output_sdf_path,)}


            # --- Extract Molecules ---
            send_progress(f"🧬 Extracting molecules from {os.path.basename(sdf_abs_path)}...", 40, node_id)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_filename = f"Extracted_{selected_db}_{num_requested}_molecules_{timestamp}.sdf"
            output_sdf_path = os.path.join(output_node_dir, output_filename)

            # Use ForwardSDMolSupplier for potentially faster iteration if file is huge, but regular is fine
            # Check if SDF file is valid and readable by RDKit early
            try:
                sdf_supplier = Chem.SDMolSupplier(sdf_abs_path)
                if not sdf_supplier: # Check if supplier initialization failed
                    raise ValueError(f"Could not open or parse SDF file: {sdf_abs_path}")
                sdf_len = len(sdf_supplier) # Get length for progress calculation
                if sdf_len == 0:
                    raise ValueError(f"SDF file is empty: {sdf_abs_path}")
            except Exception as e:
                 raise ValueError(f"Error opening/parsing SDF file '{sdf_abs_path}' with RDKit: {e}")

            sdf_writer = Chem.SDWriter(output_sdf_path)

            # Convert indices to a set for efficient lookup
            indices_set = set(indices)
            max_index_requested = max(indices_set) if indices_set else -1

            processed_count = 0
            # Update progress roughly every 5% or for smaller sets more frequently
            update_interval = max(1, sdf_len // 20)

            send_progress(f"   Iterating through {sdf_len} records in SDF...", 45, node_id)

            for current_idx, mol in enumerate(sdf_supplier):
                  # Optimization: Stop if we've processed beyond the highest requested index
                  if current_idx > max_index_requested and not indices_set:
                      send_progress(f"   Highest requested index {max_index_requested} processed. Stopping early.", 95, node_id)
                      break

                  if current_idx in indices_set:
                      if mol is not None:
                           sdf_writer.write(mol)
                           num_extracted += 1
                      else:
                           print(f"[ComfyQSAR Screening] Warning: Invalid molecule record at index {current_idx} in SDF.")
                      indices_set.remove(current_idx) # Remove processed index

                  processed_count += 1
                  if processed_count % update_interval == 0:
                       # Progress based on file records processed
                       progress = 45 + int(50 * (processed_count / sdf_len))
                       send_progress(f"   Processed {processed_count}/{sdf_len} records...", min(progress, 95), node_id)

                  # Optimization: Stop if all requested indices are found
                  if not indices_set:
                       send_progress(f"   All {num_requested} requested indices found and processed.", 95, node_id)
                       break

            sdf_writer.close()
            send_progress(f"   Extraction complete. Extracted {num_extracted} valid molecules.", 98, node_id)

            if indices_set: # Check if any requested indices were not found
                 print(f"[ComfyQSAR Screening] Warning: Could not find records for the following indices: {sorted(list(indices_set))}")


            # --- Generate Summary ---
            summary_lines = [
                "**Molecule Extraction Completed!**",
                f"Database: {selected_db}",
                f"Source SDF: {os.path.basename(sdf_abs_path)}",
                f"Index File: {os.path.basename(index_file)}",
                f"Indices Requested: {num_requested}",
                f"Molecules Extracted: {num_extracted}",
                f"Output SDF File: {os.path.basename(output_sdf_path)}"
            ]
            if indices_set:
                 summary_lines.append(f"Indices Not Found: {len(indices_set)}")

            text_container_content = create_text_container(*summary_lines)
            send_progress("🎉 Molecule extraction process finished.", 100, node_id)

            # Return the absolute path to the created SDF file
            return {"ui": {"text": text_container_content},
                    "result": (str(output_sdf_path),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File/Directory Not Found Error: {str(fnf_e)}."
            send_progress(error_msg, 100, node_id)
            return {"ui": {"text": create_text_container(error_msg)}, "result": ("",)}
        except ValueError as ve:
             error_msg = f"❌ Value Error: {str(ve)}"
             send_progress(error_msg, 100, node_id)
             return {"ui": {"text": create_text_container(error_msg)}, "result": ("",)}
        except ImportError as ie: # Should be caught by RDKIT_AVAILABLE check, but double check
             error_msg = f"❌ Import Error: {str(ie)}. RDKit is required for molecule extraction."
             send_progress(error_msg, 100, node_id)
             return {"ui": {"text": create_text_container(error_msg)}, "result": ("",)}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred during extraction: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            send_progress(error_msg, 100, node_id)
            return {"ui": {"text": create_text_container(error_msg)}, "result": ("",)}


# +++ Custom Screening Node +++
class CustomScreening:
    @classmethod
    def INPUT_TYPES(cls):
         # Check if PadelPy is available before defining inputs
         if not PADELPY_AVAILABLE:
              print("[ComfyQSAR Screening] padelpy not found, CustomScreening node disabled.")
              return {"required": {}} # Return empty if dependency missing
         return {
            "required": {
                "model_path": ("STRING", {"default": "ComfyUI/models/qsar_models/your_model.pkl", "description": "Path to the trained QSAR model (.pkl)"}),
                "descriptors_path": ("STRING", {"default": "ComfyUI/output/qsar_selection/selected_descriptors.txt", "description": "Path to the selected descriptors list (.txt)"}),
                "input_compound_file": ("STRING", {"default": folder_paths.get_input_directory() + "/compounds_to_screen.sdf", "description": "Path to the input compound file (SDF or CSV/SMI)"}),
                "input_type": (["SDF", "SMILES"], {"default": "SDF", "description": "Format of the input compound file"}),
                "task_type": (["Regression", "Classification"], {"default": "Regression", "description": "Type of QSAR task the model performs"}),
            },
            "optional": {
                 # Optional parameters matching notebook function flexibility
                 "smiles_column": ("STRING", {"default": "SMILES", "description": "Name of the SMILES column (if input_type is SMILES)"}),
                 "padel_xml_path": ("STRING", {"default": "", "description": "Optional: Path to custom PaDEL XML descriptor types file"}),
                 "padel_dir_path": ("STRING", {"default": "", "description": "Optional: Path to the directory containing PaDEL-Descriptor.jar"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    # Return paths to the predictions and the calculated descriptors
    RETURN_NAMES = ("PREDICTIONS_CSV", "CALCULATED_DESCRIPTORS_CSV",)
    FUNCTION = "screen_custom"
    CATEGORY = "QSAR/Screening"
    OUTPUT_NODE = True

    def screen_custom(self, model_path, descriptors_path, input_compound_file, input_type, task_type,
                         smiles_column="SMILES", padel_xml_path="", padel_dir_path=""):

        if not PADELPY_AVAILABLE:
             # This check is technically redundant due to INPUT_TYPES check, but good practice
             raise ImportError("padelpy library is not installed or PaDEL-Descriptor is not configured. This node requires them.")

        node_id = os.environ.get("NODE_ID")
        send_progress(f"🚀 Starting Custom Screening (Input: {input_type}, Task: {task_type})...", 0, node_id)
        predictions_csv_path = ""
        calculated_descriptors_path = "" # Path to the intermediate descriptor file

        try:
            # Ensure output directories exist
            os.makedirs(output_node_dir, exist_ok=True)
            # Also create the temp directory if it doesn't exist
            os.makedirs(temp_node_dir, exist_ok=True)
            send_progress(f"📂 Output directory: {output_node_dir}, Temp directory: {temp_node_dir}", 2, node_id)

            # Define intermediate file path within the temp directory
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            base_input_name = os.path.splitext(os.path.basename(input_compound_file))[0]
            desc_filename = f"CustomScreening_{base_input_name}_Descriptors_{timestamp}.csv"
            calculated_descriptors_path = os.path.join(temp_node_dir, desc_filename)

            # --- Validate Inputs --- (Basic checks)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(descriptors_path):
                raise FileNotFoundError(f"Descriptors file not found: {descriptors_path}")
            if not os.path.exists(input_compound_file):
                raise FileNotFoundError(f"Input compound file not found: {input_compound_file}")
            if padel_xml_path and not os.path.exists(padel_xml_path):
                raise FileNotFoundError(f"PaDEL XML file not found: {padel_xml_path}")
            if padel_dir_path and not os.path.isdir(padel_dir_path):
                # Try to be helpful if padelpy is installed but PaDEL dir isn't specified/found
                if 'PADEL_PY_PADEL_PATH' in os.environ:
                     print(f"[ComfyQSAR Screening] Info: Using PaDEL directory from environment variable PADEL_PY_PADEL_PATH: {os.environ['PADEL_PY_PADEL_PATH']}")
                else:
                     print(f"[ComfyQSAR Screening] Warning: PaDEL directory not specified or found: {padel_dir_path}. Relying on padelpy default setup.")
                 # Allow proceeding, padelpy might find it elsewhere
                 # raise FileNotFoundError(f"PaDEL directory not found: {padel_dir_path}")
            send_progress("   Input paths validated.", 5, node_id)

            # --- Load Model and Selected Descriptors ---
            send_progress(f"⏳ Loading QSAR model from {os.path.basename(model_path)}...", 10, node_id)
            model = joblib.load(model_path)
            send_progress("   Model loaded.", 15, node_id)
            send_progress(f"⏳ Loading selected descriptors from {os.path.basename(descriptors_path)}...", 20, node_id)
            with open(descriptors_path, "r") as f:
                selected_descriptors = [line.strip() for line in f if line.strip()]
            if not selected_descriptors:
                raise ValueError("No descriptors found in the selected descriptors file.")
            send_progress(f"   Loaded {len(selected_descriptors)} selected descriptors.", 25, node_id)

            # --- Calculate Descriptors using PaDEL ---
            send_progress(f"⚙️ Calculating descriptors for {os.path.basename(input_compound_file)} using PaDEL... (This may take time)", 30, node_id)
            # Configure PaDEL arguments
            padel_kwargs = {
                "d_file": calculated_descriptors_path,
                "d_2d": True, "d_3d": False, # Default to 2D, user can override with XML
                "detectaromaticity": True, "standardizenitro": True,
                "removesalt": True,
                "retainorder": True, "threads": -1, # Use all available threads
                "waitingjobs": -1, "log": False, "headless": True,
                # "maxruntime": 36000 # Example: Max 10 hours, adjust if needed
            }
            # Add optional PaDEL paths if provided
            if padel_xml_path: padel_kwargs['descriptortypes'] = padel_xml_path
            if padel_dir_path: padel_kwargs['padeldir'] = padel_dir_path # Allow user to specify PaDEL location

            padel_input_arg = {}
            temp_smiles_file = None # Keep track of temp file for cleanup

            if input_type.upper() == "SDF":
                padel_input_arg['mol_dir'] = input_compound_file
                # Use filename as mol name for SDF by default
                padel_kwargs['usefilenameasmolname'] = True
            elif input_type.upper() == "SMILES":
                 send_progress(f"   Preparing SMILES input (Column: '{smiles_column}')...", 32, node_id)
                 try:
                     # Assume CSV or similar delimited file for SMILES
                     # Try common delimiters
                     try: mols_df = pd.read_csv(input_compound_file)
                     except pd.errors.ParserError:
                         try: mols_df = pd.read_csv(input_compound_file, sep='\t')
                         except pd.errors.ParserError:
                              mols_df = pd.read_csv(input_compound_file, delim_whitespace=True)

                     if smiles_column not in mols_df.columns:
                          raise ValueError(f"SMILES column '{smiles_column}' not found in {input_compound_file}. Available columns: {list(mols_df.columns)}")
                     if mols_df[smiles_column].isnull().any():
                          raise ValueError(f"Missing values found in SMILES column '{smiles_column}'. Please clean the input file.")

                     # Create a temporary SMI file for PaDEL
                     temp_smiles_filename = f"CustomScreening_{base_input_name}_PadelInput_{timestamp}.smi"
                     temp_smiles_file = os.path.join(temp_node_dir, temp_smiles_filename)
                     # Save *only* the SMILES column, no header, required by PaDEL
                     mols_df[smiles_column].to_csv(temp_smiles_file, index=False, header=False)
                     padel_input_arg['mol_dir'] = temp_smiles_file
                     # Do not use filename as mol name for SMILES input
                     padel_kwargs['usefilenameasmolname'] = False
                 except Exception as e:
                     raise ValueError(f"Error processing SMILES file '{input_compound_file}': {e}")
            else:
                raise ValueError("Invalid input_type. Choose 'SDF' or 'SMILES'.")

            # --- Run PaDEL --- #
            try:
                 send_progress("   Running PaDEL-Descriptor...", 35, node_id)
                 # Ensure the target descriptor file does not exist before running PaDEL
                 if os.path.exists(calculated_descriptors_path):
                      os.remove(calculated_descriptors_path)
                 padeldescriptor(**padel_input_arg, **padel_kwargs)
                 send_progress("   PaDEL descriptor calculation finished.", 55, node_id)
            except Exception as e:
                 # Clean up temp file if created, even on error
                 if temp_smiles_file and os.path.exists(temp_smiles_file): os.remove(temp_smiles_file)
                 raise RuntimeError(f"PaDEL-Descriptor execution failed: {e}. Check PaDEL setup (Java installation, PADEL_PY_PADEL_PATH environment variable or padel_dir_path input) and input file format.")

            # Clean up temp smiles file if it was created
            if temp_smiles_file and os.path.exists(temp_smiles_file):
                 try: os.remove(temp_smiles_file)
                 except Exception as te: print(f"[ComfyQSAR Screening] Warning: Could not remove temp SMILES file {temp_smiles_file}: {te}")

            # Verify descriptor file creation
            if not os.path.exists(calculated_descriptors_path) or os.path.getsize(calculated_descriptors_path) == 0:
                 raise RuntimeError(f"PaDEL execution finished, but the descriptor output file ({calculated_descriptors_path}) was not created or is empty. Check PaDEL logs/output and input file compatibility.")

            # --- Load and Prepare Calculated Descriptors ---
            send_progress(f"⏳ Loading calculated descriptors from {os.path.basename(calculated_descriptors_path)}...", 60, node_id)
            descriptors_df = pd.read_csv(calculated_descriptors_path)
            # Check if 'Name' column exists (PaDEL output) for identification
            if 'Name' not in descriptors_df.columns:
                 print("[ComfyQSAR Screening] Warning: 'Name' column not found in PaDEL output. Compound identification might be less reliable.")
                 # If Name is missing, use the index as a fallback identifier
                 descriptors_df.insert(0, 'Original_Index', descriptors_df.index)
                 id_column = 'Original_Index'
            else:
                 id_column = 'Name' # Use PaDEL's Name column

            send_progress("⚙️ Preparing calculated descriptors (selecting features, handling NaN/Inf)...", 65, node_id)
            missing_model_descriptors = [desc for desc in selected_descriptors if desc not in descriptors_df.columns]
            if missing_model_descriptors:
                # Provide detailed error message
                available_descriptors = list(descriptors_df.columns)
                raise ValueError(f"Selected model descriptors missing in PaDEL output: {missing_model_descriptors}. \nAvailable descriptors in PaDEL output ({len(available_descriptors)}): {available_descriptors[:50]}... \nPlease check if the selected descriptors (.txt) are compatible with the PaDEL calculation settings (e.g., using a specific XML).")

            X_custom = descriptors_df[selected_descriptors]
            # Handle NaN/Inf before prediction
            X_custom = X_custom.replace([np.inf, -np.inf], np.nan)
            if X_custom.isnull().values.any():
                  send_progress("   Imputing NaN/Inf values in calculated descriptors (median)...", 70, node_id)
                  imputer = SimpleImputer(strategy='median')
                  X_custom_imputed = imputer.fit_transform(X_custom)
                  # Keep original index if possible, ensure columns are preserved
                  X_custom = pd.DataFrame(X_custom_imputed, columns=selected_descriptors, index=descriptors_df.index)
            send_progress("   Calculated descriptors prepared for prediction.", 75, node_id)

            # --- Perform Predictions --- #
            send_progress("🤖 Performing predictions...", 80, node_id)
            predictions = None
            if task_type == "Classification":
                 if hasattr(model, "predict_proba"):
                     predictions = model.predict_proba(X_custom)[:, 1] # Probability of class 1
                     pred_type = "Probability"
                 else:
                     predictions = model.predict(X_custom)
                     pred_type = "Label"
                 send_progress(f"   Classification predictions ({pred_type}) completed.", 85, node_id)
            elif task_type == "Regression":
                 if hasattr(model, "predict"):
                     predictions = model.predict(X_custom)
                 else: raise AttributeError("Loaded model lacks 'predict' method required for regression.")
                 send_progress("   Regression predictions completed.", 85, node_id)
            else: raise ValueError(f"Invalid task_type: {task_type}. Choose 'Regression' or 'Classification'.")


            # --- Save Results --- #
            send_progress("💾 Saving prediction results...", 90, node_id)
            # Create the results DataFrame starting with identifiers and predictions
            results_df = pd.DataFrame({
                id_column: descriptors_df[id_column],
                "Prediction": predictions
            })
            # Optionally merge back original descriptors if needed, but keep output clean
            # results_df = pd.merge(results_df, descriptors_df, on=id_column)

            # Define output filename
            results_filename = f"CustomScreening_{base_input_name}_{task_type}_Predictions_{timestamp}.csv"
            predictions_csv_path = os.path.join(output_node_dir, results_filename)
            results_df.to_csv(predictions_csv_path, index=False)
            send_progress(f"   Prediction results saved: {results_filename}", 94, node_id)

            # --- Generate Summary --- #
            send_progress("📝 Generating summary...", 95, node_id)
            summary_lines = [
                "**Custom Screening Completed!**",
                f"Input File: {os.path.basename(input_compound_file)} ({input_type})",
                f"Model File: {os.path.basename(model_path)}",
                f"Task Type: {task_type}",
                f"Compounds Processed: {results_df.shape[0]}",
                "--- Saved Files ---",
                f"Prediction Results CSV: {os.path.basename(predictions_csv_path)}",
                f"Calculated Descriptors (Temp): {os.path.basename(calculated_descriptors_path)}"
            ]
            text_container_content = create_text_container(*summary_lines)
            send_progress("🎉 Custom screening process finished successfully!", 100, node_id)

            # Return path to final predictions and the intermediate descriptor file
            return {"ui": {"text": text_container_content},
                    "result": (str(predictions_csv_path), str(calculated_descriptors_path),)}

        except FileNotFoundError as fnf_e:
             error_msg = f"❌ File/Directory Not Found Error: {str(fnf_e)}."; send_progress(error_msg, 100, node_id)
             return {"ui": {"text": create_text_container(error_msg)}, "result": ("", "")} # Return empty strings for paths
        except ValueError as ve:
             error_msg = f"❌ Value Error: {str(ve)}"; send_progress(error_msg, 100, node_id)
             return {"ui": {"text": create_text_container(error_msg)}, "result": ("", "")}
        except ImportError as ie: # padelpy missing (should be caught earlier)
             error_msg = f"❌ Import Error: {str(ie)}. Requires 'padelpy'."; send_progress(error_msg, 100, node_id)
             return {"ui": {"text": create_text_container(error_msg)}, "result": ("", "")}
        except RuntimeError as rte: # PaDEL execution error
             error_msg = f"❌ PaDEL Runtime Error: {str(rte)}"; send_progress(error_msg, 100, node_id)
             return {"ui": {"text": create_text_container(error_msg)}, "result": ("", "")}
        except AttributeError as ae: # Model prediction error
             error_msg = f"❌ Model Error: {str(ae)}"; send_progress(error_msg, 100, node_id)
             return {"ui": {"text": create_text_container(error_msg)}, "result": ("", "")}
        except Exception as e:
             error_msg = f"❌ An unexpected error occurred during custom screening: {str(e)}"
             import traceback; error_msg += f"\nTraceback:\n{traceback.format_exc()}"
             send_progress(error_msg, 100, node_id)
             return {"ui": {"text": create_text_container(error_msg)}, "result": ("", "")}
        finally:
             # Optional: Decide whether to keep the intermediate descriptor file.
             # Keeping it in the temp dir allows inspection after run.
             # If you want to auto-delete: uncomment below
             # if calculated_descriptors_path and os.path.exists(calculated_descriptors_path):
             #      try: os.remove(calculated_descriptors_path)
             #      except Exception as del_e: print(f"[ComfyQSAR Screening] Warning: Failed to delete temp descriptor file {calculated_descriptors_path}: {del_e}")
             pass


# --- Node Mappings ---
# (Keep existing mappings for VirtualScreening and ExtractScreenedMolecules)
NODE_CLASS_MAPPINGS = {
    "VirtualScreening": VirtualScreening,
    # Conditionally added RDKit node below
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VirtualScreening": "Virtual Screening",
    # Conditionally added RDKit node below
}

# Conditionally add RDKit dependent node
if RDKIT_AVAILABLE:
     NODE_CLASS_MAPPINGS["ExtractScreenedMolecules"] = ExtractScreenedMolecules
     NODE_DISPLAY_NAME_MAPPINGS["ExtractScreenedMolecules"] = "Extract Screened Molecules (SDF)"
else:
     print("[ComfyQSAR Screening] ExtractScreenedMolecules node registration skipped (RDKit not found).")

# Conditionally add PadelPy dependent node
if PADELPY_AVAILABLE:
    NODE_CLASS_MAPPINGS["CustomScreening"] = CustomScreening
    NODE_DISPLAY_NAME_MAPPINGS["CustomScreening"] = "Custom Screening (PaDEL)"
else:
     print("[ComfyQSAR Screening] CustomScreening node registration skipped (padelpy not found).")


