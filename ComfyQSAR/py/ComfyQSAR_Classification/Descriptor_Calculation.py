import os
import pandas as pd
from padelpy import padeldescriptor
from rdkit import Chem
from .Data_Loader import create_text_container

class Descriptor_Calculations_Classification:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive_path": ("STRING",),
                "negative_path": ("STRING",),
                "advanced": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "descriptor_type": (["2D", "3D"], {"default": "2D"}),
                "detect_aromaticity": ("BOOLEAN", {"default": True}),
                "remove_salt": ("BOOLEAN", {"default": True}),
                "standardize_nitro": ("BOOLEAN", {"default": True}),
                "use_file_name_as_molname": ("BOOLEAN", {"default": True}),
                "retain_order": ("BOOLEAN", {"default": True}),
                "threads": ("INT", {"default": -1, "min": -1, "max": 64, "step": 1}),
                "waiting_jobs": ("INT", {"default": -1, "min": -1, "max": 64, "step": 1}),
                "max_runtime": ("INT", {"default": 10000, "min": 1000, "max": 100000, "step": 1000}),
                "max_cpd_per_file": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1000}),
                "headless": ("BOOLEAN", {"default": True}),
                "log": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("DESCRIPTORS_PATH",)
    FUNCTION = "calculate_and_merge_descriptors"
    CATEGORY = "QSAR/CLASSIFICATION/CALCULATION"
    OUTPUT_NODE = True
    
    @staticmethod
    def calculate_and_merge_descriptors(positive_path, negative_path, descriptor_type, detect_aromaticity, remove_salt, standardize_nitro, use_file_name_as_molname, retain_order, threads, waiting_jobs, max_runtime, max_cpd_per_file, headless, log, advanced):
        
        output_dir = "QSAR/Descriptor_Calculation"
        os.makedirs(output_dir, exist_ok=True)

        # PaDEL 옵션 딕셔너리 생성
        padel_options = {
            "d_2d": descriptor_type == "2D",
            "d_3d": descriptor_type == "3D",
            "detectaromaticity": detect_aromaticity,
            "removesalt": remove_salt,
            "standardizenitro": standardize_nitro,
            "usefilenameasmolname": use_file_name_as_molname,
            "retainorder": retain_order,
            "threads": threads,
            "waitingjobs": waiting_jobs,
            "maxruntime": max_runtime,
            "maxcpdperfile": max_cpd_per_file,
            "headless": headless,
            "log": log,
        }

        def process_file(input_path, tag, label):
            """주어진 파일을 처리하여 디스크립터를 계산하고 라벨을 추가합니다."""
            mol_dir_for_padel = input_path
            smiles_path = None # 임시 SMILES 파일 경로 초기화
            desc_file = os.path.join(output_dir, f"{tag}_descriptors.csv")

            try:
                if input_path.endswith('.sdf'):
                    # SDF 파일은 직접 사용
                    mol_dir_for_padel = input_path
                elif input_path.endswith(('.csv', '.smi')):
                    # CSV 또는 SMI 파일 처리
                    df = pd.read_csv(input_path)
                    if "SMILES" not in df.columns:
                        raise ValueError(f"❌ SMILES column is not found in the file: {input_path}")

                    # 임시 SMILES 파일 생성
                    smiles_path = os.path.join(output_dir, f"{tag}_smiles_temp.smi")
                    df[["SMILES"]].to_csv(smiles_path, index=False, header=False)
                    mol_dir_for_padel = smiles_path
                else:
                    raise ValueError(f"Unsupported file format: {input_path}")

                # PaDEL 디스크립터 계산 실행
                padeldescriptor(mol_dir=mol_dir_for_padel, d_file=desc_file, **padel_options)
                
                # 디스크립터 파일 존재 확인
                if not os.path.exists(desc_file):
                    raise FileNotFoundError(f"❌ Descriptor file is not created: {desc_file}")

                # 디스크립터 로드 및 라벨 추가
                df_desc = pd.read_csv(desc_file)
                df_desc["Label"] = label
                return df_desc
            
            finally:
                # 임시 SMILES 파일이 생성되었으면 삭제
                if smiles_path and os.path.exists(smiles_path):
                    os.remove(smiles_path)
        
        try:
            # Positive 및 Negative 데이터 처리
            df_positive = process_file(positive_path, "positive", label=1)
            df_negative = process_file(negative_path, "negative", label=0)

            # 결과 병합 및 저장
            df_final = pd.concat([df_positive, df_negative], ignore_index=True)
            final_file = os.path.join(output_dir, "final_merged_descriptors.csv")
            df_final.to_csv(final_file, index=False)
                
            # 최종 결과 텍스트 생성
            text_container = create_text_container(
                    "🔹 **Descriptor Calculation & Merge Done!** 🔹",
                    f"✅ Positive Molecules: {len(df_positive)}",
                    f"✅ Negative Molecules: {len(df_negative)}",
                    f"📊 Total: {len(df_final)}",
                    "📂 Format: descriptors + Label column (1=positive, 0=negative)"
                )
                
            return {
                "ui": {"text": text_container},
                "result": (str(final_file),)
            }
        
        except Exception as e:
            # 오류 발생 시 오류 메시지 생성
            error_container = create_text_container(
                "❌ **Descriptor Calculation Failed!** ❌",
                f"Error: {str(e)}"
            )
            return {
                "ui": {"text": error_container},
                "result": (str(""),)
            }

# 노드 등록
NODE_CLASS_MAPPINGS = {
    "Descriptor_Calculations_Classification": Descriptor_Calculations_Classification
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Descriptor_Calculations_Classification": "Descriptor Calculation(Classification)"
} 