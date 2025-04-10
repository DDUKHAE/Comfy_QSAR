import pandas as pd
import os
from rdkit import Chem

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
                "smiles_file_path": ("STRING",),
                "biological_value_file_path": ("STRING",),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("DATA",)
    FUNCTION = "load_data"
    CATEGORY = "QSAR/REGRESSION/LOAD & STANDARDIZATION"
    OUTPUT_NODE = True
    
    def load_data(self, smiles_file_path, biological_value_file_path):

        os.makedirs("QSAR/Load_Data", exist_ok=True)

        #파일 로드
        smiles_df = pd.read_csv(smiles_file_path, sep="\t", header=None, names=["SMILES"])
        value_df = pd.read_csv(biological_value_file_path, sep="\t", header=None, names=["value"])

        if len(smiles_df) != len(value_df):
            raise ValueError(f" Error: Mismatched row count between SMILES ({len(smiles_df)}) and values ({len(value_df)})!"
                                )
            
        # 데이터 결합
        combined_df = pd.concat([smiles_df, value_df], axis=1)

        combined_df_path = os.path.join("QSAR/Load_Data", "Prepared_Input_Data.csv")
        combined_df.to_csv(combined_df_path, index=False)
        
        # 텍스트 컨테이너 생성
        text_container = create_text_container(
            "🔹 Data Loading & Merging Completed! 🔹",
            f"📂 SMILES and Value data successfully merged! ✅",
            f"📊 Total Records: {len(combined_df)}"
        )

        return {"ui": {"text": text_container},
                "result": (str(combined_df_path),)}
    
class Standardization_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": ("STRING",),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("DATA",)
    FUNCTION = "standardize_data"
    CATEGORY = "QSAR/REGRESSION/LOAD & STANDARDIZATION"
    OUTPUT_NODE = True

    def standardize_data(self, data):
        METAL_IONS = {
        'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
        'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Th', 'Pa', 'U'
        }

        os.makedirs("QSAR/Standardization", exist_ok=True)

        data = pd.read_csv(data)

        if "SMILES" not in data.columns or "value" not in data.columns:
            raise ValueError("Error: Required columns 'SMILES' and 'value' not found in the input data!")
        
        def filter_molecule(mol):
            if mol is None:
                return False
            
            atom_symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
            if atom_symbols.issubset(METAL_IONS):
                return False
            
            num_fragments = len(Chem.GetMolFrags(mol))
            if num_fragments > 1:
                return False
            
            return True
        
        data["RDKit_Mol"] = data["SMILES"].apply(lambda x: Chem.MolFromSmiles(x) if pd.notna(x) else None)
        filtered_data = data[data["RDKit_Mol"].apply(filter_molecule)].drop(columns=["RDKit_Mol"])

        filtered_data_path = os.path.join("QSAR/Standardization", "Standardized_Data.csv")
        filtered_data.to_csv(filtered_data_path, index=False)

        # 텍스트 컨테이너 생성
        text_container = create_text_container(
            "🔹 Standardization Completed! 🔹",
            "📂 Filtered dataset successfully saved! ✅"
        )
        
        return {"ui": {"text": text_container},
                "result": (str(filtered_data_path),)}
    
class Load_and_Standardize_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "smiles_file_path": ("STRING",),
                "biological_value_file_path": ("STRING",),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("DATA",)
    FUNCTION = "load_and_standardize_data"
    CATEGORY = "QSAR/REGRESSION/LOAD & STANDARDIZATION"
    OUTPUT_NODE = True
    
    def load_and_standardize_data(self, smiles_file_path, biological_value_file_path):

        METAL_IONS = {
        'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
        'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Th', 'Pa', 'U'
        }

        def filter_molecule(mol):
            if mol is None:
                return False
            
            atom_symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
            if atom_symbols.issubset(METAL_IONS):
                return False
            
            num_fragments = len(Chem.GetMolFrags(mol))
            if num_fragments > 1:
                return False
            
            return True
                
        os.makedirs("QSAR/Load_and_Standardize", exist_ok=True)
        #파일 로드
        smiles_df = pd.read_csv(smiles_file_path, sep="\t", header=None, names=["SMILES"])
        value_df = pd.read_csv(biological_value_file_path, sep="\t", header=None, names=["value"])

        if len(smiles_df) != len(value_df):
            raise ValueError(f" Error: Mismatched row count between SMILES ({len(smiles_df)}) and values ({len(value_df)})!"
                                )
            
        combined_df = pd.concat([smiles_df, value_df], axis=1)

        if "SMILES" not in combined_df.columns or "value" not in combined_df.columns:
            raise ValueError("Error: Required columns 'SMILES' and 'value' not found in the input data!")
        
        combined_df["RDKit_Mol"] = combined_df["SMILES"].apply(lambda x: Chem.MolFromSmiles(x) if pd.notna(x) else None)
        filtered_data = combined_df[combined_df["RDKit_Mol"].apply(filter_molecule)].drop(columns=["RDKit_Mol"])

        filtered_data_path = os.path.join("QSAR/Load_and_Standardize", "Standardized_Data.csv")
        filtered_data.to_csv(filtered_data_path, index=False)

        # 텍스트 컨테이너 생성
        text_container = create_text_container(
            "🔹 Loading & Standardization Completed! 🔹",
            f"📊 Total Records: {len(combined_df)}",
            "📂 Filtered dataset successfully saved! ✅"
        )
        
        return {"ui": {"text": text_container},
                "result": (str(filtered_data_path),)}
        
NODE_CLASS_MAPPINGS = {
    "Data_Loader_Regression": Data_Loader_Regression,
    "Standardization_Regression": Standardization_Regression,
    "Load_and_Standardize_Regression": Load_and_Standardize_Regression,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Data_Loader_Regression": "Data Loader(Regression)",
    "Standardization_Regression": "Standardization(Regression)",
    "Load_and_Standardize_Regression": "Load & Standardization(Regression)",
}