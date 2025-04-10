import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import SDWriter

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
                "positive_file_path": ("STRING",),
                "negative_file_path": ("STRING",),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("POSITIVE_PATH", "NEGATIVE_PATH",)
    FUNCTION = "load_data"
    CATEGORY = "QSAR/CLASSIFICATION/LOAD &STANDARDIZATION"
    OUTPUT_NODE = True

    def load_data(self, positive_file_path, negative_file_path):
        os.makedirs("QSAR/Load_Data", exist_ok=True)
        
        # 양성/음성 데이터 로드 및 확인
        if not os.path.exists(positive_file_path):
            raise FileNotFoundError(f"❌ Positive file not found: {positive_file_path}")
        if not os.path.exists(negative_file_path):
            raise FileNotFoundError(f"❌ Negative file not found: {negative_file_path}")

        # SMILES 또는 SDF 파일 확인
        if not (positive_file_path.endswith('.smi') or positive_file_path.endswith('.csv') or positive_file_path.endswith('.sdf')):
            raise ValueError("❌ Unsupported positive file format. Use .smi, .csv, or .sdf.")
        if not (negative_file_path.endswith('.smi') or negative_file_path.endswith('.csv') or negative_file_path.endswith('.sdf')):
            raise ValueError("❌ Unsupported negative file format. Use .smi, .csv, or .sdf.")
        
        # 파일 확인 및 개수 계산 함수
        def count_molecules(file_path):
            if file_path.endswith('.sdf'):
                # SDF 파일 처리
                suppl = Chem.SDMolSupplier(file_path, removeHs=False, strictParsing=False)
                return sum(1 for mol in suppl if mol is not None)
            elif file_path.endswith('.smi'):
                # SMI 파일 처리
                df = pd.read_csv(file_path, header=None)
                return len(df)
            elif file_path.endswith('.csv'):
                # CSV 파일 처리
                df = pd.read_csv(file_path)
                if "SMILES" not in df.columns:
                    raise ValueError(f"CSV file {file_path} must contain a 'SMILES' column")
                return len(df)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        
        # 양성/음성 데이터 개수 계산
        pos_count = count_molecules(positive_file_path)
        neg_count = count_molecules(negative_file_path)
        total_count = pos_count + neg_count
        
        # 로그 메시지
        text_container = create_text_container(
            "🔹 Classification Data Loaded! 🔹",
            f"✅ Positive Compounds: {pos_count}",
            f"✅ Negative Compounds: {neg_count}",
            f"📊 Total: {total_count} molecules",
        )

        return {
            "ui": {"text": text_container},
            "result": (str(positive_file_path), str(negative_file_path))
        }

class Standardization_Classification:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_path": ("STRING",),
                "negative_path": ("STRING",),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("POSITIVE_PATH", "NEGATIVE_PATH",)
    FUNCTION = "standardize_data"
    CATEGORY = "QSAR/CLASSIFICATION/LOAD & STANDARDIZATION"
    OUTPUT_NODE = True
    
    def standardize_data(self, positive_path, negative_path):
        METAL_IONS = {
            'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
            'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Th', 'Pa', 'U'
        }
        
        os.makedirs("QSAR/Standardization", exist_ok=True)
        
        # 분자 필터링 함수
        def filter_molecule(mol):
            if mol is None:
                return False
            
            # 금속 이온만 포함된 분자 필터링
            atom_symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
            if atom_symbols.issubset(METAL_IONS):
                return False
            
            # 다중 조각 구조 필터링
            num_fragments = len(Chem.GetMolFrags(mol))
            if num_fragments > 1:
                return False 
            return True
        
        # 파일 처리 함수
        def process_file(file_path, output_name):
            if file_path.endswith('.sdf'):
                # SDF 파일 처리
                suppl = Chem.SDMolSupplier(file_path, removeHs=True)
                filtered = [mol for mol in suppl if filter_molecule(mol)]
                
                output_file = os.path.join("QSAR/Standardization", f"{output_name}.sdf")
                with Chem.SDWriter(output_file) as writer:
                    for mol in filtered:
                        writer.write(mol)

                return output_file, len(filtered)
                
            elif file_path.endswith('.smi') or file_path.endswith('.csv'):
                # SMI 또는 CSV 파일 처리
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    if "SMILES" not in df.columns:
                        raise ValueError(f"CSV file {file_path} must contain a 'SMILES' column")
                    smiles_col = "SMILES"
                else:
                    df = pd.read_csv(file_path, header=None, names=["SMILES"])
                    smiles_col = "SMILES"
                
                # RDKit 분자 객체 생성 및 필터링
                df["RDKit_Mol"] = df[smiles_col].apply(lambda x: Chem.MolFromSmiles(x) if pd.notna(x) else None)
                filtered_df = df[df["RDKit_Mol"].apply(filter_molecule)]
                filtered_df = filtered_df.drop(columns=["RDKit_Mol"])
                
                # 필터링 결과 저장
                output_file = os.path.join("QSAR/Standardization", f"{output_name}.csv")
                filtered_df.to_csv(output_file, index=False)
                
                return output_file, len(filtered_df)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        
        # 양성/음성 데이터 처리
        positive_output, pos_filtered_count = process_file(positive_path, "positive_standardized")
        negative_output, neg_filtered_count = process_file(negative_path, "negative_standardized")
        
        # 로그 메시지
        text_container = create_text_container(
            "🔹 Standardization Completed! 🔹",
            f"✅ Positive Molecules: {pos_filtered_count}",
            f"✅ Negative Molecules: {neg_filtered_count}",
            f"📊 Total: {pos_filtered_count + neg_filtered_count} molecules",
        )

        return {
            "ui": {"text": text_container},
            "result": (str(positive_output), str(negative_output))
        }

class Load_and_Standardize_Classification:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_file_path": ("STRING",),
                "negative_file_path": ("STRING",),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("POSITIVE_PATH", "NEGATIVE_PATH",)
    FUNCTION = "load_and_standardize"
    CATEGORY = "QSAR/CLASSIFICATION/LOAD & STANDARDIZATION"
    OUTPUT_NODE = True

    def load_and_standardize(self, positive_file_path, negative_file_path):
        METAL_IONS = {
            'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
            'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Th', 'Pa', 'U'
        }
        
        os.makedirs("QSAR/Load_and_Standardize", exist_ok=True)
        
        # 파일 확인
        if not os.path.exists(positive_file_path):
            raise FileNotFoundError(f"❌ Positive file not found: {positive_file_path}")
        if not os.path.exists(negative_file_path):
            raise FileNotFoundError(f"❌ Negative file not found: {negative_file_path}")

        # 파일 형식 확인
        if not (positive_file_path.endswith('.smi') or positive_file_path.endswith('.csv') or positive_file_path.endswith('.sdf')):
            raise ValueError("❌ Unsupported positive file format. Use .smi, .csv, or .sdf.")
        if not (negative_file_path.endswith('.smi') or negative_file_path.endswith('.csv') or negative_file_path.endswith('.sdf')):
            raise ValueError("❌ Unsupported negative file format. Use .smi, .csv, or .sdf.")
        
        # 분자 필터링 함수
        def filter_molecule(mol):
            if mol is None:
                return False
            
            # 금속 이온만 포함된 분자 필터링
            atom_symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
            if atom_symbols.issubset(METAL_IONS):
                return False
            
            # 다중 조각 구조 필터링
            num_fragments = len(Chem.GetMolFrags(mol))
            if num_fragments > 1:
                return False
            
            return True
        
        # 파일 로드 및 필터링 함수
        def process_file(file_path, output_name):
            if file_path.endswith('.sdf'):
                # SDF 파일 처리
                suppl = Chem.SDMolSupplier(file_path, removeHs=True)
                all_count = 0
                valid_molecules = []
                
                for mol in suppl:
                    all_count += 1
                    if filter_molecule(mol):
                        valid_molecules.append(mol)
                
                output_file = os.path.join("QSAR/Load_and_Standardize", f"{output_name}.sdf")
                writer = Chem.SDWriter(output_file)
                for mol in valid_molecules:
                    writer.write(mol)
                writer.close()
                
                return output_file, all_count, len(valid_molecules)
                
            elif file_path.endswith('.smi') or file_path.endswith('.csv'):
                # SMI 또는 CSV 파일 처리
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    if "SMILES" not in df.columns:
                        raise ValueError(f"CSV file {file_path} must contain a 'SMILES' column")
                    smiles_col = "SMILES"
                else:
                    df = pd.read_csv(file_path, header=None, names=["SMILES"])
                    smiles_col = "SMILES"
                
                all_count = len(df)
                
                # RDKit 분자 객체 생성 및 필터링
                df["RDKit_Mol"] = df[smiles_col].apply(lambda x: Chem.MolFromSmiles(x) if pd.notna(x) else None)
                filtered_df = df[df["RDKit_Mol"].apply(filter_molecule)]
                filtered_df = filtered_df.drop(columns=["RDKit_Mol"])
                
                # 필터링 결과 저장
                output_file = os.path.join("QSAR/Load_and_Standardize", f"{output_name}.csv")
                filtered_df.to_csv(output_file, index=False)
                
                return output_file, all_count, len(filtered_df)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        
        # 양성/음성 데이터 처리
        positive_output, pos_orig_count, pos_filtered_count = process_file(positive_file_path, "positive_standardized")
        negative_output, neg_orig_count, neg_filtered_count = process_file(negative_file_path, "negative_standardized")
        
        # 로그 메시지
        text_container = create_text_container(
            "🔹 Load & Standardization Completed! 🔹",
            f"📊 Original Data:",
            f"  - Positive: {pos_orig_count}",
            f"  - Negative: {neg_orig_count}",
            f"  - Total: {pos_orig_count + neg_orig_count}",
            f"📊 After Standardization:",
            f"  - Positive: {pos_filtered_count}",
            f"  - Negative: {neg_filtered_count}",
            f"  - Total: {pos_filtered_count + neg_filtered_count}",
        )

        return {
            "ui": {"text": text_container},
            "result": (str(positive_output), str(negative_output))
        }

NODE_CLASS_MAPPINGS = {
    "Data_Loader_Classification": Data_Loader_Classification,
    "Standardization_Classification": Standardization_Classification,
    "Load_and_Standardize_Classification": Load_and_Standardize_Classification,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Data_Loader_Classification": "Data Loader(Classification)",
    "Standardization_Classification": "Standardization(Classification)",
    "Load_and_Standardize_Classification": "Load & Standardization(Classification)",
}