import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import SDWriter

METAL_IONS = {
    'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
    'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Th', 'Pa', 'U'
}

#function
def count_molecules(file_path):
    if file_path.endswith('.smi'):
        try:
            df = pd.read_csv(file_path, header=None)
            return len(df)
        except Exception:
            return 0
    elif file_path.endswith('.sdf'):
        try:
            suppl = Chem.SDMolSupplier(file_path, removeHs=False, strictParsing=False)
            return sum(1 for mol in suppl if mol is not None)
        except Exception as e:
            return 0
    else:
        raise ValueError("Unsupported file format.")
    
def filter_molecule(mol):
    if mol is None:
        return False
    atom_symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
    if atom_symbols.issubset(METAL_IONS):
        return False
    if len(Chem.GetMolFrags(mol)) > 1:
        return False
    return True

#node
class Data_Loader_Classification:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_file_path": ("STRING", 
                                       {"placeholder": "input/your/positive.sdf, .csv or .smi/path",
                                        "tooltip": "Path to the positive file"}), # Added default examples
                "negative_file_path": ("STRING", 
                                       {"placeholder": "input/your/negative.sdf, .csv or .smi/path",
                                        "tooltip": "Path to the negative file"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("POSITIVE", "NEGATIVE",)
    FUNCTION = "load_data"
    CATEGORY = "QSAR/CLASSIFICATION/LOAD&STANDARDIZATION"
    OUTPUT_NODE = True

    def load_data(self, positive_file_path, negative_file_path):
        try:

            # Check positive file
            if not os.path.exists(positive_file_path):
                raise FileNotFoundError(f"Positive file not found: {positive_file_path}")
            if not (positive_file_path.endswith('.smi') or positive_file_path.endswith('.csv') or positive_file_path.endswith('.sdf')):
                raise ValueError("Unsupported positive file format. Use .smi, .csv, or .sdf.")

            # Check negative file
            if not os.path.exists(negative_file_path):
                raise FileNotFoundError(f"Negative file not found: {negative_file_path}")
            if not (negative_file_path.endswith('.smi') or negative_file_path.endswith('.csv') or negative_file_path.endswith('.sdf')):
                raise ValueError("Unsupported negative file format. Use .smi, .csv, or .sdf.")

        except (FileNotFoundError, ValueError) as e:
            error_msg = f"❌ Error checking input files: {str(e)}"
            return {"ui": {"text": "❌ Data Loading Error", "text2": error_msg}, "result": ("", "")}

        try:
            # Count molecules
            pos_count = count_molecules(positive_file_path)
            neg_count = count_molecules(negative_file_path)
            total_count = pos_count + neg_count

            # Log message
            log_message = (
                "========================================\n"
                "🔹 Classification Data Loaded! 🔹\n"
                "========================================\n"
                f"✅ Positive Compounds: {pos_count}\n"
                f"✅ Negative Compounds: {neg_count}\n"
                f"📊 Total: {total_count} molecules\n"
                "📂 Ready to pass paths to next module ✅\n"
                "========================================"
                )
            
            return {
                "ui": {"text": log_message},
                "result": (str(positive_file_path), str(negative_file_path))
            }

        except Exception as e:
            error_message = (
                "========================================\n"
                "❌ **Data Loading Error!** ❌\n"
                "========================================\n"
                f"Error: {str(e)}\n"
                "Please check the file path and format.\n"
                "========================================"
            )
            return {"ui": {"text": error_message}, "result": ("", "")}


class Standardization_Classification:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_path": ("STRING", 
                                  {"placeholder": "connect/or/input/load/data/path",
                                   "tooltip": "Path to the positive file"}),
                "negative_path": ("STRING", 
                                  {"placeholder": "connect/or/input/load/data/path",
                                   "tooltip": "Path to the negative file"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("POSITIVE_STD", "NEGATIVE_STD",) # Renamed for clarity
    FUNCTION = "standardize_data"
    CATEGORY = "QSAR/CLASSIFICATION/LOAD&STANDARDIZATION"
    OUTPUT_NODE = True
    
    def standardize_data(self, positive_path, negative_path):
        output_dir = "QSAR/Standardization"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            error_msg = f"❌ Error creating output directory: {str(e)}"
            return {"ui": {"text": "❌ Standardization Error", "text2": error_msg}, "result": ("", "")}

        def process_file(file_path, output_name):
            output_file = ""
            filtered_count = 0
            try:
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

                return output_file, filtered_count

            except Exception as e:
                error_msg = f"❌ Error standardizing {output_name} file ({os.path.basename(file_path)}): {str(e)}"
                raise RuntimeError(error_msg) from e


        try:
            # Process files
            positive_output, pos_filtered_count = process_file(positive_path, "positive")
            negative_output, neg_filtered_count = process_file(negative_path, "negative")

            # Log message
            log_message = (
                "========================================\n"
                "🔹 Standardization Completed! 🔹\n"
                "========================================\n"
                f"✅ Positive Molecules Standardized: {pos_filtered_count}\n"
                f"✅ Negative Molecules Standardized: {neg_filtered_count}\n"
                f"💾 Output Dir: `{os.path.abspath(output_dir)}`\n"
                "✅ Invalid structures were removed during filtering.\n"
                "========================================"
            )
            
            return {
                "ui": {"text": log_message},
                "result": (str(positive_output), str(negative_output))
            }

        except Exception as e:
            error_message = (
                "========================================\n"
                "❌ **Standardization Error!** ❌\n"
                "========================================\n"
                f"Error: {str(e)}\n"
                "Please check the file path and format.\n"
                "========================================"
            )
            return {"ui": {"text": error_message}, "result": ("", "")}


class Load_and_Standardize_Classification:
    @classmethod
    def INPUT_TYPES(cls):
         return {
            "required": {
                "positive_file_path": ("STRING", 
                                       {"placeholder": "input/your/positive.sdf, .csv or .smi/path",
                                        "tooltip": "Path to the positive file"}),
                "negative_file_path": ("STRING", 
                                       {"placeholder": "input/your/negative.sdf, .csv or .smi/path",
                                        "tooltip": "Path to the negative file"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("POSITIVE_STD", "NEGATIVE_STD",) # Renamed for clarity
    FUNCTION = "load_and_standardize"
    CATEGORY = "QSAR/CLASSIFICATION/LOAD&STANDARDIZATION"
    OUTPUT_NODE = True

    def load_and_standardize(self, positive_file_path, negative_file_path):
        output_dir = "QSAR/Load_and_Standardize" # Different output dir
        try:
            os.makedirs(output_dir, exist_ok=True)

            if not os.path.exists(positive_file_path):
                raise FileNotFoundError(f"Positive file not found: {positive_file_path}")
            if not (positive_file_path.endswith('.smi') or positive_file_path.endswith('.csv') or positive_file_path.endswith('.sdf')):
                raise ValueError("Unsupported positive file format. Use .smi, .csv, or .sdf.")

            if not os.path.exists(negative_file_path):
                raise FileNotFoundError(f"Negative file not found: {negative_file_path}")
            if not (negative_file_path.endswith('.smi') or negative_file_path.endswith('.csv') or negative_file_path.endswith('.sdf')):
                raise ValueError("Unsupported negative file format. Use .smi, .csv, or .sdf.")

        except (FileNotFoundError, ValueError) as e:
            error_msg = f"❌ Error checking input files: {str(e)}"
            return {"ui": {"text": "❌ Load & Standardization Error", "text2": error_msg}, "result": ("", "")}

        def process_file(file_path, output_name):
            output_file = ""
            filtered_count = 0
            try:
                if file_path.endswith('.sdf'):
                    suppl = Chem.SDMolSupplier(file_path, removeHs=True, strictParsing=False)
                    valid_molecules = []
                    for mol in suppl:
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

                    df["RDKit_Mol"] = df[smiles_col].apply(lambda x: Chem.MolFromSmiles(x) if pd.notna(x) else None)
                    filtered_df = df[df["RDKit_Mol"].apply(filter_molecule)].copy()
                    filtered_df.drop(columns=["RDKit_Mol"], inplace=True)
                    filtered_count = len(filtered_df)

                    output_file = os.path.join(output_dir, f"{output_name}_standardized.csv")
                    filtered_df.to_csv(output_file, index=False)
                else:
                    raise ValueError(f"Unsupported file format: {file_path}")

                return output_file, filtered_count

            except Exception as e:
                error_msg = f"❌ Error processing {output_name} file ({os.path.basename(file_path)}): {str(e)}"
                raise RuntimeError(error_msg) from e

        try:
            # Process files
            positive_output, pos_filtered_count = process_file(positive_file_path, "positive")
            negative_output, neg_filtered_count = process_file(negative_file_path, "negative")

            # Log message
            log_message = (
                "========================================\n"
                "🔹 Load & Standardization Completed! 🔹\n"
                "========================================\n"
                f"✅ Positive Molecules Standardized: {pos_filtered_count}\n"
                f"✅ Negative Molecules Standardized: {neg_filtered_count}\n"
                f"💾 Output Dir: `{os.path.abspath(output_dir)}`\n"
                "✅ Invalid structures were removed during filtering.\n"
                "========================================"
            )
            
            return {
                "ui": {"text": log_message},
                "result": (str(positive_output), str(negative_output))
            }
        except Exception as e:
            error_message = (
                "========================================\n"
                "❌ **Load & Standardization Error!** ❌\n"
                "========================================\n"
                f"Error: {str(e)}\n"
                "Please check the file path and format.\n"
                "========================================"
            )
            return {"ui": {"text": error_message}, "result": ("", "")}


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