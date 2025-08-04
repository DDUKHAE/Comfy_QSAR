import pandas as pd
import os
from rdkit import Chem

METAL_IONS = {
    'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
    'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Th', 'Pa', 'U'
}

#function
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

#node
class Data_Loader_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "smiles_file_path": ("STRING", {"placeholder": "smiles.tsv", 
                                                "tooltip": "Path to the SMILES file"}),
                "biological_value_file_path": ("STRING", {"placeholder": "values.tsv", 
                                                          "tooltip": "Path to the biological value file"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("COMBINED_DATA_PATH",)
    FUNCTION = "load_data"
    CATEGORY = "QSAR/REGRESSION/LOAD & STANDARDIZATION"
    OUTPUT_NODE = True

    def load_data(self, smiles_file_path, biological_value_file_path):
        output_dir = "QSAR/Load_Data"
        combined_df_path = ""

        try:
            os.makedirs(output_dir, exist_ok=True)

            if not os.path.exists(smiles_file_path):
                raise FileNotFoundError(f"SMILES file not found: {smiles_file_path}")
            smiles_df = pd.read_csv(smiles_file_path, sep="\t", header=None, names=["SMILES"])

            if not os.path.exists(biological_value_file_path):
                raise FileNotFoundError(f"Biological value file not found: {biological_value_file_path}")
            value_df = pd.read_csv(biological_value_file_path, sep="\t", header=None, names=["value"])

            if len(smiles_df) != len(value_df):
                raise ValueError(f"Mismatched row count between SMILES ({len(smiles_df)}) and values ({len(value_df)})!")

            combined_df = pd.concat([smiles_df, value_df], axis=1)

            combined_df_path = os.path.join(output_dir, "regression_combined_input_data.csv")
            combined_df.to_csv(combined_df_path, index=False)

            log_message = (
                "========================================\n"
                "🔹 Regression Data Loading & Merging Completed! 🔹\n"
                "========================================\n"
                f"✅ SMILES File: {os.path.basename(smiles_file_path)} ({len(smiles_df)} records)\n"
                f"✅ Values File: {os.path.basename(biological_value_file_path)} ({len(value_df)} records)\n"
                f"✅ Total Merged Records: {len(combined_df)}\n"
                f"💾 Saved Combined Data: {combined_df_path}\n"
                "========================================\n"
            )
            return {"ui": {"text": log_message},
                    "result": (str(combined_df_path),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}. Please check input file paths."
            return {"ui": {"text": error_msg}, "result": (",")}
        except ValueError as ve:
            error_msg = f"❌ Value Error: {str(ve)}."
            return {"ui": {"text": error_msg}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred during data loading/merging: {str(e)}"

            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": error_msg}, "result": (",")} 


class Standardization_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_data_path": ("STRING", {"placeholder": "regression_input_data.csv", 
                                               "tooltip": "Path to the input data file"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STANDARDIZED_DATA_PATH",)
    FUNCTION = "standardize_data"
    CATEGORY = "QSAR/REGRESSION/LOAD & STANDARDIZATION"
    OUTPUT_NODE = True

    def standardize_data(self, input_data_path):
        output_dir = "QSAR/Standardization"
        filtered_data_path = ""

        try:
            os.makedirs(output_dir, exist_ok=True)

            data = pd.read_csv(input_data_path)
            original_count = len(data)

            if "SMILES" not in data.columns or "value" not in data.columns:
                raise ValueError("Required columns 'SMILES' and/or 'value' not found in the input data!")

            data["RDKit_Mol"] = data["SMILES"].apply(lambda x: Chem.MolFromSmiles(x) if pd.notna(x) else None)
            filtered_data = data[data["RDKit_Mol"].apply(filter_molecule)].drop(columns=["RDKit_Mol"])
            filtered_count = len(filtered_data)
            removed_count = original_count - filtered_count

            filtered_data_path = os.path.join(output_dir, f"regression_standardized_{original_count}_to_{filtered_count}.csv")
            filtered_data.to_csv(filtered_data_path, index=False)

            log_message = (
                "========================================\n"
                "🔹 Regression Data Standardization Completed! 🔹\n"
                "========================================\n"
                f"✅ Input File: {os.path.basename(input_data_path)} ({original_count} records)\n"
                f"✅ Records Kept: {filtered_count}\n"
                f"✅ Records Removed (Invalid SMILES, Metals, Fragments): {removed_count}\n"
                f"💾 Saved Standardized Data: {filtered_data_path}\n"
                "========================================\n"
            )

            return {"ui": {"text": log_message},
                    "result": (str(filtered_data_path),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}. Please check input file path."
            return {"ui": {"text": error_msg}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred during standardization: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": error_msg}, "result": (",")}


class Load_and_Standardize_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "smiles_file_path": ("STRING", {"placeholder": "smiles.tsv", 
                                                "tooltip": "Path to the SMILES file"}),
                "biological_value_file_path": ("STRING", {"placeholder": "values.tsv", 
                                                          "tooltip": "Path to the biological value file"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STANDARDIZED_DATA_PATH",)
    FUNCTION = "load_and_standardize_data"
    CATEGORY = "QSAR/REGRESSION/LOAD & STANDARDIZATION"
    OUTPUT_NODE = True

    def load_and_standardize_data(self, smiles_file_path, biological_value_file_path):
        output_dir = "QSAR/Load_and_Standardize"
        filtered_data_path = ""
        original_count, filtered_count = 0, 0

        try:
            os.makedirs(output_dir, exist_ok=True)

            if not os.path.exists(smiles_file_path): raise FileNotFoundError(f"SMILES file not found: {smiles_file_path}")
            smiles_df = pd.read_csv(smiles_file_path, sep="\t", header=None, names=["SMILES"])

            if not os.path.exists(biological_value_file_path): raise FileNotFoundError(f"Value file not found: {biological_value_file_path}")
            value_df = pd.read_csv(biological_value_file_path, sep="\t", header=None, names=["value"])

            if len(smiles_df) != len(value_df): raise ValueError(f"Mismatched rows: SMILES ({len(smiles_df)}) vs values ({len(value_df)})!")

            combined_df = pd.concat([smiles_df, value_df], axis=1)
            original_count = len(combined_df)

            if "SMILES" not in combined_df.columns or "value" not in combined_df.columns:
                 raise ValueError("Internal Error: Required columns 'SMILES'/'value' missing after merge!")

            combined_df["RDKit_Mol"] = combined_df["SMILES"].apply(lambda x: Chem.MolFromSmiles(x) if pd.notna(x) else None)
            filtered_data = combined_df[combined_df["RDKit_Mol"].apply(filter_molecule)].drop(columns=["RDKit_Mol"])
            filtered_count = len(filtered_data)
            removed_count = original_count - filtered_count

            filtered_data_path = os.path.join(output_dir, f"regression_loaded_standardized_{original_count}_to_{filtered_count}.csv")
            filtered_data.to_csv(filtered_data_path, index=False)

            log_message = (
                "========================================\n"
                "🔹 Regression Load & Standardization Completed! 🔹\n"
                "========================================\n"
                f"✅ Input SMILES: {os.path.basename(smiles_file_path)}\n"
                f"✅ Input Values: {os.path.basename(biological_value_file_path)}\n"
                f"✅ Original Records: {original_count}\n"
                f"✅ Records Kept After Standardization: {filtered_count}\n"
                f"✅ Records Removed: {removed_count}\n"
                f"💾 Saved Standardized Data: {filtered_data_path}\n"
                "========================================\n"
            )
            return {"ui": {"text": log_message},
                    "result": (str(filtered_data_path),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}."
            return {"ui": {"text": error_msg}, "result": (",")}
        except ValueError as ve:
            error_msg = f"❌ Value Error: {str(ve)}."
            return {"ui": {"text": error_msg}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": error_msg}, "result": (",")}


NODE_CLASS_MAPPINGS = {
    "Data_Loader_Regression": Data_Loader_Regression,
    "Standardization_Regression": Standardization_Regression,
    "Load_and_Standardize_Regression": Load_and_Standardize_Regression,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Data_Loader_Regression": "Data Loader (Regression)",
    "Standardization_Regression": "Standardization (Regression)",
    "Load_and_Standardize_Regression": "Load & Standardization (Regression)",
}