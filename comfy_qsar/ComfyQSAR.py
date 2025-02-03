import sys
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import SDWriter
from padelpy import padeldescriptor
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, RFE, SelectFromModel
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Lasso
from xgboost import XGBClassifier
import ast
from server import PromptServer
import folder_paths

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    import aiohttp
    from aiohttp import web
except ImportError:
    print("Module 'aiohttp' not installed. Please install it via:")
    print("pip install aiohttp")
    print("or")
    print("pip install -r requirements.txt")
    sys.exit()

class LoadTextAsset:
    @classmethod
    def INPUT_TYPES(cls):
        input_sub_folder = "QSAR"  # 기본값 설정
        input_dir = os.path.join(folder_paths.get_input_directory(), input_sub_folder)
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

        return {"required": {
                    "input_sub_folder": ("STRING", {"default": "QSAR"}),
                    "file": (sorted(files), {"save_textasset": True}),
                    "overwrite": ([True, False],),
                }
            }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "load_asset"
    CATEGORY = "QSAR"

    def load_asset(self, input_sub_folder, file, overwrite):
        text_path = os.path.join(folder_paths.get_input_directory(), file)
        return {"result": (text_path,)}

    @classmethod
    def IS_CHANGED(cls, input_sub_folder, file, overwrite):
        received_file_path = os.path.join(folder_paths.get_input_directory(), file)
        return received_file_path

    @classmethod
    def VALIDATE_INPUTS(cls, input_sub_folder, file, overwrite):
        if not input_sub_folder:
            raise ValueError("input_sub_folder is required")
        if not file:
            raise ValueError("file is required")
        if not isinstance(overwrite, bool):
            raise ValueError("overwrite must be a boolean")
        return True

@PromptServer.instance.routes.post("/upload/textasset")
async def upload_textasset(request):
    post = await request.post()
    response = save_received_textasset(post)
    return response

def get_dir_by_type(dir_type):
    if dir_type is None:
        dir_type = "input"

    if dir_type == "input":
        type_dir = folder_paths.get_input_directory()
    elif dir_type == "temp":
        type_dir = folder_paths.get_temp_directory()
    elif dir_type == "output":
        type_dir = folder_paths.get_output_directory()

    return type_dir, os.path.join(type_dir, "QSAR")

def save_received_textasset(post, received_file_save_function=None):
    received_file = post.get("file")
    overwrite = post.get("overwrite")

    received_file_upload_type = post.get("type")
    upload_dir, received_file_upload_type = get_dir_by_type(received_file_upload_type)

    if received_file and received_file.file:
        filename = received_file.filename
        if not filename:
            return web.Response(status=400)

        subfolder = post.get("subfolder", "")
        full_output_folder = os.path.join(upload_dir, os.path.normpath(subfolder))
        if not os.path.exists(full_output_folder):
            os.makedirs(full_output_folder)
        filepath = os.path.join(full_output_folder, filename)

        if overwrite is not None and (overwrite == "true" or overwrite == "1"):
            pass
        else:
            i = 1
            while os.path.exists(filepath):
                filename = f"{os.path.splitext(filename)[0]}_({i}){os.path.splitext(filename)[1]}"
                filepath = os.path.join(full_output_folder, filename)
                i += 1

        if received_file_save_function is not None:
            received_file_save_function(received_file, post, filepath)
        else:
            with open(filepath, "wb") as f:
                f.write(received_file.file.read())

        return web.json_response({"name": filename, "subfolder": subfolder, "type": received_file_upload_type, "path": filepath})
    else:
        return web.Response(status=400)

class LOAD_FILE():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "positive_path": ("STRING", {"multiline": False, "default": ""}),
                "negative_path": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("POSITIVE DATA", "NEGATIVE DATA")
    FUNCTION = "load_classification_data"
    CATEGORY = "QSAR"
    OUTPUT_NODE = True
    
    @staticmethod
    def load_classification_data(positive_path, negative_path):
        def load_sdf_file(file_path):
            suppl = Chem.SDMolSupplier(file_path)
            molecules = [mol for mol in suppl if mol is not None]
            return molecules

        def load_smiles_file(file_path):
            data = pd.read_csv(file_path)
            return data
        
        result_text = []
        positive_data = None
        negative_data = None
        
        # Positive data 로딩 시도
        if positive_path:
            if positive_path.endswith('.csv'):
                positive_data = load_smiles_file(positive_path)
                result_text.extend(["Positive data loaded successfully."])
            elif positive_path.endswith('.sdf'):
                positive_data = load_sdf_file(positive_path)
                result_text.extend(["Positive data loaded successfully."])
            else:
                raise ValueError("Unsupported file format for Positive dataset. Use .csv or .sdf.")

        # Negative data 로딩 시도
        if negative_path:
            if negative_path.endswith('.csv'):
                negative_data = load_smiles_file(negative_path)
                result_text.extend(["Negative data loaded successfully."])
            elif negative_path.endswith('.sdf'):
                negative_data = load_sdf_file(negative_path)
                result_text.extend(["Negative data loaded successfully."])
            else:
                raise ValueError("Unsupported file format for Negative dataset. Use .csv or .sdf.")

        # 데이터가 하나도 로드되지 않은 경우
        if positive_data is None and negative_data is None:
            raise ValueError("At least one dataset (positive or negative) must be provided.")

        return {"ui": {"text": result_text},
                "result": (positive_data, negative_data)}

class STANDARD():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required" :{
                "output_dir" : ("STRING", {"multiline" : False, "default" : ""}),
            },
            "optional" : {
                "positive_data" : ("STRING", {"multiline": False, "default": ""}),
                "negative_data" : ("STRING", {"multiline": False, "default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("POSITIVE DATA", "NEGATIVE DATA",)
    FUNCTION = "standardize_classification_data"
    CATEGORY = "QSAR"
    OUTPUT_NODE = True
    
    @staticmethod
    def standardize_classification_data(positive_data, negative_data, output_dir):
        METAL_IONS = {'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Th', 'Pa', 'U'}
    
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
       
        os.makedirs(output_dir, exist_ok=True)
        result_text = []
        positive_output = None
        negative_output = None

        # Positive data 처리
        if positive_data:
            if isinstance(positive_data, pd.DataFrame):
                if 'SMILES' not in positive_data.columns:
                    raise ValueError("The input DataFrame does not contain a 'SMILES' column.")
                
                positive_data['RDKit_Mol'] = positive_data['SMILES'].apply(lambda x: Chem.MolFromSmiles(x) if x else None)
                filtered_positive = positive_data[positive_data['RDKit_Mol'].apply(filter_molecule)]
                filtered_positive = filtered_positive.drop(columns=['RDKit_Mol'])
                positive_output = os.path.join(output_dir, "positive_standardized.csv")
                filtered_positive.to_csv(positive_output, index=False)
                result_text.extend([f"Positive data standardized and saved at: {positive_output}"])
            else:
                filtered_positive = [mol for mol in positive_data if filter_molecule(mol)]
                positive_output = os.path.join(output_dir, "positive_standardized.sdf")
                with SDWriter(positive_output) as writer:
                    for mol in filtered_positive:
                        writer.write(mol)
                result_text.extend([f"Positive data standardized and saved at: {positive_output}"])

        # Negative data 처리
        if negative_data:
            if isinstance(negative_data, pd.DataFrame):
                if 'SMILES' not in negative_data.columns:
                    raise ValueError("The input DataFrame does not contain a 'SMILES' column.")
                
                negative_data['RDKit_Mol'] = negative_data['SMILES'].apply(lambda x: Chem.MolFromSmiles(x) if x else None)
                filtered_negative = negative_data[negative_data['RDKit_Mol'].apply(filter_molecule)]
                filtered_negative = filtered_negative.drop(columns=['RDKit_Mol'])
                negative_output = os.path.join(output_dir, "negative_standardized.csv")
                filtered_negative.to_csv(negative_output, index=False)
                result_text.extend([f"Negative data standardized and saved at: {negative_output}"])
            else:
                filtered_negative = [mol for mol in negative_data if filter_molecule(mol)]
                negative_output = os.path.join(output_dir, "negative_standardized.sdf")
                with SDWriter(negative_output) as writer:
                    for mol in filtered_negative:
                        writer.write(mol)
                result_text.extend([f"Negative data standardized and saved at: {negative_output}"])

        # 데이터가 하나도 처리되지 않은 경우
        if not positive_output and not negative_output:
            raise ValueError("At least one dataset (positive or negative) must be provided.")
      
        return {"ui": {"text": result_text},
                "result": (positive_output, negative_output)
                }

#CALCULATION
class CALCULATE_DESCRIPTORS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_dir": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "positive_data": ("STRING", {"multiline": False, "default": ""}),
                "negative_data": ("STRING", {"multiline": False, "default": ""}),
                "d_2d": ([True, False], {"default": True}),
                "d_3d": ([False, True], {"default": False}),
                "detectaromaticity": ([True, False], {"default": True}),
                "log": ([True, False], {"default": True}),
                "removesalt": ([True, False], {"default": True}),
                "standardizenitro": ([True, False], {"default": True}),
                "usefilenameasmolname": ([True, False], {"default": True}),
                "retainorder": ([True, False], {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("POSITIVE DESCRIPTORS", "NEGATIVE DESCRIPTORS")
    FUNCTION = "calculate_descriptors"
    CATEGORY = "QSAR"
    OUTPUT_NODE = True
    
    def calculate_descriptors(self, positive_data, negative_data, output_dir, 
                            d_2d=True, d_3d=False, detectaromaticity=True, 
                            log=True, removesalt=True, standardizenitro=True, 
                            usefilenameasmolname=True, retainorder=True):
        result_text = []
        positive_descriptors = None
        negative_descriptors = None

        # Positive 파일 처리
        if positive_data:
            positive_descriptors = os.path.join(output_dir, "positive_descriptors.csv")
            padeldescriptor(mol_dir=positive_data, d_file=positive_descriptors,
                          d_2d=d_2d, d_3d=d_3d, detectaromaticity=detectaromaticity,
                          log=log, removesalt=removesalt, standardizenitro=standardizenitro,
                          usefilenameasmolname=usefilenameasmolname, retainorder=retainorder,
                          threads=-1, waitingjobs=-1, maxruntime=10000,
                          maxcpdperfile=0, headless=True)
            result_text.extend([
                "Positive descriptors calculation completed \n Saved at: {positive_descriptors}"])

        # Negative 파일 처리
        if negative_data:
            negative_descriptors = os.path.join(output_dir, "negative_descriptors.csv")
            padeldescriptor(mol_dir=negative_data, d_file=negative_descriptors,
                          d_2d=d_2d, d_3d=d_3d, detectaromaticity=detectaromaticity,
                          log=log, removesalt=removesalt, standardizenitro=standardizenitro,
                          usefilenameasmolname=usefilenameasmolname, retainorder=retainorder,
                          threads=-1, waitingjobs=-1, maxruntime=10000,
                          maxcpdperfile=0, headless=True)
            result_text.extend([
                "Negative descriptors calculation completed \n Saved at: {negative_descriptors}"])

        return {"ui": {"text": result_text},
                "result": (positive_descriptors, negative_descriptors)}

#PREPROCESSING
class FILTER_COMPOUNDS_BY_NAN_DUAL():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_data": ("STRING", {"forceInput": True}),
                "negative_data": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "output_dir": ("STRING", {"multiline": False, "default": ""}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("positive_filtered_file",
                    "negative_filtered_file",
                    "positive_removed_file",
                    "negative_removed_file",
                    )
    FUNCTION = "filter_compounds_by_nan_dual"
    CATEGORY = "QSAR"
    OUTPUT_NODE = True
    
    def filter_compounds_by_nan_dual(self, positive_data, negative_data, output_dir, threshold):
        os.makedirs(output_dir, exist_ok=True)

        # Positive data processing
        df_positive = pd.read_csv(positive_data)
        positive_nan_counts = df_positive.isna().sum(axis=1)
        total_descriptors_positive = df_positive.shape[1] - 2  # Exclude 'Label' and 'Name'
        df_positive['NaN_Percentage'] = positive_nan_counts / total_descriptors_positive

        positive_filtered = df_positive[df_positive['NaN_Percentage'] <= threshold].drop(columns=['NaN_Percentage'])
        positive_removed = df_positive[df_positive['NaN_Percentage'] > threshold]

        positive_filtered_file = os.path.join(output_dir, "positive_filtered_compound.csv")
        positive_removed_file = os.path.join(output_dir, "positive_removed_compound.csv")

        positive_filtered.to_csv(positive_filtered_file, index=False)
        positive_removed.to_csv(positive_removed_file, index=False)

        positive_retained_count = positive_filtered.shape[0]
        positive_removed_count = positive_removed.shape[0]

        # Negative data processing
        df_negative = pd.read_csv(negative_data)
        negative_nan_counts = df_negative.isna().sum(axis=1)
        total_descriptors_negative = df_negative.shape[1] - 2  # Exclude 'Label' and 'Name'
        df_negative['NaN_Percentage'] = negative_nan_counts / total_descriptors_negative

        negative_filtered = df_negative[df_negative['NaN_Percentage'] <= threshold].drop(columns=['NaN_Percentage'])
        negative_removed = df_negative[df_negative['NaN_Percentage'] > threshold]

        negative_filtered_file = os.path.join(output_dir, "negative_filtered_compound.csv")
        negative_removed_file = os.path.join(output_dir, "negative_removed_compound.csv")

        negative_filtered.to_csv(negative_filtered_file, index=False)
        negative_removed.to_csv(negative_removed_file, index=False)

        negative_retained_count = negative_filtered.shape[0]
        negative_removed_count = negative_removed.shape[0]

        text_output = [
            f"Positive compounds - Retained: {positive_retained_count}", 
            f"Saved at : {positive_filtered_file}",
            f"Positive compounds - Removed: {positive_removed_count}",
            f"Saved at : {positive_removed_file}",
            f"Negative compounds - Retained: {negative_retained_count}", 
            f"Saved at : {negative_filtered_file}",
            f"Negative compounds - Removed: {negative_removed_count}",
            f"Saved at : {negative_removed_file}"
        ]
        return {
            "ui": {"text": text_output},
            "result": (positive_filtered_file, negative_filtered_file, positive_removed_file, negative_removed_file)
        }


class REMOVE_HIGH_NAN_DESCRIPTORS():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_data": ("STRING", {"forceInput": True}),
                "negative_data": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "output_dir": ("STRING", {"multiline": False, "default": ""}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("positive_filtered_file",
                    "negative_filtered_file",
                    "positive_removed_file",
                    "negative_removed_file",
                    )
    FUNCTION = "remove_high_nan_descriptors"
    CATEGORY = "QSAR"
    OUTPUT_NODE = True

    def remove_high_nan_descriptors(self, positive_data, negative_data, output_dir, threshold):
        os.makedirs(output_dir, exist_ok=True)

        # Positive data processing
        df_positive = pd.read_csv(positive_data)
        positive_nan_percentage = df_positive.isna().mean()
        positive_retained = positive_nan_percentage[positive_nan_percentage <= threshold].index.tolist()
        positive_removed = positive_nan_percentage[positive_nan_percentage > threshold].index.tolist()

        positive_filtered = df_positive[positive_retained]
        positive_removed_df = df_positive[positive_removed]

        positive_filtered_file = os.path.join(output_dir, "positive_filtered_descriptors.csv")
        positive_removed_file = os.path.join(output_dir, "positive_removed_descriptors.csv")

        positive_filtered.to_csv(positive_filtered_file, index=False)
        positive_removed_df.to_csv(positive_removed_file, index=False)

        positive_retained_count = len(positive_retained)
        positive_removed_count = len(positive_removed)

        # Negative data processing
        df_negative = pd.read_csv(negative_data)
        negative_nan_percentage = df_negative.isna().mean()
        negative_retained = negative_nan_percentage[negative_nan_percentage <= threshold].index.tolist()
        negative_removed = negative_nan_percentage[negative_nan_percentage > threshold].index.tolist()

        negative_filtered = df_negative[negative_retained]
        negative_removed_df = df_negative[negative_removed]

        negative_filtered_file = os.path.join(output_dir, "negative_filtered_descriptors.csv")
        negative_removed_file = os.path.join(output_dir, "negative_removed_descriptors.csv")

        negative_filtered.to_csv(negative_filtered_file, index=False)
        negative_removed_df.to_csv(negative_removed_file, index=False)

        negative_retained_count = len(negative_retained)
        negative_removed_count = len(negative_removed)

        text_output = [
            f"Positive compounds - Retained: {positive_retained_count}", 
            f"Saved at : {positive_filtered_file}",
            f"Positive compounds - Removed: {positive_removed_count}",
            f"Saved at : {positive_removed_file}",
            f"Negative compounds - Retained: {negative_retained_count}", 
            f"Saved at : {negative_filtered_file}",
            f"Negative compounds - Removed: {negative_removed_count}",
            f"Saved at : {negative_removed_file}"
        ]
        return {
            "ui": {"text": text_output},
            "result": (positive_filtered_file, negative_filtered_file, positive_removed_file, negative_removed_file)
        }

class IMPUTE_MISSING_VALUES():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_data": ("STRING", {"forceInput": True}),
                "negative_data": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "output_dir": ("STRING", {"multiline": False, "default": ""}),
                "method": (["mean", "median", "most_frequent"],),
                "include_zero": ([True, False],),
                "specific_columns": ("STRING", {"multiline": False, "default": "None"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("positive_imputed_file", "negative_imputed_file")
    FUNCTION = "impute_missing_values"
    CATEGORY = "QSAR"
    OUTPUT_NODE = True
    
    def impute_missing_values(self, positive_data, negative_data, output_dir, method, include_zero, specific_columns):
        os.makedirs(output_dir, exist_ok=True)

        def process_file(input_file, output_file, method, include_zero, specific_columns):
            df = pd.read_csv(input_file)

            if specific_columns:
                specific_columns = ast.literal_eval(specific_columns)

            target_columns = specific_columns if specific_columns else [col for col in df.columns if col != "Name"]

            imputer = SimpleImputer(strategy=method)
            df[target_columns] = imputer.fit_transform(df[target_columns])
            df.to_csv(output_file, index=False)

        positive_imputed_file = os.path.join(output_dir, "positive_imputed.csv")
        process_file(positive_data, positive_imputed_file, method, include_zero, specific_columns)

        negative_imputed_file = os.path.join(output_dir, "negative_imputed.csv")
        process_file(negative_data, negative_imputed_file, method, include_zero, specific_columns)

        return {"ui": {"text": [
            f"Missing values imputed using {method} method",
            f"Imputed Positive file saved at:: {positive_imputed_file}",
            f"Imputed Negative file saved at:: {negative_imputed_file}"
        ]}, "result": (positive_imputed_file, negative_imputed_file)}


class MERGE_IMPUTED_DATA():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_data": ("STRING", {"forceInput": True}),
                "negative_data": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "output_dir": ("STRING", {"multiline": False, "default": ""}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("OUTPUT_FILE",)
    FUNCTION = "merge_imputed_data"
    CATEGORY = "QSAR"
    OUTPUT_NODE = True
    
    def merge_imputed_data(self, positive_data, negative_data, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        df_positive = pd.read_csv(positive_data)
        df_positive["Label"] = 1

        df_negative = pd.read_csv(negative_data)
        df_negative["Label"] = 0

        df_merged = pd.concat([df_positive, df_negative], ignore_index=True)
        output_file = os.path.join(output_dir, "merged_data.csv")
        df_merged.to_csv(output_file, index=False)

        return {"ui": {"text": [
            f"Merged dataset saved at: {output_file}",
            "Preview of the merged dataset:",
            str(df_merged.head())
        ]},
        "result": (output_file,)}

#OPTIMIZATION
class REMOVE_LOW_VARIANCE_FEATURES:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING", {"forceInput": True}),
                "output_file": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "threshold": ("FLOAT", {"default": 0.95,"step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("FEATURE",)
    FUNCTION = "remove_low_variance_features"
    CATEGORY = "QSAR"
    OUTPUT_NODE = True
    
    @staticmethod
    def remove_low_variance_features(input_file, output_file, threshold):
        df = pd.read_csv(input_file)
        initial_feature_count = len(df.columns) - 1  # Excluding Label column
        
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)

        try:
            df = pd.read_csv(input_file)
        except FileNotFoundError:
            return {"ui": {"text": [f"The file {input_file} was not found."]}}
        except pd.errors.EmptyDataError:
            return {"ui": {"text": [f"The file {input_file} is empty."]}}

        # Separate Label column
        if "Label" not in df.columns:
            return {"ui": {"text": ["The dataset must contain a 'Label' column."]}}

        label_column = df["Label"]
        feature_columns = df.drop(columns=["Label"])

        selector = VarianceThreshold(threshold=threshold)
        selected_features = selector.fit_transform(feature_columns)

        # Get remaining column names
        retained_columns = feature_columns.columns[selector.get_support()]

        # Create new DataFrame with retained features
        df_retained = pd.DataFrame(selected_features, columns=retained_columns)
        df_retained["Label"] = label_column

        # Save the resulting DataFrame
        df_retained.to_csv(output_file, index=False)

        final_feature_count = len(retained_columns)  # Fixed: selected_columns -> retained_columns
        
        return {"ui": {"text": [
            f"Initial number of features: {initial_feature_count}",
            f"Remaining number of features: {final_feature_count}",
            f"Features removed: {initial_feature_count - final_feature_count}"
        ]}, "result": (output_file,)}

class REMOVE_HIGH_CORRELATION_FEATURES:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING", {"forceInput": True}),
                "output_file": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "threshold": ("FLOAT", {"default": 0.90, "step": 0.01}),
                "mode": (["upper", "lower"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("FEATURE",)
    FUNCTION = "remove_high_correlation_features"
    CATEGORY = "QSAR"
    OUTPUT_NODE = True
    
    @staticmethod
    def remove_high_correlation_features(input_file, output_file, threshold, mode):
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)

        try:
            df = pd.read_csv(input_file)
        except FileNotFoundError:
            raise ValueError(f"The file {input_file} was not found.")
        except pd.errors.EmptyDataError:
            raise ValueError(f"The file {input_file} is empty.")

        # Separate Label column
        if "Label" not in df.columns:
            raise ValueError("The dataset must contain a 'Label' column.")

        label_column = df["Label"]
        feature_columns = df.drop(columns=["Label"])

        correlation_matrix = np.corrcoef(feature_columns.T)
        mask = np.abs(correlation_matrix) > threshold
        
        if mode == "upper":
            mask = np.triu(mask, k=1)
        else:
            mask = np.tril(mask, k=-1)
        
        to_remove = set(feature_columns.columns[mask.any(axis=0)])

        # Retain only non-removed columns
        retained_columns = [col for col in feature_columns.columns if col not in to_remove]
        df_retained = feature_columns[retained_columns]
        df_retained["Label"] = label_column

        # Save the resulting DataFrame
        df_retained.to_csv(output_file, index=False)
        
        initial_feature_count = feature_columns.shape[1]
        final_feature_count = len(retained_columns)

        return {"ui": {"text": [
            f"High correlation features removed with threshold {threshold} using {mode} mode",
            f"Selected features saved to: {output_file}",
            f"Initial number of features: {initial_feature_count}",
            f"Remaining number of features: {final_feature_count}",
            f"Features removed: {initial_feature_count - final_feature_count}",
            "Preview of the dataset after High Correlation Feature Removal:",
            str(df_retained.head())
        ]}, "result": (output_file,)}

#SELECTION
class LASSO_FEATURE_SELECTION:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_data": ("STRING", {"forceInput": True}),
                "target_column": ("STRING", {"multiline": False, "default": "Label"}),
                "alpha": ("FLOAT", {"multiline": False, "default": 0.01, "step": 0.01}),
                "max_iter": ("INT", {"multiline": False, "default": 1000}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("FEATURE",)
    FUNCTION = "lasso_feature_selection"
    CATEGORY = "QSAR"
    OUTPUT_NODE = True
    
    @staticmethod
    def lasso_feature_selection(input_data, target_column, alpha, max_iter):
        df = pd.read_csv(input_data)
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        lasso = Lasso(alpha=alpha, max_iter=max_iter, random_state=42)
        lasso.fit(X, y)
        selected_columns = X.columns[lasso.coef_ != 0]
        selected_features = X[selected_columns]
        selected_features[target_column] = y.reset_index(drop=True)
        return {"ui": {"text": [
            f"Lasso feature selection completed",
            f"Selected {len(selected_columns)} features"
        ]}, "result": (selected_features,)}


class TREE_FEATURE_SELECTION:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_data": ("STRING", {"forceInput": True}),
                "target_column": ("STRING", {"multiline": False, "default": "Label"}),
                "n_features": ("INT", {"multiline": False, "default": 10}),
                "n_estimators": ("INT", {"multiline": False, "default": 100}),
                "max_depth": ("INT", {"multiline": False, "default": None}),
                "min_samples_split": ("INT", {"multiline": False, "default": 2}),
                "criterion": (["gini", "entropy"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("FEATURE",)
    FUNCTION = "tree_feature_selection"
    CATEGORY = "QSAR"
    OUTPUT_NODE = True
    
    @staticmethod
    def tree_feature_selection(input_data, target_column, n_features, n_estimators, max_depth, min_samples_split, criterion):
        df = pd.read_csv(input_data)
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                      min_samples_split=min_samples_split, criterion=criterion, random_state=42)
        model.fit(X, y)
        feature_importances = model.feature_importances_
        important_indices = feature_importances.argsort()[-n_features:]
        selected_columns = X.columns[important_indices]
        selected_features = X[selected_columns]
        selected_features[target_column] = y.reset_index(drop=True)
        return {"ui": {"text": [
            f"Tree feature selection completed",
            f"Selected top {n_features} features using {criterion} criterion"
        ]}, "result": (selected_features,)}


class XGBOOST_FEATURE_SELECTION:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_data": ("STRING", {"forceInput": True}),
                "target_column": ("STRING", {"multiline": False, "default": "Label"}),
                "n_features": ("INT", {"multiline": False, "default": 10}),
                "n_estimators": ("INT", {"multiline": False, "default": 10}),
                "max_depth": ("INT", {"multiline": False, "default":""}),
                "learning_rate": ("FLOAT", {"multiline": False, "default": 0.1, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("FEATURE",)
    FUNCTION = "xgboost_feature_selection"
    CATEGORY = "QSAR"
    OUTPUT_NODE = True
    
    @staticmethod
    def xgboost_feature_selection(input_data, target_column, n_features, n_estimators, max_depth, learning_rate):
        df = pd.read_csv(input_data)
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
        model.fit(X, y)
        feature_importances = model.feature_importances_
        important_indices = feature_importances.argsort()[-n_features:]
        selected_columns = X.columns[important_indices]
        selected_features = X[selected_columns]
        selected_features[target_column] = y.reset_index(drop=True)
        return {"ui": {"text": [
            f"XGBoost feature selection completed",
            f"Selected top {n_features} features"
        ]}, "result": (selected_features,)}


class RFE_FEATURE_SELECTION:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_data": ("STRING", {"forceInput": True}),
                "target_column": ("STRING", {"multiline": False, "default": "Label"}),
                "n_features": ("INT", {"multiline": False, "default": 10}),
                "step": ("INT", {"multiline": False, "default": 1}),
                "cv": ("INT", {"multiline": False, "default": None}),
                "verbose": ("INT", {"multiline": False, "default": 0})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("FEATURE",)
    FUNCTION = "rfe_feature_selection"
    CATEGORY = "QSAR"
    OUTPUT_NODE = True
    
    @staticmethod
    def rfe_feature_selection(input_data, target_column, n_features, step, cv, verbose):
        df = pd.read_csv(input_data)
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        model = RandomForestClassifier(random_state=42)
        selector = RFE(estimator=model, n_features_to_select=n_features, step=step, verbose = verbose)
        selector.fit(X, y)
        selected_columns = X.columns[selector.get_support()]
        selected_features = X[selected_columns]
        selected_features[target_column] = y.reset_index(drop=True)
        return {"ui": {"text": [
            f"RFE feature selection completed",
            f"Selected {n_features} features using step size {step}"
        ]}, "result": (selected_features,)}


class SELECTFROMMODEL_FEATURE_SELECTION:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_data": ("STRING", {"forceInput": True}),
                "target_column": ("STRING", {"multiline": False, "default": "Label"}),
                "n_features": ("INT", {"multiline": False, "default": 10}),
                "threshold_auto": ([True, False],),
                "threshold": ("FLOAT", {"multiline": False, "default": None, "step": 0.01}),
                "prefit": ([True, False],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("FEATURE",)
    FUNCTION = "selectfrommodel_feature_selection"
    CATEGORY = "QSAR"
    OUTPUT_NODE = True
    
    @staticmethod
    def selectfrommodel_feature_selection(input_data, target_column, n_features, threshold_auto, threshold, prefit):
        df = pd.read_csv(input_data)
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        model = RandomForestClassifier(random_state=42)
        if threshold_auto == False:
            selector = SelectFromModel(estimator=model, threshold=threshold, max_features=n_features, prefit=prefit)
            
        else:
            selector = SelectFromModel(estimator=model, threshold="median", max_features=n_features, prefit=prefit)

        if prefit:
            model.fit(X, y)
            selector.fit(X, y)
            selected_columns = X.columns[selector.get_support()]
        else:
            selector.fit(X, y)
            selected_columns = X.columns[selector.get_support()]
        selected_features = X[selected_columns]
        selected_features[target_column] = y.reset_index(drop=True)
        return {"ui": {"text": [
            f"SelectFromModel feature selection completed",
            f"Selected {len(selected_columns)} features",
            f"Threshold mode: {'auto' if threshold_auto else 'manual'}"
        ]}, "result": (selected_features,)}

#GRID SEARCH
class GRID_SEARCH_HYPERPARAMETERS():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "X_train": ("DATAFRAME", {"multiline": False, "default": ""}),
                "y_train": ("DATAFRAME", {"multiline": False, "default": ""}),
            },
            "optional": {
                "algorithm": (["xgboost", "random_forest", "svm"],),
                "param_grid": (["dict", None],),
                "random_state": ("INT", {"step": 1, "default" : 42}),
                "verbose": ("INT", {"step": 0.1, "default": 2}),
            }
        }

    RETURN_TYPES = ("MODEL", "DICT", "DICT",)
    RETURN_NAMES = ("BEST MODEL", "BEST PARAMS", "GRID SEARCH RESULTS")
    FUNCTION = "grid_search"
    CATEGORY = "QSAR"
    OUTPUT_NODE = True
    
    def grid_search(self, X_train, y_train, algorithm, param_grid, random_state, verbose):
        #DEFINE THE ALGORITHM
        if algorithm == "xgboost":
            model = XGBClassifier(objective="binary:logistic", random_state=random_state)
            default_param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.6, 0.8, 1.0],
                'reg_alpha': [0.1, 1, 10],
                'reg_lambda': [1, 10, 100],
            }
        elif algorithm == "random_forest":
            model = RandomForestClassifier(random_state=random_state)
            default_param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False],
            }
        elif algorithm == "svm":
            model = SVC(probability=True, random_state=random_state)
            default_param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto'],
            }
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Choose from 'xgboost', 'random_forest', 'svm'.")

        # Use provided param_grid if available
        if param_grid is None:
            param_grid = default_param_grid
            
        #DEFINE SCORING METRICS
        scoring = {
        'accuracy': 'accuracy',
        'f1': make_scorer(f1_score),
        'roc_auc': 'roc_auc',
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'specificity': make_scorer(
            lambda y_true, y_pred: recall_score(y_true, y_pred, pos_label=0)  # Specificity 계산
        ),
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        
        #PERFORM GRID SEARCH CV
        grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit='accuracy',  # Default refit by accuracy
        return_train_score=True,
        verbose=verbose
        )

        print(f"Starting GridSearchCV for {algorithm}...")
        try:
            grid_search.fit(X_train, y_train)
        except Exception as e:
            print(f"An error occurred during grid search: {e}")
            return None

        return {
            "best_model": grid_search.best_estimator_,
            "best_params": grid_search.best_params_,
            "grid_search_results": grid_search.cv_results_,
        }
        
#TRAIN
class TRAIN_AND_EVALUATE_MODEL():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "best_model": ("DATAFRAME", {"multiline": False, "default": ""}),
                "X_test": ("DATAFRAME", {"multiline": False, "default": ""}),
                "y_test": ("DATAFRAME", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("MODEL", "DICT", "DICT",)
    RETURN_NAMES = ("BEST MODEL", "BEST PARAMS", "GRID SEARCH RESULTS")
    FUNCTION = "train_and_evaluate_model"
    CATEGORY = "QSAR"
    OUTPUT_NODE = True
    
    def train_and_evaluate_model(self, best_model, X_test, y_test):
        # Evaluate on the test set
        print("Evaluating the best model on the test set...")
        predictions = best_model.predict(X_test)
        probabilities = best_model.predict_proba(X_test)[:, 1]

        test_accuracy = best_model.score(X_test, y_test)
        test_f1 = f1_score(y_test, predictions)
        test_roc_auc = roc_auc_score(y_test, probabilities)
        test_precision = precision_score(y_test, predictions)
        test_recall = recall_score(y_test, predictions)
        test_specificity = recall_score(y_test, predictions, pos_label=0)  # Specificity 계산

        print("\nTest Set Results:")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"F1-Score: {test_f1:.4f}")
        print(f"ROC-AUC: {test_roc_auc:.4f}")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall (Sensitivity): {test_recall:.4f}")
        print(f"Specificity: {test_specificity:.4f}")

        return {
            "test_accuracy": test_accuracy,
            "test_f1": test_f1,
            "test_roc_auc": test_roc_auc,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_specificity": test_specificity,
        }
        
NODE_CLASS_MAPPINGS = {
    "LoadTextAsset": LoadTextAsset,
    "LOAD_FILE" : LOAD_FILE,
    "STANDARD" : STANDARD,
    "CALCULATE_DESCRIPTORS" : CALCULATE_DESCRIPTORS,
    "FILTER_COMPOUNDS_BY_NAN_DUAL" : FILTER_COMPOUNDS_BY_NAN_DUAL,
    "REMOVE_HIGH_NAN_DESCRIPTORS" : REMOVE_HIGH_NAN_DESCRIPTORS,
    "IMPUTE_MISSING_VALUES" : IMPUTE_MISSING_VALUES,
    "MERGE_IMPUTED_DATA" : MERGE_IMPUTED_DATA,
    "REMOVE_LOW_VARIANCE_FEATURES": REMOVE_LOW_VARIANCE_FEATURES,
    "REMOVE_HIGH_CORRELATION_FEATURES": REMOVE_HIGH_CORRELATION_FEATURES,
    "LASSO_FEATURE_SELECTION": LASSO_FEATURE_SELECTION,
    "TREE_FEATURE_SELECTION": TREE_FEATURE_SELECTION,
    "XGBOOST_FEATURE_SELECTION": XGBOOST_FEATURE_SELECTION,
    "RFE_FEATURE_SELECTION": RFE_FEATURE_SELECTION,
    "SELECTFROMMODEL_FEATURE_SELECTION": SELECTFROMMODEL_FEATURE_SELECTION,
    "GRID_SEARCH_HYPERPARAMETERS" : GRID_SEARCH_HYPERPARAMETERS,
    "TRAIN_AND_EVALUATE_MODEL": TRAIN_AND_EVALUATE_MODEL,
} 

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadTextAsset": "FILE LOADER",
    "LOAD_FILE" : "LOAD FILE",
    "STANDARD" : "STANDARDZATION",
    "CALCULATE_DESCRIPTORS" : "DESCRIPTORS CALCULATION",
    "FILTER_COMPOUNDS_BY_NAN_DUAL" : "FILTER COMPOUNDS BY NAN DUAL",
    "REMOVE_HIGH_NAN_DESCRIPTORS" : "REMOVE HIGH NAN DESCRIPTORS",
    "IMPUTE_MISSING_VALUES" : "IMPUTE MISSING VALUES",
    "MERGE_IMPUTED_DATA" : "MERGE IMPUTED DATA",
    "REMOVE_LOW_VARIANCE_FEATURES": "REMOVE_LOW_VARIANCE_FEATURES",
    "REMOVE_HIGH_CORRELATION_FEATURES": "REMOVE_HIGH_CORRELATION_FEATURES",
    "LASSO_FEATURE_SELECTION": "LASSO FEATURE SELECTION",
    "TREE_FEATURE_SELECTION": "TREE FEATURE SELECTION",
    "XGBOOST_FEATURE_SELECTION": "XGBOOST FEATURE SELECTION",
    "RFE_FEATURE_SELECTION": "RFE FEATURE SELECTION",
    "SELECTFROMMODEL_FEATURE_SELECTION": "SELECT FROM MODEL FEATURE SELECTION",
    "GRID_SEARCH_HYPERPARAMETERS" : "GRID SEARCH HYPERPARAMETERS",
    "TRAIN_AND_EVALUATE_MODEL": "TRAIN AND EVALUATE MODEL",
}
