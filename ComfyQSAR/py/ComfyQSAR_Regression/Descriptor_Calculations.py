import os
import pandas as pd
from padelpy import padeldescriptor
from .Data_Loader import create_text_container

class Descriptor_Calculations_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filtered_data": ("STRING",),
                "advanced": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "descriptor_type": (["2D", "3D"], {"default": "2D"}),
                "detect_aromaticity": ("BOOLEAN", {"default": True}),
                "log": ("BOOLEAN", {"default": True}),
                "remove_salt": ("BOOLEAN", {"default": True}),
                "standardize_nitro": ("BOOLEAN", {"default": True}),
                "use_filename_as_mol_name": ("BOOLEAN", {"default": True}),
                "retain_order": ("BOOLEAN", {"default": True}),
                "threads": ("INT", {"default": -1, "min": -1, "max": 64, "step": 1}),
                "waiting_jobs": ("INT", {"default": -1, "min": -1, "max": 64, "step": 1}),
                "max_runtime": ("INT", {"default": 10000, "min": 1000, "max": 100000, "step": 1000}),
                "max_cpd_per_file": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1000}),
                "headless": ("BOOLEAN", {"default": True}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("DESCRIPTOR",)
    FUNCTION = "calculate_descriptors"
    CATEGORY = "QSAR/REGRESSION/CALCULATION"
    OUTPUT_NODE = True
    
    @staticmethod
    def calculate_descriptors(filtered_data, descriptor_type, detect_aromaticity, remove_salt, standardize_nitro, use_filename_as_mol_name, retain_order, threads, waiting_jobs, max_runtime, max_cpd_per_file, headless, log, advanced):

        os.makedirs("QSAR/Descriptor_Calculation", exist_ok=True)

        if descriptor_type == "2D":
            d_2d = True
            d_3d = False
        else:
            d_2d = False
            d_3d = True

        filtered_data = pd.read_csv(filtered_data)

        smiles_file = os.path.join("QSAR/Descriptor_Calculation", "smiles_for_calculation.smi")
        filtered_data["SMILES"].to_csv(smiles_file, index=False, header=None)
        
        descriptor_file = os.path.join("QSAR/Descriptor_Calculation", "descriptors_output.csv")

        padeldescriptor(mol_dir=smiles_file, d_file=descriptor_file,
                    d_2d=d_2d, d_3d=d_3d, detectaromaticity=detect_aromaticity,
                    log=log, removesalt=remove_salt, standardizenitro=standardize_nitro,
                    usefilenameasmolname=use_filename_as_mol_name, retainorder=retain_order,
                    threads=threads, waitingjobs=waiting_jobs, maxruntime=max_runtime,
                    maxcpdperfile=max_cpd_per_file, headless=headless)
        
        descriptors_df = pd.read_csv(descriptor_file)
        
        final_data = pd.concat([filtered_data.reset_index(drop=True), descriptors_df.reset_index(drop=True)], axis=1)

        final_file = os.path.join("QSAR/Descriptor_Calculation", "molecular_descriptors_with_label.csv")
        final_data.to_csv(final_file, index=False)

        text_container = create_text_container(
            "🔹 **Descriptor Calculation Done!** 🔹",
            f"✅ Molecules: {len(filtered_data)}",
            f"📊 Total: {len(final_data)}",
            "📂 Format: descriptors + Label column (1=positive, 0=negative)"
        )
        
        os.remove(descriptor_file)

        return {"ui": {"text": text_container},
                "result": (str(final_file),)}
        
NODE_CLASS_MAPPINGS = {
    "Descriptor_Calculations_Regression": Descriptor_Calculations_Regression,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Descriptor_Calculations_Regression": "Descriptor Calculation(Regression)",
}