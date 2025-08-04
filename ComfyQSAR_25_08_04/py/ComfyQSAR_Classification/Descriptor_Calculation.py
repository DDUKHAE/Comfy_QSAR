import os
import pandas as pd
from padelpy import padeldescriptor
import multiprocessing

class Descriptor_Calculations_Classification:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive_std": ("STRING", {"tooltip": "Path to positive compounds file (SDF/CSV/SMI)"}),
                "negative_std": ("STRING", {"tooltip": "Path to negative compounds file (SDF/CSV/SMI)"}),
                "advanced": ("BOOLEAN", {"default": False, 
                                         "tooltip": "Show advanced settings for fine-tuning"}),
                "descriptor_type": ("BOOLEAN", {"default": True,
                                                "label_on": "2D",
                                                "label_off": "3D",
                                                "tooltip": "Choose descriptor type: 2D (faster) or 3D (more detailed)"}),
            },
            "optional": {
                "detect_aromaticity": ("BOOLEAN", {"default": True, 
                                                   "tooltip": "Detect and handle aromatic structures"}),
                "remove_salt": ("BOOLEAN", {"default": True, 
                                            "tooltip": "Remove salt components from molecules"}),
                "standardize_nitro": ("BOOLEAN", {"default": True, 
                                                  "tooltip": "Standardize nitro groups"}),
                "log": ("BOOLEAN", {"default": False, 
                                    "tooltip": "Apply log transformation to descriptors"}),
                "use_file_name_as_molname": ("BOOLEAN", {"default": False, 
                                                         "tooltip": "Use filename instead of SMILES as molecule name"}),
                "retain_order": ("BOOLEAN", {"default": True, 
                                             "tooltip": "Keep original molecule order"}),
                "max_runtime": ("INT", {"default": 10000, "min": 1000, "max": 100000, "step": 1000, 
                                        "tooltip": "Maximum calculation time per molecule (seconds)"}),
                "max_cpd_per_file": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1000, 
                                             "tooltip": "Split large files (0 = no splitting)"}),
                "headless": ("BOOLEAN", {"default": True, 
                                         "tooltip": "Run without GUI (recommended for servers)"}),
                "threads": ("INT", {"default": min(4, multiprocessing.cpu_count()), "min": 1, "max": multiprocessing.cpu_count(), "step": 1, "display" : "slider",
                                    "tooltip": f"Number of CPU threads (1-{multiprocessing.cpu_count()})"}),
                "waiting_jobs": ("INT", {"default": min(4, multiprocessing.cpu_count()), "min": 1, "max": multiprocessing.cpu_count(), "step": 1, "display" : "slider",
                                         "tooltip": "Number of concurrent jobs in queue"}),
                
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("DESCRIPTORS",)
    FUNCTION = "calculate_and_merge_descriptors"
    CATEGORY = "QSAR/CLASSIFICATION/CALCULATION"
    OUTPUT_NODE = True

    def calculate_and_merge_descriptors(self, positive_std, negative_std, descriptor_type, detect_aromaticity, remove_salt, standardize_nitro, use_file_name_as_molname, retain_order, threads, waiting_jobs, max_runtime, max_cpd_per_file, headless, log, advanced):
        if descriptor_type == True:
            d_2d = True
            d_3d = False
        else:
            d_2d = False
            d_3d = True

        output_dir = "QSAR/Descriptor_Calculation"
        os.makedirs(output_dir, exist_ok=True)

        padel_options = {
            "d_2d": d_2d,
            "d_3d": d_3d,
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
            if input_path.endswith('.sdf'):
                desc_file = os.path.join(output_dir, f"{tag}_descriptors.csv")
                padeldescriptor(mol_dir=input_path, d_file=desc_file,
                                **padel_options)

            elif input_path.endswith('.csv') or input_path.endswith('.smi'):
                df = pd.read_csv(input_path)
                if "SMILES" not in df.columns:
                    raise ValueError(f"❌ The file {input_path} does not contain a 'SMILES' column.")

                smiles_path = os.path.join(output_dir, f"{tag}_smiles_temp.smi")
                df[["SMILES"]].to_csv(smiles_path, index=False, header=False)

                desc_file = os.path.join(output_dir, f"{tag}_descriptors.csv")
                padeldescriptor(mol_dir=smiles_path, d_file=desc_file,
                                **padel_options)
                os.remove(smiles_path)

            else:
                raise ValueError(f"Unsupported file format for {tag}: {input_path}")

            df_desc = pd.read_csv(desc_file)
            df_desc["Label"] = label
            return df_desc

        df_positive = process_file(positive_std, "positive", label=1)
        df_negative = process_file(negative_std, "negative", label=0)

        df_final = pd.concat([df_positive, df_negative], ignore_index=True)
        final_file = os.path.join(output_dir, "final_merged_descriptors.csv")
        df_final.to_csv(final_file, index=False)

        log_message = (
            "========================================\n"
            "🔹 **Descriptor Calculation & Merge Done!** 🔹\n"
            "========================================\n"
            f"✅ Positive Molecules: {len(df_positive)}\n"
            f"✅ Negative Molecules: {len(df_negative)}\n"
            f"📊 Total: {len(df_final)}\n"
            f"💾 Output: `{final_file}`\n"
            "📂 Format: descriptors + Label column (1=positive, 0=negative)\n"
            "========================================"
        )

        return { "ui": {"text": log_message}, "result": (str(final_file),) }


NODE_CLASS_MAPPINGS = {
    "Descriptor_Calculations_Classification": Descriptor_Calculations_Classification
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Descriptor_Calculations_Classification": "Descriptor Calculation (Classification)"
} 