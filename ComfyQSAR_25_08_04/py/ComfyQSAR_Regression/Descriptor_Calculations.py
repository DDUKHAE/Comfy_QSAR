import os
import pandas as pd
from padelpy import padeldescriptor
import time # For potential delays if needed
import multiprocessing

CPU_COUNT = multiprocessing.cpu_count()

class Descriptor_Calculations_Regression:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_data_path": ("STRING", {"placeholder": "regression_input_data.csv", 
                                               "tooltip": "Path to the input data file"}),
                "advanced": ("BOOLEAN", {"default": False, 
                                         "tooltip": "Show advanced settings for fine-tuning"}),
                "descriptor_type": ("BOOLEAN", {"default": True,
                                                "label_on": "2D",
                                                "label_off": "3D",
                                                "tooltip": "Choose descriptor type: 2D (faster) or 3D (more detailed)"}),
            },
            "optional": {
                # === Basic Settings (shown when advanced=True) ===
                "detect_aromaticity": ("BOOLEAN", {"default": True, 
                                                    "tooltip": "Detect and handle aromatic structures"}),
                "remove_salt": ("BOOLEAN", {"default": True, 
                                            "tooltip": "Remove salt components from molecules"}),
                "standardize_nitro": ("BOOLEAN", {"default": True, 
                                                    "tooltip": "Standardize nitro groups"}),
                "log": ("BOOLEAN", {"default": False, 
                                    "tooltip": "Apply log transformation to descriptors"}),
                # === Advanced Settings ===
                "use_filename_as_mol_name": ("BOOLEAN", {"default": False, 
                                                         "tooltip": "Use filename instead of SMILES as molecule name"}),
                "retain_order": ("BOOLEAN", {"default": True, 
                                            "tooltip": "Keep original molecule order"}),
                "max_runtime": ("INT", {"default": 10000, "min": 1000, "max": 100000, "step": 1000,
                                       "tooltip": "Maximum calculation time per molecule (seconds)"}),
                "max_cpd_per_file": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1000,
                                             "tooltip": "Split large files (0 = no splitting)"}),
                "headless": ("BOOLEAN", {"default": True, 
                                        "tooltip": "Run without GUI (recommended for servers)"}),
                # === Performance Settings ===
                "threads": ("INT", {"default": min(4, CPU_COUNT), "min": 1, "max": CPU_COUNT, "step": 1, "display" : "slider",
                                    "tooltip": f"Number of CPU threads (1-{CPU_COUNT})"}),
                "waiting_jobs": ("INT", {"default": min(4, CPU_COUNT), "min": 1, "max": CPU_COUNT, "step": 1, "display" : "slider",
                                         "tooltip": "Number of concurrent jobs in queue"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("DESCRIPTORS_WITH_VALUE_PATH",) # Updated name
    FUNCTION = "calculate_descriptors"
    CATEGORY = "QSAR/REGRESSION/CALCULATION"
    OUTPUT_NODE = True
    
    def calculate_descriptors(self, input_data_path, descriptor_type, detect_aromaticity, remove_salt, standardize_nitro, use_filename_as_mol_name, retain_order, threads, waiting_jobs, max_runtime, max_cpd_per_file, headless, log, advanced):
        output_dir = "QSAR/Descriptor_Calculation"
        smiles_file = ""
        descriptor_file = ""
        final_file = ""
    
        try:
            os.makedirs(output_dir, exist_ok=True)

            # Determine descriptor dimensions
            if descriptor_type == True:
                d_2d = True
                d_3d = False
            else:
                d_2d = False
                d_3d = True

            input_data_df = pd.read_csv(input_data_path)
            original_count = len(input_data_df)

            if "SMILES" not in input_data_df.columns:
                raise ValueError("Input data must contain a 'SMILES' column.")
            if "value" not in input_data_df.columns:
                raise ValueError("Input data must contain a 'value' column.")

            smiles_file = os.path.join(output_dir, "temp_smiles_for_padel_reg.smi")
            # Ensure SMILES column exists before attempting to save
            if "SMILES" in input_data_df:
                input_data_df["SMILES"].to_csv(smiles_file, index=False, header=False) # No header for .smi
            else:
                 # This case should ideally be caught earlier, but handle defensively
                 raise ValueError("Cannot proceed without 'SMILES' column.")

            descriptor_file = os.path.join(output_dir, "temp_descriptors_output_reg.csv")

            # PaDEL Call - Wrap in try-except? PaDEL itself might raise errors.
            try:
                 padeldescriptor(mol_dir=smiles_file, d_file=descriptor_file,
                                d_2d=d_2d, d_3d=d_3d, detectaromaticity=detect_aromaticity,
                                log=log, removesalt=remove_salt, standardizenitro=standardize_nitro,
                                usefilenameasmolname=use_filename_as_mol_name, retainorder=retain_order,
                                threads=threads, waitingjobs=waiting_jobs, maxruntime=max_runtime,
                                maxcpdperfile=max_cpd_per_file, headless=headless)
                 # Simple delay to ensure file system catches up and UI updates
                 time.sleep(0.5)

            except Exception as padel_e:
                 # Catch potential errors from the padelpy call itself
                 error_msg = f"❌ Error during PaDEL-Descriptor execution: {str(padel_e)}"
                 # Include more details if possible, e.g., check if descriptor_file was created
                 if not os.path.exists(descriptor_file) or os.path.getsize(descriptor_file) == 0:
                      error_msg += "\n   PaDEL output file might be missing or empty. Check PaDEL logs/installation."
                 # Clean up temporary file if it exists
                 if os.path.exists(smiles_file): os.remove(smiles_file)
                 return {"ui": {"text": error_msg}, "result": (",")}


            if not os.path.exists(descriptor_file):
                 raise FileNotFoundError("PaDEL output file not found after execution.")
            descriptors_df = pd.read_csv(descriptor_file)
            # Check if descriptor calculation produced expected output
            if descriptors_df.empty or len(descriptors_df) != original_count:
                  # Decide how to handle: error out or try to merge anyway? Let's try merging but warn heavily.
                  # For safety, maybe error out if counts differ significantly?
                  if abs(len(descriptors_df) - original_count) > 0: # Error if any mismatch for now
                       raise ValueError(f"Mismatch between input count ({original_count}) and PaDEL output count ({len(descriptors_df)}). Cannot reliably merge.")


            # Reset index for safe concatenation, especially if retain_order was false in PaDEL
            input_data_reset = input_data_df.reset_index(drop=True)
            descriptors_reset = descriptors_df.reset_index(drop=True)
            # Select only necessary columns from input for merging (e.g., SMILES and value)
            merge_cols = ['SMILES']
            if 'value' in input_data_reset.columns:
                 merge_cols.append('value')
            input_subset = input_data_reset[merge_cols]

            # Concatenate the input subset with the descriptors
            final_data = pd.concat([input_subset, descriptors_reset.drop(columns=['Name'], errors='ignore')], axis=1) # Drop 'Name' from descriptors if it exists

            final_file = os.path.join(output_dir, "regression_descriptors_with_value.csv")
            final_data.to_csv(final_file, index=False)


            if os.path.exists(smiles_file):
                os.remove(smiles_file)
            if os.path.exists(descriptor_file):
                os.remove(descriptor_file)


            log_message = (
                "========================================\n"
                "🔹 **Descriptor Calculation & Merge Done!** 🔹\n"
                "========================================\n"
                f"✅ Input Records: {original_count}\n"
                f"✅ Calculated Descriptors: {descriptors_df.shape[1] - 1 if 'Name' in descriptors_df else descriptors_df.shape[1]}\n"
                f"✅ Output Records: {len(final_data)}\n"
                f"💾 Output File: `{final_file}`\n"
                "📂 Format: descriptors + value column\n"
                "========================================"
            )

            return {"ui": {"text": log_message}, "result": (str(final_file),) }

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}. Please check input file paths."
            # Clean up potentially created temp files on error
            if os.path.exists(smiles_file): os.remove(smiles_file)
            if os.path.exists(descriptor_file): os.remove(descriptor_file)
            return {"ui": {"text": error_msg}, "result": (",")}
        except ValueError as ve:
            error_msg = f"❌ Value Error: {str(ve)}."
            return {"ui": {"text": error_msg}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred during descriptor calculation: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": error_msg}, "result": (",")}

NODE_CLASS_MAPPINGS = {
    "Descriptor_Calculations_Regression": Descriptor_Calculations_Regression,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Descriptor_Calculations_Regression": "Descriptor Calculation (Regression)", # Updated
}