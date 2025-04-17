import os
import pandas as pd
from padelpy import padeldescriptor
import time # For potential delays if needed

# --- Common Utility Import ---
try:
    from .Data_Loader import send_progress, create_text_container
except ImportError:
    print("[ComfyQSAR Regression Calculation] Warning: Could not import progress_utils. Progress updates might not work.")
    # Fallback functions
    def send_progress(message, progress=None, node_id=None):
        print(f"[Progress Fallback] {message}" + (f" ({progress}%)" if progress is not None else ""))
    def create_text_container(*lines):
        return "\n".join(lines)

class Descriptor_Calculations_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_data_path": ("STRING",), # Renamed for clarity
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
    RETURN_NAMES = ("DESCRIPTORS_WITH_VALUE_PATH",) # Updated name
    FUNCTION = "calculate_descriptors"
    CATEGORY = "QSAR/REGRESSION/CALCULATION"
    OUTPUT_NODE = True
    
    def calculate_descriptors(self, input_data_path, descriptor_type, detect_aromaticity, remove_salt, standardize_nitro, use_filename_as_mol_name, retain_order, threads, waiting_jobs, max_runtime, max_cpd_per_file, headless, log, advanced):
        send_progress("🚀 Starting Regression Descriptor Calculation...", 0)
        output_dir = "QSAR/Descriptor_Calculation"
        smiles_file = ""
        descriptor_file = ""
        final_file = ""

        try:
            os.makedirs(output_dir, exist_ok=True)
            send_progress(f"📂 Output directory checked/created: {output_dir}", 2)

            # Determine descriptor dimensions
            d_2d = descriptor_type == "2D"
            d_3d = descriptor_type == "3D"
            send_progress(f"🧬 Descriptor type set: {'2D' if d_2d else ''}{' & ' if d_2d and d_3d else ''}{'3D' if d_3d else ''}", 5)

            send_progress(f"⏳ Loading input data from: {input_data_path}", 8)
            input_data_df = pd.read_csv(input_data_path)
            original_count = len(input_data_df)
            send_progress(f"   Loaded {original_count} records.", 10)

            if "SMILES" not in input_data_df.columns:
                raise ValueError("Input data must contain a 'SMILES' column.")
            if "value" not in input_data_df.columns:
                send_progress("   Warning: 'value' column not found in input, will be added if possible after calculation.", 11)


            send_progress("💾 Preparing temporary SMILES file for PaDEL...", 12)
            smiles_file = os.path.join(output_dir, "temp_smiles_for_padel_reg.smi")
            # Ensure SMILES column exists before attempting to save
            if "SMILES" in input_data_df:
                input_data_df["SMILES"].to_csv(smiles_file, index=False, header=False) # No header for .smi
                send_progress(f"   Temporary SMILES file saved: {smiles_file}", 15)
            else:
                 # This case should ideally be caught earlier, but handle defensively
                 raise ValueError("Cannot proceed without 'SMILES' column.")

            descriptor_file = os.path.join(output_dir, "temp_descriptors_output_reg.csv")

            send_progress("⏳ Calculating descriptors using PaDEL-Descriptor... (This may take time)", 20)
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
                 send_progress("✅ PaDEL calculation finished.", 80)

            except Exception as padel_e:
                 # Catch potential errors from the padelpy call itself
                 error_msg = f"❌ Error during PaDEL-Descriptor execution: {str(padel_e)}"
                 # Include more details if possible, e.g., check if descriptor_file was created
                 if not os.path.exists(descriptor_file) or os.path.getsize(descriptor_file) == 0:
                      error_msg += "\n   PaDEL output file might be missing or empty. Check PaDEL logs/installation."
                 send_progress(error_msg)
                 # Clean up temporary file if it exists
                 if os.path.exists(smiles_file): os.remove(smiles_file)
                 return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}


            send_progress("⏳ Loading calculated descriptors...", 82)
            if not os.path.exists(descriptor_file):
                 raise FileNotFoundError("PaDEL output file not found after execution.")
            descriptors_df = pd.read_csv(descriptor_file)
            # Check if descriptor calculation produced expected output
            if descriptors_df.empty or len(descriptors_df) != original_count:
                  send_progress(f"   Warning: Descriptor output rows ({len(descriptors_df)}) mismatch original ({original_count}). Check PaDEL logs.", 84)
                  # Decide how to handle: error out or try to merge anyway? Let's try merging but warn heavily.
                  # For safety, maybe error out if counts differ significantly?
                  if abs(len(descriptors_df) - original_count) > 0: # Error if any mismatch for now
                       raise ValueError(f"Mismatch between input count ({original_count}) and PaDEL output count ({len(descriptors_df)}). Cannot reliably merge.")
            send_progress(f"   Loaded {descriptors_df.shape[1]} descriptors for {len(descriptors_df)} records.", 85)


            send_progress("⚙️ Merging descriptors with original data (SMILES, value)...", 88)
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
            send_progress("   Merging complete.", 90)

            send_progress("💾 Saving final data (descriptors + value)...", 92)
            final_file = os.path.join(output_dir, "regression_descriptors_with_value.csv")
            final_data.to_csv(final_file, index=False)
            send_progress(f"   Final data saved to: {final_file}", 94)


            send_progress("🧹 Cleaning up temporary files...", 96)
            if os.path.exists(smiles_file):
                os.remove(smiles_file)
                send_progress(f"   Removed {smiles_file}", 97)
            if os.path.exists(descriptor_file):
                os.remove(descriptor_file)
                send_progress(f"   Removed {descriptor_file}", 98)


            send_progress("📝 Generating summary...", 99)
            text_container_content = create_text_container(
                "🔹 **Regression Descriptor Calculation Completed!** 🔹",
                f"Input Records: {original_count}",
                f"Calculated Descriptors: {descriptors_df.shape[1] - 1 if 'Name' in descriptors_df else descriptors_df.shape[1]}", # Exclude 'Name' if present
                f"Output Records: {len(final_data)}",
                f"Output File: {final_file}",
                "(Contains SMILES, value, and calculated descriptors)"
            )
            send_progress("🎉 Descriptor calculation process finished.", 100)

            return {"ui": {"text": text_container_content},
                    "result": (str(final_file),)}

        except FileNotFoundError as fnf_e:
            error_msg = f"❌ File Not Found Error: {str(fnf_e)}. Please check input file paths."
            send_progress(error_msg)
            # Clean up potentially created temp files on error
            if os.path.exists(smiles_file): os.remove(smiles_file)
            if os.path.exists(descriptor_file): os.remove(descriptor_file)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except ValueError as ve:
            error_msg = f"❌ Value Error: {str(ve)}."
            send_progress(error_msg)
            if os.path.exists(smiles_file): os.remove(smiles_file)
            if os.path.exists(descriptor_file): os.remove(descriptor_file)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred during descriptor calculation: {str(e)}"
            import traceback
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
            send_progress(error_msg)
            if os.path.exists(smiles_file): os.remove(smiles_file)
            if os.path.exists(descriptor_file): os.remove(descriptor_file)
            return {"ui": {"text": create_text_container(error_msg)}, "result": (",")}


NODE_CLASS_MAPPINGS = {
    "Descriptor_Calculations_Regression": Descriptor_Calculations_Regression,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Descriptor_Calculations_Regression": "Descriptor Calculation (Regression)", # Updated
}