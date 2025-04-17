"""
ComfyQSAR - QSAR 모델링을 위한 ComfyUI 확장
"""

import os
import traceback
import importlib.util
import sys
import subprocess
import time

# --- Path Definitions ---
QSAR_PATH = os.path.dirname(os.path.realpath(__file__))
PY_PATH = os.path.join(QSAR_PATH, "py")
WEB_DIRECTORY = "web" # Note: This variable is defined but not used later for web files.
REQUIREMENTS_PATH = os.path.join(QSAR_PATH, "requirements.txt")

# --- Dependency Management ---
def install_dependencies():
    if not os.path.exists(REQUIREMENTS_PATH):
        print(f"ComfyQSAR: requirements.txt not found at {REQUIREMENTS_PATH}")
        return False

    print("ComfyQSAR: Installing dependencies from requirements.txt...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_PATH],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # Quieter install
        print("ComfyQSAR: Dependencies installed successfully.")
        time.sleep(1) # Give pip some time
        return True
    except subprocess.CalledProcessError as e:
        print(f"ComfyQSAR: Error installing dependencies: {str(e)}")
        # Attempt to provide more specific feedback if possible
        try:
            # Rerun with output captured to show the error
            result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_PATH],
                                    capture_output=True, text=True, check=False)
            print("--- pip install output ---")
            print(result.stdout)
            print(result.stderr)
            print("--------------------------")
        except Exception as inner_e:
            print(f"ComfyQSAR: Could not capture pip output: {inner_e}")
        return False
    except FileNotFoundError:
        print(f"ComfyQSAR: Error: '{sys.executable} -m pip' command not found. Is Python/pip installed correctly?")
        return False

def check_and_install_dependencies():
    required = [
        "numpy", "pandas", "scikit-learn", "rdkit", "matplotlib",
        "seaborn", "joblib", "scipy", "padelpy", "statsmodels",
        "xgboost", "lightgbm", "catboost"
    ]
    missing = []
    for dep in required:
        try:
            importlib.import_module(dep)
        except ImportError:
            missing.append(dep)

    if missing:
        print(f"ComfyQSAR: Missing dependencies: {', '.join(missing)}")
        if install_dependencies():
            # Re-check after installation
            still_missing = []
            for dep in missing: # Only check those that were initially missing
                try:
                    # Force reload in case it was partially imported before failing
                    importlib.invalidate_caches()
                    importlib.import_module(dep)
                except ImportError:
                    still_missing.append(dep)
            if still_missing:
                print(f"ComfyQSAR: Failed to install some dependencies: {', '.join(still_missing)}")
                print("ComfyQSAR: Some nodes may not work correctly. Please install manually:")
                print(f"'{sys.executable} -m pip install -r {REQUIREMENTS_PATH}'")
            else:
                print("ComfyQSAR: All required dependencies are now available.")
        else:
            print("ComfyQSAR: Failed to automatically install dependencies.")
            print("ComfyQSAR: Please install manually:")
            # Using the simpler print version
            print("  Run: " + sys.executable + " -m pip install -r \"" + REQUIREMENTS_PATH + "\"")
    else:
        print("ComfyQSAR: All dependencies verified.")


# --- Node Loading Logic ---
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def load_dynamic_module(module_path, spec_name):
    """Helper function to load a module dynamically and get mappings."""
    loaded_class_mappings = {}
    loaded_display_mappings = {}
    try:
        spec = importlib.util.spec_from_file_location(spec_name, module_path)
        if spec is None or spec.loader is None:
            print(f"ComfyQSAR: Could not create import spec for {module_path}")
            return None, None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec_name] = module
        spec.loader.exec_module(module)

        if hasattr(module, "NODE_CLASS_MAPPINGS"):
            loaded_class_mappings = module.NODE_CLASS_MAPPINGS
        if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
            loaded_display_mappings = module.NODE_DISPLAY_NAME_MAPPINGS
        return loaded_class_mappings, loaded_display_mappings

    except FileNotFoundError:
        print(f"ComfyQSAR: Module file not found: {module_path}")
        return None, None
    except Exception as e:
        # Simplify print statement using format()
        print("ComfyQSAR: Error loading module '{}' from {}: {}".format(spec_name, module_path, e))
        traceback.print_exc()
        return None, None

def load_modules_from_directory(directory_path, prefix):
    """Loads all valid Python modules from a directory."""
    global NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    if not os.path.isdir(directory_path):
        # print(f"ComfyQSAR: Module directory not found: {directory_path}") # Optional: Warn if dir is missing
        return

    for filename in os.listdir(directory_path):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]
            module_path = os.path.join(directory_path, filename)
            spec_name = f"{prefix}.{module_name}" if prefix else module_name # Create a unique spec name

            class_mappings, display_mappings = load_dynamic_module(module_path, spec_name)

            if class_mappings is not None: # Check if loading was successful
                NODE_CLASS_MAPPINGS.update(class_mappings)
            if display_mappings is not None:
                NODE_DISPLAY_NAME_MAPPINGS.update(display_mappings)

def load_all_qsar_modules():
    """Loads all modules for ComfyQSAR in a specific order."""
    global NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    NODE_CLASS_MAPPINGS = {} # Reset mappings
    NODE_DISPLAY_NAME_MAPPINGS = {}

    print("ComfyQSAR: Starting module loading...")
    # 1. Load main node categories
    print("ComfyQSAR: Loading Regression nodes...")
    regression_path = os.path.join(PY_PATH, "ComfyQSAR_Regression")
    load_modules_from_directory(regression_path, "ComfyQSAR.Regression")

    print("ComfyQSAR: Loading Classification nodes...")
    classification_path = os.path.join(PY_PATH, "ComfyQSAR_Classification")
    load_modules_from_directory(classification_path, "ComfyQSAR.Classification")

    # 2. Load Utility and other nodes
    print("ComfyQSAR: Loading Other node(s)...")
    
    # Define paths to load
    files_to_load = {
        "Screening.py": "ComfyQSAR.Screening",
        "show_text.py": "ComfyQSAR.show_text",
    }
    
    # Try loading each file
    for file_name, spec_name in files_to_load.items():
        file_path = os.path.join(PY_PATH, file_name)
        if os.path.isfile(file_path):
            class_mappings, display_mappings = load_dynamic_module(file_path, spec_name)
            if class_mappings is not None:
                NODE_CLASS_MAPPINGS.update(class_mappings)
            if display_mappings is not None:
                NODE_DISPLAY_NAME_MAPPINGS.update(display_mappings)
        else:
            print(f"ComfyQSAR: {file_name} not found at {file_path}")

    print(f"ComfyQSAR: Module loading complete. Found {len(NODE_CLASS_MAPPINGS)} node classes.")
    if not NODE_CLASS_MAPPINGS:
        print("ComfyQSAR: Warning - No node classes were loaded. Check module paths and file contents.")
        return False
    return True

# --- Initialization ---
print("\n=== ComfyQSAR Extension Initializing ===")
check_and_install_dependencies()
modules_loaded = load_all_qsar_modules()

print(f"ComfyQSAR Initialization Status: Modules Loaded - {'SUCCESS' if modules_loaded else 'FAILED'}")
print("======================================\n")

# --- Export Mappings ---
# Ensure __all__ is defined only once at the end
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
 