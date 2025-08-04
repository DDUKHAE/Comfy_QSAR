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
WEB_DIRECTORY = "./web"
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
        
# --- Export Mappings ---
# Ensure __all__ is defined only once at the end
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"] 