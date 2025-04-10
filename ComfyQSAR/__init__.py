"""
ComfyQSAR - QSAR 모델링을 위한 ComfyUI 확장
"""

import os
import traceback
import importlib.util
import sys
import subprocess
import time

# 경로 정의
QSAR_PATH = os.path.dirname(os.path.realpath(__file__))
PY_PATH = os.path.join(QSAR_PATH, "py")
WEB_DIRECTORY = "web"
# REQUIREMENTS_PATH = "requirements.txt"  # 이전 코드 주석 처리 또는 삭제
REQUIREMENTS_PATH = os.path.join(QSAR_PATH, "requirements.txt") # 수정된 라인

# 종속성 설치 함수
def install_dependencies():
    if not os.path.exists(REQUIREMENTS_PATH):
        print(f"ComfyQSAR: requirements.txt 파일을 찾을 수 없습니다: {REQUIREMENTS_PATH}")
        return False
    
    print("ComfyQSAR: 필요한 종속성을 설치합니다. 잠시 기다려주세요...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_PATH])
        print("ComfyQSAR: 모든 종속성이 성공적으로 설치되었습니다.")
        time.sleep(1)  # 설치 완료 메시지를 읽을 시간을 줍니다
        return True
    except subprocess.CalledProcessError as e:
        print(f"ComfyQSAR: 종속성 설치 중 오류가 발생했습니다: {str(e)}")
        return False

# 필요한 종속성 확인
required_dependencies = [
    "numpy", "pandas", "scikit-learn", "rdkit", "matplotlib", 
    "seaborn", "joblib", "scipy", "padelpy", "statsmodels",
    "xgboost", "lightgbm", "catboost"
]

missing_dependencies = []
for dependency in required_dependencies:
    try:
        importlib.import_module(dependency)
    except ImportError:
        missing_dependencies.append(dependency)

# 누락된 종속성이 있으면 자동 설치
if missing_dependencies:
    print(f"ComfyQSAR: 필요한 종속성이 누락되었습니다: {', '.join(missing_dependencies)}")
    print("ComfyQSAR: 자동으로 requirements.txt를 설치합니다...")
    
    if install_dependencies():
        # 종속성 다시 확인
        still_missing = []
        for dependency in missing_dependencies:
            try:
                importlib.import_module(dependency)
            except ImportError:
                still_missing.append(dependency)
        
        if still_missing:
            print(f"ComfyQSAR: 설치 후에도 누락된 종속성이 있습니다: {', '.join(still_missing)}")
            print("ComfyQSAR: 일부 기능이 제대로 작동하지 않을 수 있습니다.")
        else:
            print("ComfyQSAR: 모든 종속성이 정상적으로 설치되었습니다.")
    else:
        print("ComfyQSAR: 자동 설치에 실패했습니다. 수동으로 설치해주세요:")
        print(f"pip install -r {REQUIREMENTS_PATH}")

# 노드 매핑 초기화
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 파이썬 모듈 로딩
def load_modules():
    try:
        # ShowText 노드 로드
        sys.path.append(PY_PATH)
        from show_text import ShowText
        NODE_CLASS_MAPPINGS["ShowText"] = ShowText
        NODE_DISPLAY_NAME_MAPPINGS["ShowText"] = "Show Text"
        
        # Regression 모듈 로드
        regression_path = os.path.join(PY_PATH, "ComfyQSAR_Regression")
        regression_modules = [f[:-3] for f in os.listdir(regression_path) 
                            if f.endswith(".py") and not f.startswith("__")]
        
        for module_name in regression_modules:
            try:
                module_path = os.path.join(regression_path, f"{module_name}.py")
                spec = importlib.util.spec_from_file_location(f"ComfyQSAR_Regression.{module_name}", module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, "NODE_CLASS_MAPPINGS"):
                    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
            except Exception as e:
                print(f"ComfyQSAR: Regression 모듈 '{module_name}' 로드 중 오류: {str(e)}")
                traceback.print_exc()
        
        # Classification 모듈 로드
        classification_path = os.path.join(PY_PATH, "ComfyQSAR_Classification")
        classification_modules = [f[:-3] for f in os.listdir(classification_path) 
                                if f.endswith(".py") and not f.startswith("__")]
        
        for module_name in classification_modules:
            try:
                module_path = os.path.join(classification_path, f"{module_name}.py")
                spec = importlib.util.spec_from_file_location(f"ComfyQSAR_Classification.{module_name}", module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, "NODE_CLASS_MAPPINGS"):
                    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
            except Exception as e:
                print(f"ComfyQSAR: Classification 모듈 '{module_name}' 로드 중 오류: {str(e)}")
                traceback.print_exc()
        
        print(f"ComfyQSAR: {len(NODE_CLASS_MAPPINGS)} 노드가 성공적으로 로드되었습니다.")
        return True
    except Exception as e:
        print(f"ComfyQSAR: 모듈 로드 중 오류 발생: {str(e)}")
        traceback.print_exc()
        return False

# 웹 파일 복사 및 모듈 로드 실행
print("\n=== ComfyQSAR 확장 초기화 중... ===")
modules_loaded = load_modules()

print(f"ComfyQSAR 초기화 완료: 모듈 로드 {'성공' if modules_loaded else '실패'}")
print("===================================\n")

# API 노출
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"] 