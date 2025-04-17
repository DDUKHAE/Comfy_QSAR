import os
import pandas as pd
from padelpy import padeldescriptor
# from server import PromptServer # send_progress에서 사용하므로 여기서 직접 임포트 불필요
import json

# --- Common Utility Import ---
from server import PromptServer
import time # time 모듈 추가

# WebSocket 이벤트 이름 정의 (모든 QSAR 노드에서 공통 사용)
QSAR_PROGRESS_EVENT = "qsar-desc-calc-progress" # 이름을 좀 더 범용적으로 변경 (선택적)
# 또는 기존 이름 유지: QSAR_DESC_CALC_PROGRESS_EVENT = "qsar-desc-calc-progress"

def send_progress(message, progress=None, node_id=None):
    """
    지정된 메시지와 진행률(0-100)을 WebSocket을 통해 프론트엔드로 전송하고,
    중간 단계 업데이트 시 짧은 지연 시간을 추가하여 UI에서 볼 수 있도록 합니다.
    Args:
        message (str): 표시할 상태 메시지.
        progress (Optional[float]): 0부터 100 사이의 진행률 값.
        node_id (Optional[str]): 특정 노드를 대상으로 할 경우 노드 ID.
    """
    payload = {"text": [message]}
    is_intermediate_update = False # 중간 업데이트 여부 플래그

    if progress is not None:
        # 진행률 값을 0과 100 사이로 제한하고 소수점 첫째 자리까지 반올림 (선택적)
        clamped_progress = max(0.0, min(100.0, float(progress)))
        payload['progress'] = round(clamped_progress, 1)
        # 100%가 아닌 진행률 업데이트인지 확인
        if clamped_progress < 100:
            is_intermediate_update = True

    # node ID 추가 (프론트엔드에서 필터링 시 사용 가능)
    # if node_id: payload['node'] = node_id

    try:
        # PromptServer 인스턴스를 통해 동기적으로 메시지 전송
        PromptServer.instance.send_sync(QSAR_PROGRESS_EVENT, payload)

        # 중간 진행률 업데이트 후 짧은 지연 시간 추가 (0.2초)
        # 최종(100%) 업데이트 시에는 지연 없음
        if is_intermediate_update:
            time.sleep(0.2) # 0.2초 대기

    except Exception as e:
        print(f"[ComfyQSAR Progress Util] WebSocket 전송 오류: {e}")

# 필요에 따라 다른 유틸리티 함수 추가 가능 (예: 시간 포맷팅 등) 

# 텍스트 컨테이너 생성 헬퍼 함수
def create_text_container(*lines):
    # 가장 긴 라인을 기준으로 구분선 길이 결정
    max_length = max(len(line) for line in lines)
    separator = "=" * max_length
    
    # 첫 구분선 추가
    result = [separator]
    
    # 각 라인 추가
    for line in lines:
        result.append(line)
    
    # 마지막 구분선 추가
    result.append(separator)
    
    # 줄바꿈으로 조인
    return "\n".join(result)

# --- Helper Functions ---
# send_progress 함수 정의 제거

class Descriptor_Calculations_Classification:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive_path": ("STRING",),
                "negative_path": ("STRING",),
                "advanced": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "descriptor_type": (["2D", "3D"], {"default": "2D"}),
                "detect_aromaticity": ("BOOLEAN", {"default": True}),
                "remove_salt": ("BOOLEAN", {"default": True}),
                "standardize_nitro": ("BOOLEAN", {"default": True}),
                "use_file_name_as_molname": ("BOOLEAN", {"default": True}),
                "retain_order": ("BOOLEAN", {"default": True}),
                "threads": ("INT", {"default": -1, "min": -1, "max": 64, "step": 1}),
                "waiting_jobs": ("INT", {"default": -1, "min": -1, "max": 64, "step": 1}),
                "max_runtime": ("INT", {"default": 10000, "min": 1000, "max": 100000, "step": 1000}),
                "max_cpd_per_file": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1000}),
                "headless": ("BOOLEAN", {"default": True}),
                "log": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("DESCRIPTORS_PATH",)
    FUNCTION = "calculate_and_merge_descriptors"
    CATEGORY = "QSAR/CLASSIFICATION/CALCULATION"
    OUTPUT_NODE = True # UI 출력을 위해 True 유지

    def calculate_and_merge_descriptors(self, positive_path, negative_path, descriptor_type, detect_aromaticity, remove_salt, standardize_nitro, use_file_name_as_molname, retain_order, threads, waiting_jobs, max_runtime, max_cpd_per_file, headless, log, advanced):

        # --- 전체 단계별 진행률 할당 (예시) ---
        PREPARE_PCT = 5
        PROCESS_POS_PCT = 40
        PROCESS_NEG_PCT = 40
        MERGE_SAVE_PCT = 15
        # 합계 = 100

        current_progress = 0
        # 이제 공통 유틸리티 함수 사용
        send_progress("🚀 Starting descriptor calculation...", current_progress)

        # --- 1. 초기 설정 및 디렉토리 생성 ---
        output_dir = "QSAR/Descriptor_Calculation"
        os.makedirs(output_dir, exist_ok=True)
        current_progress = PREPARE_PCT
        send_progress(f"📂 Output directory prepared: {output_dir}", current_progress)

        # PaDEL 옵션 딕셔너리 생성
        padel_options = {
            "d_2d": descriptor_type == "2D",
            "d_3d": descriptor_type == "3D",
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

        # --- 2. 내부 파일 처리 함수 정의 ---
        def process_file(input_path, tag, label, progress_start, progress_total_allocation):
            """주어진 파일을 처리하고 해당 단계 내에서의 진행률 업데이트"""

            # 단계별 예상 가중치
            READ_CHECK_WEIGHT = 10
            PADEL_WEIGHT = 80
            LABEL_SAVE_WEIGHT = 10

            def update_sub_progress(sub_step_progress, message):
                """전체 진행률 기준으로 업데이트"""
                new_progress = progress_start + (sub_step_progress / 100.0) * progress_total_allocation
                # 공통 유틸리티 함수 사용
                send_progress(message, new_progress)


            base_message = f"⏳ Processing {tag} file: {os.path.basename(input_path)}..."
            # update_sub_progress(0, base_message)

            mol_dir_for_padel = input_path
            smiles_path = None
            desc_file = os.path.join(output_dir, f"{tag}_descriptors.csv")

            try:
                # 1. 파일 읽기 및 확인
                sub_progress = 0
                if input_path.endswith('.sdf'):
                    update_sub_progress(sub_progress + READ_CHECK_WEIGHT * 0.5, f"   Format: SDF. Using directly.")
                    mol_dir_for_padel = input_path
                elif input_path.endswith(('.csv', '.smi')):
                    update_sub_progress(sub_progress + READ_CHECK_WEIGHT * 0.2, f"   Format: {input_path.split('.')[-1]}. Reading SMILES...")
                    try:
                        df = pd.read_csv(input_path)
                        if "SMILES" not in df.columns: raise ValueError(f"❌ SMILES column not found: {input_path}")
                    except Exception as read_e: raise ValueError(f"❌ Error reading file {input_path}: {read_e}")
                    smiles_path = os.path.join(output_dir, f"{tag}_smiles_temp.smi")
                    df[["SMILES"]].to_csv(smiles_path, index=False, header=False)
                    update_sub_progress(sub_progress + READ_CHECK_WEIGHT, f"   Created temporary SMILES file: {smiles_path}")
                    mol_dir_for_padel = smiles_path
                else: raise ValueError(f"Unsupported file format: {input_path}")
                sub_progress += READ_CHECK_WEIGHT

                # 2. PaDEL 실행
                update_sub_progress(sub_progress + PADEL_WEIGHT * 0.05, f"   Running PaDEL-Descriptor for {tag}...")
                padeldescriptor(mol_dir=mol_dir_for_padel, d_file=desc_file, **padel_options)
                sub_progress += PADEL_WEIGHT
                update_sub_progress(sub_progress, f"   PaDEL finished for {tag}. Output: {desc_file}")

                if not os.path.exists(desc_file): raise FileNotFoundError(f"❌ Descriptor file not created: {desc_file}")

                # 3. 라벨 추가 및 후처리
                df_desc = pd.read_csv(desc_file)
                df_desc["Label"] = label
                sub_progress += LABEL_SAVE_WEIGHT * 0.5
                update_sub_progress(sub_progress, f"   Added 'Label' column ({label}) to {tag} descriptors.")
                update_sub_progress(100, f"   Finished processing {tag} file.")
                return df_desc

            finally:
                if smiles_path and os.path.exists(smiles_path):
                    os.remove(smiles_path)
                    # send_progress 호출 제거 (이미 완료 메시지 전송됨)

        # --- 3. 메인 처리 로직 ---
        try:
            # Positive 데이터 처리
            start_pos_progress = current_progress
            df_positive = process_file(positive_path, "positive", label=1, progress_start=start_pos_progress, progress_total_allocation=PROCESS_POS_PCT)
            current_progress = start_pos_progress + PROCESS_POS_PCT
            send_progress(f"✅ Positive file processing complete. {len(df_positive)} molecules.", current_progress)

            # Negative 데이터 처리
            start_neg_progress = current_progress
            df_negative = process_file(negative_path, "negative", label=0, progress_start=start_neg_progress, progress_total_allocation=PROCESS_NEG_PCT)
            current_progress = start_neg_progress + PROCESS_NEG_PCT
            send_progress(f"✅ Negative file processing complete. {len(df_negative)} molecules.", current_progress)

            # 결과 병합 및 저장
            merge_start_progress = current_progress
            merge_progress_allocation = 100 - merge_start_progress

            send_progress("🔗 Merging positive and negative sets...", merge_start_progress + merge_progress_allocation * 0.2)
            df_final = pd.concat([df_positive, df_negative], ignore_index=True)
            send_progress("💾 Saving final merged descriptors...", merge_start_progress + merge_progress_allocation * 0.6)
            final_file = os.path.join(output_dir, "final_merged_descriptors.csv")
            df_final.to_csv(final_file, index=False)
            current_progress = 98
            send_progress(f"📊 Total molecules processed: {len(df_final)}. File saved to: {final_file}", current_progress)

            # 최종 결과 텍스트 생성
            text_container = create_text_container(
                    "🔹 **Descriptor Calculation & Merge Done!** 🔹",
                    f"✅ Positive Molecules: {len(df_positive)}",
                    f"✅ Negative Molecules: {len(df_negative)}",
                    f"📊 Total Molecules: {len(df_final)}",
                    "📂 Format: descriptors + Label column (1=positive, 0=negative)",
                    f"💾 Output File: {final_file}"
                )

            # 완료 메시지 (100%)
            current_progress = 100
            send_progress("🎉 Calculation and merge finished successfully!", current_progress)

            # 최종 결과 반환
            return { "ui": {"text": text_container}, "result": (str(final_file),) }

        except Exception as e:
            error_message = f"❌ **Descriptor Calculation Failed!** ❌\nError: {str(e)}"
            send_progress(error_message) # 실패 시 progress 없이
            error_container = create_text_container(
                "❌ **Descriptor Calculation Failed!** ❌",
                f"Error: {str(e)}"
            )
            return { "ui": {"text": error_container}, "result": (str(""),) }


# 노드 등록 (기존과 동일)
NODE_CLASS_MAPPINGS = {
    "Descriptor_Calculations_Classification": Descriptor_Calculations_Classification
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Descriptor_Calculations_Classification": "Descriptor Calculation(Classification)"
} 