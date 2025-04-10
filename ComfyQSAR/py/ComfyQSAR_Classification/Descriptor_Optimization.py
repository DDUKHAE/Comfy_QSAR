import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from .Descriptor_Preprocessing import create_text_container
class Remove_Low_Variance_Features_Classification:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_file": ("STRING",),
                "threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("OUTPUT_FILE",)
    FUNCTION = "remove_low_variance"
    CATEGORY = "QSAR/CLASSIFICATION/OPTIMIZATION"
    OUTPUT_NODE = True
    
    def remove_low_variance(self, input_file, threshold=0.05):
        """
        Remove low variance features from a dataset.
        """
        # 출력 디렉토리 생성
        os.makedirs("QSAR/Optimization", exist_ok=True)

        # 데이터 로드
        df = pd.read_csv(input_file)

        # Label 열 분리
        if "Label" not in df.columns:
            raise ValueError("The dataset must contain a 'Label' column.")

        target_column = df["Label"]
        feature_columns = df.drop(columns=["Label"])
        
        # 무한값 및 큰 값 처리
        # inf 값을 NaN으로 변환 후 NaN 값을 중앙값으로 대체
        feature_columns = feature_columns.replace([np.inf, -np.inf], np.nan)
        for col in feature_columns.columns:
            if feature_columns[col].isnull().any():
                median_val = feature_columns[col].median()
                feature_columns[col] = feature_columns[col].fillna(median_val)

        # 저분산 특성 제거
        selector = VarianceThreshold(threshold=threshold)
        selected_features = selector.fit_transform(feature_columns)
        
        # 남은 열 이름 가져오기
        retained_columns = feature_columns.columns[selector.get_support()]

        # 선택된 특성으로 새 DataFrame 생성
        df_retained = pd.DataFrame(selected_features, columns=retained_columns)
        df_retained["Label"] = target_column

        # 파일명 동적 생성
        initial_count = feature_columns.shape[1]
        final_count = len(retained_columns)
        output_file = os.path.join("QSAR/Optimization", f"low_variance_results_({initial_count}_{final_count}).csv")
        
        # 저장
        df_retained.to_csv(output_file, index=False)

        # 로그 메시지
        text_container = create_text_container(
            "🔹 Low Variance Features Removed! 🔹",
            f"📊 Initial Features: {initial_count}",
            f"📉 Remaining Features: {final_count}",
            f"🗑️ Removed: {initial_count - final_count}",
        )

        return {
            "ui": {"text": text_container},
            "result": (str(output_file),)
        }

class Remove_High_Correlation_Features_Classification:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_file": ("STRING",),
                "threshold": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01}),
                "mode": (["target_based", "upper", "lower"],),
                "importance_model": (["lasso", "random_forest"],)
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("OUTPUT_FILE",)
    FUNCTION = "remove_high_correlation"
    CATEGORY = "QSAR/CLASSIFICATION/OPTIMIZATION"
    OUTPUT_NODE = True
    
    def remove_high_correlation(self, input_file, threshold=0.95, mode="target_based", importance_model="lasso"):
        """
        Remove highly correlated features from a classification dataset while preserving the most informative ones.
        """
        # 출력 디렉토리 생성
        os.makedirs("QSAR/Optimization", exist_ok=True)
        
        # 데이터 로드
        df = pd.read_csv(input_file)

        # Label 열 체크
        if "Label" not in df.columns:
            raise ValueError("The dataset must contain a 'Label' column.")

        # 라벨 및 이름 열 분리
        target_column = df["Label"]
        feature_columns = df.drop(columns=["Label"])
        
        # 무한값 및 큰 값 처리
        # inf 값을 NaN으로 변환 후 NaN 값을 중앙값으로 대체
        feature_columns = feature_columns.replace([np.inf, -np.inf], np.nan)
        for col in feature_columns.columns:
            if feature_columns[col].isnull().any():
                median_val = feature_columns[col].median()
                feature_columns[col] = feature_columns[col].fillna(median_val)

        # 상관 행렬 계산
        correlation_matrix = feature_columns.corr()
        to_remove = set()

        if mode == "target_based":
            # 타겟 변수와의 상관관계 계산
            feature_target_corr = feature_columns.corrwith(target_column).abs()
            
            # 특성 중요도 계산 (지정된 경우)
            feature_importance = {}

            if importance_model in ["lasso", "random_forest"]:
                X, y = feature_columns, target_column

                if importance_model == "lasso":
                    # 데이터 스케일링 전처리 추가
                    scaler = StandardScaler()
                    try:
                        # 추가 전처리: 무한값과 NaN을 제거한 후 스케일링
                        X_copy = X.copy()
                        X_copy = X_copy.replace([np.inf, -np.inf], np.nan)
                        for col in X_copy.columns:
                            if X_copy[col].isnull().any():
                                median_val = X_copy[col].median()
                                X_copy[col] = X_copy[col].fillna(median_val)
                        
                        X_scaled = scaler.fit_transform(X_copy)
                        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
                        
                        # 1단계: 기본 파라미터로 시도
                        print("LASSO 1단계: 기본 파라미터로 학습 시도...")
                        model = Lasso(random_state=42)
                        model.fit(X_scaled_df, y)
                        importance_values = np.abs(model.coef_)
                        print("✅ LASSO 1단계 학습 성공!")
                    except Exception as e:
                        print(f"LASSO 1단계 학습 중 오류 발생: {e}")
                        try:
                            # 2단계: alpha 0.1, max_iter 10,000으로 시도
                            print("LASSO 2단계: alpha=0.1, max_iter=10,000으로 학습 시도...")
                            model = Lasso(alpha=0.1, max_iter=10000, random_state=42)
                            model.fit(X_scaled_df, y)
                            importance_values = np.abs(model.coef_)
                            print("✅ LASSO 2단계 학습 성공!")
                        except Exception as e2:
                            print(f"LASSO 2단계 학습 중 오류 발생: {e2}")
                            try:
                                # 3단계: alpha 1.0, max_iter 20,000으로 시도
                                print("LASSO 3단계: alpha=1.0, max_iter=20,000으로 학습 시도...")
                                model = Lasso(alpha=1.0, max_iter=20000, random_state=42)
                                model.fit(X_scaled_df, y)
                                importance_values = np.abs(model.coef_)
                                print("✅ LASSO 3단계 학습 성공!")
                            except Exception as e3:
                                print(f"LASSO 3단계 학습 중 오류 발생: {e3}")
                                # 오류 발생 시 RandomForest로 대체
                                try:
                                    print("대체 모델: RandomForest 사용...")
                                    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
                                    model.fit(X, y)
                                    importance_values = model.feature_importances_
                                    print("✅ RandomForest 대체 모델 학습 성공!")
                                except Exception as e4:
                                    print(f"RandomForest 대체 모델도 실패: {e4}")
                                    # 모든 시도가 실패하면 동일한 중요도 부여
                                    print("⚠️ 모든 모델 학습 실패. 동일한 특성 중요도를 사용합니다.")
                                    importance_values = np.ones(X.shape[1]) / X.shape[1]
                else:
                    # RandomForest의 경우 범위가 크거나 무한인 값을 처리
                    model = RandomForestClassifier(n_estimators=200, random_state=42)
                    try:
                        model.fit(X, y)
                        importance_values = model.feature_importances_
                    except Exception as e:
                        print(f"RandomForest 모델 학습 중 오류 발생: {e}")
                        # 추가적인 스케일링 시도
                        try:
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)
                            X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
                            model.fit(X_scaled_df, y)
                            importance_values = model.feature_importances_
                        except Exception as e2:
                            print(f"스케일링 후에도 오류 발생: {e2}")
                            # 기본값으로 모든 특성에 동일한 중요도 부여
                            importance_values = np.ones(X.shape[1]) / X.shape[1]
                
                feature_importance = dict(zip(feature_columns.columns, importance_values))

                
            # 높은 상관관계를 가진 쌍 찾기
            rows, cols = np.where(np.abs(np.triu(correlation_matrix, k=1)) > threshold)
            for row, col in zip(rows, cols):
                f1 = correlation_matrix.columns[row]
                f2 = correlation_matrix.columns[col]

                # 타겟 변수와의 상관관계 비교
                if feature_target_corr[f1] > feature_target_corr[f2]:
                    weaker = f2
                elif feature_target_corr[f1] < feature_target_corr[f2]:
                    weaker = f1
                else:
                    # 같은 경우 특성 중요도 사용 (가능한 경우)
                    weaker = f2 if feature_importance.get(f1, 0) > feature_importance.get(f2, 0) else f1

                to_remove.add(weaker)

        else:
            # "upper" 또는 "lower" 모드 사용
            tri = np.triu(correlation_matrix, k=1) if mode == "upper" else np.tril(correlation_matrix, k=-1)
            rows, cols = np.where(np.abs(tri) > threshold)
            for row, col in zip(rows, cols):
                f1 = correlation_matrix.columns[row]
                f2 = correlation_matrix.columns[col]
                to_remove.add(f2 if mode == "upper" else f1)

        # 제거되지 않은 열만 유지
        retained_columns = [c for c in feature_columns.columns if c not in to_remove]
        df_retained = feature_columns[retained_columns]
        df_retained["Label"] = target_column

        # 파일명 동적 생성
        initial_count = feature_columns.shape[1]
        final_count = len(retained_columns)
        output_file = os.path.join("QSAR/Optimization", f"high_correlation_results_({initial_count}_{final_count}).csv")
        
        # 저장
        df_retained.to_csv(output_file, index=False)

        # 로그 메시지
        text_container = create_text_container(
            "🔹 High Correlation Features Removed! 🔹",
            f"📊 Initial Features: {initial_count}",
            f"📉 Remaining Features: {final_count}",
            f"🗑️ Removed: {initial_count - final_count}",
            f"🔧 Mode: {mode}, Model: {importance_model if mode=='target_based' else 'N/A'}",
        )

        return {
            "ui": {"text": text_container},
            "result": (str(output_file),)
        }

class Descriptor_Optimization_Classification:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_file": ("STRING",),
                "variance_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "correlation_threshold": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01}),
                "correlation_mode": (["target_based", "upper", "lower"],),
                "importance_model": (["lasso", "random_forest"],)
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("OUTPUT_FILE",)
    FUNCTION = "optimize_descriptors"
    CATEGORY = "QSAR/CLASSIFICATION/OPTIMIZATION"
    OUTPUT_NODE = True
    
    def optimize_descriptors(self, input_file, variance_threshold=0.05, 
                             correlation_threshold=0.95, correlation_mode="target_based", 
                             importance_model="lasso"):
        """
        Complete descriptor optimization pipeline:
        1. Remove low variance features
        2. Remove highly correlated features
        """
        # 출력 디렉토리 생성
        os.makedirs("QSAR/Optimization", exist_ok=True)
        
        # 데이터에 inf 값이 있는지 확인하고 처리
        try:
            df = pd.read_csv(input_file)
            if "Label" in df.columns:
                target_column = df["Label"].copy()
                feature_columns = df.drop(columns=["Label"])
                
                # inf 값을 NaN으로 변환 후 중앙값으로 대체
                feature_columns = feature_columns.replace([np.inf, -np.inf], np.nan)
                for col in feature_columns.columns:
                    if feature_columns[col].isnull().any():
                        median_val = feature_columns[col].median()
                        feature_columns[col] = feature_columns[col].fillna(median_val)
                
                # 처리된 데이터 다시 저장
                processed_df = feature_columns.copy()
                processed_df["Label"] = target_column
                
                temp_file = os.path.join("QSAR/Optimization", "temp_preprocessed.csv")
                processed_df.to_csv(temp_file, index=False)
                
                # 전처리된 파일을 입력으로 사용
                input_file = temp_file
        except Exception as e:
            print(f"데이터 전처리 중 오류 발생: {e}")
            # 오류 발생 시 원본 파일 그대로 사용
        
        # 1. 저분산 특성 제거
        variance_remover = Remove_Low_Variance_Features_Classification()
        variance_result = variance_remover.remove_low_variance(input_file, threshold=variance_threshold)
        variance_output = variance_result["result"][0]
        
        # 2. 고상관 특성 제거
        correlation_remover = Remove_High_Correlation_Features_Classification()
        correlation_result = correlation_remover.remove_high_correlation(
            variance_output, threshold=correlation_threshold, 
            mode=correlation_mode, importance_model=importance_model
        )
        final_output = correlation_result["result"][0]
        
        # 최종 데이터 로드
        final_data = pd.read_csv(final_output)
        
        # 초기 데이터 로드 (통계 비교용)
        initial_data = pd.read_csv(input_file)
        
        # 임시 파일 삭제
        if os.path.exists(os.path.join("QSAR/Optimization", "temp_preprocessed.csv")):
            try:
                os.remove(os.path.join("QSAR/Optimization", "temp_preprocessed.csv"))
            except:
                pass
        
        # 특성 수 계산
        initial_features = initial_data.shape[1] - (1 if "Label" in initial_data.columns else 0) - (1 if "Name" in initial_data.columns else 0)
        final_features = final_data.shape[1] - (1 if "Label" in final_data.columns else 0) - (1 if "Name" in final_data.columns else 0)
        
        # 로그 메시지
        text_container = create_text_container(
            "🔹 Complete Descriptor Optimization Done! 🔹",
            f"📊 Initial Features: {initial_features}",
            f"📉 Final Features: {final_features}",
            f"🗑️ Total Removed: {initial_features - final_features} ({(initial_features - final_features) / initial_features * 100:.1f}%)",
            f"🔧 Optimization Pipeline:",
            f"   1. Removed low variance features (threshold: {variance_threshold})",
            f"   2. Removed highly correlated features (threshold: {correlation_threshold})",
            f"      Mode: {correlation_mode}, Model: {importance_model if correlation_mode=='target_based' else 'N/A'}",
            )

        return {
            "ui": {"text": text_container},
            "result": (str(final_output),)
        }

# 노드 등록
NODE_CLASS_MAPPINGS = {
    "Remove_Low_Variance_Features_Classification": Remove_Low_Variance_Features_Classification,
    "Remove_High_Correlation_Features_Classification": Remove_High_Correlation_Features_Classification,
    "Descriptor_Optimization_Classification": Descriptor_Optimization_Classification
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Remove_Low_Variance_Features_Classification": "Remove Low Variance Features(Classification)",
    "Remove_High_Correlation_Features_Classification": "Remove High Correlation Features(Classification)",
    "Descriptor_Optimization_Classification": "Descriptor Optimization(Classification)"
} 