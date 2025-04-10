# ComfyQSAR

ComfyQSAR는 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 환경에서 정량적 구조-활성 관계(Quantitative Structure-Activity Relationship, QSAR) 모델링을 수행하기 위한 커스텀 노드 확장입니다. 화학 구조 데이터로부터 분자 기술자를 계산하고, 데이터를 전처리하며, 머신러닝 모델을 훈련, 평가, 최적화하는 워크플로우를 ComfyUI의 시각적 인터페이스 내에서 구축할 수 있습니다.

## 기능

*   **모듈식 워크플로우:** 데이터 로딩, 기술자 계산, 전처리, 모델 훈련, 하이퍼파라미터 튜닝 등 QSAR 모델링의 각 단계를 개별 노드로 제공하여 유연한 워크플로우 구성이 가능합니다.
*   **다양한 머신러닝 모델 지원:** Regression 및 Classification 작업 모두에 대해 다양한 알고리즘(예: XGBoost, LightGBM, RandomForest, SVM, Linear Models 등)을 지원합니다.
*   **자동 종속성 관리:** 필요한 파이썬 라이브러리를 자동으로 감지하고 설치를 시도합니다. (`requirements.txt`)
*   **ComfyUI 통합:** ComfyUI의 노드 기반 인터페이스를 활용하여 직관적인 모델링 파이프라인 구축 및 실행이 가능합니다.

## 설치

1.  **ComfyUI 설치:** 아직 ComfyUI를 설치하지 않았다면, [ComfyUI GitHub 리포지토리](https://github.com/comfyanonymous/ComfyUI)의 지침에 따라 설치합니다.
2.  **ComfyQSAR 다운로드:**
    *   이 리포지토리를 ComfyUI의 `custom_nodes` 디렉토리 안에 클론합니다.
        ```bash
        cd ComfyUI/custom_nodes/
        git clone <ComfyQSAR 리포지토리 주소> ComfyQSAR
        ```
    *   또는, 리포지토리의 zip 파일을 다운로드하고 압축을 해제하여 `ComfyQSAR` 폴더를 `ComfyUI/custom_nodes/` 디렉토리 안에 위치시킵니다.
3.  **종속성 설치:**
    *   ComfyUI를 시작합니다. ComfyQSAR가 처음 로드될 때 필요한 파이썬 라이브러리(`requirements.txt` 파일 기준)를 자동으로 확인하고 설치를 시도합니다.
    *   자동 설치에 실패할 경우, 터미널에서 다음 명령어를 직접 실행하여 수동으로 설치할 수 있습니다:
        ```bash
        # ComfyUI 가상환경 활성화 (필요한 경우)
        cd ComfyUI/custom_nodes/ComfyQSAR/
        pip install -r requirements.txt
        ```
4.  **ComfyUI 재시작:** 종속성 설치 후 ComfyUI를 재시작합니다.

## 노드 설명

ComfyQSAR는 **Regression**과 **Classification** 작업을 위한 노드 그룹을 제공합니다. 각 노드는 특정 QSAR 모델링 단계를 수행하며, 입력과 출력을 연결하여 전체 워크플로우를 구성합니다.

### 공통 노드

*   **Show Text (`ShowText`)**:
    *   **기능:** 입력된 텍스트 데이터를 ComfyUI 인터페이스에 표시합니다. 주로 다른 노드의 결과(예: 데이터 로딩 상태, 모델 평가 결과 요약)를 확인하거나 디버깅 목적으로 사용됩니다.
    *   **입력:** `text` (STRING) - 표시할 텍스트 내용.
    *   **출력:** 없음. (OUTPUT_NODE = True)

### ComfyQSAR Regression 노드 (QSAR/REGRESSION)

연속적인 값(예: 화합물의 IC50, logP 등)을 예측하는 Regression 모델링을 위한 노드들입니다.

#### LOAD & STANDARDIZATION

*   **Data Loader (Regression) (`Data_Loader_Regression`)**:
    *   **기능:** SMILES 파일과 해당 분자의 생물학적 활성값 파일을 로드하여 하나의 데이터셋으로 결합합니다.
    *   **입력:** `smiles_file_path` (STRING), `biological_value_file_path` (STRING) - 각각 SMILES와 활성값 파일 경로 (.tsv 또는 .csv 형태).
    *   **출력:** `DATA` (STRING) - 결합된 데이터 파일 경로 (CSV).
*   **Standardization (Regression) (`Standardization_Regression`)**:
    *   **기능:** 입력된 데이터셋에서 RDKit을 사용하여 분자를 표준화합니다. 유효하지 않은 분자, 금속 이온만 포함된 분자, 다중 조각 구조를 필터링합니다.
    *   **입력:** `data` (STRING) - 'SMILES'와 'value' 컬럼을 포함하는 CSV 파일 경로.
    *   **출력:** `DATA` (STRING) - 표준화되고 필터링된 데이터 파일 경로 (CSV).
*   **Load and Standardize (Regression) (`Load_and_Standardize_Regression`)**:
    *   **기능:** `Data Loader`와 `Standardization` 노드의 기능을 통합하여, SMILES/활성값 파일을 로드하고 바로 표준화/필터링을 수행합니다.
    *   **입력:** `smiles_file_path` (STRING), `biological_value_file_path` (STRING).
    *   **출력:** `DATA` (STRING) - 로드되고 표준화된 데이터 파일 경로 (CSV).

#### CALCULATION

*   **Descriptor Calculation (Regression) (`Descriptor_Calculations_Regression`)**:
    *   **기능:** 입력된 데이터의 SMILES 문자열로부터 PaDEL-Descriptor를 사용하여 분자 기술자(2D 또는 3D)를 계산하고, 원본 데이터(value 포함)와 병합합니다.
    *   **입력:** `filtered_data` (STRING) - 'SMILES'와 'value' 컬럼을 포함하는 CSV 파일 경로. `advanced` (BOOLEAN) 및 다양한 PaDEL 옵션 (선택 사항).
    *   **출력:** `DESCRIPTOR` (STRING) - 계산된 기술자와 value가 포함된 데이터 파일 경로 (CSV).

#### PREPROCESSING

*   **Replace inf with nan (Regression) (`Replace_inf_with_nan_Regression`)**:
    *   **기능:** 데이터셋 내의 무한대(inf, -inf) 값을 NaN(Not a Number)으로 대체합니다. 무한대 값이 포함된 컬럼 리포트를 생성할 수 있습니다.
    *   **입력:** `input_file` (STRING) - CSV 파일 경로.
    *   **출력:** `PREPROCESSED_DATA` (STRING) - 무한대 값이 처리된 데이터 파일 경로 (CSV).
*   **Remove high nan compounds (Regression) (`Remove_high_nan_compounds_Regression`)**:
    *   **기능:** 각 화합물(행)별로 NaN 값의 비율을 계산하여, 지정된 임계값(threshold)을 초과하는 화합물을 제거합니다.
    *   **입력:** `input_file` (STRING), `threshold` (FLOAT) - 허용할 NaN 비율 임계값 (0.0 ~ 1.0).
    *   **출력:** `PREPROCESSED_DATA` (STRING) - NaN 비율이 높은 화합물이 제거된 데이터 파일 경로 (CSV).
*   **Remove high nan descriptors (Regression) (`Remove_high_nan_descriptors_Regression`)**:
    *   **기능:** 각 기술자(열)별로 NaN 값의 비율을 계산하여, 지정된 임계값(threshold)을 초과하는 기술자를 제거합니다. 'SMILES'와 'value' 컬럼은 유지됩니다.
    *   **입력:** `input_file` (STRING), `threshold` (FLOAT).
    *   **출력:** `PREPROCESSED_DATA` (STRING) - NaN 비율이 높은 기술자가 제거된 데이터 파일 경로 (CSV).
*   **Impute missing values (Regression) (`Impute_missing_values_Regression`)**:
    *   **기능:** 데이터셋 내의 NaN 값을 지정된 전략(평균, 중앙값, 최빈값)으로 대체(impute)합니다. 'SMILES'와 'value' 컬럼은 제외하고 처리합니다.
    *   **입력:** `input_file` (STRING), `method` (STRING: "mean", "median", "most_frequent").
    *   **출력:** `PREPROCESSED_DATA` (STRING) - 결측치가 대체된 데이터 파일 경로 (CSV).
*   **Descriptor preprocessing (Regression) (`Descriptor_preprocessing_Regression`)**:
    *   **기능:** 위 전처리 단계(inf->NaN, high NaN 행/열 제거, 결측치 대체)를 통합하여 한 번에 수행합니다.
    *   **입력:** `input_file` (STRING), `compounds_nan_threshold` (FLOAT), `descriptors_nan_threshold` (FLOAT), `imputation_method` (STRING).
    *   **출력:** `INTEGRATED_PREPROCESSED_DATA` (STRING) - 모든 전처리 단계가 적용된 데이터 파일 경로 (CSV).

#### OPTIMIZATION

*   **Remove Low Variance Descriptors (Regression) (`Remove_Low_Variance_Descriptors_Regression`)**:
    *   **기능:** 분산이 낮은 기술자(거의 변화가 없는 특성)를 제거합니다. 분산 임계값(threshold)을 기준으로 제거 대상을 결정합니다.
    *   **입력:** `input_file` (STRING) - 'value' 컬럼 포함 CSV. `threshold` (FLOAT).
    *   **출력:** `DATA` (STRING) - 저분산 기술자가 제거된 데이터 파일 경로 (CSV).
*   **Remove High Correlation Features (Regression) (`Remove_High_Correlation_Features_Regression`)**:
    *   **기능:** 서로 상관관계가 높은 기술자 쌍 중에서 하나를 제거합니다. 제거 기준은 지정된 모드(`target_based`, `upper`, `lower`)와 중요도 모델(`lasso`, `random_forest`)에 따라 결정됩니다.
    *   **입력:** `input_file` (STRING), `threshold` (FLOAT) - 상관계수 임계값. `mode` (STRING), `importance_model` (STRING).
    *   **출력:** `DATA` (STRING) - 상관관계 높은 기술자가 제거된 데이터 파일 경로 (CSV).
*   **Descriptor Optimization (Regression) (`Descriptor_Optimization_Regression`)**:
    *   **기능:** 저분산 제거와 고상관 제거를 순차적으로 통합 수행합니다.
    *   **입력:** `input_file` (STRING), `variance_threshold` (FLOAT), `correlation_threshold` (FLOAT), `correlation_mode` (STRING), `importance_model` (STRING).
    *   **출력:** `OPTIMIZED_DATA` (STRING) - 최적화된 기술자 데이터 파일 경로 (CSV).

#### SELECTION

*   **Feature Selection (Regression) (`Feature_Selection_Regression`)**:
    *   **기능:** 다양한 기법(Lasso, RandomForest, DecisionTree, XGBoost, LightGBM, RFE, SelectFromModel)을 사용하여 모델 성능에 중요한 기술자를 선택합니다. 방법과 모델, 파라미터(n_features, threshold 등)를 지정할 수 있습니다.
    *   **입력:** `input_file` (STRING), `method` (STRING), `target_column` (STRING), `n_features` (INT), `threshold` (FLOAT or STRING), `model` (STRING), `advanced` (BOOLEAN) 및 알고리즘별 파라미터 (선택 사항).
    *   **출력:** `SELECTED_DESCRIPTORS` (STRING) - 선택된 기술자와 target 컬럼만 포함된 데이터 파일 경로 (CSV).

#### COMBINATION

*   **Get Best Descriptor Combinations RF (`Get_Best_Descriptor_Combinations_RF`)**:
    *   **기능:** RandomForestRegressor 모델을 사용하여 지정된 최대 특성 수(`max_features`)까지 모든 기술자 조합을 평가하고, R² 점수가 가장 높은 상위 N개(`top_n`) 조합과 각 크기별 최적 조합을 찾습니다. 멀티프로세싱을 지원합니다.
    *   **입력:** `input_csv` (STRING), `max_features` (INT), `num_cores` (INT), `top_n` (INT).
    *   **출력:** `DATA` (STRING) - 전체 조합 중 가장 성능이 좋은 기술자 조합 데이터 파일 경로 (CSV).
*   **Get Best Descriptor Combinations (`Get_Best_Descriptor_Combinations`)**:
    *   **기능:** LinearRegression 모델과 MinMaxScaler를 사용하여 기술자 조합을 평가합니다. (RF 버전과 유사하나 평가 모델과 스케일링 적용 여부가 다름)
    *   **입력:** `input_csv` (STRING), `max_features` (INT), `num_cores` (INT), `top_n` (INT).
    *   **출력:** `DATA` (STRING) - 최적 기술자 조합 데이터 파일 경로 (CSV).

#### GRID SEARCH

*   **Hyperparameter Grid Search (Regression) (`Hyperparameter_Grid_Search_Regression`)**:
    *   **기능:** 선택된 Regression 알고리즘(XGBoost, RF, DT, LightGBM, SVM, Ridge, Lasso, ElasticNet, LinearRegression)에 대해 지정된 하이퍼파라미터 조합으로 Grid Search를 수행하여 최적의 모델을 찾습니다. 데이터를 Train/Test로 분할하고, 교차 검증(Cross-validation)을 통해 성능을 평가합니다.
    *   **입력:** `input_file` (STRING), `algorithm` (STRING), `advanced` (BOOLEAN), `target_column` (STRING), `test_size` (FLOAT), `num_cores` (INT), `cv_splits` (INT), `verbose` (INT), `random_state` (INT), 그리고 알고리즘별 하이퍼파라미터 범위 (STRING 형태의 리스트, 예: "[100, 200]").
    *   **출력:** `MODEL_PATH` (STRING) - 최적 하이퍼파라미터로 훈련된 모델 파일 경로 (.joblib). `SELECTED_DESCRIPTORS_PATH` (STRING) - 사용된 기술자 목록 파일 경로. `X_TEST_PATH` (STRING), `Y_TEST_PATH` (STRING) - 분할된 테스트 데이터 파일 경로 (CSV).

#### VALIDATION

*   **Model Validation (Regression) (`Model_Validation_Regression`)**:
    *   **기능:** Grid Search 등으로 훈련된 모델을 테스트 데이터셋으로 평가합니다. R², MSE, RMSE, MAE, MAPE 등의 평가 지표를 계산하고, 실제값과 예측값을 비교하는 파일을 생성합니다.
    *   **입력:** `model_path` (STRING), `selected_descriptors` (STRING) - 사용할 기술자 목록 파일 경로. `X_test` (STRING), `y_test` (STRING) - 테스트 데이터 파일 경로.
    *   **출력:** `EVALUATION_PATH` (STRING) - 평가 결과 파일 경로 (CSV). `PREDICTION_PATH` (STRING) - 실제값 vs 예측값 파일 경로 (CSV).

### ComfyQSAR Classification 노드 (QSAR/CLASSIFICATION)

이산적인 범주(예: 활성/비활성, 독성/비독성)를 예측하는 Classification 모델링을 위한 노드들입니다. Regression 노드와 유사한 구조를 가지지만, 분류 문제에 특화된 기능을 제공합니다.

#### LOAD & STANDARDIZATION

*   **Data Loader (Classification) (`Data_Loader_Classification`)**:
    *   **기능:** 양성(Positive) 및 음성(Negative) 클래스에 해당하는 분자 데이터 파일(SMILES, CSV, SDF 형식 지원)을 각각 로드합니다.
    *   **입력:** `positive_file_path` (STRING), `negative_file_path` (STRING).
    *   **출력:** `POSITIVE_PATH` (STRING), `NEGATIVE_PATH` (STRING) - 원본 파일 경로를 그대로 전달. (UI에 로드된 분자 수 표시)
*   **Standardization (Classification) (`Standardization_Classification`)**:
    *   **기능:** 양성/음성 데이터 파일 각각에 대해 RDKit을 사용한 분자 표준화 및 필터링을 수행합니다. (Regression 버전과 필터링 로직 동일)
    *   **입력:** `positive_path` (STRING), `negative_path` (STRING).
    *   **출력:** `POSITIVE_PATH` (STRING), `NEGATIVE_PATH` (STRING) - 표준화/필터링된 데이터 파일 경로 (SDF 또는 CSV).
*   **Load and Standardize (Classification) (`Load_and_Standardize_Classification`)**:
    *   **기능:** `Data Loader`와 `Standardization` 기능을 통합하여 양성/음성 데이터 로딩 및 표준화/필터링을 한 번에 수행합니다.
    *   **입력:** `positive_file_path` (STRING), `negative_file_path` (STRING).
    *   **출력:** `POSITIVE_PATH` (STRING), `NEGATIVE_PATH` (STRING) - 로드되고 표준화된 데이터 파일 경로.

#### CALCULATION

*   **Descriptor Calculation (Classification) (`Descriptor_Calculations_Classification`)**:
    *   **기능:** 표준화된 양성/음성 데이터 각각에 대해 PaDEL-Descriptor를 사용하여 기술자를 계산하고, 두 데이터셋을 병합합니다. 병합된 데이터에는 'Label' 컬럼이 추가됩니다 (양성=1, 음성=0).
    *   **입력:** `positive_path` (STRING), `negative_path` (STRING). `advanced` (BOOLEAN) 및 PaDEL 옵션 (선택 사항).
    *   **출력:** `DESCRIPTORS_PATH` (STRING) - 양성/음성 기술자가 병합되고 'Label' 컬럼이 추가된 데이터 파일 경로 (CSV).

#### PREPROCESSING

*   **(Regression과 동일)** `Replace_inf_with_nan_Classification`, `Remove_high_nan_compounds_Classification`, `Remove_high_nan_descriptors_Classification`, `Impute_missing_values_Classification`, `Descriptor_preprocessing_Classification` 노드가 제공되며, Regression 버전과 기능은 거의 동일합니다. 다만, 처리 시 'Label' 컬럼을 고려합니다.

#### OPTIMIZATION

*   **(Regression과 동일)** `Remove_Low_Variance_Features_Classification`, `Remove_High_Correlation_Features_Classification`, `Descriptor_Optimization_Classification` 노드가 제공되며, Regression 버전과 기능은 거의 동일합니다. 다만, 처리 시 'Label' 컬럼을 타겟 변수로 고려합니다. ('target_based' 모드 등)

#### SELECTION

*   **Feature Selection (Classification) (`Feature_Selection_Classification`)**:
    *   **기능:** 다양한 분류 모델 기반 기법(Lasso(Logistic), RandomForest, DecisionTree, XGBoost, LightGBM, RFE, SelectFromModel)을 사용하여 분류 성능에 중요한 기술자를 선택합니다. 데이터 스케일링이 내부적으로 적용될 수 있습니다.
    *   **입력:** `input_file` (STRING), `method` (STRING), `target_column` (STRING, 기본값 "Label"), `n_features` (INT), `threshold` (STRING or FLOAT), `model` (STRING), `advanced` (BOOLEAN) 및 알고리즘별 파라미터 (선택 사항).
    *   **출력:** `OUTPUT_FILE` (STRING) - 선택된 기술자와 'Label' 컬럼만 포함된 데이터 파일 경로 (CSV).

#### COMBINATION

*   **Feature Combination Search (`Feature_Combination_Search`)**:
    *   **기능:** LogisticRegression 모델을 사용하여 지정된 최대 특성 수(`max_features`)까지 모든 기술자 조합을 평가하고, Accuracy 점수가 가장 높은 상위 N개(`top_n`) 조합과 각 크기별 최적 조합을 찾습니다. 멀티프로세싱을 지원합니다.
    *   **입력:** `input_file` (STRING), `max_features` (INT), `num_cores` (INT), `top_n` (INT).
    *   **출력:** `BEST_FEATURE_SET` (STRING) - 전체 조합 중 가장 성능이 좋은 기술자 조합 데이터 파일 경로 (CSV).

#### GRID SEARCH

*   **Hyperparameter Grid Search (Classification) (`Hyperparameter_Grid_Search_Classification`)**:
    *   **기능:** 선택된 Classification 알고리즘(XGBoost, RF, DT, LightGBM, Logistic, Lasso, SVM)에 대해 Grid Search를 수행하여 최적 모델을 찾습니다. 데이터를 Train/Test로 분할(Stratified)하고 교차 검증을 수행합니다.
    *   **입력:** `input_file` (STRING), `algorithm` (STRING), `advanced` (BOOLEAN), `target_column` (STRING, 기본값 "Label"), `test_size` (FLOAT), `num_cores` (INT), `cv_splits` (INT), `verbose` (INT), `random_state` (INT), 그리고 알고리즘별 하이퍼파라미터 범위 (STRING 형태 리스트).
    *   **출력:** `MODEL_PATH` (STRING) - 최적 모델 파일 경로 (.joblib). `DESCRIPTORS_PATH` (STRING) - 사용된 기술자 목록 파일 경로. `X_TEST_PATH` (STRING), `Y_TEST_PATH` (STRING) - 분할된 테스트 데이터 파일 경로 (CSV).

#### VALIDATION

*   **Model Validation (Classification) (`Model_Validation_Classification`)**:
    *   **기능:** 훈련된 분류 모델을 테스트 데이터셋으로 평가합니다. Accuracy, F1-Score, Precision, Recall(Sensitivity), Specificity, ROC-AUC 등의 지표를 계산하고, 실제값과 예측값을 비교하는 파일을 생성합니다.
    *   **입력:** `model_path` (STRING), `selected_descriptors_path` (STRING) - 사용할 기술자 목록 파일 경로. `X_test_path` (STRING), `y_test_path` (STRING).
    *   **출력:** `EVALUATION_PATH` (STRING) - 평가 결과 파일 경로 (CSV). `PREDICTION_PATH` (STRING) - 실제값 vs 예측값 (+예측 확률) 파일 경로 (CSV).

## 사용 예시

*(워크플로우 예시 이미지나 json 파일을 여기에 추가하면 좋습니다.)*

**Regression 워크플로우 예시:**

1.  `Load and Standardize (Regression)`: SMILES와 활성값 파일 로드 및 표준화.
2.  `Descriptor Calculation (Regression)`: 표준화된 데이터로 기술자 계산.
3.  `Descriptor preprocessing (Regression)`: 결측치 및 무한값 처리.
4.  `Descriptor Optimization (Regression)`: 저분산/고상관 기술자 제거.
5.  `Feature Selection (Regression)`: 중요 기술자 선택.
6.  `Hyperparameter Grid Search (Regression)`: 선택된 기술자로 모델 훈련 및 최적화.
7.  `Model Validation (Regression)`: 최적 모델을 테스트 데이터로 최종 평가.

**Classification 워크플로우 예시:**

1.  `Load and Standardize (Classification)`: 양성/음성 데이터 로드 및 표준화.
2.  `Descriptor Calculation (Classification)`: 기술자 계산 및 데이터 병합 (Label 추가).
3.  `Descriptor preprocessing (Classification)`: 전처리 수행.
4.  `Descriptor Optimization (Classification)`: 기술자 최적화.
5.  `Feature Selection (Classification)`: 중요 기술자 선택.
6.  `Hyperparameter Grid Search (Classification)`: 모델 훈련 및 최적화.
7.  `Model Validation (Classification)`: 최종 모델 평가.

## 기여

버그를 발견하거나 새로운 기능을 제안하고 싶으시면 GitHub 이슈를 열어주세요. 풀 리퀘스트도 환영합니다!

## 라이선스

*(프로젝트에 적용할 라이선스를 명시하세요. 예: MIT License)*
