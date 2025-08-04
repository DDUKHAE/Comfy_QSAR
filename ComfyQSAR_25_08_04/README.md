# ComfyQSAR

ComfyQSAR는 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 환경에서 정량적 구조-활성 관계(Quantitative Structure-Activity Relationship, QSAR) 모델링을 수행하기 위한 커스텀 노드 확장입니다. 화학 구조 데이터로부터 분자 기술자를 계산하고, 데이터를 전처리하며, 머신러닝 모델을 훈련, 평가, 최적화하는 워크플로우를 ComfyUI의 시각적 인터페이스 내에서 구축할 수 있습니다.

## 주요 기능

ComfyQSAR는 QSAR 모델링 프로세스를 위한 다양한 노드를 제공합니다.

### 모듈식 워크플로우 구축 (Regression & Classification)
![워크플로우 예시 이미지](images/workflow_example.png)
데이터 로딩부터 모델 평가까지, Regression 및 Classification 작업에 필요한 각 단계를 노드로 연결하여 유연하고 시각적인 QSAR 파이프라인을 구성할 수 있습니다.

### 데이터 로딩 및 표준화
![데이터 로딩 노드 이미지](images/data_loading_nodes.png)
SMILES, CSV, SDF 등 다양한 형식의 화학 데이터를 로드하고, RDKit을 이용한 표준화 과정을 통해 모델링에 적합한 형태로 데이터를 정제합니다.

### 강력한 분자 기술자 계산
![기술자 계산 노드 이미지](images/descriptor_calculation.png)
외부 도구인 PaDEL-Descriptor를 연동하여 2D 및 3D 분자 기술자를 손쉽게 계산하고 데이터셋에 통합합니다.

### 체계적인 데이터 전처리
![전처리 노드 이미지](images/preprocessing_nodes.png)
결측치 처리(행/열 제거, 대체), 무한대 값 처리 등 모델 성능 향상을 위한 필수적인 전처리 단계를 세분화된 노드로 제공합니다.

### 기술자 최적화 및 선택
![최적화/선택 노드 이미지](images/optimization_selection_nodes.png)
낮은 분산이나 높은 상관관계를 가진 기술자를 제거하고, 다양한 기법(Lasso, Random Forest 등)을 통해 모델 성능에 중요한 핵심 기술자만 선택하여 모델을 효율화합니다.

### 자동화된 모델 훈련 및 하이퍼파라미터 튜닝
![그리드 서치 노드 이미지](images/grid_search_node.png)
XGBoost, LightGBM, RandomForest, SVM 등 다양한 머신러닝 알고리즘을 지원하며, Grid Search를 통해 최적의 하이퍼파라미터를 자동으로 탐색하고 모델을 훈련합니다.

### 상세한 모델 검증
![모델 검증 노드 이미지](images/validation_node.png)
훈련된 모델의 성능을 R², MSE, Accuracy, F1-Score, ROC-AUC 등 Regression/Classification 문제에 맞는 다양한 지표로 상세하게 평가하고 결과를 시각화합니다.

### ComfyUI 완벽 통합
![ComfyUI 통합 예시](images/comfyui_integration.png)
모든 기능은 ComfyUI의 노드 기반 인터페이스 내에서 완벽하게 작동하여 직관적인 사용성을 제공합니다.

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

ComfyQSAR는 **Regression**과 **Classification** 작업을 위한 노드 그룹을 제공합니다. 각 노드는 QSAR 모델링 파이프라인의 특정 단계를 수행합니다. 노드들을 순서대로 연결하여 데이터 준비부터 모델 평가까지 전체 과정을 시각적으로 구성할 수 있습니다.

### 공통 노드

*   **Show Text (`ShowText`)**:
    *   **사용 시나리오:** 다른 노드에서 생성된 텍스트 정보(예: 데이터 로드 결과, 모델 평가 지표 요약)를 ComfyUI 화면에서 직접 확인하고 싶을 때 사용합니다. 워크플로우 중간중간 결과를 확인하거나 디버깅할 때 유용합니다.
    *   **사용법:** 텍스트 출력이 있는 노드의 `text` 출력을 이 노드의 `text` 입력에 연결합니다.

### ComfyQSAR Regression 노드 (QSAR/REGRESSION)

연속적인 값(예: 약물의 효능 수치, 분자의 물리화학적 속성)을 예측하는 Regression 모델링 파이프라인을 구성할 때 사용합니다.

#### LOAD & STANDARDIZATION (데이터 로딩 및 표준화)

*   **Data Loader (Regression) (`Data_Loader_Regression`)**:
    *   **사용 시나리오:** QSAR 모델링을 시작할 때, 분자 구조 정보(SMILES)와 해당 분자의 실험값(활성도, 속성값 등)이 별도의 파일(.tsv, .csv 등)에 저장되어 있는 경우 사용합니다. 두 파일을 읽어 하나의 데이터셋으로 통합합니다.
    *   **사용법:** SMILES 파일 경로와 활성값 파일 경로를 각각 입력으로 지정합니다. 노드는 두 데이터를 합친 CSV 파일의 경로를 출력합니다.
        *   **SMILES 파일 예시 (`smiles.smi` 또는 `smiles.csv`):**
            ```
            CHEMBL12345	CCO
            CHEMBL67890	c1ccccc1
            ...
            ```
            (탭 또는 쉼표로 구분된 ID와 SMILES 문자열)
        *   **활성값 파일 예시 (`activity.csv`):**
            ```
            ID,Value
            CHEMBL12345,5.6
            CHEMBL67890,8.1
            ...
            ```
            (쉼표로 구분된 ID와 숫자형 활성값, 헤더 포함)
*   **Standardization (Regression) (`Standardization_Regression`)**:
    *   **사용 시나리오:** 로드된 분자 데이터를 모델링에 적합하도록 정제할 때 사용합니다. RDKit 라이브러리를 이용해 화학적으로 유효하지 않거나 모델링에 부적합한 구조(예: 염 제거 실패, 다중 성분 등)를 필터링합니다.
    *   **사용법:** `Data Loader` 또는 다른 노드에서 출력된 CSV 파일 경로를 입력으로 받습니다. 표준화 및 필터링된 분자 데이터가 포함된 새 CSV 파일 경로를 출력합니다.
        *   **입력 CSV 파일 예시:**
            ```csv
            ID,SMILES,Value
            CHEMBL12345,CCO,5.6
            CHEMBL67890,c1ccccc1,8.1
            ...
            ```
*   **Load and Standardize (Regression) (`Load_and_Standardize_Regression`)**:
    *   **사용 시나리오:** SMILES/활성값 파일 로딩과 분자 표준화/필터링을 한 번에 처리하고 싶을 때 사용합니다. 위 두 노드(`Data Loader`, `Standardization`)를 순차적으로 사용하는 것과 동일한 작업을 수행합니다.
    *   **사용법:** SMILES 파일 경로와 활성값 파일 경로를 입력으로 지정합니다. 로드, 병합, 표준화, 필터링이 완료된 데이터의 CSV 파일 경로를 출력합니다.
        *   **SMILES 파일 예시 (`smiles.smi` 또는 `smiles.csv`):** (위 Data Loader 예시 참고)
        *   **활성값 파일 예시 (`activity.csv`):** (위 Data Loader 예시 참고)

#### CALCULATION (기술자 계산)

*   **Descriptor Calculation (Regression) (`Descriptor_Calculations_Regression`)**:
    *   **사용 시나리오:** 표준화된 분자 데이터로부터 머신러닝 모델이 학습할 특성, 즉 분자 기술자(Molecular Descriptors)를 계산할 때 사용합니다. PaDEL-Descriptor 소프트웨어를 활용하여 2D 또는 3D 기술자를 계산하고, 이를 원본 데이터('value' 포함)와 병합합니다.
    *   **사용법:** 표준화된 데이터 CSV 파일 경로를 입력으로 받습니다. 필요에 따라 PaDEL 옵션(2D/3D, 염 제거 등)을 설정합니다. 계산된 기술자가 추가된 CSV 파일 경로를 출력합니다.
        *   **입력 CSV 파일 예시 (Standardization 출력):**
            ```csv
            ID,SMILES,Value
            CHEMBL12345,CCO,5.6
            CHEMBL67890,c1ccccc1,8.1
            ...
            ```
        *   **출력 CSV 파일 예시 (기술자 추가됨):**
            ```csv
            ID,SMILES,Value,Descriptor1,Descriptor2,...
            CHEMBL12345,CCO,5.6,12.3,45.6,...
            CHEMBL67890,c1ccccc1,8.1,78.9,10.1,...
            ...
            ```

#### PREPROCESSING (데이터 전처리)

*   **Replace inf with nan (Regression) (`Replace_inf_with_nan_Regression`)**:
    *   **사용 시나리오:** 기술자 계산 과정에서 발생할 수 있는 무한대(infinity) 값을 후속 처리(예: 결측치 처리)가 가능한 NaN(Not a Number)으로 바꿀 때 사용합니다.
    *   **사용법:** 기술자가 포함된 CSV 파일 경로를 입력받습니다. 무한대 값이 NaN으로 대체된 CSV 파일 경로를 출력합니다.
*   **Remove high nan compounds (Regression) (`Remove_high_nan_compounds_Regression`)**:
    *   **사용 시나리오:** 데이터셋에서 결측치(NaN)가 너무 많은 분자(행)를 제거하고 싶을 때 사용합니다. 특정 비율 이상의 NaN을 가진 분자는 모델 성능에 부정적인 영향을 줄 수 있습니다.
    *   **사용법:** CSV 파일 경로와 허용할 최대 NaN 비율(0.0~1.0)을 입력합니다. 기준을 초과하는 분자가 제거된 CSV 파일 경로를 출력합니다.
*   **Remove high nan descriptors (Regression) (`Remove_high_nan_descriptors_Regression`)**:
    *   **사용 시나리오:** 데이터셋에서 결측치(NaN)가 너무 많은 기술자(열)를 제거하고 싶을 때 사용합니다. 대부분의 값이 누락된 기술자는 모델 학습에 유용하지 않을 수 있습니다.
    *   **사용법:** CSV 파일 경로와 허용할 최대 NaN 비율(0.0~1.0)을 입력합니다. 기준을 초과하는 기술자가 제거된 CSV 파일 경로를 출력합니다. ('SMILES', 'value' 컬럼은 유지됩니다.)
*   **Impute missing values (Regression) (`Impute_missing_values_Regression`)**:
    *   **사용 시나리오:** 전처리 후에도 남아있는 결측치(NaN)를 특정 값으로 채워 모델 학습이 가능하도록 만들 때 사용합니다. 평균(mean), 중앙값(median), 최빈값(most_frequent) 중 하나를 선택하여 결측치를 대체합니다.
    *   **사용법:** CSV 파일 경로와 결측치 대체 방법을 입력합니다. 결측치가 채워진 CSV 파일 경로를 출력합니다.
*   **Descriptor preprocessing (Regression) (`Descriptor_preprocessing_Regression`)**:
    *   **사용 시나리오:** 위의 4가지 전처리 단계(inf->NaN, high NaN 행 제거, high NaN 열 제거, 결측치 대체)를 순서대로 한 번에 적용하고 싶을 때 사용합니다. 각 단계의 임계값과 방법을 파라미터로 설정할 수 있습니다.
    *   **사용법:** CSV 파일 경로와 각 전처리 단계에 필요한 파라미터(임계값, 방법)를 입력합니다. 모든 전처리가 완료된 CSV 파일 경로를 출력합니다.
        *   **입력 CSV 파일 예시 (기술자 포함):**
            ```csv
            ID,SMILES,Value,Desc1,Desc2,Desc3,...
            CHEMBL12345,CCO,5.6,12.3,NaN,45.6,...
            CHEMBL67890,c1ccccc1,8.1,78.9,10.1,inf,...
            ...
            ```

#### OPTIMIZATION (기술자 최적화)

*   **Remove Low Variance Descriptors (Regression) (`Remove_Low_Variance_Descriptors_Regression`)**:
    *   **사용 시나리오:** 데이터셋 내에서 값의 변화가 거의 없는 (분산이 매우 낮은) 기술자를 제거하여 모델의 복잡성을 줄이고 과적합 위험을 낮추고 싶을 때 사용합니다.
    *   **사용법:** CSV 파일 경로와 분산 임계값을 입력합니다. 임계값 미만의 분산을 가진 기술자가 제거된 CSV 파일 경로를 출력합니다.
*   **Remove High Correlation Features (Regression) (`Remove_High_Correlation_Features_Regression`)**:
    *   **사용 시나리오:** 기술자들 간의 상관관계가 매우 높아 다중공선성(multicollinearity) 문제가 우려될 때 사용합니다. 상관관계가 높은 기술자 쌍 중에서 정보량이 적거나 타겟 변수와의 관련성이 낮은 기술자를 제거합니다.
    *   **사용법:** CSV 파일 경로, 상관관계 임계값, 제거 기준 모드(target_based 등), 중요도 평가 모델(lasso, random_forest)을 입력합니다. 중복 정보가 제거된 CSV 파일 경로를 출력합니다.
*   **Descriptor Optimization (Regression) (`Descriptor_Optimization_Regression`)**:
    *   **사용 시나리오:** 저분산 기술자 제거와 고상관 기술자 제거를 순차적으로 한 번에 적용하여 기술자 집합을 최적화하고 싶을 때 사용합니다.
    *   **사용법:** CSV 파일 경로와 각 최적화 단계에 필요한 파라미터(임계값, 모드, 모델)를 입력합니다. 최적화된 기술자가 포함된 CSV 파일 경로를 출력합니다.
        *   **입력 CSV 파일 예시 (전처리 완료):**
            ```csv
            ID,SMILES,Value,Desc1,Desc2,Desc_low_variance,Desc_high_corr,...
            CHEMBL12345,CCO,5.6,12.3,10.0,0.0,12.4,...
            CHEMBL67890,c1ccccc1,8.1,78.9,20.0,0.0,78.8,...
            ...
            ```
*   **Feature Selection (Regression) (`Feature_Selection_Regression`)**:
    *   **사용 시나리오:** 최적화된 기술자 집합에서 모델 성능에 가장 중요하게 기여하는 기술자들만 최종적으로 선택하고 싶을 때 사용합니다. 다양한 머신러닝 기법(Lasso, Random Forest 등)을 기반으로 중요도를 평가하고, 지정된 개수(n_features) 또는 중요도 임계값(threshold)에 따라 기술자를 선택합니다.
    *   **사용법:** CSV 파일 경로, 사용할 선택 방법(method), 선택할 기술자 수 또는 임계값, 기반 모델(선택 사항) 등을 입력합니다. 선택된 핵심 기술자들만 포함된 CSV 파일 경로를 출력합니다.
        *   **입력 CSV 파일 예시 (최적화 완료):**
            ```csv
            ID,SMILES,Value,SelectedDesc1,SelectedDesc2,LessImportantDesc,...
            CHEMBL12345,CCO,5.6,12.3,10.0,5.5,...
            CHEMBL67890,c1ccccc1,8.1,78.9,20.0,1.2,...
            ...
            ```

#### COMBINATION (기술자 조합 탐색)

*   **Get Best Descriptor Combinations RF (`Get_Best_Descriptor_Combinations_RF`)**:
    *   **사용 시나리오:** 개별 기술자의 중요도뿐만 아니라, 여러 기술자의 '조합'이 모델 성능에 미치는 영향을 탐색하고 싶을 때 사용합니다. Random Forest 모델을 사용하여 가능한 모든 조합(지정된 최대 개수까지)을 평가하고 최적의 조합을 찾습니다. 계산량이 많을 수 있어 멀티코어 활용이 권장됩니다.
    *   **사용법:** CSV 파일 경로, 탐색할 최대 기술자 조합 개수(max_features), 사용할 CPU 코어 수(num_cores), 저장할 상위 조합 개수(top_n)를 입력합니다. 가장 성능이 좋았던 기술자 조합으로 구성된 CSV 파일 경로를 출력합니다. (추가 결과 파일도 생성됨)
*   **Get Best Descriptor Combinations (`Get_Best_Descriptor_Combinations`)**:
    *   **사용 시나리오:** `Get Best Descriptor Combinations RF`와 유사하지만, 평가 모델로 Linear Regression을 사용합니다. 더 빠르지만 비선형 관계를 잘 포착하지 못할 수 있습니다.
    *   **사용법:** RF 버전과 동일한 파라미터를 입력합니다. 최적 조합 CSV 파일 경로를 출력합니다.

#### GRID SEARCH (하이퍼파라미터 탐색)

*   **Hyperparameter Grid Search (Regression) (`Hyperparameter_Grid_Search_Regression`)**:
    *   **사용 시나리오:** 선택된 기술자 집합을 사용하여 특정 머신러닝 모델(예: XGBoost, RandomForest, SVM)의 성능을 최적화하는 하이퍼파라미터 조합을 찾고 싶을 때 사용합니다. 지정된 범위 내의 여러 하이퍼파라미터 조합을 시도하고, 교차 검증(Cross-validation)을 통해 가장 성능이 좋은 조합을 찾습니다. 이 과정에서 데이터는 자동으로 Train/Test 세트로 분리됩니다.
    *   **사용법:** 최종 기술자 선택 또는 조합 탐색 결과 CSV 파일 경로, 사용할 알고리즘, 탐색할 하이퍼파라미터 범위(문자열 리스트 형태), 교차 검증 설정 등을 입력합니다. 최적의 하이퍼파라미터로 학습된 모델 파일(.joblib), 사용된 기술자 목록, 그리고 평가를 위한 Test 데이터 파일들의 경로를 출력합니다.
        *   **입력 CSV 파일 예시 (최종 기술자 포함):**
            ```csv
            ID,SMILES,Value,FinalDesc1,FinalDesc2,...
            CHEMBL12345,CCO,5.6,12.3,10.0,...
            CHEMBL67890,c1ccccc1,8.1,78.9,20.0,...
            ...
            ```

#### VALIDATION (모델 검증)

*   **Model Validation (Regression) (`Model_Validation_Regression`)**:
    *   **사용 시나리오:** `Grid Search`를 통해 얻은 최적 모델의 최종 성능을 이전에 분리해 둔 Test 데이터셋으로 평가할 때 사용합니다. R², MSE, RMSE 등 다양한 평가 지표를 계산하여 모델의 일반화 성능을 확인합니다.
    *   **사용법:** `Grid Search`에서 출력된 모델 파일 경로(.joblib), 기술자 목록 파일(.txt 또는 .csv) 경로, Test 데이터 파일(X_test.csv, y_test.csv) 경로를 입력합니다. 모델의 평가 결과가 담긴 CSV 파일 경로와 실제값-예측값 비교 CSV 파일 경로를 출력합니다.
        *   **입력 파일:** 모델 파일 (`.joblib`), 기술자 목록 파일, X_test CSV, y_test CSV

### ComfyQSAR Classification 노드 (QSAR/CLASSIFICATION)

분자를 특정 범주(예: 활성/비활성, 독성/비독성)로 분류하는 Classification 모델링 파이프라인을 구성할 때 사용합니다. Regression 노드들과 유사한 흐름을 가지지만, 분류 문제의 특성(데이터 로딩 방식, 평가 지표 등)에 맞게 조정되었습니다.

#### LOAD & STANDARDIZATION (데이터 로딩 및 표준화)

*   **Data Loader (Classification) (`Data_Loader_Classification`)**:
    *   **사용 시나리오:** 분류 모델링을 시작할 때, 각 클래스(예: Positive, Negative)에 해당하는 분자 데이터가 별도의 파일(SMILES, CSV, SDF)로 제공되는 경우 사용합니다. 각 클래스 파일을 로드하고 분자 수를 확인합니다.
    *   **사용법:** Positive 클래스 파일 경로와 Negative 클래스 파일 경로를 각각 입력합니다. 각 파일 경로를 그대로 출력하며, UI에 로드된 분자 수를 표시합니다.
        *   **Positive/Negative 파일 예시 (`pos.smi`, `neg.smi`):**
            ```
            ID1	SMILES1
            ID2	SMILES2
            ...
            ```
            (탭 또는 쉼표로 구분된 ID와 SMILES. CSV나 SDF 형식도 지원)
*   **Standardization (Classification) (`Standardization_Classification`)**:
    *   **사용 시나리오:** 로드된 Positive/Negative 분자 데이터를 각각 표준화하고 유효하지 않은 구조를 필터링할 때 사용합니다. (Regression의 Standardization과 동일한 로직 적용)
    *   **사용법:** `Data Loader`에서 출력된 Positive/Negative 파일 경로를 입력받습니다. 표준화/필터링된 데이터 파일 경로를 각각 출력합니다.
        *   **입력 파일 예시:** 위 Data Loader 예시 참고
*   **Load and Standardize (Classification) (`Load_and_Standardize_Classification`)**:
    *   **사용 시나리오:** Positive/Negative 데이터 파일 로딩과 표준화/필터링을 한 번에 처리하고 싶을 때 사용합니다.
    *   **사용법:** Positive/Negative 파일 경로를 입력합니다. 표준화/필터링된 데이터 파일 경로를 각각 출력합니다.
        *   **입력 파일 예시:** 위 Data Loader 예시 참고

#### CALCULATION (기술자 계산)

*   **Descriptor Calculation (Classification) (`Descriptor_Calculations_Classification`)**:
    *   **사용 시나리오:** 표준화된 Positive/Negative 데이터로부터 각각 분자 기술자를 계산하고, 두 데이터셋을 하나로 병합할 때 사용합니다. 병합된 데이터에는 각 분자가 어느 클래스에 속하는지 나타내는 'Label' 컬럼(Positive=1, Negative=0)이 자동으로 추가됩니다.
    *   **사용법:** 표준화된 Positive/Negative 데이터 파일 경로를 입력받습니다. PaDEL 옵션을 설정할 수 있습니다. 기술자 계산 및 병합, Label 추가가 완료된 CSV 파일 경로를 출력합니다.
        *   **입력 파일:** 표준화된 Positive 파일, 표준화된 Negative 파일
        *   **출력 CSV 파일 예시:**
            ```csv
            ID,SMILES,Label,Descriptor1,Descriptor2,...
            PosID1,PosSMILES1,1,12.3,45.6,...
            NegID1,NegSMILES1,0,78.9,10.1,...
            ...
            ```

#### PREPROCESSING (데이터 전처리)

*   **Regression 노드들과 거의 동일:** `Replace_inf_with_nan_Classification`, `Remove_high_nan_compounds_Classification`, `Remove_high_nan_descriptors_Classification`, `Impute_missing_values_Classification`, `Descriptor_preprocessing_Classification` 노드가 제공됩니다. Regression 버전과 사용법 및 기능이 유사하며, 'Label' 컬럼을 유지하면서 전처리를 수행합니다.

#### OPTIMIZATION (기술자 최적화)

*   **Regression 노드들과 거의 동일:** `Remove_Low_Variance_Features_Classification`, `Remove_High_Correlation_Features_Classification`, `Descriptor_Optimization_Classification` 노드가 제공됩니다. Regression 버전과 사용법 및 기능이 유사하며, 'Label' 컬럼을 타겟 변수로 간주하여 최적화를 수행합니다(예: 'target_based' 상관관계 제거).

#### SELECTION (기술자 선택)

*   **Feature Selection (Classification) (`Feature_Selection_Classification`)**:
    *   **사용 시나리오:** 최적화된 기술자 집합에서 분류 모델 성능에 중요한 기술자만 선택할 때 사용합니다. 다양한 분류 모델 기반 기법(Logistic Regression(Lasso), RandomForest 등)을 활용합니다.
    *   **사용법:** CSV 파일 경로, 선택 방법, 기반 모델, 선택할 기술자 수/임계값 등을 입력합니다. 선택된 기술자와 'Label' 컬럼만 포함된 CSV 파일 경로를 출력합니다.
        *   **입력 CSV 파일 예시 (최적화 완료):**
            ```csv
            ID,SMILES,Label,SelectedDesc1,SelectedDesc2,LessImportantDesc,...
            PosID1,PosSMILES1,1,12.3,10.0,5.5,...
            NegID1,NegSMILES1,0,78.9,20.0,1.2,...
            ...
            ```

#### COMBINATION (기술자 조합 탐색)

*   **Feature Combination Search (`Feature_Combination_Search`)**:
    *   **사용 시나리오:** 분류 모델의 정확도(Accuracy)를 기준으로 최적의 기술자 조합을 탐색할 때 사용합니다. Logistic Regression 모델로 조합을 평가하며, 멀티코어를 활용할 수 있습니다.
    *   **사용법:** CSV 파일 경로, 최대 조합 개수, 사용할 코어 수, 저장할 상위 조합 개수를 입력합니다. 최적 조합 CSV 파일 경로를 출력합니다.

#### GRID SEARCH (하이퍼파라미터 탐색)

*   **Hyperparameter Grid Search (Classification) (`Hyperparameter_Grid_Search_Classification`)**:
    *   **사용 시나리오:** 분류 모델(XGBoost, RF, SVM 등)의 성능을 최적화하는 하이퍼파라미터를 찾을 때 사용합니다. Grid Search와 교차 검증(Stratified K-Fold 사용)을 통해 최적 조합을 찾고, Train/Test 데이터 분할을 수행합니다.
    *   **사용법:** 최종 기술자 CSV 파일 경로, 알고리즘, 하이퍼파라미터 범위, 교차 검증 설정 등을 입력합니다. 최적 모델 파일(.joblib), 기술자 목록, Test 데이터 파일들의 경로를 출력합니다.
        *   **입력 CSV 파일 예시 (최종 기술자 포함):**
            ```csv
            ID,SMILES,Label,FinalDesc1,FinalDesc2,...
            PosID1,PosSMILES1,1,12.3,10.0,...
            NegID1,NegSMILES1,0,78.9,20.0,...
            ...
            ```

#### VALIDATION (모델 검증)

*   **Model Validation (Classification) (`Model_Validation_Classification`)**:
    *   **사용 시나리오:** `Grid Search`로 찾은 최적 분류 모델의 최종 성능을 Test 데이터셋으로 평가할 때 사용합니다. Accuracy, F1-Score, ROC-AUC 등 분류 문제에 적합한 다양한 지표를 계산합니다.
    *   **사용법:** 모델 파일 경로(.joblib), 기술자 목록 파일(.txt 또는 .csv) 경로, Test 데이터 파일 경로(X_test.csv, y_test.csv)를 입력합니다. 평가 결과 CSV 파일 경로와 실제값-예측값 비교 CSV 파일 경로를 출력합니다.
        *   **입력 파일:** 모델 파일 (`.joblib`), 기술자 목록 파일, X_test CSV, y_test CSV

## 사용 예시

`add_image`

**Regression 워크플로우 예시:**

1.  **데이터 준비:** `Load and Standardize (Regression)` 노드로 시작하여 SMILES와 활성값 데이터를 로드하고 표준화합니다.
2.  **기술자 계산:** `Descriptor Calculation (Regression)` 노드를 연결하여 분자 기술자를 계산합니다.
3.  **전처리:** `Descriptor preprocessing (Regression)` 노드를 연결하여 결측치 등을 처리합니다.
4.  **최적화/선택:** (선택 사항) `Descriptor Optimization` 또는 `Feature Selection` 노드를 사용하여 기술자 수를 줄입니다.
5.  **모델 훈련 및 최적화:** `Hyperparameter Grid Search (Regression)` 노드를 연결하여 최적의 모델을 찾습니다. 입력으로 기술자 데이터 파일 경로를, 사용할 알고리즘과 하이퍼파라미터 범위를 지정합니다.
6.  **최종 평가:** `Model Validation (Regression)` 노드를 연결하고, Grid Search에서 나온 모델, 기술자 목록, Test 데이터를 입력하여 최종 성능을 확인합니다.

**Classification 워크플로우 예시:**

1.  **데이터 준비:** `Load and Standardize (Classification)` 노드로 Positive/Negative 데이터를 로드하고 표준화합니다.
2.  **기술자 계산:** `Descriptor Calculation (Classification)` 노드를 연결하여 기술자를 계산하고 'Label'이 추가된 단일 데이터셋을 만듭니다.
3.  **전처리/최적화/선택:** Regression과 유사하게 `Descriptor preprocessing`, `Optimization`, `Selection` 노드를 필요에 따라 연결합니다.
4.  **모델 훈련 및 최적화:** `Hyperparameter Grid Search (Classification)` 노드를 사용하여 최적의 분류 모델을 찾습니다.
5.  **최종 평가:** `Model Validation (Classification)` 노드를 사용하여 최종 모델의 분류 성능(Accuracy, F1 등)을 평가합니다.

## 기여

버그를 발견하거나 새로운 기능을 제안하고 싶으시면 GitHub 이슈를 열어주세요. 풀 리퀘스트도 환영합니다!

## 라이선스

*(프로젝트에 적용할 라이선스를 명시하세요. 예: MIT License)*