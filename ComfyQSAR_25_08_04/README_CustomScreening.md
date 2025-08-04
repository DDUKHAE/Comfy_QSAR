# ComfyQSAR 커스텀 스크리닝 노드 사용법

## 개요

이 문서는 Jupyter 노트북 `10_Custom_screening.ipynb`의 기능을 ComfyUI 노드로 구현한 커스텀 스크리닝 워크플로우 사용법을 설명합니다.

## 노드 구성

커스텀 스크리닝 워크플로우는 4개의 주요 노드로 구성됩니다:

### 1. Custom DB - Standardization
**카테고리:** `QSAR/Custom Screening`
**기능:** SDF 파일의 분자 구조를 표준화하고 유효하지 않은 분자를 제거

**입력:**
- `input_sdf_path`: 입력 SDF 파일 경로

**출력:**
- `STANDARDIZED_SDF`: 표준화된 SDF 파일 경로
- `LOG_MESSAGE`: 처리 결과 로그

**처리 과정:**
- 금속 이온만으로 구성된 분자 제거
- 다중 프래그먼트 분자 제거
- 유효하지 않은 RDKit 분자 제거

### 2. Custom DB - Descriptor Calculation
**카테고리:** `QSAR/Custom Screening`
**기능:** PaDEL을 사용하여 분자 디스크립터 계산

**입력:**
- `input_sdf`: 표준화된 SDF 파일 경로 (이전 노드에서 연결)

**출력:**
- `DESCRIPTORS_CSV`: 계산된 디스크립터 CSV 파일 경로
- `LOG_MESSAGE`: 처리 결과 로그

**필요 조건:**
- PaDEL-Descriptor 설치 (`pip install padelpy`)

### 3. Custom DB - Descriptor Preprocessing
**카테고리:** `QSAR/Custom Screening`
**기능:** 계산된 디스크립터 전처리 (결측값 처리, 필터링)

**입력:**
- `descriptor_csv`: 원시 디스크립터 CSV 파일 경로
- `nan_threshold`: NaN 임계값 (기본값: 0.5)
- `impute_method`: 결측값 대치 방법 (mean, median, most_frequent)

**출력:**
- `PREPROCESSED_CSV`: 전처리된 디스크립터 CSV 파일 경로
- `LOG_MESSAGE`: 처리 결과 로그

**처리 단계:**
1. 무한값을 NaN으로 변환
2. 높은 NaN 비율의 화합물 제거
3. 높은 NaN 비율의 디스크립터 제거
4. 결측값 대치

### 4. Custom DB - Screening
**카테고리:** `QSAR/Custom Screening`
**기능:** 훈련된 QSAR 모델을 사용하여 화합물 스크리닝

**입력:**
- `model_path`: 훈련된 QSAR 모델 파일 경로 (.pkl)
- `descriptors_path`: 선택된 디스크립터 목록 파일 경로 (.txt)
- `descriptor_csv`: 전처리된 디스크립터 CSV 파일 경로
- `original_sdf`: 원본 표준화된 SDF 파일 경로
- `task_type`: QSAR 작업 유형 (Classification 또는 Regression)
- `threshold`: 선택 임계값

**출력:**
- `PREDICTIONS_CSV`: 예측 결과 CSV 파일 경로
- `SELECTED_SDF`: 선택된 화합물 SDF 파일 경로
- `LOG_MESSAGE`: 처리 결과 로그

## 사용 예시

### 기본 워크플로우

1. **데이터 준비**
   - 입력 SDF 파일 준비 (예: `PTP1B_positive_compounds_BindingDB(200).sdf`)
   - 훈련된 QSAR 모델 파일 준비 (예: `PTP1B_prediction_QSAR_model.pkl`)
   - 선택된 디스크립터 목록 파일 준비 (예: `selected_features_V3.txt`)

2. **노드 연결 순서**
   ```
   Custom DB - Standardization
           ↓
   Custom DB - Descriptor Calculation
           ↓
   Custom DB - Descriptor Preprocessing
           ↓
   Custom DB - Screening
   ```

3. **파라미터 설정**
   - Standardization: 입력 SDF 파일 경로 입력
   - Descriptor Calculation: 자동으로 이전 노드 출력 연결
   - Preprocessing: NaN 임계값 및 대치 방법 선택
   - Screening: 모델 파일, 디스크립터 파일 경로 및 임계값 설정

### 노트북과 동일한 예시

노트북에서 사용된 예시를 ComfyUI로 재현:

**Standardization 노드:**
- `input_sdf_path`: `"PTP1B_positive_compounds_BindingDB(200).sdf"`

**Preprocessing 노드:**
- `nan_threshold`: `0.5`
- `impute_method`: `"mean"`

**Screening 노드:**
- `model_path`: `"./PTP1B_prediction_QSAR_model.pkl"`
- `descriptors_path`: `"./selected_features_V3.txt"`
- `task_type`: `"Classification"`
- `threshold`: `0.5`

## 출력 파일

실행 완료 후 다음 파일들이 생성됩니다:

```
ComfyUI/output/qsar_custom_screening/
├── standardized/
│   └── standardized_input.sdf
├── descriptors/
│   └── molecular_descriptors.csv
├── preprocessed/
│   ├── step1_cleaned_data.csv
│   ├── step2_filtered_compounds_[count].csv
│   ├── step3_filtered_descriptors_[count].csv
│   └── preprocessed_data.csv
└── screening_results/
    ├── Custom_Screening_Predictions.csv
    └── Custom_Screening_Selected_Molecules.sdf
```

## 예상 결과

노트북 예시 기준:
- **입력 화합물:** 200개
- **표준화 후:** 200개 (유효한 분자)
- **전처리 후:** 198개 (2개 제거)
- **선택된 화합물:** 195개 (임계값 0.5 기준)

## 필요 조건

### 소프트웨어
- ComfyUI
- Python 3.7+
- RDKit (`conda install -c conda-forge rdkit` 또는 `pip install rdkit`)
- PaDEL-Descriptor (`pip install padelpy`)
- scikit-learn, pandas, numpy, joblib

### 하드웨어
- 메모리: 최소 4GB RAM (대용량 데이터셋의 경우 더 많이 필요)
- 저장공간: 임시 파일 및 결과 파일을 위한 충분한 공간

## 문제 해결

### 일반적인 오류

1. **RDKit 설치 오류**
   ```bash
   conda install -c conda-forge rdkit
   ```

2. **PaDEL 설치 오류**
   ```bash
   pip install padelpy
   ```

3. **메모리 부족**
   - 입력 데이터 크기 줄이기
   - 시스템 메모리 증가

4. **파일 경로 오류**
   - 절대 경로 사용
   - 파일 존재 여부 확인

### 로그 확인

각 노드의 `LOG_MESSAGE` 출력을 확인하여 처리 과정과 결과를 모니터링할 수 있습니다.

## 성능 최적화

- **PaDEL 계산:** 멀티스레딩 활용 (기본값: 모든 CPU 코어 사용)
- **대용량 데이터:** 배치 처리 고려
- **SSD 사용:** I/O 성능 향상

## 추가 정보

더 자세한 정보는 다음 파일들을 참조하세요:
- `10_Custom_screening.ipynb`: 원본 노트북
- `custom_screening.py`: 노드 구현 코드
- ComfyQSAR 문서: 전체 워크플로우 가이드 