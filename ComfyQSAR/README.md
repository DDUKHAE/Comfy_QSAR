# ComfyQSAR

ComfyQSAR는 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 환경에서 정량적 구조-활성 관계(Quantitative Structure-Activity Relationship, QSAR) 모델링을 수행하기 위한 커스텀 노드 확장입니다. 화학 구조 데이터로부터 분자 기술자를 계산하고, 데이터를 전처리하며, 머신러닝 모델을 훈련, 평가, 최적화하는 워크플로우를 ComfyUI의 시각적 인터페이스 내에서 구축할 수 있습니다.

## 기능

*   **모듈식 워크플로우:** 데이터 로딩, 기술자 계산, 전처리, 모델 훈련, 하이퍼파라미터 튜닝 등 QSAR 모델링의 각 단계를 개별 노드로 제공하여 유연한 워크플로우 구성이 가능합니다.
*   **다양한 머신러닝 모델 지원:** Regression 및 Classification 작업 모두에 대해 다양한 알고리즘(예: XGBoost, LightGBM, CatBoost, Scikit-learn 모델 등)을 지원합니다. (구체적인 지원 모델은 각 노드 설명 참조)
*   **자동 종속성 관리:** 필요한 파이썬 라이브러리를 자동으로 감지하고 설치를 시도합니다.
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
        cd ComfyUI/custom_nodes/ComfyQSAR/
        pip install -r requirements.txt
        ```
4.  **ComfyUI 재시작:** 종속성 설치 후 ComfyUI를 재시작합니다.

## 노드 설명

ComfyQSAR는 Regression과 Classification 작업을 위한 노드 그룹을 제공합니다.

### 공통 노드

*   **Show Text:** 입력된 텍스트 데이터를 화면에 표시합니다. (디버깅 또는 결과 확인용)

### ComfyQSAR Regression 노드

Regression(회귀) 모델링을 위한 노드들입니다. 주로 연속적인 값(예: 화합물의 특정 활성도 수치)을 예측하는 데 사용됩니다.

*   **Data Loader:** Regression 모델링에 사용할 데이터셋(예: CSV 파일)을 로드합니다. 데이터 분할(Train/Test split) 기능 등을 포함할 수 있습니다.
*   **Descriptor Calculations:** 로드된 분자 데이터로부터 다양한 분자 기술자(Molecular Descriptors)를 계산합니다. (예: RDKit, PadelPy 사용)
*   **Descriptor Preprocessing:** 계산된 기술자 데이터를 전처리합니다. (예: 스케일링, 결측치 처리)
*   **Descriptor Selection:** 중요한 기술자를 선택하거나 차원을 축소하는 기법을 적용합니다. (예: Feature selection 알고리즘)
*   **Descriptor Combination:** 여러 종류의 기술자 또는 데이터 소스를 결합합니다.
*   **Descriptor Optimization:** 기술자 선택 또는 생성 과정을 최적화합니다.
*   **Grid Search Hyperparameter:** Regression 모델의 최적 하이퍼파라미터를 Grid Search 방식으로 탐색합니다.
*   **Train and Validation:** Regression 모델을 훈련하고 검증 데이터셋으로 성능을 평가합니다.

### ComfyQSAR Classification 노드

Classification(분류) 모델링을 위한 노드들입니다. 주로 이산적인 범주(예: 활성/비활성, 독성/비독성)를 예측하는 데 사용됩니다.

*   **Data Loader:** Classification 모델링에 사용할 데이터셋을 로드하고 분할합니다.
*   **Descriptor Calculation:** 분자 기술자를 계산합니다.
*   **Descriptor Preprocessing:** 기술자 데이터를 전처리합니다.
*   **Descriptor Selection:** 중요한 기술자를 선택하거나 차원을 축소합니다.
*   **Descriptor Combination:** 기술자 또는 데이터를 결합합니다.
*   **Descriptor Optimization:** 기술자 관련 프로세스를 최적화합니다.
*   **Grid Search Hyperparameter:** Classification 모델의 최적 하이퍼파라미터를 Grid Search 방식으로 탐색합니다.
*   **Train and Validation:** Classification 모델을 훈련하고 성능을 평가합니다.

*(참고: 위 노드 설명은 파일명을 기반으로 유추한 일반적인 기능입니다. 각 노드의 정확한 기능, 입력/출력, 파라미터 등 상세 내용은 실제 구현을 확인해야 합니다.)*

## 사용 예시

*(워크플로우 예시 이미지나 json 파일을 여기에 추가하면 좋습니다.)*

1.  `Data Loader` 노드로 데이터를 로드합니다.
2.  `Descriptor Calculation` 노드로 분자 기술자를 계산합니다.
3.  `Descriptor Preprocessing` 노드로 데이터를 정제합니다.
4.  (선택 사항) `Descriptor Selection` 노드로 주요 기술자를 선택합니다.
5.  `Train and Validation` 노드에 데이터를 연결하여 모델을 훈련하고 평가합니다.
6.  (선택 사항) `Grid Search Hyperparameter` 노드를 사용하여 모델 성능을 최적화합니다.

## 기여

버그를 발견하거나 새로운 기능을 제안하고 싶으시면 GitHub 이슈를 열어주세요. 풀 리퀘스트도 환영합니다!

## 라이선스

*(프로젝트에 적용할 라이선스를 명시하세요. 예: MIT License)*
