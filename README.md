# ComfyQSAR
**ComfyQSAR** is a custom node extension for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) designed to perform Quantitative Structure-Activity Relationship (QSAR) modeling. It allows you to build flexible, visual pipelines for calculating molecular descriptors, preprocessing data, and training, evaluating, and optimizing machine learning models directly within ComfyUI.
---
## Features
**ComfyQSAR** provides a comprehensive set of nodes to cover the entire QSAR modeling process:
*   **Modular Workflow**: Build flexible pipelines for both **Regression** and **Classification** tasks by connecting specialized nodes.
*   **Data Loading & Standardization**: Load chemical data (SMILES, CSV, SDF) and standardize molecular structures using RDKit.
*   **Descriptor Calculation**: seamless integration with PaDEL-Descriptor to calculate 2D and 3D molecular descriptors.
*   **Data Preprocessing**: Handling missing values (NaN, infinity), removing high-NaN rows/columns, and imputing values.
*   **Feature Optimization & Selection**: Reduce dimensionality by removing low-variance or high-correlation features, and select key descriptors using Lasso or Random Forest.
*   **AutoML & Grid Search**: Automatically search for optimal hyperparameters for algorithms like XGBoost, LightGBM, Random Forest, and SVM.
*   **Model Validation**: Evaluate model performance with detailed metrics (R², MSE, Accuracy, F1-Score, ROC-AUC) and visualizations.
## Visualization & Analysis
ComfyQSAR treats data analysis as a visual flow. You can inspect the state of your data at any point, visualize model performance, and iteratively improve your models.
![Workflow Example](images/workflow_example.png)
*Example of a QSAR workflow*
## Installation
### Installation via git
1.  Navigate to the `custom_nodes` directory within your ComfyUI installation.
2.  Run the following command:
    ```bash
    git clone https://github.com/YourUsername/ComfyQSAR.git
    ```
    This will create a new subdirectory `ComfyQSAR`.
3.  **Install Dependencies**:
    *   Start ComfyUI. The node attempts to automatically install required Python packages listed in `requirements.txt` upon first load.
    *   If automatic installation fails, install manually:
        ```bash
        cd ComfyUI/custom_nodes/ComfyQSAR
        pip install -r requirements.txt
        ```
    *   **Note**: This extension relies on `rdkit`, `scikit-learn`, `pandas`, `xgboost`, `lightgbm`, and others.
4.  Restart ComfyUI.
## How to Use
ComfyQSAR provides separate node categories for **Regression** and **Classification**. Connect nodes in a logical sequence: **Load -> Calculate -> Preprocess -> Optimize -> Train -> Validate**.
### Common Nodes
*   **Show Text**: Display text output (e.g., loaded data summary, evaluation metrics) directly in the workflow.
### Regression Nodes (QSAR/REGRESSION)
For predicting continuous values (e.g., potency, physical properties).
*   **Load & Standardize**:
    *   `Data Loader`: Load SMILES and Activity files.
    *   `Standardization`: Clean and filter molecules using RDKit.
    *   `Load and Standardize`: Combine both steps.
*   **Calculation**:
    *   `Descriptor Calculation`: Calculate molecular descriptors (PaDEL).
*   **Preprocessing**:
    *   `Replace inf with nan`, `Remove high nan compounds/descriptors`, `Impute missing values`.
    *   `Descriptor Preprocessing`: All-in-one preprocessing node.
*   **Optimization**:
    *   `Remove Low Variance`, `Remove High Correlation`.
    *   `Feature Selection`: Select top features using Lasso/RF.
    *   `Descriptor Optimization`: All-in-one optimization node.
*   **Grid Search**:
    *   `Hyperparameter Grid Search`: Find best model parameters (XGBoost, RF, SVM, etc.).
*   **Validation**:
    *   `Model Validation`: Evaluate the trained model on a test set.
### Classification Nodes (QSAR/CLASSIFICATION)
For predicting categories (e.g., Active/Inactive, Toxic/Non-toxic).
*   **Load & Standardize**:
    *   `Data Loader`: Load separate Positive and Negative class files.
    *   `Standardization` & `Load and Standardize`: Similar to regression but handles class labels.
*   **Calculation**:
    *   `Descriptor Calculation`: Calculates descriptors and assigns labels (Pos=1, Neg=0).
*   **Preprocessing & Optimization**:
    *   Similar counterparts to regression nodes, adapted for labeled data.
*   **Grid Search**:
    *   `Hyperparameter Grid Search`: optimized for classification metrics (Accuracy, F1, etc.).
*   **Validation**:
    *   `Model Validation`: Evaluate classification performance.
## Contribution
Based on [ComfyUI-Data-Analysis](https://github.com/HowToSD/ComfyUI-Data-Analysis).
If you have feature requests or find bugs, please create an Issue.
## License
MIT License
