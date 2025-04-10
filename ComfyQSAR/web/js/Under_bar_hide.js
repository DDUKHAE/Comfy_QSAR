import { app } from "../../../scripts/app.js";

// 적용되는 노드 목록 정의 (ComfyQSAR_TEXT.js와 동일하게 유지)
const TARGET_NODES = Object.freeze([
    //Data Loader
    "Data_Loader_Regression",
    "Data_Loader_Classification",
    //Standardization
    "Standardization_Regression",
    "Standardization_Classification",
    "Load_and_Standardize_Regression",
    "Load_and_Standardize_Classification",
    //Descriptor Calculations
    "Descriptor_calculations_Regression",
    "Descriptor_calculations_Classification",
    //Descriptor Preprocessing
    "Replace_inf_with_nan_Regression",
    "Replace_inf_with_nan_Classification",
    "Remove_high_nan_compounds_Regression",
    "Remove_high_nan_compounds_Classification",
    "Remove_high_nan_descriptors_Regression",
    "Remove_high_nan_descriptors_Classification",
    //Descriptor Optimization
    "Remove_Low_Variance_Descriptors_Regression",
    "Remove_Low_Variance_Descriptors_Classification",
    "Remove_High_Correlation_Features_Regression",
    "Remove_High_Correlation_Features_Classification",
    "Descriptor_Optimization_Regression",
    "Descriptor_Optimization_Classification",
    //FEATURE SELECTION
    "Feature_Selection_Regression",
    "Feature_Selection_Classification",
    //GRID SEARCH HYPERPARAMETER
    "Hyperparameter_Grid_Search_Regression",
    "Hyperparameter_Grid_Search_Classification",
    //MODEL VALIDATION
    "Model_Validation_Regression",
    "Model_Validation_Classification",
    // 추가 노드
    "ShowText"
]);

app.registerExtension({
    name: "HIDE_UNDERSCORE",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (TARGET_NODES.includes(nodeData.name)) {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                // 모든 위젯을 순회
                for (const widget of this.widgets) {
                    // 원래 라벨 텍스트를 저장
                    const originalLabel = widget.name;
                    
                    // '_' 문자를 공백으로 대체한 새 라벨 생성
                    const newLabel = originalLabel.replace(/_/g, ' ');
                    
                    // 위젯의 표시 이름 변경
                    if (widget.label) {
                        widget.label = newLabel;
                    }
                    
                    // DOM 요소가 있는 경우 텍스트 업데이트
                    if (widget.element) {
                        const labelElement = widget.element.querySelector('.widget-label');
                        if (labelElement) {
                            labelElement.textContent = newLabel;
                        }
                    }
                }
                
                return result;
            };
        }
    }
});