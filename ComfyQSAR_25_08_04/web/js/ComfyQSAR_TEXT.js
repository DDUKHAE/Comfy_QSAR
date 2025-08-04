import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// 상수 정의
const TEXT_WIDGET_HEIGHT = 120;  // 출력 위젯 높이
const TEXT_WIDGET_MARGIN = 10;  // 출력 위젯 여백

const TARGET_NODES = Object.freeze([
    "Data_Loader_Regression", "Data_Loader_Classification",
    "Standardization_Regression", "Standardization_Classification",
    "Load_and_Standardize_Regression", "Load_and_Standardize_Classification",
    "Descriptor_Calculations_Regression", "Descriptor_Calculations_Classification",
    "Replace_inf_with_nan_Regression", "Replace_inf_with_nan_Classification",
    "Remove_high_nan_compounds_Regression", "Remove_high_nan_compounds_Classification",
    "Remove_high_nan_descriptors_Regression", "Remove_high_nan_descriptors_Classification",
    "Impute_missing_values_Regression", "Impute_missing_values_Classification",
    "Descriptor_preprocessing_Regression", "Descriptor_preprocessing_Classification",
    "Remove_Low_Variance_Features_Regression", "Remove_Low_Variance_Features_Classification",
    "Remove_High_Correlation_Features_Regression", "Remove_High_Correlation_Features_Classification",
    "Descriptor_Optimization_Regression", "Descriptor_Optimization_Classification",
    "Feature_Selection_Regression", "Feature_Selection_Classification",
    "Hyperparameter_Grid_Search_Regression", "Hyperparameter_Grid_Search_Classification",
    "Model_Validation_Regression", "Model_Validation_Classification",
]);

// 출력 위젯 스타일 적용 함수
function applyTextWidgetStyles(textWidget) {
    const styles = {
        width: "100%",
        height: `${TEXT_WIDGET_HEIGHT}px`,
        whiteSpace: "pre-wrap",
        overflow: "auto",
        wordBreak: "break-word",
        textAlign: "left",
        padding: "8px",
        fontFamily: "monospace",
        fontSize: "10px",
        fontWeight: "normal",
        color: "black",
        borderRadius: "6px",
        border: "1px solid #f5f5f5",
        background: "#f5f5f5",
        transition: "none",
        marginTop: `${TEXT_WIDGET_MARGIN}px`,
        marginBottom: `${TEXT_WIDGET_MARGIN}px`,
        position: "relative",
        zIndex: "1",
        display: "block",
        clear: "both"
    };
    Object.assign(textWidget.inputEl.style, styles);
    textWidget.inputEl.readOnly = true;
    textWidget.inputEl.style.opacity = "1.0";
}

// 출력 위젯 생성 함수
function createTextWidget(node) {
    try {
        const textWidget = ComfyWidgets["STRING"](node, "text2", ["STRING", { multiline: true }], app).widget;
        applyTextWidgetStyles(textWidget);
        return textWidget;
    } catch (error) {
        console.error(`[${node.title}] Error creating text2 widget:`, error);
        return null;
    }
}

// 출력 위젯 내용 업데이트 함수
function updateTextWidget(node, message) {
    const textWidget = node.widgets?.find(w => w.name === "text2");
    if (textWidget) {
        if (message && message.text) {
            const outputText = Array.isArray(message.text) ? message.text.join('') : message.text;
            if (outputText && outputText.trim() !== '') {
                textWidget.value = outputText;
                textWidget.inputEl.style.background = "#f5f5f5";
            } else {
                textWidget.value = "No output available";
                textWidget.inputEl.style.background = "#f5f5f5";
            }
        } else {
            textWidget.value = "No output available";
            textWidget.inputEl.style.background = "#f5f5f5";
        }
    }
}

// ComfyUI 확장 등록
app.registerExtension({
    name: "ComfyQSAR_TEXT",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!TARGET_NODES.includes(nodeData.name)) return;

        // 노드 생성 시 처리 - 출력 위젯 생성
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            // 상태 변수 초기화
            this.currentMessage = "";
            this.currentVisualProgress = 0;
            this.currentTargetProgress = 0;
            this.animationFrameId = null;
            this.hasTextWidget = true; // 출력 위젯 존재 여부 플래그

            // 출력 위젯 생성
            const textWidget = createTextWidget(this);
            if (textWidget) {
                textWidget.value = "No output available";
                textWidget.inputEl.style.background = "#f5f5f5";
            }

            return result;
        };

        // 노드 실행 시 처리 - 출력 위젯 내용 업데이트
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            // 출력 위젯 내용 업데이트
            updateTextWidget(this, message);
        };

        // 노드 설정 시 처리
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);

            // 애니메이션 중지
            if (this.animationFrameId) {
                cancelAnimationFrame(this.animationFrameId);
                this.animationFrameId = null;
            }

            // 상태 초기화
            this.currentVisualProgress = 0;
            this.currentTargetProgress = 0;
            this.currentMessage = "";

            // 출력 위젯 내용 초기화
            updateTextWidget(this, null);
        };
    }
});