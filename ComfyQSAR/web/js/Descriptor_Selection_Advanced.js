import { app } from "../../../scripts/app.js";

let origProps = {};

const findWidgetByName = (node, name) => node.widgets.find((w) => w.name === name);

const doesInputWithNameExist = (node, name) => node.inputs ? node.inputs.some((input) => input.name === name) : false;

function updateNodeHeight(node) {
    node.setSize([node.size[0], node.computeSize()[1]]);
    app.canvas.dirty_canvas = true;
}

const hiddenType = "hiddenWidget";

function toggleWidget(node, widget, show = false, suffix = "") {
    if (!widget || doesInputWithNameExist(node, widget.name)) return;
    if (!origProps[widget.name]) {
        origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize, origComputedHeight: widget.computedHeight };    
    }
    const origSize = node.size;

    widget.type = show ? origProps[widget.name].origType : hiddenType + suffix;
    widget.computeSize = show ? origProps[widget.name].origComputeSize : () => [0, -4];
    widget.computedHeight = show ? origProps[widget.name].origComputedHeight : 0;

    widget.linkedWidgets?.forEach(w => toggleWidget(node, w, show, ":" + widget.name));    

    const height = show ? Math.max(node.computeSize()[1], origSize[1]) : node.size[1];
    node.setSize([node.size[0], height]);
    app.canvas.dirty_canvas = true;
}

function widgetLogic(node, widget) {
    // 필요한 위젯 상태 가져오기
    const isAdvanced = findWidgetByName(node, 'advanced')?.value === true;
    const methodValue = findWidgetByName(node, 'method')?.value || "";
    const modelValue = findWidgetByName(node, 'model')?.value || "";

    // 메서드 및 모델 값 로깅 (디버깅용)
    console.log(`Method: ${methodValue}, Model: ${modelValue}, Advanced: ${isAdvanced}`);

    // 위젯 토글 헬퍼 함수
    const toggleWidgets = (widgets, condition) => {
        widgets.forEach(widgetName => {
            const widget = findWidgetByName(node, widgetName);
            if (widget) {
                toggleWidget(node, widget, condition);
            }
        });
    };

    // model 위젯의 값 범위 조정 함수
    const updateModelOptions = (options) => {
        const modelWidget = findWidgetByName(node, 'model');
        if (!modelWidget) return;
        
        // model 위젯의 옵션 업데이트
        if (modelWidget.options) {
            // 현재 선택된 값이 새 옵션에 없으면 첫번째 옵션으로 선택
            if (!options.includes(modelWidget.value)) {
                modelWidget.value = options[0];
            }
            // options 배열 대신 ComfyUI 형식으로 변환
            modelWidget.options = { values: options };
        }
    };

    switch (widget.name) {
        case 'advanced':
        case 'method':
        case 'model': {
            // 1. method 값에 따라 model 위젯 토글 및 옵션 설정
            const modelWidget = findWidgetByName(node, 'model');
            
            if (methodValue === "RFE") {
                // RFE일 때는 모델 위젯 표시하고 RandomForest와 DecisionTree만 선택 가능
                if (modelWidget) {
                    toggleWidget(node, modelWidget, true);
                    updateModelOptions(["RandomForest", "DecisionTree"]);
                }
            } else if (methodValue === "SelectFromModel") {
                // SelectFromModel일 때는 모델 위젯 표시하고 모든 모델 선택 가능
                if (modelWidget) {
                    toggleWidget(node, modelWidget, true);
                    updateModelOptions(["Lasso", "RandomForest", "DecisionTree", "XGBoost", "LightGBM"]);
                }
            } else {
                // 그 외의 경우 모델 위젯 숨김
                if (modelWidget) {
                    toggleWidget(node, modelWidget, false);
                }
            }

            // 2. 공통 위젯 (항상 advanced 여부에 따라 토글)
            toggleWidgets(['n_features', 'target_column'], isAdvanced);

            // 3. 모델별 파라미터 위젯 표시 여부 설정
            // 현재 method가 무엇인지 확인하고, RFE나 SelectFromModel인 경우는 model 값에 따라 결정
            const activeModel = (methodValue === "RFE" || methodValue === "SelectFromModel") ? modelValue : methodValue;
            
            // 각 모델별 파라미터 위젯 토글
            // Lasso 관련 위젯
            toggleWidgets(['alpha', 'max_iter'], isAdvanced && 
                (activeModel === "Lasso" || (methodValue === "SelectFromModel" && modelValue === "Lasso")));
            
            // RandomForest, XGBoost, LightGBM 공통 위젯
            toggleWidgets(['n_estimators'], isAdvanced && 
                (activeModel === "RandomForest" || activeModel === "XGBoost" || activeModel === "LightGBM"));
            
            // Decision Tree, Random Forest 공통 위젯
            toggleWidgets(['max_depth'], isAdvanced && 
                (activeModel === "DecisionTree" || activeModel === "RandomForest" || 
                 activeModel === "XGBoost" || activeModel === "LightGBM"));
            
            toggleWidgets(['min_samples_split', 'criterion'], isAdvanced && 
                (activeModel === "DecisionTree" || activeModel === "RandomForest"));
            
            // n_iterations는 RandomForest와 DecisionTree에서만 사용
            toggleWidgets(['n_iterations'], isAdvanced && 
                (activeModel === "RandomForest" || activeModel === "DecisionTree"));
            
            // XGBoost, LightGBM 관련 위젯
            toggleWidgets(['learning_rate'], isAdvanced && 
                (activeModel === "XGBoost" || activeModel === "LightGBM"));
            
            // RFE 관련 위젯
            toggleWidgets(['step', 'verbose'], isAdvanced && methodValue === "RFE");
            
            // SelectFromModel 관련 위젯
            toggleWidgets(['threshold', 'max_features', 'prefit'], 
                isAdvanced && methodValue === "SelectFromModel");
            
            break;
        }
    }
}

// 모니터링할 위젯 목록
const getSetWidgets = ['advanced', 'method', 'model'];

// 디스크립터 선택 노드들의 타이틀 및 타입 목록
const getSetTitles = [
    // Regression 버전
    "Feature Selection(Regression)",
    "Feature_Selection_Regression", 
    
    // Classification 버전
    "Feature Selection(Classification)",
    "Feature_Selection_Classification"
];

function getSetters(node) {
    if (node.widgets) {
        for (const w of node.widgets) {
            if (getSetWidgets.includes(w.name)) {
                widgetLogic(node, w);
                let widgetValue = w.value;

                Object.defineProperty(w, 'value', {
                    get() {
                        return widgetValue;
                    },
                    set(newVal) {
                        if (newVal !== widgetValue) {
                            widgetValue = newVal;
                            widgetLogic(node, w);
                        }
                    }
                });
            }
        }
    }
}

app.registerExtension({
    name: "Descriptor_Selection_Advanced",

    nodeCreated(node) {
        const nodeTitle = node.constructor.title;
        const nodeType = node.type;
        
        // 노드의 타이틀이나 타입이 목록에 있는지 확인
        if (getSetTitles.includes(nodeTitle) || 
            getSetTitles.some(title => nodeType.includes(title)) ||
            nodeType.includes("Feature_Selection")) {
            
            console.log(`적용됨: ${nodeTitle} (${nodeType})`);
            getSetters(node);
        }
    }
});
