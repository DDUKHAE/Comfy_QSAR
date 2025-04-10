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
    const algorithmValue = findWidgetByName(node, 'algorithm')?.value || "";
    
    // 위젯 값 로깅 (디버깅용)
    console.log(`Algorithm: ${algorithmValue}, Advanced: ${isAdvanced}`);

    // 위젯 토글 헬퍼 함수
    const toggleWidgets = (widgets, condition) => {
        widgets.forEach(widgetName => {
            const widget = findWidgetByName(node, widgetName);
            if (widget) {
                toggleWidget(node, widget, condition);
            }
        });
    };

    switch (widget.name) {
        case 'advanced':
        case 'algorithm': {
            // 1. 공통 위젯 토글 (advanced가 true일 때만 표시)
            const commonWidgets = [
                'test_size', 'num_cores', 'cv_splits', 'verbose', 'random_state', 'target_column'
            ];
            
            // 공통 위젯 토글
            toggleWidgets(commonWidgets, isAdvanced);

            // 2. 알고리즘별 위젯 토글 (advanced가 true이고 해당 알고리즘이 선택된 경우만 표시)
            
            // XGBoost 관련 위젯
            toggleWidgets(['n_estimators', 'learning_rate', 'max_depth', 'subsample', 'reg_alpha', 'reg_lambda'], 
                isAdvanced && algorithmValue === "xgboost");
            
            // LightGBM 관련 위젯
            toggleWidgets(['n_estimators', 'learning_rate', 'max_depth', 'num_leaves', 'reg_alpha', 'reg_lambda'],
                isAdvanced && algorithmValue === "lightgbm");
            
            // Random Forest 관련 위젯
            toggleWidgets(['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'],
                isAdvanced && algorithmValue === "random_forest");
            
            // Decision Tree 관련 위젯
            toggleWidgets(['max_depth', 'min_samples_split', 'min_samples_leaf', 'criterion'],
                isAdvanced && algorithmValue === "decision_tree");
            
            // SVM 관련 위젯
            toggleWidgets(['C', 'kernel', 'gamma', 'epsilon'],
                isAdvanced && algorithmValue === "svm");
            
            // Ridge 관련 위젯
            toggleWidgets(['alpha'],
                isAdvanced && algorithmValue === "ridge");
            
            // Lasso 관련 위젯
            toggleWidgets(['alpha'],
                isAdvanced && algorithmValue === "lasso");
            
            // ElasticNet 관련 위젯
            toggleWidgets(['alpha', 'l1_ratio'],
                isAdvanced && algorithmValue === "elasticnet");
            
            // Linear Regression은 특별한 파라미터가 없음
            break;
        }
    }
}

// 모니터링할 위젯 목록
const getSetWidgets = ['advanced', 'algorithm'];

// Grid Search 노드들의 타이틀 및 타입 목록
const getSetTitles = [
    // 원래 이름
    "Grid Search Hyperparameter",
    
    // Regression 버전
    "Grid Search Hyperparameter (Regression)",
    "Hyperparameter_Grid_Search_Regression",
    
    // Classification 버전
    "Grid Search Hyperparameter (Classification)",
    "Hyperparameter_Grid_Search_Classification"
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
    name: "Grid_Search_Advanced",

    nodeCreated(node) {
        const nodeTitle = node.constructor.title;
        const nodeType = node.type;
        
        // 노드의 타이틀이나 타입이 목록에 있는지 확인
        if (getSetTitles.includes(nodeTitle) || 
            getSetTitles.some(title => nodeType.includes(title)) ||
            nodeType.includes("Grid_search_hyperparameter") ||
            nodeType.includes("Hyperparameter_Grid_Search")) {
            
            console.log(`적용됨: ${nodeTitle} (${nodeType})`);
            getSetters(node);
        }
    }
});
