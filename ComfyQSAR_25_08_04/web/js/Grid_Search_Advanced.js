import { app } from "../../../scripts/app.js";

// ComfyUI 표준 위젯 숨김 타입
const HIDDEN_TYPE = "hiddenWidget";

// 원본 위젯 속성 저장
const origProps = {};

// 위젯 토글 함수 (ComfyUI 표준 방식)
function toggleWidget(node, widget, show = false, nodeKey = "") {
    if (!widget) return;
    
    const propKey = `${nodeKey}_${widget.name}`;
    
    // 원본 속성 저장 (한 번만)
    if (!origProps[propKey]) {
        origProps[propKey] = { 
            origType: widget.type, 
            origComputeSize: widget.computeSize,
            origComputedHeight: widget.computedHeight || 0
        };
    }
    
    const props = origProps[propKey];
    
    if (show) {
        // 위젯 표시
        widget.type = props.origType;
        widget.computeSize = props.origComputeSize;
        widget.computedHeight = props.origComputedHeight;
        console.log(`[Grid Search] ✅ Shown: ${widget.name}`);
    } else {
        // 위젯 숨김
        widget.type = HIDDEN_TYPE;
        widget.computeSize = () => [0, -4];
        widget.computedHeight = 0;
        console.log(`[Grid Search] ❌ Hidden: ${widget.name}`);
    }
}

// 노드 크기 업데이트
function updateNodeSize(node) {
    setTimeout(() => {
        const newSize = node.computeSize();
        node.setSize(newSize);
        app.canvas.setDirty(true, true);
        console.log(`[Grid Search] Node resized to: ${newSize}`);
    }, 10);
}

// 노드 타입 감지
function isGridSearchNode(nodeData) {
    const targetTypes = [
        "Hyperparameter_Grid_Search_Regression",
        "Hyperparameter_Grid_Search_Classification",
        "Grid_search_hyperparameter"
    ];
    return targetTypes.some(type => nodeData.name.includes(type)) || 
           nodeData.name.includes("Grid_Search") ||
           nodeData.name.includes("Hyperparameter");
}

// 공통 위젯 목록 (advanced가 true일 때만 표시)
function getCommonWidgets() {
    return [
        'test_size', 'num_cores', 'cv_splits', 'verbose', 'random_state', 'target_column'
    ];
}

// 알고리즘별 위젯 매핑
function getAlgorithmWidgets(algorithm) {
    const algorithmWidgets = {
        "xgboost": ['n_estimators', 'learning_rate', 'max_depth', 'subsample', 'reg_alpha', 'reg_lambda'],
        "lightgbm": ['n_estimators', 'learning_rate', 'max_depth', 'num_leaves', 'reg_alpha', 'reg_lambda'],
        "random_forest": ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'],
        "decision_tree": ['max_depth', 'min_samples_split', 'min_samples_leaf', 'criterion'],
        "svm": ['C', 'kernel', 'gamma', 'epsilon'],
        "ridge": ['alpha'],
        "lasso": ['alpha'],
        "elasticnet": ['alpha', 'l1_ratio']
        // linear regression은 특별한 파라미터가 없음
    };
    return algorithmWidgets[algorithm] || [];
}

// 모든 조건부 위젯 목록
function getAllConditionalWidgets() {
    const commonWidgets = getCommonWidgets();
    const allAlgorithmWidgets = [
        ...getAlgorithmWidgets("xgboost"),
        ...getAlgorithmWidgets("lightgbm"),
        ...getAlgorithmWidgets("random_forest"),
        ...getAlgorithmWidgets("decision_tree"),
        ...getAlgorithmWidgets("svm"),
        ...getAlgorithmWidgets("ridge"),
        ...getAlgorithmWidgets("lasso"),
        ...getAlgorithmWidgets("elasticnet")
    ];
    
    // 중복 제거
    return [...new Set([...commonWidgets, ...allAlgorithmWidgets])];
}

// ComfyUI 확장 등록
app.registerExtension({
    name: "QSAR.GridSearchAdvanced",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (isGridSearchNode(nodeData)) {
            console.log(`[Grid Search] Registering advanced control for: ${nodeData.name}`);
            
            const originalNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                originalNodeCreated?.apply(this, arguments);
                
                console.log(`[Grid Search] Node created: ${this.title || nodeData.name}`);
                
                // 노드별 고유 키
                const nodeKey = `${nodeData.name}_${this.id || Date.now()}`;
                this.nodeKey = nodeKey;
                
                // 모든 조건부 위젯 목록
                const allConditionalWidgets = getAllConditionalWidgets();
                
                // 제어 위젯들 찾기
                const advancedWidget = this.widgets.find(w => w.name === "advanced");
                const algorithmWidget = this.widgets.find(w => w.name === "algorithm");
                
                if (!advancedWidget) {
                    console.log(`[Grid Search] Advanced widget not found`);
                    return;
                }
                
                if (!algorithmWidget) {
                    console.log(`[Grid Search] Algorithm widget not found`);
                    return;
                }
                
                console.log(`[Grid Search] Control widgets found - advanced: ${advancedWidget.value}, algorithm: ${algorithmWidget.value}`);
                console.log(`[Grid Search] Available widgets: ${this.widgets.map(w => w.name).join(', ')}`);
                console.log(`[Grid Search] Conditional widgets: ${allConditionalWidgets.join(', ')}`);
                
                // 모든 조건부 위젯의 원본 속성 저장
                allConditionalWidgets.forEach(widgetName => {
                    const widget = this.widgets.find(w => w.name === widgetName);
                    if (widget) {
                        const propKey = `${nodeKey}_${widgetName}`;
                        origProps[propKey] = { 
                            origType: widget.type, 
                            origComputeSize: widget.computeSize,
                            origComputedHeight: widget.computedHeight || 0
                        };
                        console.log(`[Grid Search] Registered widget: ${widgetName} (type: ${widget.type})`);
                    } else {
                        console.warn(`[Grid Search] Widget not found: ${widgetName}`);
                    }
                });
                
                // 위젯 가시성 업데이트 함수
                const updateWidgetVisibility = (isAdvanced, selectedAlgorithm) => {
                    console.log(`[Grid Search] 🔄 Updating visibility - advanced: ${isAdvanced}, algorithm: ${selectedAlgorithm}`);
                    
                    let changedCount = 0;
                    
                    // 1. 공통 위젯들 (advanced가 true일 때만 표시)
                    const commonWidgets = getCommonWidgets();
                    commonWidgets.forEach(widgetName => {
                        const widget = this.widgets.find(w => w.name === widgetName);
                        if (widget) {
                            toggleWidget(this, widget, isAdvanced, nodeKey);
                            changedCount++;
                        }
                    });
                    
                    // 2. 알고리즘별 위젯들 (advanced가 true이고 해당 알고리즘이 선택된 경우만 표시)
                    const allAlgorithms = ["xgboost", "lightgbm", "random_forest", "decision_tree", "svm", "ridge", "lasso", "elasticnet"];
                    
                    allAlgorithms.forEach(algorithm => {
                        const algorithmWidgets = getAlgorithmWidgets(algorithm);
                        const shouldShow = isAdvanced && selectedAlgorithm === algorithm;
                        
                        algorithmWidgets.forEach(widgetName => {
                            const widget = this.widgets.find(w => w.name === widgetName);
                            if (widget) {
                                toggleWidget(this, widget, shouldShow, nodeKey);
                                changedCount++;
                            }
                        });
                    });
                    
                    if (changedCount > 0) {
                        updateNodeSize(this);
                    }
                    
                    console.log(`[Grid Search] ✅ Updated ${changedCount} widgets for advanced: ${isAdvanced}, algorithm: ${selectedAlgorithm}`);
                };
                
                // advanced 값 변경 감지 (Property descriptor 방식)
                const advancedDesc = Object.getOwnPropertyDescriptor(advancedWidget, "value") || {};
                let advancedValue = advancedWidget.value;
                
                Object.defineProperty(advancedWidget, "value", {
                    get() {
                        return advancedDesc.get ? advancedDesc.get.call(advancedWidget) : advancedValue;
                    },
                    set(newVal) {
                        console.log(`[Grid Search] 🔀 Advanced changed from ${advancedValue} to: ${newVal}`);
                        
                        if (advancedDesc.set) advancedDesc.set.call(advancedWidget, newVal);
                        else advancedValue = newVal;
                        
                        // 위젯 가시성 업데이트
                        updateWidgetVisibility(newVal, algorithmWidget.value);
                    }
                });
                
                // algorithm 값 변경 감지 (Property descriptor 방식)
                const algorithmDesc = Object.getOwnPropertyDescriptor(algorithmWidget, "value") || {};
                let algorithmValue = algorithmWidget.value;
                
                Object.defineProperty(algorithmWidget, "value", {
                    get() {
                        return algorithmDesc.get ? algorithmDesc.get.call(algorithmWidget) : algorithmValue;
                    },
                    set(newVal) {
                        console.log(`[Grid Search] 🔀 Algorithm changed from "${algorithmValue}" to "${newVal}"`);
                        
                        if (algorithmDesc.set) algorithmDesc.set.call(algorithmWidget, newVal);
                        else algorithmValue = newVal;
                        
                        // 위젯 가시성 업데이트
                        updateWidgetVisibility(advancedWidget.value, newVal);
                    }
                });
                
                // 초기 상태 설정
                console.log(`[Grid Search] 🚀 Setting initial state...`);
                updateWidgetVisibility(advancedWidget.value || false, algorithmWidget.value || "");
            };
            
            // 노드 제거 시 정리
            const originalOnRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
                console.log(`[Grid Search] 🗑️ Cleaning up node: ${this.nodeKey}`);
                
                // 이 노드의 원본 속성들 정리
                if (this.nodeKey) {
                    Object.keys(origProps).forEach(key => {
                        if (key.startsWith(this.nodeKey)) {
                            delete origProps[key];
                        }
                    });
                }
                
                originalOnRemoved?.apply(this, arguments);
            };
        }
    }
});

console.log("🎯 QSAR Grid Search Advanced Extension Loaded (Fixed Version)");
