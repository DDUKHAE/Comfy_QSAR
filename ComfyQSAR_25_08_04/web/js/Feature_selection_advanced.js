import { app } from "../../../scripts/app.js";

// ComfyUI 표준 위젯 숨김 타입
const HIDDEN_TYPE = "hiddenWidget";

// 원본 위젯 속성 저장
const origProps = {};

// Method별 파라미터 매핑 (실제 Python 파일과 일치)
function getMethodParameters(method) {
    const parameterMappings = {
        "Lasso": ["alpha", "max_iter"],
        "RandomForest": ["n_estimators", "max_depth", "min_samples_split", "criterion", "n_iterations"],
        "DecisionTree": ["max_depth", "min_samples_split", "criterion", "n_iterations"],
        "XGBoost": ["n_estimators", "max_depth", "learning_rate"],
        "LightGBM": ["n_estimators", "max_depth", "learning_rate"],
        "RFE": ["base_model_for_selection", "n_features"],
        "SelectFromModel": ["base_model_for_selection", "threshold"]
    };
    return parameterMappings[method] || [];
}

// advanced가 false일 때 숨길 공통 위젯들
function getAdvancedOnlyWidgets() {
    return [
        // 모든 method별 파라미터들
        "alpha", "max_iter", "n_estimators", "max_depth", "min_samples_split", 
        "criterion", "learning_rate", "n_iterations", "base_model_for_selection", 
        "n_features", "threshold"
    ];
}

// 위젯 토글 함수 (ComfyUI 표준 방식)
function toggleWidget(node, widget, show = false, nodeKey = "") {
    if (!widget) {
        console.warn(`[Calculation Advanced] toggleWidget called with null widget`);
        return;
    }
    
    console.log(`[Calculation Advanced] 🔄 Toggling widget "${widget.name}" to ${show ? 'SHOW' : 'HIDE'}`);
    console.log(`[Calculation Advanced] Current widget type: ${widget.type}, computeSize: ${typeof widget.computeSize}`);
    
    const propKey = `${nodeKey}_${widget.name}`;
    
    // 원본 속성 저장 (한 번만)
    if (!origProps[propKey]) {
        origProps[propKey] = { 
            origType: widget.type, 
            origComputeSize: widget.computeSize,
            origComputedHeight: widget.computedHeight || 0,
            origDisabled: widget.disabled || false,
            origDisplay: widget.options?.display || "number"
        };
        console.log(`[Calculation Advanced] 💾 Saved original props for ${widget.name}: type=${widget.type}, display=${widget.options?.display || 'none'}`);
    }
    
    const props = origProps[propKey];
    
    if (show) {
        // 위젯 표시
        widget.type = props.origType;
        widget.computeSize = props.origComputeSize;
        widget.computedHeight = props.origComputedHeight;
        // 원본 disabled 상태 복원
        widget.disabled = props.origDisabled;
        // 원본 display 속성 복원
        if (widget.options && props.origDisplay) {
            widget.options.display = props.origDisplay;
        }
        // 위젯 표시 속성 복원
        widget.visible = true;
        widget.hidden = false;
        console.log(`[Calculation Advanced] ✅ Shown: ${widget.name} (restored type: ${props.origType}, disabled: ${widget.disabled}, display: ${widget.options?.display || 'none'})`);
    } else {
        // 위젯 숨김 - slider 위젯도 완전히 숨김
        widget.type = HIDDEN_TYPE;
        widget.computeSize = () => [0, -4];
        widget.computedHeight = 0;
        // 추가로 위젯을 완전히 비활성화
        widget.disabled = true;
        // slider 위젯의 경우 추가 처리
        if (widget.options && widget.options.display === "slider") {
            widget.options.display = "number"; // slider를 number로 변경
        }
        // 추가로 위젯을 완전히 숨기기 위한 강제 처리
        widget.visible = false;
        widget.hidden = true;
        console.log(`[Calculation Advanced] ❌ Hidden: ${widget.name} (type changed to: ${HIDDEN_TYPE}, disabled: ${widget.disabled})`);
    }
}

// 노드 크기 업데이트
function updateNodeSize(node) {
    setTimeout(() => {
        const newSize = node.computeSize();
        node.setSize(newSize);
        app.canvas.setDirty(true, true);
        console.log(`[Calculation Advanced] Node resized to: ${newSize}`);
    }, 10);
}

// ComfyUI 확장 등록
app.registerExtension({
    name: "QSAR.CalculationAdvanced",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Feature Selection 관련 노드들 확인
        const isFeatureSelectionNode = nodeData.name === "Feature_Selection_Regression" || 
                                       nodeData.name === "Feature_Selection_Classification" ||
                                       nodeData.name.includes("Feature_Selection");
        
        if (isFeatureSelectionNode) {
            console.log(`[Calculation Advanced] Registering dynamic control for: ${nodeData.name}`);
            
            const originalNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                originalNodeCreated?.apply(this, arguments);
                
                console.log(`[Calculation Advanced] Node created: ${this.title || nodeData.name}`);
                
                // 노드별 고유 키
                const nodeKey = `${nodeData.name}_${this.id || Date.now()}`;
                this.nodeKey = nodeKey;
                
                // 모든 조건부 파라미터 목록 (실제 Python 파일과 일치)
                const allConditionalParams = [
                    "n_features", "threshold", "alpha", "max_iter",
                    "n_estimators", "max_depth", "min_samples_split", 
                    "criterion", "learning_rate", "n_iterations",
                    "base_model_for_selection"
                ];
                
                // 제어 위젯들 찾기 (method 또는 model 등)
                const methodWidget = this.widgets.find(w => w.name === "method") ||
                                   this.widgets.find(w => w.name === "model") ||
                                   this.widgets.find(w => w.name === "base_model");
                const advancedWidget = this.widgets.find(w => w.name === "advanced");
                
                if (!methodWidget) {
                    console.warn(`[Calculation Advanced] No method/model widget found in ${nodeData.name}`);
                    return;
                }
                
                if (!advancedWidget) {
                    console.warn(`[Calculation Advanced] Advanced widget not found - advanced mode control disabled`);
                }
                
                console.log(`[Calculation Advanced] Control widgets found - method: ${methodWidget.value}, advanced: ${advancedWidget?.value || 'N/A'}`);
                console.log(`[Calculation Advanced] Available widgets: ${this.widgets.map(w => w.name).join(', ')}`);
                console.log(`[Calculation Advanced] Conditional widgets: ${allConditionalParams.join(', ')}`);
                
                // 모든 조건부 위젯의 원본 속성 저장
                allConditionalParams.forEach(paramName => {
                    const widget = this.widgets.find(w => w.name === paramName);
                    if (widget) {
                        const propKey = `${nodeKey}_${paramName}`;
                        origProps[propKey] = { 
                            origType: widget.type, 
                            origComputeSize: widget.computeSize,
                            origComputedHeight: widget.computedHeight || 0,
                            origDisabled: widget.disabled || false,
                            origDisplay: widget.options?.display || "number"
                        };
                        console.log(`[Calculation Advanced] Registered widget: ${paramName} (type: ${widget.type})`);
                    } else {
                        console.warn(`[Calculation Advanced] Widget not found: ${paramName}`);
                    }
                });
                
                // 위젯 가시성 업데이트 함수 (이중 조건부)
                const updateParameterVisibility = (selectedMethod, isAdvanced) => {
                    console.log(`[Calculation Advanced] 🔄 Updating visibility - method: ${selectedMethod}, advanced: ${isAdvanced}`);
                    
                    const activeParams = getMethodParameters(selectedMethod);
                    const advancedOnlyWidgets = getAdvancedOnlyWidgets();
                    
                    console.log(`[Calculation Advanced] Active params for ${selectedMethod}:`, activeParams);
                    
                    let changedCount = 0;
                    allConditionalParams.forEach(paramName => {
                        const widget = this.widgets.find(w => w.name === paramName);
                        if (widget) {
                            // 이중 조건: advanced가 true이고 해당 method에 필요한 파라미터
                            const isMethodParam = activeParams.includes(paramName);
                            const isAdvancedParam = advancedOnlyWidgets.includes(paramName);
                            const shouldShow = isAdvanced && isMethodParam;
                            
                            console.log(`[Calculation Advanced] 🔧 Processing widget: ${paramName} (type: ${widget.type}, display: ${widget.options?.display || 'none'})`);
                            toggleWidget(this, widget, shouldShow, nodeKey);
                            changedCount++;
                        }
                    });
                    
                    if (changedCount > 0) {
                        updateNodeSize(this);
                    }
                    
                    console.log(`[Calculation Advanced] ✅ Updated ${changedCount} widgets for method: ${selectedMethod}, advanced: ${isAdvanced}`);
                };
                
                // method 값 변경 감지 (Property descriptor 방식)
                const methodDesc = Object.getOwnPropertyDescriptor(methodWidget, "value") || {};
                let methodValue = methodWidget.value;
                
                Object.defineProperty(methodWidget, "value", {
                    get() {
                        return methodDesc.get ? methodDesc.get.call(methodWidget) : methodValue;
                    },
                    set(newVal) {
                        console.log(`[Calculation Advanced] 🔀 Method changed from "${methodValue}" to "${newVal}"`);
                        
                        if (methodDesc.set) methodDesc.set.call(methodWidget, newVal);
                        else methodValue = newVal;
                        
                        // 파라미터 가시성 업데이트
                        updateParameterVisibility(newVal, advancedWidget?.value || false);
                    }
                });
                
                // advanced 값 변경 감지 (있는 경우에만)
                if (advancedWidget) {
                    const advancedDesc = Object.getOwnPropertyDescriptor(advancedWidget, "value") || {};
                    let advancedValue = advancedWidget.value;
                    
                    Object.defineProperty(advancedWidget, "value", {
                        get() {
                            return advancedDesc.get ? advancedDesc.get.call(advancedWidget) : advancedValue;
                        },
                        set(newVal) {
                            console.log(`[Calculation Advanced] 🔀 Advanced changed from ${advancedValue} to: ${newVal}`);
                            
                            if (advancedDesc.set) advancedDesc.set.call(advancedWidget, newVal);
                            else advancedValue = newVal;
                            
                            // 파라미터 가시성 업데이트
                            updateParameterVisibility(methodWidget.value, newVal);
                        }
                    });
                }
                
                // 초기 상태 설정
                console.log(`[Calculation Advanced] 🚀 Setting initial state...`);
                updateParameterVisibility(methodWidget.value || "Lasso", advancedWidget?.value || false);
            };
            
            // 노드 제거 시 정리
            const originalOnRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
                console.log(`[Calculation Advanced] 🗑️ Cleaning up node: ${this.nodeKey}`);
                
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

console.log("🎯 QSAR Calculation Advanced Extension Loaded (Enhanced Version)");
