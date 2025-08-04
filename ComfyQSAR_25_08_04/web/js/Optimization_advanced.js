import { app } from "../../../scripts/app.js";

// 원본 위젯 속성 저장
const origProps = {};

// correlation_mode별 파라미터 매핑
function getCorrelationModeParameters(correlationMode) {
    const parameterMappings = {
        "target_based": ["importance_model"],
        "upper": [],
        "lower": []
    };
    return parameterMappings[correlationMode] || [];
}

// importance_model별 파라미터 매핑
function getImportanceModelParameters(importanceModel) {
    const parameterMappings = {
        "lasso": ["alpha", "max_iter"],
        "random_forest": ["n_estimators"]
    };
    return parameterMappings[importanceModel] || [];
}

// 모든 조건부 파라미터 목록
function getAllConditionalParams() {
    return [
        "importance_model", "alpha", "max_iter", "n_estimators"
    ];
}

// 위젯 토글 함수 (ComfyUI 표준 방식)
function toggleWidget(node, widget, show = false, nodeKey = "") {
    if (!widget) {
        console.warn(`[Feature Selection Advanced] toggleWidget called with null widget`);
        return;
    }
    
    console.log(`[Feature Selection Advanced] 🔄 Toggling widget "${widget.name}" to ${show ? 'SHOW' : 'HIDE'}`);
    
    const propKey = `${nodeKey}_${widget.name}`;
    
    // 원본 속성 저장 (한 번만)
    if (!origProps[propKey]) {
        origProps[propKey] = { 
            origType: widget.type, 
            origComputeSize: widget.computeSize,
            origComputedHeight: widget.computedHeight || 0,
            origHidden: widget.hidden || false
        };
        console.log(`[Feature Selection Advanced] 💾 Saved original props for ${widget.name}: type=${widget.type}, hidden=${widget.hidden}`);
    }
    
    const props = origProps[propKey];
    
    if (show) {
        // 위젯 표시
        widget.type = props.origType;
        widget.computeSize = props.origComputeSize;
        widget.computedHeight = props.origComputedHeight;
        widget.hidden = props.origHidden;
        
        // DOM 요소가 있다면 표시
        if (widget.element) {
            widget.element.style.display = "";
        }
        
        console.log(`[Feature Selection Advanced] ✅ Shown: ${widget.name} (restored type: ${props.origType})`);
    } else {
        // 위젯 완전 숨김
        widget.type = null;
        widget.computeSize = () => [0, 0];
        widget.computedHeight = 0;
        widget.hidden = true;
        
        // DOM 요소가 있다면 숨김
        if (widget.element) {
            widget.element.style.display = "none";
        }
        
        console.log(`[Feature Selection Advanced] ❌ Hidden: ${widget.name} (type: null, hidden: true)`);
    }
}

// 노드 크기 업데이트
function updateNodeSize(node) {
    setTimeout(() => {
        // 위젯들의 computeSize를 강제로 다시 계산
        node.widgets?.forEach(widget => {
            if (widget.computeSize && typeof widget.computeSize === 'function') {
                widget.computeSize();
            }
        });
        
        const newSize = node.computeSize();
        node.setSize(newSize);
        
        // 캔버스 강제 리프레시
        app.canvas.setDirty(true, true);
        app.graph.setDirtyCanvas(true, true);
        
        console.log(`[Feature Selection Advanced] Node resized to: ${newSize}`);
    }, 20);
}

// ComfyUI 확장 등록
app.registerExtension({
    name: "QSAR.FeatureSelectionAdvanced",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Descriptor Optimization 관련 노드들 확인
        const isDescriptorOptimizationNode = nodeData.name === "Descriptor_Optimization_Classification" || 
                                           nodeData.name === "Descriptor_Optimization_Regression" ||
                                           nodeData.name === "Remove_High_Correlation_Features_Classification" ||
                                           nodeData.name === "Remove_High_Correlation_Features_Regression" ||
                                           nodeData.name.includes("Descriptor_Optimization") ||
                                           nodeData.name.includes("Remove_High_Correlation");
        
        if (isDescriptorOptimizationNode) {
            console.log(`[Feature Selection Advanced] Registering dynamic control for: ${nodeData.name}`);
            
            const originalNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                originalNodeCreated?.apply(this, arguments);
                
                console.log(`[Feature Selection Advanced] Node created: ${this.title || nodeData.name}`);
                
                // 노드별 고유 키
                const nodeKey = `${nodeData.name}_${this.id || Date.now()}`;
                this.nodeKey = nodeKey;
                
                // 모든 조건부 파라미터 목록
                const allConditionalParams = getAllConditionalParams();
                
                // 제어 위젯들 찾기
                const correlationModeWidget = this.widgets.find(w => w.name === "correlation_mode");
                const importanceModelWidget = this.widgets.find(w => w.name === "importance_model");
                
                if (!correlationModeWidget) {
                    console.warn(`[Feature Selection Advanced] No correlation_mode widget found in ${nodeData.name}`);
                    return;
                }
                
                console.log(`[Feature Selection Advanced] Control widgets found - correlation_mode: ${correlationModeWidget.value}, importance_model: ${importanceModelWidget?.value || 'N/A'}`);
                console.log(`[Feature Selection Advanced] Available widgets: ${this.widgets.map(w => w.name).join(', ')}`);
                console.log(`[Feature Selection Advanced] Conditional widgets: ${allConditionalParams.join(', ')}`);
                
                // 모든 조건부 위젯의 원본 속성 저장
                allConditionalParams.forEach(paramName => {
                    const widget = this.widgets.find(w => w.name === paramName);
                    if (widget) {
                        const propKey = `${nodeKey}_${paramName}`;
                        origProps[propKey] = { 
                            origType: widget.type, 
                            origComputeSize: widget.computeSize,
                            origComputedHeight: widget.computedHeight || 0
                        };
                        console.log(`[Feature Selection Advanced] Registered widget: ${paramName} (type: ${widget.type})`);
                    } else {
                        console.warn(`[Feature Selection Advanced] Widget not found: ${paramName}`);
                    }
                });
                
                // 위젯 가시성 업데이트 함수 (다중 조건부)
                const updateParameterVisibility = (correlationMode, importanceModel) => {
                    console.log(`[Feature Selection Advanced] 🔄 Updating visibility - correlation_mode: ${correlationMode}, importance_model: ${importanceModel}`);
                    
                    const correlationParams = getCorrelationModeParameters(correlationMode);
                    const importanceParams = getImportanceModelParameters(importanceModel);
                    
                    console.log(`[Feature Selection Advanced] Active params for correlation_mode ${correlationMode}:`, correlationParams);
                    console.log(`[Feature Selection Advanced] Active params for importance_model ${importanceModel}:`, importanceParams);
                    
                    let changedCount = 0;
                    allConditionalParams.forEach(paramName => {
                        const widget = this.widgets.find(w => w.name === paramName);
                        if (widget) {
                            // 조건부 표시 로직
                            let shouldShow = false;
                            
                            if (paramName === "importance_model") {
                                // importance_model은 correlation_mode가 target_based일 때만 표시
                                shouldShow = correlationMode === "target_based";
                            } else if (["alpha", "max_iter", "n_estimators"].includes(paramName)) {
                                // 이 파라미터들은 correlation_mode가 target_based이고 해당 importance_model에 속할 때만 표시
                                shouldShow = correlationMode === "target_based" && importanceParams.includes(paramName);
                            }
                            
                            console.log(`[Feature Selection Advanced] 🔧 Processing widget: ${paramName} (type: ${widget.type}) - shouldShow: ${shouldShow}`);
                            toggleWidget(this, widget, shouldShow, nodeKey);
                            changedCount++;
                        }
                    });
                    
                    if (changedCount > 0) {
                        console.log(`[Feature Selection Advanced] Triggering node resize...`);
                        updateNodeSize(this);
                    }
                    
                    console.log(`[Feature Selection Advanced] ✅ Updated ${changedCount} widgets for correlation_mode: ${correlationMode}, importance_model: ${importanceModel}`);
                };
                
                // correlation_mode 값 변경 감지 (Property descriptor 방식)
                const correlationDesc = Object.getOwnPropertyDescriptor(correlationModeWidget, "value") || {};
                let correlationValue = correlationModeWidget.value;
                
                // 이미 정의된 속성인지 확인
                if (!correlationModeWidget._valueRedefined) {
                    try {
                        Object.defineProperty(correlationModeWidget, "value", {
                            get() {
                                return correlationDesc.get ? correlationDesc.get.call(correlationModeWidget) : correlationValue;
                            },
                            set(newVal) {
                                console.log(`[Feature Selection Advanced] 🔀 correlation_mode changed from "${correlationValue}" to "${newVal}"`);
                                
                                if (correlationDesc.set) correlationDesc.set.call(correlationModeWidget, newVal);
                                else correlationValue = newVal;
                                
                                // 파라미터 가시성 업데이트
                                updateParameterVisibility(newVal, importanceModelWidget?.value || "lasso");
                            }
                        });
                        correlationModeWidget._valueRedefined = true;
                    } catch (error) {
                        console.warn(`[Feature Selection Advanced] Could not redefine correlation_mode widget value property: ${error.message}`);
                    }
                }
                
                // importance_model 값 변경 감지 (있는 경우에만)
                if (importanceModelWidget && !importanceModelWidget._valueRedefined) {
                    const importanceDesc = Object.getOwnPropertyDescriptor(importanceModelWidget, "value") || {};
                    let importanceValue = importanceModelWidget.value;
                    
                    try {
                        Object.defineProperty(importanceModelWidget, "value", {
                            get() {
                                return importanceDesc.get ? importanceDesc.get.call(importanceModelWidget) : importanceValue;
                            },
                            set(newVal) {
                                console.log(`[Feature Selection Advanced] 🔀 importance_model changed from ${importanceValue} to: ${newVal}`);
                                
                                if (importanceDesc.set) importanceDesc.set.call(importanceModelWidget, newVal);
                                else importanceValue = newVal;
                                
                                // 파라미터 가시성 업데이트
                                updateParameterVisibility(correlationModeWidget.value, newVal);
                            }
                        });
                        importanceModelWidget._valueRedefined = true;
                    } catch (error) {
                        console.warn(`[Feature Selection Advanced] Could not redefine importance_model widget value property: ${error.message}`);
                    }
                }
                
                // 초기 상태 설정
                console.log(`[Feature Selection Advanced] 🚀 Setting initial state for correlation_mode: ${correlationModeWidget.value}, importance_model: ${importanceModelWidget?.value || 'lasso'}`);
                updateParameterVisibility(correlationModeWidget.value || "target_based", importanceModelWidget?.value || "lasso");
            };
            
            // 노드 제거 시 정리
            const originalOnRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
                console.log(`[Feature Selection Advanced] 🗑️ Cleaning up node: ${this.nodeKey}`);
                
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

console.log("🎯 QSAR Feature Selection Advanced Extension Loaded (Enhanced Version)");
