import { app } from "../../../scripts/app.js";

// ComfyUI 표준 위젯 숨김 타입
const HIDDEN_TYPE = "hiddenWidget";

// 원본 위젯 속성 저장
const origProps = {};

// 위젯 토글 함수 (ComfyUI 표준 방식)
function toggleWidget(node, widget, show = false, nodeKey = "") {
    if (!widget) {
        console.warn(`[Parameter Hide] toggleWidget called with null widget`);
        return;
    }
    
    console.log(`[Parameter Hide] 🔄 Toggling widget "${widget.name}" to ${show ? 'SHOW' : 'HIDE'}`);
    console.log(`[Parameter Hide] Current widget type: ${widget.type}, computeSize: ${typeof widget.computeSize}`);
    
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
        console.log(`[Parameter Hide] 💾 Saved original props for ${widget.name}: type=${widget.type}, display=${widget.options?.display || 'none'}`);
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
        console.log(`[Parameter Hide] ✅ Shown: ${widget.name} (restored type: ${props.origType}, disabled: ${widget.disabled}, display: ${widget.options?.display || 'none'})`);
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
        console.log(`[Parameter Hide] ❌ Hidden: ${widget.name} (type changed to: ${HIDDEN_TYPE}, disabled: ${widget.disabled})`);
    }
}

// 노드 크기 업데이트
function updateNodeSize(node) {
    setTimeout(() => {
        const newSize = node.computeSize();
        node.setSize(newSize);
        app.canvas.setDirty(true, true);
        console.log(`[Parameter Hide] Node resized to: ${newSize}`);
    }, 10);
}

// 노드 타입 감지
function isTargetNode(nodeData) {
    const targetTypes = [
        "Descriptor_Calculations_Classification",
        "Descriptor_Calculations_Regression"
    ];
    return targetTypes.includes(nodeData.name);
}

// 노드별 숨길 위젯 목록
function getHideableWidgets(nodeDataName) {
    const widgetMappings = {
        "Descriptor_Calculations_Classification": [
            "descriptor_type", "detect_aromaticity", "log",
            "remove_salt", "standardize_nitro", "use_file_name_as_molname",
            "retain_order", "threads", "waiting_jobs",
            "max_runtime", "max_cpd_per_file", "headless"
        ],
        "Descriptor_Calculations_Regression": [
            "descriptor_type", "detect_aromaticity", "log",
            "remove_salt", "standardize_nitro", "use_filename_as_mol_name",
            "retain_order", "threads", "waiting_jobs",
            "max_runtime", "max_cpd_per_file", "headless"
        ]
    };
    return widgetMappings[nodeDataName] || [];
}

// ComfyUI 확장 등록
app.registerExtension({
    name: "ComfyQSAR_PARAMETER_HIDE",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (isTargetNode(nodeData)) {
            console.log(`[Parameter Hide] Registering advanced control for: ${nodeData.name}`);
            
            const originalNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                originalNodeCreated?.apply(this, arguments);
                
                console.log(`[Parameter Hide] Node created: ${this.title || nodeData.name}`);
                
                // 노드별 고유 키
                const nodeKey = `${nodeData.name}_${this.id || Date.now()}`;
                this.nodeKey = nodeKey;
                
                // 숨길 위젯 목록
                const hideableWidgets = getHideableWidgets(nodeData.name);
                
                // advanced 위젯 찾기
                const advancedWidget = this.widgets.find(w => w.name === "advanced");
                if (!advancedWidget) {
                    console.log(`[Parameter Hide] Advanced widget not found`);
                    return;
                }
                
                console.log(`[Parameter Hide] Advanced widget found, initial value: ${advancedWidget.value}`);
                console.log(`[Parameter Hide] Available widgets: ${this.widgets.map(w => w.name).join(', ')}`);
                console.log(`[Parameter Hide] Available widget types: ${this.widgets.map(w => `${w.name}:${w.type}`).join(', ')}`);
                console.log(`[Parameter Hide] Hideable widgets: ${hideableWidgets.join(', ')}`);
                
                // 각 위젯의 존재 여부 개별 확인
                hideableWidgets.forEach(widgetName => {
                    const widget = this.widgets.find(w => w.name === widgetName);
                    if (widget) {
                        console.log(`[Parameter Hide] ✅ Found widget: ${widgetName} (type: ${widget.type}, value: ${widget.value}, display: ${widget.options?.display || 'none'})`);
                    } else {
                        console.warn(`[Parameter Hide] ❌ Widget NOT FOUND: ${widgetName}`);
                        console.log(`[Parameter Hide] Available similar names: ${this.widgets.filter(w => w.name.includes(widgetName.split('_')[0]) || w.name.includes(widgetName.split('_')[1] || '')).map(w => w.name).join(', ')}`);
                    }
                });
                
                // 모든 숨길 위젯의 원본 속성 저장
                hideableWidgets.forEach(widgetName => {
                    const widget = this.widgets.find(w => w.name === widgetName);
                    if (widget) {
                        const propKey = `${nodeKey}_${widgetName}`;
                        origProps[propKey] = { 
                            origType: widget.type, 
                            origComputeSize: widget.computeSize,
                            origComputedHeight: widget.computedHeight || 0
                        };
                        console.log(`[Parameter Hide] Registered widget: ${widgetName} (type: ${widget.type})`);
                    } else {
                        console.warn(`[Parameter Hide] Widget not found: ${widgetName}`);
                    }
                });
                
                // 위젯 가시성 업데이트 함수
                const updateWidgetVisibility = (isAdvanced) => {
                    console.log(`[Parameter Hide] 🔄 Updating visibility for advanced: ${isAdvanced}`);
                    
                    let changedCount = 0;
                    hideableWidgets.forEach(widgetName => {
                        const widget = this.widgets.find(w => w.name === widgetName);
                        if (widget) {
                            console.log(`[Parameter Hide] 🔧 Processing widget: ${widgetName} (type: ${widget.type}, display: ${widget.options?.display || 'none'})`);
                            toggleWidget(this, widget, isAdvanced, nodeKey);
                            changedCount++;
                        } else {
                            console.warn(`[Parameter Hide] ❌ Widget not found: ${widgetName}`);
                        }
                    });
                    
                    if (changedCount > 0) {
                        updateNodeSize(this);
                    }
                    
                    console.log(`[Parameter Hide] ✅ Updated ${changedCount} widgets, advanced: ${isAdvanced}`);
                };
                
                // advanced 값 변경 감지 (Property descriptor 방식)
                const desc = Object.getOwnPropertyDescriptor(advancedWidget, "value") || {};
                let widgetValue = advancedWidget.value;
                
                // 이미 정의된 속성인지 확인
                if (!advancedWidget._valueRedefined) {
                    try {
                        Object.defineProperty(advancedWidget, "value", {
                            get() {
                                return desc.get ? desc.get.call(advancedWidget) : widgetValue;
                            },
                            set(newVal) {
                                console.log(`[Parameter Hide] 🔀 Advanced changed from ${widgetValue} to: ${newVal}`);
                                
                                if (desc.set) desc.set.call(advancedWidget, newVal);
                                else widgetValue = newVal;
                                
                                // 위젯 가시성 업데이트
                                updateWidgetVisibility(newVal);
                            }
                        });
                        advancedWidget._valueRedefined = true;
                    } catch (error) {
                        console.warn(`[Parameter Hide] Could not redefine advanced widget value property: ${error.message}`);
                    }
                }
                
                // 초기 상태 설정
                console.log(`[Parameter Hide] 🚀 Setting initial state...`);
                updateWidgetVisibility(advancedWidget.value || false);
            };
            
            // 노드 제거 시 정리
            const originalOnRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
                console.log(`[Parameter Hide] 🗑️ Cleaning up node: ${this.nodeKey}`);
                
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

console.log("🎯 ComfyQSAR Parameter Hide Extension Loaded (Enhanced Version)");
