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
    
    // 디버깅을 위한 로깅
    console.log(`[Descriptor_Optimization_Advanced] 위젯 ${widget.name} ${show ? '표시' : '숨김'}`);
}

function widgetLogic(node, widget) {
    // 디버깅을 위해 로그 출력
    console.log(`[Descriptor_Optimization_Advanced] widgetLogic 호출됨 - 위젯: ${widget.name}, 값: ${widget.value}`);

    const toggleWidgets = (widgets, condition) => {
        widgets.forEach(widgetName => {
            const widget = findWidgetByName(node, widgetName);
            if (widget) {  // 위젯이 존재하는 경우에만 토글
                toggleWidget(node, widget, condition);
            }
        });
    };

    switch (widget.name) {
        case 'correlation_mode': {
            // target_based 모드에서만 importance_model 표시
            const isTargetBased = widget.value === "target_based";
            console.log(`[Descriptor_Optimization_Advanced] correlation_mode = ${widget.value}, isTargetBased = ${isTargetBased}`);
            
            const importanceWidget = findWidgetByName(node, "importance_model");
            if (importanceWidget) {
                console.log(`[Descriptor_Optimization_Advanced] importance_model 위젯 찾음, 토글 실행`);
                toggleWidget(node, importanceWidget, isTargetBased);
                updateNodeHeight(node); // 노드 높이 업데이트 추가
            } else {
                console.log(`[Descriptor_Optimization_Advanced] importance_model 위젯을 찾을 수 없음`);
            }
            break;
        }
    }
}

// 모니터링할 위젯 목록
const getSetWidgets = ['correlation_mode'];

// 모든 디스크립터 최적화 노드 타이틀/타입 목록
const getSetTitles = [
    // Regression 노드 타이틀 및 타입
    "Descriptor Optimization(Regression)",
    "Descriptor_Optimization_Regression",
    "Remove High Correlation Features(Regression)",
    "Remove_High_Correlation_Features_Regression",
    
    // Classification 노드 타이틀 및 타입
    "Descriptor Optimization(Classification)",
    "Descriptor_Optimization_Classification",
    "Remove High Correlation Features(Classification)",
    "Remove_High_Correlation_Features_Classification"
];

function getSetters(node) {
    console.log(`[Descriptor_Optimization_Advanced] getSetters 호출됨 - 노드: ${node.type}`);
    
    if (node.widgets) {
        for (const w of node.widgets) {
            if (getSetWidgets.includes(w.name)) {
                console.log(`[Descriptor_Optimization_Advanced] 위젯 ${w.name} 감시 설정 중`);
                
                // 중복 설정 방지
                if (w._hasCustomGetter) {
                    console.log(`[Descriptor_Optimization_Advanced] 위젯 ${w.name}은 이미 감시 중`);
                    continue;
                }
                
                // 초기 실행
                widgetLogic(node, w);
                let widgetValue = w.value;

                Object.defineProperty(w, 'value', {
                    get() {
                        return widgetValue;
                    },
                    set(newVal) {
                        console.log(`[Descriptor_Optimization_Advanced] 위젯 ${w.name} 값 변경: ${widgetValue} -> ${newVal}`);
                        if (newVal !== widgetValue) {
                            widgetValue = newVal;
                            widgetLogic(node, w);
                        }
                    }
                });
                
                // 중복 설정 방지 플래그
                w._hasCustomGetter = true;
            }
        }
    }
    
    // 초기 상태 확인 및 적용
    setTimeout(() => {
        const correlationModeWidget = findWidgetByName(node, "correlation_mode");
        if (correlationModeWidget) {
            console.log(`[Descriptor_Optimization_Advanced] 초기화 - correlation_mode=${correlationModeWidget.value}`);
            widgetLogic(node, correlationModeWidget);
        }
    }, 100);
}

app.registerExtension({
    name: "Descriptor_Optimization_Advanced",

    nodeCreated(node) {
        const nodeTitle = node.constructor.title;
        const nodeType = node.type;
        
        // 노드 이름 또는 타입이 목록에 있는지 확인
        if (getSetTitles.includes(nodeTitle) || 
            getSetTitles.some(title => nodeType.includes(title)) ||
            nodeType.includes("Correlation_Features") || 
            nodeType.includes("Descriptor_Optimization")) {
            
            console.log(`[Descriptor_Optimization_Advanced] 감시 적용: ${nodeTitle} (${nodeType})`);
            getSetters(node);
            
            // 노드 구성 변경 시 (재로딩, 값 변경 등) 위젯 상태 확인
            if (!node._configureObserverAdded) {
                const onConfigure = node.onConfigure;
                node.onConfigure = function() {
                    const result = onConfigure?.apply(this, arguments);
                    console.log(`[Descriptor_Optimization_Advanced] 노드 구성됨: ${this.type}`);
                    
                    // 약간의 지연 후 위젯 상태 확인
                    setTimeout(() => {
                        const correlationModeWidget = findWidgetByName(this, "correlation_mode");
                        if (correlationModeWidget) {
                            widgetLogic(this, correlationModeWidget);
                        }
                    }, 200);
                    
                    return result;
                };
                node._configureObserverAdded = true;
            }
        }
    }
});
