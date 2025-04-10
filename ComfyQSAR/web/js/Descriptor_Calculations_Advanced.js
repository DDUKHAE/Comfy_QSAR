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
    // advanced 위젯의 값을 확인
    const isAdvanced = findWidgetByName(node, 'advanced').value === true;

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
        case 'advanced': {
            // 공통 위젯 목록 정의 (Regression과 Classification에 모두 적용)
            const commonWidgets = [
                'descriptor_type', 'detect_aromaticity', 'log', 'remove_salt', 
                'standardize_nitro', 'use_filename_as_mol_name', 'retain_order',
                'threads', 'waiting_jobs', 'max_runtime', 'max_cpd_per_file', 'headless',
                'use_file_name_as_molname'
            ];
            // 공통 위젯 토글
            toggleWidgets(commonWidgets, isAdvanced);
            break;
        }
    }
}

const getSetWidgets = ['advanced'];

// 모든 타입의 디스크립터 계산 노드 타이틀을 포함하도록 수정
const getSetTitles = [
    "Descriptor Calculation_Regression",  // 원래 타이틀
    "Descriptor Calculation(Regression)", // 실제 노드 표시 이름
    "Descriptor calculations_Regression", // 클래스 이름 기반
    "Descriptor_calculations_Regression", // 클래스 이름 기반 (언더스코어)
    "Descriptor Calculation_Classification", // 원래 타이틀 
    "Descriptor Calculation(Classification)", // 실제 노드 표시 이름
    "Descriptor_Calculation_Classification" // 클래스 이름 기반
];

function getSetters(node) {
    if (node.widgets)
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

app.registerExtension({
    name: "Descriptor_calculations_advanced",

    nodeCreated(node) {
        const nodeTitle = node.constructor.title;
        const nodeType = node.type;
        
        // 클래스 이름과 표시 이름을 모두 확인
        if (getSetTitles.includes(nodeTitle) || 
            nodeType.includes("Descriptor_Calculation") || 
            nodeType.includes("Descriptor_calculations")) {
            console.log(`Applied advanced widget logic to node: ${nodeTitle} (${nodeType})`);
            getSetters(node);
        }
    }
});
