import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

const TARGET_NODES = Object.freeze([
    "LOAD_FILE", 
    "STANDARD",
    "CALCULATE_DESCRIPTORS",
    "FILTER_COMPOUNDS_BY_NAN_DUAL",
    "REMOVE_HIGH_NAN_DESCRIPTORS",
    "IMPUTE_MISSING_VALUES",
    "MERGE_IMPUTED_DATA", 
    "REMOVE_LOW_VARIANCE_FEATURES",
    "REMOVE_HIGH_CORRELATION_FEATURES",
    "LASSO_FEATURE_SELECTION",
    "TREE_FEATURE_SELECTION", 
    "XGBOOST_FEATURE_SELECTION",
    "RFE_FEATURE_SELECTION",
    "SELECTFROMMODEL_FEATURE_SELECTION"
]);

app.registerExtension({
    name: "ComfyQSAR_TEXT",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (TARGET_NODES.includes(nodeData.name)) {
            function populate(text) {
                if (this.widgets) {
                    // 처음 생성한 노드의 위젯은 제거하지 않음
                    const initialWidgetsCount = this.initialWidgetsCount || this.widgets.length;
                    this.initialWidgetsCount = initialWidgetsCount;

                    for (let i = this.widgets.length - 1; i >= initialWidgetsCount; i--) {
                        this.widgets[i].onRemove?.();
                        this.widgets.pop();
                    }
                }

                const v = [...text];
                if (!v[0]) {
                    v.shift();
                }
                for (const list of v) {
                    const w = ComfyWidgets["STRING"](this, "text2", ["STRING", { multiline: true }], app).widget;
                    w.inputEl.readOnly = true;
                    w.inputEl.style.opacity = 0.6;
                    w.value = list;
                }

                requestAnimationFrame(() => {
                    const sz = this.computeSize();
                    if (sz[0] < this.size[0]) {
                        sz[0] = this.size[0];
                    }
                    if (sz[1] < this.size[1]) {
                        sz[1] = this.size[1];
                    }
                    this.onResize?.(sz);
                    app.graph.setDirtyCanvas(true, false);
                });
            }

            // When the node is executed we will be sent the input text, display this in the widget
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                populate.call(this, message.text);
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                onConfigure?.apply(this, arguments);
                if (this.widgets_values?.length) {
                    populate.call(this, this.widgets_values.slice(+this.widgets_values.length > 1));
                }
            };
        }
    },
});