import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

ComfyWidgets["TEXTASSETUPLOAD"] = function(node, inputName, inputData, app) {
    const pathWidget = node.widgets.find((w) => w.name === "file");
    const subFolderWidget = node.widgets.find((w) => w.name === "input_sub_folder");

    function showContent(content) {
        if (node.widgets) {
            for (let i = 4; i < node.widgets.length; i++) {
                node.widgets[i].onRemove?.();
            }
            node.widgets.length = 4;
        }

        if (content) {
            const v = content;
            const w = ComfyWidgets["STRING"](node, "text", ["STRING", { multiline: true }], app).widget;
            w.inputEl.readOnly = true;
            w.inputEl.style.opacity = 0.6;
            w.value = v;
            node.widgets.push(w);
        }

        requestAnimationFrame(() => {
            const sz = node.computeSize();
            if (sz[0] < node.size[0]) {
                sz[0] = node.size[0];
            }
            if (sz[1] < node.size[1]) {
                sz[1] = node.size[1];
            }
            node.onResize?.(sz);
            app.graph.setDirtyCanvas(true, false);
        });
    }

    var default_value = pathWidget.value;
    Object.defineProperty(pathWidget, "value", {
        set : function(value) {
            this._real_value = value;
        },
        get : function() {
            let value = "";
            if (this._real_value) {
                value = this._real_value;
            } else {
                return default_value;
            }

            if (value.filename) {
                let real_value = value;
                value = "";
                if (real_value.subfolder && real_value.type !== subFolderWidget.value) {
                    value = real_value.subfolder + "/";
                }

                value += real_value.filename;

                if(real_value.type && real_value.type !== "input")
                    value += ` [${real_value.type}]`;
            }
            return value;
        }
    });

    const show_content_widget = node.widgets.find((w) => w.name === "show_content");

    // Add our own callback to the combo widget to render an image when it changes
    const cb = node.callback;
    pathWidget.callback = function () {
        inputData[1] = pathWidget.value;
        if(show_content_widget?.value) {
            showContent(pathWidget.value);
        }else{
            showContent(null);
        }
        if (cb) {
            return cb.apply(this, arguments);
        }
    };

    async function uploadFile(file, overwrite, updateNode, pasted = false) {
        // Wrap file in formdata so it includes filename
        const body = new FormData();
        body.append("file", file);
        body.append("overwrite", overwrite);
        body.append("subfolder", subFolderWidget.value);
        const resp = await api.fetchApi("/upload/textasset", {
            method: "POST",
            body,
        });

        if (resp.status === 200) {
            const data = await resp.json();
            // Add the file to the dropdown list and update the widget value
            let path = data.name;
            if (data.subfolder)
                path = data.subfolder + "/" + path;

            if (!pathWidget.options.values.includes(path)) {
                pathWidget.options.values.push(path);
            }

            if (updateNode) {
                pathWidget.value = path;
            }
            inputData[1] = path;
            if(show_content_widget?.value) {
                showContent(path);
            }else{
                showContent(null);
            }
            console.log("[upload]", path);
        } else {
            alert(resp.status + " - " + resp.statusText);
        }
    }

    const overwriteWidget = node.widgets.find((w) => w.name === "overwrite");
    const fileInput = document.createElement("input");
    Object.assign(fileInput, {
        type: "file",
        accept: "",
        style: "display: none",
        onchange: async () => {
            if (fileInput.files.length) {
                await uploadFile(fileInput.files[0], overwriteWidget.value, true);
            }
        },
    });
    document.body.append(fileInput);

    // Create the button widget for selecting the files
    let uploadWidget = node.addWidget("button", inputName, "file", () => {
        fileInput.click();
    });
    uploadWidget.label = "choose file to upload";
    uploadWidget.serialize = false;

    return { widget: uploadWidget };
};

app.registerExtension({
    name: "LoadTextAsset",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.input?.required?.file?.[1]?.save_textasset === true) {
            nodeData.input.required.upload = ["TEXTASSETUPLOAD"];
        }
    }
});