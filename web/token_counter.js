// SPDX-License-Identifier: GPL-3.0-only
import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "K2.TokenCounter",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "K2TokenCounter") return;
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            if (!Array.isArray(message?.text)) return;
            let output = this.widgets?.find((widget) => widget.name === "token_count_report");
            if (!output) {
                output = ComfyWidgets.STRING(this, "token_count_report", ["STRING", { multiline: true }], app).widget;
                output.inputEl.readOnly = true;
                output.options.serialize = false;
            }
            output.value = message.text.join("\n");
            const size = this.computeSize();
            this.setSize([Math.max(this.size[0], size[0]), Math.max(this.size[1], size[1])]);
            app.graph.setDirtyCanvas(true, false);
        };
    },
});
