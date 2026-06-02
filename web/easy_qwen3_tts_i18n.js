const { app } = window.comfyAPI.app;

const EXTENSION_NAME = "eastmoe.ComfyEasyQwen3TTS.i18n";
const LOCALIZATION_URL = "/easy_qwen3_tts/local/zh-cn/nodes.json";

let localizationPromise = null;

function loadLocalization() {
  if (!localizationPromise) {
    localizationPromise = fetch(LOCALIZATION_URL)
      .then((response) => (response.ok ? response.json() : {}))
      .catch((error) => {
        console.warn("[Comfy-Easy-Qwen3-TTS] Failed to load zh-cn localization:", error);
        return {};
      });
  }
  return localizationPromise;
}

function chainCallback(target, name, callback) {
  const original = target[name];
  target[name] = function (...args) {
    const result = original?.apply(this, args);
    callback.apply(this, args);
    return result;
  };
}

function getInputTranslation(data, inputName, translations) {
  return data?.inputs?.[inputName] ?? data?.ui?.[inputName] ?? translations?.input_labels?.[inputName];
}

function labelSlot(slot, label) {
  if (!slot || !label) return;
  slot.label = label;
  slot.localized_name = label;
}

function applyNodeLabels(node, translations) {
  const nodeClass = node.constructor?.comfyClass ?? node.type;
  const nodeData = translations?.nodes?.[nodeClass] ?? {};
  const title = nodeData.display_name ?? translations?.node_display_names?.[nodeClass];
  if (title) node.title = title;

  for (const input of node.inputs ?? []) {
    const entry = getInputTranslation(nodeData, input.name, translations) ?? getInputTranslation(nodeData, input.label, translations);
    labelSlot(input, entry?.display_name);
  }

  for (const output of node.outputs ?? []) {
    const entry = nodeData?.outputs?.[output.name] ?? nodeData?.outputs?.[output.label];
    labelSlot(output, entry?.display_name ?? entry);
  }

  for (const widget of node.widgets ?? []) {
    const entry = getInputTranslation(nodeData, widget.name, translations) ?? getInputTranslation(nodeData, widget.label, translations);
    if (!entry?.display_name) continue;
    widget.label = entry.display_name;
    widget.localized_name = entry.display_name;
  }

  app.graph?.setDirtyCanvas(true, true);
}

function applyNodeDataTranslations(nodeData, translations) {
  const nodeClass = nodeData?.name;
  const nodeTranslation = translations?.nodes?.[nodeClass] ?? {};
  const title = nodeTranslation.display_name ?? translations?.node_display_names?.[nodeClass];
  if (title) nodeData.display_name = title;
  if (Array.isArray(nodeData.output_name)) {
    nodeData.output_name = nodeData.output_name.map((name) => {
      const entry = nodeTranslation?.outputs?.[name];
      return entry?.display_name ?? entry ?? name;
    });
  }

  for (const section of ["required", "optional", "hidden"]) {
    const inputs = nodeData.input?.[section];
    if (!inputs) continue;
    for (const [name, spec] of Object.entries(inputs)) {
      const entry = getInputTranslation(nodeTranslation, name, translations);
      if (!entry?.display_name || !Array.isArray(spec)) continue;
      const options = spec[1] ?? {};
      options.display_name = entry.display_name;
      options.label = entry.display_name;
      if (entry.tooltip) options.tooltip = entry.tooltip;
      spec[1] = options;
    }
  }
}

app.registerExtension({
  name: EXTENSION_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const translations = await loadLocalization();
    if (!translations?.nodes?.[nodeData?.name] && !translations?.node_display_names?.[nodeData?.name]) return;

    applyNodeDataTranslations(nodeData, translations);

    chainCallback(nodeType.prototype, "onNodeCreated", function () {
      applyNodeLabels(this, translations);
    });

    chainCallback(nodeType.prototype, "onConfigure", function () {
      applyNodeLabels(this, translations);
    });
  },

  async nodeCreated(node) {
    const translations = await loadLocalization();
    applyNodeLabels(node, translations);
  },
});
