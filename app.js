import * as ort from "./vendor/ort.min.mjs";

const MODEL_PATH = "./assets/detector.onnx";
const MATCHER_PATH = "./assets/matcher.onnx";
const REFERENCE_EMBED_PATH = "./assets/reference_embeddings.json";
const CHARACTER_DATA_PATH = "./data/characters.json";
const UNIT_DATA_PATH = "./data/units.json";
const MODEL_SIZE = 960;
const MATCHER_SIZE = 160;
const COLORS = {
  box: "#b34d2e",
  textBg: "#8d381e",
  text: "#fffaf5",
};

const fileInput = document.querySelector("#file-input");
const openFileButton = document.querySelector("#open-file-button");
const openRecognitionButton = document.querySelector("#open-recognition-button");
const startCameraButton = document.querySelector("#start-camera-button");
const capturePhotoButton = document.querySelector("#capture-photo-button");
const addPieceButton = document.querySelector("#add-piece-button");
const showRecommendedButton = document.querySelector("#show-recommended-button");
const cameraVideo = document.querySelector("#camera-video");
const canvas = document.querySelector("#result-canvas");
const ctx = canvas.getContext("2d");
const statusLabel = document.querySelector("#status-label");
const handCountLabel = document.querySelector("#hand-count-label");
const recognitionNote = document.querySelector("#recognition-note");
const attributeSelect = document.querySelector("#attribute-select");
const initialSelect = document.querySelector("#initial-select");
const searchInput = document.querySelector("#search-input");
const pickerContext = document.querySelector("#picker-context");
const characterList = document.querySelector("#character-list");
const handList = document.querySelector("#hand-list");
const recommendedUnits = document.querySelector("#recommended-units");
const recognitionModal = document.querySelector("#recognition-modal");
const pickerModal = document.querySelector("#picker-modal");
const recommendedModal = document.querySelector("#recommended-modal");

let detectorSession;
let matcherSession;
let referenceEmbeddings = [];
let currentImage = null;
let charactersByName = {};
let characterRecords = [];
let units = [];
let unitsByCharacter = new Map();
let handEntries = [];
let activeEntryId = null;
let expandedEntryId = null;
let nextEntryId = 1;
let cameraStream = null;
let pendingNewEntryId = null;

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;

async function loadApp() {
  const [characterMap, loadedUnits] = await Promise.all([
    fetch(CHARACTER_DATA_PATH).then((response) => response.json()),
    fetch(UNIT_DATA_PATH).then((response) => response.json()),
  ]);

  charactersByName = characterMap;
  characterRecords = Object.entries(characterMap)
    .map(([name, meta]) => ({
      name,
      attribute: meta.attribute,
      reading: meta.reading,
      initial: meta.reading?.[0] ?? "#",
    }))
    .sort(compareCharacters);
  units = loadedUnits;
  unitsByCharacter = buildUnitsByCharacter(units);

  populateAttributeSelect();
  populateInitialSelect();
  drawResult(null, []);
  renderCharacterPicker();
  renderHandList();
  renderRecommendedUnits();

  detectorSession = await ort.InferenceSession.create(MODEL_PATH, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  matcherSession = await ort.InferenceSession.create(MATCHER_PATH, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  referenceEmbeddings = await loadReferenceEmbeddings();

  setStatus("準備完了");
}

function compareCharacters(a, b) {
  return a.reading.localeCompare(b.reading, "ja") || a.name.localeCompare(b.name, "ja");
}

function buildUnitsByCharacter(unitRows) {
  const map = new Map();
  for (const unit of unitRows) {
    for (const name of unit.characters) {
      if (!map.has(name)) {
        map.set(name, []);
      }
      map.get(name).push(unit);
    }
  }
  return map;
}

function populateAttributeSelect() {
  const attributes = ["すべて", ...new Set(characterRecords.map((record) => record.attribute))];
  attributeSelect.innerHTML = attributes
    .map((attribute, index) => {
      const value = index === 0 ? "all" : attribute;
      return `<option value="${escapeHtml(value)}">${escapeHtml(attribute)}</option>`;
    })
    .join("");
}

function populateInitialSelect() {
  const initials = ["すべて", ...new Set(characterRecords.map((record) => record.initial))];
  initialSelect.innerHTML = initials
    .map((initial, index) => {
      const value = index === 0 ? "all" : initial;
      return `<option value="${escapeHtml(value)}">${escapeHtml(initial)}</option>`;
    })
    .join("");
}

function resetPickerFilters() {
  attributeSelect.value = "all";
  initialSelect.value = "all";
  searchInput.value = "";
}

function setStatus(message) {
  statusLabel.textContent = message;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function letterbox(image, size) {
  const ratio = Math.min(size / image.width, size / image.height);
  const resizedWidth = Math.round(image.width * ratio);
  const resizedHeight = Math.round(image.height * ratio);
  const dw = (size - resizedWidth) / 2;
  const dh = (size - resizedHeight) / 2;
  const stage = document.createElement("canvas");
  stage.width = size;
  stage.height = size;
  const stageCtx = stage.getContext("2d");
  stageCtx.fillStyle = "rgb(114,114,114)";
  stageCtx.fillRect(0, 0, size, size);
  stageCtx.drawImage(image, dw, dh, resizedWidth, resizedHeight);
  return { canvas: stage, ratio, dw, dh };
}

function toTensor(stageCanvas) {
  const imageData = stageCanvas.getContext("2d").getImageData(0, 0, stageCanvas.width, stageCanvas.height);
  const { data, width, height } = imageData;
  const tensor = new Float32Array(3 * width * height);
  const plane = width * height;

  for (let i = 0; i < width * height; i += 1) {
    const src = i * 4;
    tensor[i] = data[src] / 255;
    tensor[plane + i] = data[src + 1] / 255;
    tensor[plane * 2 + i] = data[src + 2] / 255;
  }

  return new ort.Tensor("float32", tensor, [1, 3, height, width]);
}

function toMatcherTensor(canvasOrImage) {
  const stage = document.createElement("canvas");
  stage.width = MATCHER_SIZE;
  stage.height = MATCHER_SIZE;
  const stageCtx = stage.getContext("2d");
  stageCtx.drawImage(canvasOrImage, 0, 0, MATCHER_SIZE, MATCHER_SIZE);
  const { data, width, height } = stageCtx.getImageData(0, 0, MATCHER_SIZE, MATCHER_SIZE);
  const tensor = new Float32Array(3 * width * height);
  const plane = width * height;
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  for (let i = 0; i < width * height; i += 1) {
    const src = i * 4;
    tensor[i] = (data[src] / 255 - mean[0]) / std[0];
    tensor[plane + i] = (data[src + 1] / 255 - mean[1]) / std[1];
    tensor[plane * 2 + i] = (data[src + 2] / 255 - mean[2]) / std[2];
  }

  return new ort.Tensor("float32", tensor, [1, 3, height, width]);
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  if (inter <= 0) {
    return 0;
  }
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter);
}

function nms(boxes, iouThreshold) {
  const order = [...boxes].sort((a, b) => b.score - a.score);
  const keep = [];
  while (order.length > 0) {
    const current = order.shift();
    keep.push(current);
    for (let i = order.length - 1; i >= 0; i -= 1) {
      if (iou(current, order[i]) > iouThreshold) {
        order.splice(i, 1);
      }
    }
  }
  return keep;
}

function filterDetectionOutliers(boxes) {
  if (boxes.length === 0) {
    return boxes;
  }

  const widths = boxes.map((box) => box.x2 - box.x1).sort((a, b) => a - b);
  const heights = boxes.map((box) => box.y2 - box.y1).sort((a, b) => a - b);
  const median = (values) => values[Math.floor(values.length / 2)];
  const medianWidth = median(widths);
  const medianHeight = median(heights);

  const filtered = boxes.filter(
    (box) => box.x2 - box.x1 >= medianWidth * 0.45 && box.y2 - box.y1 >= medianHeight * 0.6
  );
  if (filtered.length < 3) {
    return filtered;
  }

  const rowThreshold = Math.max(12, medianHeight * 0.6);
  filtered.sort((a, b) => (a.y1 + a.y2) / 2 - (b.y1 + b.y2) / 2 || (a.x1 + a.x2) / 2 - (b.x1 + b.x2) / 2);

  const rows = [];
  for (const box of filtered) {
    const cy = (box.y1 + box.y2) / 2;
    if (rows.length === 0) {
      rows.push([box]);
      continue;
    }
    const lastRow = rows[rows.length - 1];
    const lastCy = lastRow.reduce((sum, item) => sum + (item.y1 + item.y2) / 2, 0) / lastRow.length;
    if (Math.abs(cy - lastCy) <= rowThreshold) {
      lastRow.push(box);
    } else {
      rows.push([box]);
    }
  }

  const rowCenters = rows.map(
    (row) => row.reduce((sum, item) => sum + (item.y1 + item.y2) / 2, 0) / row.length
  );
  const keptRows = [];
  for (let index = 0; index < rows.length; index += 1) {
    let row = rows[index];
    if (row.length === 1) {
      const prevGap = index > 0 ? rowCenters[index] - rowCenters[index - 1] : null;
      const nextGap = index + 1 < rowCenters.length ? rowCenters[index + 1] - rowCenters[index] : null;
      const isolatedBetween =
        prevGap !== null && nextGap !== null && prevGap > medianHeight * 0.9 && nextGap > medianHeight * 0.9;
      const isolatedEdge =
        (prevGap === null && nextGap !== null && nextGap > medianHeight * 0.9) ||
        (nextGap === null && prevGap !== null && prevGap > medianHeight * 0.9);
      if (isolatedBetween || isolatedEdge) {
        continue;
      }
    }
    if (row.length >= 4) {
      const byScore = [...row].sort((a, b) => a.score - b.score);
      if (byScore[0].score < 0.25 && byScore[1].score - byScore[0].score > 0.5) {
        row = row.filter((box) => box !== byScore[0]);
      }
    }
    keptRows.push(row);
  }

  return keptRows.flat();
}

function toRows(output) {
  const { dims, data } = output;

  if (dims.length === 3 && dims[0] === 1 && dims[1] === 5) {
    const rows = dims[2];
    return Array.from({ length: rows }, (_, index) => ({
      x: data[index],
      y: data[rows + index],
      w: data[rows * 2 + index],
      h: data[rows * 3 + index],
      score: data[rows * 4 + index],
    }));
  }

  if (dims.length === 3 && dims[0] === 1 && (dims[2] === 5 || dims[2] === 6)) {
    const rows = dims[1];
    const stride = dims[2];
    return Array.from({ length: rows }, (_, index) => {
      const base = index * stride;
      return {
        x: data[base],
        y: data[base + 1],
        w: data[base + 2],
        h: data[base + 3],
        score: data[base + 4],
      };
    });
  }

  throw new Error(`unsupported output dims: ${JSON.stringify(dims)}`);
}

function decode(output, original, ratio, dw, dh, confThreshold, iouThreshold) {
  const rows = toRows(output);
  const boxes = [];

  for (const row of rows) {
    const { x, y, w, h, score } = row;
    if (score < confThreshold) {
      continue;
    }

    const left = (x - w / 2 - dw) / ratio;
    const top = (y - h / 2 - dh) / ratio;
    const right = (x + w / 2 - dw) / ratio;
    const bottom = (y + h / 2 - dh) / ratio;
    boxes.push({
      x1: clamp(left, 0, original.width),
      y1: clamp(top, 0, original.height),
      x2: clamp(right, 0, original.width),
      y2: clamp(bottom, 0, original.height),
      score,
    });
  }

  return filterDetectionOutliers(nms(boxes, iouThreshold));
}

function sortBoxesForHand(boxes) {
  const sorted = [...boxes].sort(
    (a, b) => (a.y1 + a.y2) / 2 - (b.y1 + b.y2) / 2 || (a.x1 + a.x2) / 2 - (b.x1 + b.x2) / 2
  );
  if (sorted.length <= 1) {
    return sorted;
  }

  const heights = sorted.map((box) => box.y2 - box.y1).sort((a, b) => a - b);
  const medianHeight = heights[Math.floor(heights.length / 2)] || 1;
  const rowThreshold = Math.max(12, medianHeight * 0.6);
  const rows = [];

  for (const box of sorted) {
    const cy = (box.y1 + box.y2) / 2;
    const row = rows.find((current) => Math.abs(current.cy - cy) <= rowThreshold);
    if (row) {
      row.items.push(box);
      row.cy = row.items.reduce((sum, item) => sum + (item.y1 + item.y2) / 2, 0) / row.items.length;
    } else {
      rows.push({ cy, items: [box] });
    }
  }

  rows.sort((a, b) => a.cy - b.cy);
  return rows.flatMap((row) => row.items.sort((a, b) => (a.x1 + a.x2) / 2 - (b.x1 + b.x2) / 2));
}

function drawResult(image, entries) {
  if (!image) {
    canvas.width = 960;
    canvas.height = 540;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#efe8db";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    return;
  }

  canvas.width = image.width;
  canvas.height = image.height;
  ctx.drawImage(image, 0, 0);
  ctx.lineWidth = Math.max(2, Math.round(image.width / 500));
  ctx.font = `${Math.max(11, Math.round(image.width / 72))}px sans-serif`;

  entries.forEach((entry, index) => {
    if (!entry.box) {
      return;
    }
    const width = entry.box.x2 - entry.box.x1;
    const height = entry.box.y2 - entry.box.y1;
    ctx.strokeStyle = COLORS.box;
    ctx.strokeRect(entry.box.x1, entry.box.y1, width, height);

    const label = `${index + 1} ${entry.name || "未設定"}`;
    const metrics = ctx.measureText(label);
    const labelHeight = Math.max(16, Math.round(image.width / 56));
    const labelWidth = metrics.width + 10;
    ctx.fillStyle = COLORS.textBg;
    ctx.fillRect(entry.box.x1, Math.max(0, entry.box.y1 - labelHeight), labelWidth, labelHeight);
    ctx.fillStyle = COLORS.text;
    ctx.fillText(label, entry.box.x1 + 5, Math.max(12, entry.box.y1 - 4));
  });
}

async function loadReferenceEmbeddings() {
  const rows = await fetch(REFERENCE_EMBED_PATH).then((response) => response.json());
  return rows.map((row) => {
    const embedding = new Float32Array(row.embedding);
    let norm = 0;
    for (let index = 0; index < embedding.length; index += 1) {
      norm += embedding[index] * embedding[index];
    }
    norm = Math.sqrt(norm) || 1;
    for (let index = 0; index < embedding.length; index += 1) {
      embedding[index] /= norm;
    }
    return { name: row.name, embedding };
  });
}

function cosineScores(embedding) {
  const bestByName = new Map();
  for (const ref of referenceEmbeddings) {
    let score = 0;
    for (let index = 0; index < embedding.length; index += 1) {
      score += embedding[index] * ref.embedding[index];
    }
    const current = bestByName.get(ref.name);
    if (!current || score > current.score) {
      bestByName.set(ref.name, { name: ref.name, score });
    }
  }
  return [...bestByName.values()].sort((a, b) => b.score - a.score);
}

function cropBoxToCanvas(image, box) {
  const left = Math.max(0, Math.floor(box.x1));
  const top = Math.max(0, Math.floor(box.y1));
  const width = Math.max(1, Math.ceil(box.x2 - box.x1));
  const height = Math.max(1, Math.ceil(box.y2 - box.y1));
  const crop = document.createElement("canvas");
  crop.width = width;
  crop.height = height;
  crop.getContext("2d").drawImage(image, left, top, width, height, 0, 0, width, height);
  return crop;
}

async function matchBoxes(image, boxes) {
  const inputName = matcherSession.inputNames[0];
  const outputName = matcherSession.outputNames[0];
  const results = [];
  for (const box of boxes) {
    const crop = cropBoxToCanvas(image, box);
    const tensor = toMatcherTensor(crop);
    const outputs = await matcherSession.run({ [inputName]: tensor });
    results.push(cosineScores(outputs[outputName].data).slice(0, 4));
  }
  return results;
}

async function loadImageFromSource(source) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = source;
  });
}

function createEntry({ name = "", matches = [], score = null, box = null } = {}) {
  return {
    id: nextEntryId++,
    name,
    matches,
    score,
    box,
  };
}

function getHandCounts() {
  const counts = new Map();
  for (const entry of handEntries) {
    if (!entry.name) {
      continue;
    }
    counts.set(entry.name, (counts.get(entry.name) || 0) + 1);
  }
  return counts;
}

function scoreUnit(unit, handCounts) {
  const owned = unit.characters.filter((name) => handCounts.has(name));
  const missing = unit.characters.filter((name) => !handCounts.has(name));
  return { unit, owned, missing };
}

function compareUnitScore(a, b) {
  return (
    a.missing.length - b.missing.length ||
    b.owned.length - a.owned.length ||
    a.unit.characters.length - b.unit.characters.length ||
    a.unit.unit_name.localeCompare(b.unit.unit_name, "ja")
  );
}

function getRecommendedUnitScores() {
  const handCounts = getHandCounts();
  return units
    .map((unit) => scoreUnit(unit, handCounts))
    .filter((row) => row.owned.length > 0)
    .sort(compareUnitScore);
}

function renderUnitCard(row, emphasisName) {
  const complete = row.missing.length === 0;
  const members = row.unit.characters
    .map((name) => {
      const classes = ["member"];
      if (row.owned.includes(name)) {
        classes.push("owned");
      }
      if (name === emphasisName) {
        classes.push("is-focus");
      }
      return `<span class="${classes.join(" ")}">${escapeHtml(name)}</span>`;
    })
    .join("");

  return `
    <article class="unit-card ${complete ? "is-complete" : ""}">
      <div class="unit-title">${escapeHtml(row.unit.unit_name)}</div>
      <div class="unit-meta">
        <span class="tag good">${row.owned.length}/${row.unit.characters.length} 枚</span>
        <span class="tag ${complete ? "good" : "missing"}">${complete ? "完成" : `不足 ${row.missing.length}`}</span>
      </div>
      <div class="unit-members">${members}</div>
    </article>
  `;
}

function renderEntryAccordion(entry) {
  const handCounts = getHandCounts();
  const focused = (unitsByCharacter.get(entry.name) || []).map((unit) => scoreUnit(unit, handCounts)).sort(compareUnitScore);

  const focusedMarkup = focused.length
    ? focused.map((row) => renderUnitCard(row, entry.name)).join("")
    : `<div class="empty">この駒を含む役はありません。</div>`;

  return `
    <div class="accordion">
      <div class="accordion-panel" ${entry.id === expandedEntryId ? "" : "hidden"}>
        <h3>この駒を含む役</h3>
        <div class="unit-list">${focusedMarkup}</div>
      </div>
    </div>
  `;
}

function renderHandList() {
  handCountLabel.textContent = `${handEntries.length} 枚`;

  if (!handEntries.length) {
    handList.innerHTML = `<div class="empty">まだ手持ちはありません。</div>`;
    return;
  }

  handList.innerHTML = handEntries
    .map((entry, index) => `
      <article class="hand-card ${entry.id === activeEntryId ? "is-active" : ""}">
        <div class="hand-main" data-action="toggle-accordion" data-entry-id="${entry.id}">
          <div class="hand-top">
            <div class="piece-index">${index + 1}</div>
            <div class="piece-name">${escapeHtml(entry.name || "未設定")}</div>
            <div class="hand-actions">
              <button type="button" class="secondary" data-action="edit-entry" data-entry-id="${entry.id}">差し替え</button>
              <button type="button" class="secondary" data-action="remove-entry" data-entry-id="${entry.id}">削除</button>
            </div>
          </div>
        </div>
        ${renderEntryAccordion(entry)}
      </article>
    `)
    .join("");
}

function getFilteredCharacters() {
  const attribute = attributeSelect.value;
  const initial = initialSelect.value;
  const query = searchInput.value.trim();

  return characterRecords.filter((record) => {
    if (attribute !== "all" && record.attribute !== attribute) {
      return false;
    }
    if (initial !== "all" && record.initial !== initial) {
      return false;
    }
    if (!query) {
      return true;
    }
    return record.name.includes(query) || record.reading.includes(query);
  });
}

function renderCharacterPicker() {
  const activeEntry = handEntries.find((entry) => entry.id === activeEntryId) || null;
  pickerContext.textContent = activeEntry ? `編集中: ${activeEntry.name || "未設定"}` : "編集中の駒を選択してください。";

  const filtered = getFilteredCharacters();
  if (!filtered.length) {
    characterList.innerHTML = `<div class="empty">条件に一致する駒がありません。</div>`;
    return;
  }

  characterList.innerHTML = filtered
    .map((record) => {
      const isSelected = activeEntry?.name === record.name;
      return `
        <button
          type="button"
          class="character-chip ${isSelected ? "is-selected" : ""}"
          data-action="pick-character"
          data-name="${escapeAttr(record.name)}"
        >
          <strong>${escapeHtml(record.name)}</strong>
          <span>${escapeHtml(record.attribute)} / ${escapeHtml(record.reading)}</span>
        </button>
      `;
    })
    .join("");
}

function renderRecommendedUnits() {
  const recommended = getRecommendedUnitScores().slice(0, 24);
  recommendedUnits.innerHTML = recommended.length
    ? recommended.map((row) => renderUnitCard(row, null)).join("")
    : `<div class="empty">まだ役候補はありません。</div>`;
}

function syncUi() {
  renderHandList();
  renderCharacterPicker();
  renderRecommendedUnits();
}

function getModalByKind(kind) {
  if (kind === "recognition") {
    return recognitionModal;
  }
  if (kind === "picker") {
    return pickerModal;
  }
  if (kind === "recommended") {
    return recommendedModal;
  }
  return null;
}

function openModal(kind) {
  const modal = getModalByKind(kind);
  if (!modal) {
    return;
  }
  modal.hidden = false;
  document.body.classList.add("modal-open");
}

function closeModal(kind) {
  const modal = getModalByKind(kind);
  if (!modal) {
    return;
  }
  modal.hidden = true;
  if (kind === "recognition") {
    stopCamera();
  }
  if (kind === "picker") {
    resetPickerFilters();
    if (pendingNewEntryId !== null) {
      handEntries = handEntries.filter((entry) => entry.id !== pendingNewEntryId);
      if (activeEntryId === pendingNewEntryId) {
        activeEntryId = handEntries[0]?.id ?? null;
      }
      if (expandedEntryId === pendingNewEntryId) {
        expandedEntryId = null;
      }
      pendingNewEntryId = null;
      syncUi();
    }
  }
  if ([recognitionModal, pickerModal, recommendedModal].every((node) => node.hidden)) {
    document.body.classList.remove("modal-open");
  }
}

function setCurrentImage(image, message) {
  currentImage = image;
  cameraVideo.hidden = true;
  drawResult(currentImage, handEntries);
  recognitionNote.textContent = message;
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    recognitionNote.textContent = "このブラウザではカメラ起動に対応していません。";
    return;
  }

  stopCamera();
  cameraStream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: "environment" } },
    audio: false,
  });
  cameraVideo.srcObject = cameraStream;
  cameraVideo.hidden = false;
  await cameraVideo.play();
  capturePhotoButton.disabled = false;
  recognitionNote.textContent = "撮影してください。";
}

function stopCamera() {
  if (cameraStream) {
    for (const track of cameraStream.getTracks()) {
      track.stop();
    }
    cameraStream = null;
  }
  cameraVideo.srcObject = null;
  cameraVideo.hidden = true;
  capturePhotoButton.disabled = true;
}

async function capturePhoto() {
  if (!cameraVideo.videoWidth || !cameraVideo.videoHeight) {
    return;
  }

  const stage = document.createElement("canvas");
  stage.width = cameraVideo.videoWidth;
  stage.height = cameraVideo.videoHeight;
  stage.getContext("2d").drawImage(cameraVideo, 0, 0, stage.width, stage.height);
  const source = stage.toDataURL("image/jpeg", 0.92);
  const image = await loadImageFromSource(source);
  stopCamera();
  setCurrentImage(image, "撮影画像を読み込みました。");
  await detectAndApplyCurrentImage();
}

function replaceHand(entries) {
  handEntries = entries;
  activeEntryId = handEntries[0]?.id ?? null;
  expandedEntryId = null;
  syncUi();
}

function assignCharacterToEntry(entryId, name) {
  handEntries = handEntries.map((entry) => (entry.id === entryId ? { ...entry, name } : entry));
  activeEntryId = entryId;
  pendingNewEntryId = pendingNewEntryId === entryId ? null : pendingNewEntryId;
  syncUi();
}

function addManualEntry() {
  const entry = createEntry();
  handEntries = [...handEntries, entry];
  activeEntryId = entry.id;
  pendingNewEntryId = entry.id;
  resetPickerFilters();
  syncUi();
  openModal("picker");
}

function removeEntry(entryId) {
  handEntries = handEntries.filter((entry) => entry.id !== entryId);
  if (activeEntryId === entryId) {
    activeEntryId = handEntries[0]?.id ?? null;
  }
  if (expandedEntryId === entryId) {
    expandedEntryId = null;
  }
  syncUi();
}

async function detectAndApplyCurrentImage() {
  if (!detectorSession || !matcherSession || !currentImage) {
    return;
  }

  setStatus("認識中");
  recognitionNote.textContent = "認識しています...";

  const { canvas: stage, ratio, dw, dh } = letterbox(currentImage, MODEL_SIZE);
  const tensor = toTensor(stage);
  const inputName = detectorSession.inputNames[0];
  const outputName = detectorSession.outputNames[0];
  const outputs = await detectorSession.run({ [inputName]: tensor });
  const boxes = sortBoxesForHand(decode(outputs[outputName], currentImage, ratio, dw, dh, 0.12, 0.4));
  const matches = await matchBoxes(currentImage, boxes);

  const entries = boxes.map((box, index) =>
    createEntry({
      box,
      score: box.score,
      matches: matches[index],
      name: matches[index]?.[0]?.name || "",
    })
  );

  replaceHand(entries);
  drawResult(currentImage, entries);
  closeModal("recognition");
  setStatus(`${entries.length} 枚認識`);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function escapeAttr(value) {
  return escapeHtml(value);
}

openRecognitionButton.addEventListener("click", () => {
  openModal("recognition");
});

showRecommendedButton.addEventListener("click", () => {
  renderRecommendedUnits();
  openModal("recommended");
});

openFileButton.addEventListener("click", () => {
  fileInput.click();
});

fileInput.addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (!file) {
    return;
  }
  const image = await loadImageFromSource(URL.createObjectURL(file));
  stopCamera();
  setCurrentImage(image, "画像を読み込みました。");
  await detectAndApplyCurrentImage();
  event.target.value = "";
});

startCameraButton.addEventListener("click", () => {
  startCamera().catch((error) => {
    console.error(error);
    recognitionNote.textContent = `カメラ起動に失敗しました: ${error.message}`;
  });
});

capturePhotoButton.addEventListener("click", () => {
  capturePhoto().catch((error) => {
    console.error(error);
    recognitionNote.textContent = `撮影に失敗しました: ${error.message}`;
  });
});

addPieceButton.addEventListener("click", () => {
  addManualEntry();
});

attributeSelect.addEventListener("change", renderCharacterPicker);
initialSelect.addEventListener("change", renderCharacterPicker);
searchInput.addEventListener("input", renderCharacterPicker);

handList.addEventListener("click", (event) => {
  const button = event.target.closest("[data-action]");
  if (!button) {
    return;
  }

  const action = button.dataset.action;
  const entryId = Number(button.dataset.entryId);

  if (action === "edit-entry") {
    activeEntryId = entryId;
    pendingNewEntryId = null;
    resetPickerFilters();
    renderCharacterPicker();
    openModal("picker");
    return;
  }
  if (action === "remove-entry") {
    removeEntry(entryId);
    return;
  }
  if (action === "toggle-accordion") {
    expandedEntryId = expandedEntryId === entryId ? null : entryId;
    activeEntryId = entryId;
    renderHandList();
  }
});

characterList.addEventListener("click", (event) => {
  const button = event.target.closest("[data-name]");
  if (!button || activeEntryId === null) {
    return;
  }
  assignCharacterToEntry(activeEntryId, button.dataset.name);
  closeModal("picker");
});

document.addEventListener("click", (event) => {
  const closeTarget = event.target.closest("[data-close-modal]");
  if (!closeTarget) {
    return;
  }
  closeModal(closeTarget.dataset.closeModal);
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") {
    return;
  }
  if (!recognitionModal.hidden) {
    closeModal("recognition");
  } else if (!pickerModal.hidden) {
    closeModal("picker");
  } else if (!recommendedModal.hidden) {
    closeModal("recommended");
  }
});

loadApp().catch((error) => {
  console.error(error);
  setStatus(`初期化失敗: ${error.message}`);
});
