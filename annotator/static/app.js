const LABELS = {
  neutral: { name: "中立" },
  positive: { name: "正面" },
  negative: { name: "负面" },
};

const state = {
  session: null,
  filter: "unlabeled",
  currentId: null,
  recentIds: [],
  pendingAction: null,
  exportMessage: "",
  busy: false,
};

const elements = {
  reviewerChip: document.getElementById("reviewer-chip"),
  topProgress: document.getElementById("top-progress"),
  progressFill: document.getElementById("progress-fill"),
  queuePosition: document.getElementById("queue-position"),
  statusDot: document.getElementById("status-dot"),
  itemStatus: document.getElementById("item-status"),
  reviewPanel: document.getElementById("review-panel"),
  itemId: document.getElementById("item-id"),
  itemSource: document.getElementById("item-source"),
  itemDomain: document.getElementById("item-domain"),
  reviewText: document.getElementById("review-text"),
  labelButtons: [...document.querySelectorAll(".label-button")],
  filterButtons: [...document.querySelectorAll("[data-filter]")],
  prevButton: document.getElementById("prev-button"),
  nextButton: document.getElementById("next-button"),
  clearButton: document.getElementById("clear-button"),
  annotatedCount: document.getElementById("annotated-count"),
  totalCount: document.getElementById("total-count"),
  remainingLabel: document.getElementById("remaining-label"),
  sideProgressFill: document.getElementById("side-progress-fill"),
  countNeutral: document.getElementById("count-neutral"),
  countPositive: document.getElementById("count-positive"),
  countNegative: document.getElementById("count-negative"),
  recentList: document.getElementById("recent-list"),
  recentCount: document.getElementById("recent-count"),
  exportButton: document.getElementById("export-button"),
  exportNote: document.getElementById("export-note"),
  dialog: document.getElementById("confirm-dialog"),
  dialogTitle: document.getElementById("dialog-title"),
  dialogMessage: document.getElementById("dialog-message"),
  dialogPreview: document.getElementById("dialog-preview"),
  dialogConfirm: document.getElementById("dialog-confirm"),
  dialogCancel: document.getElementById("dialog-cancel"),
  dialogClose: document.getElementById("dialog-close"),
  toast: document.getElementById("toast"),
};

let toastTimer = null;

async function api(path, options = {}) {
  const request = {
    headers: { "Content-Type": "application/json" },
    ...options,
  };
  const response = await fetch(path, request);
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    throw new Error(payload?.error || `请求失败 (${response.status})`);
  }
  return payload?.result;
}

function currentItem() {
  if (!state.session || !state.currentId) return null;
  return state.session.items.find((item) => item.id === state.currentId) || null;
}

function queueItems() {
  if (!state.session) return [];
  if (state.filter === "all") return state.session.items;
  return state.session.items.filter((item) => !item.label);
}

function syncCurrentToQueue() {
  const queue = queueItems();
  if (!queue.length) {
    state.currentId = null;
    return;
  }
  if (!queue.some((item) => item.id === state.currentId)) {
    state.currentId = queue[0].id;
  }
}

function metrics() {
  const items = state.session?.items || [];
  const counts = { neutral: 0, positive: 0, negative: 0 };
  for (const item of items) {
    if (item.label && counts[item.label] !== undefined) counts[item.label] += 1;
  }
  const annotated = counts.neutral + counts.positive + counts.negative;
  return { counts, annotated, total: items.length, remaining: items.length - annotated };
}

function render() {
  if (!state.session) return;
  syncCurrentToQueue();
  renderStats();
  renderRecent();
  renderQueuePosition();
  renderItem();
}

function renderStats() {
  const { counts, annotated, total, remaining } = metrics();
  const percentage = total ? (annotated / total) * 100 : 0;

  elements.reviewerChip.textContent = `标注员 ${state.session.annotator.toUpperCase()}`;
  elements.topProgress.textContent = `${annotated} / ${total}`;
  elements.progressFill.style.width = `${percentage}%`;
  elements.annotatedCount.textContent = String(annotated);
  elements.totalCount.textContent = `/ ${total}`;
  elements.remainingLabel.textContent = `剩余 ${remaining} 条`;
  elements.sideProgressFill.style.width = `${percentage}%`;
  elements.countNeutral.textContent = String(counts.neutral);
  elements.countPositive.textContent = String(counts.positive);
  elements.countNegative.textContent = String(counts.negative);
  elements.exportButton.disabled = state.busy;
  elements.exportNote.textContent = state.exportMessage || (
    annotated
      ? `已确认 ${annotated} 条，可同步到 CSV`
      : "暂无已确认标注，可导出当前空白状态"
  );
}

function renderQueuePosition() {
  const queue = queueItems();
  const index = queue.findIndex((item) => item.id === state.currentId);
  if (!queue.length) {
    elements.queuePosition.textContent = "当前队列已清空";
  } else {
    elements.queuePosition.textContent = `第 ${index + 1} / ${queue.length} 条`;
  }
  elements.prevButton.disabled =
    state.busy || !queue.length || index <= 0;
  elements.nextButton.disabled =
    state.busy || !queue.length || index < 0 || index >= queue.length - 1;
}

function renderItem() {
  const item = currentItem();
  const queue = queueItems();
  const isDone = !item;

  elements.reviewPanel.classList.toggle("is-empty", isDone);
  elements.labelButtons.forEach((button) => {
    const active = item?.label === button.dataset.label;
    button.classList.toggle("is-selected", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
    button.disabled = isDone || state.busy;
  });

  if (isDone) {
    elements.itemId.textContent = "--";
    elements.itemSource.textContent = "--";
    elements.itemDomain.textContent = "--";
    elements.itemStatus.textContent = queue.length ? "未标注" : "队列已清空";
    elements.statusDot.classList.remove("is-labeled");
    elements.reviewText.textContent =
      state.filter === "unlabeled" && metrics().remaining === 0
        ? "全部评论已完成标注。"
        : "当前筛选条件下没有可展示的评论。";
    elements.clearButton.hidden = true;
    return;
  }

  const labelName = item.label ? LABELS[item.label].name : "未标注";
  elements.itemId.textContent = item.id;
  elements.itemSource.textContent = item.source || "未知来源";
  elements.itemDomain.textContent = item.domain || "未分类";
  elements.itemStatus.textContent = labelName;
  elements.statusDot.classList.toggle("is-labeled", Boolean(item.label));
  elements.reviewText.textContent = item.text;
  elements.clearButton.hidden = !item.label;
  elements.clearButton.disabled = state.busy;
}

function renderRecent() {
  const items = state.session?.items || [];
  const byId = new Map(items.map((item) => [item.id, item]));
  state.recentIds = state.recentIds.filter((id) => byId.get(id)?.label);
  const recent = state.recentIds
    .map((id) => byId.get(id))
    .filter(Boolean)
    .slice(0, 5);

  elements.recentCount.textContent = String(recent.length);
  elements.recentList.replaceChildren();
  if (!recent.length) {
    const empty = document.createElement("li");
    empty.className = "recent-empty";
    empty.textContent = "尚无历史记录";
    elements.recentList.append(empty);
    return;
  }

  for (const item of recent) {
    const row = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.recentId = item.id;
    button.innerHTML = "";

    const id = document.createElement("span");
    id.className = "recent-id";
    id.textContent = `#${item.id}`;
    const label = document.createElement("span");
    label.className = `recent-label ${item.label}`;
    label.textContent = LABELS[item.label].name;
    button.append(id, label);
    button.addEventListener("click", () => {
      state.filter = "all";
      state.currentId = item.id;
      updateFilterButtons();
      render();
    });
    row.append(button);
    elements.recentList.append(row);
  }
}

function updateFilterButtons() {
  elements.filterButtons.forEach((button) => {
    const active = button.dataset.filter === state.filter;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });
}

function move(direction) {
  const queue = queueItems();
  if (!queue.length) return;
  let index = queue.findIndex((item) => item.id === state.currentId);
  if (index < 0) index = 0;
  const nextIndex = Math.max(0, Math.min(queue.length - 1, index + direction));
  state.currentId = queue[nextIndex].id;
  render();
}

function nextUnlabeledAfterCurrent() {
  const items = state.session.items;
  const currentIndex = items.findIndex((item) => item.id === state.currentId);
  const ordered = items.slice(currentIndex + 1).concat(items.slice(0, currentIndex + 1));
  return ordered.find((item) => !item.label) || null;
}

function requestLabel(label) {
  const item = currentItem();
  if (!item || state.busy) return;
  state.pendingAction = { type: "label", label };
  const previous = item.label ? `当前为“${LABELS[item.label].name}”，` : "";
  elements.dialogTitle.textContent = "确认标注";
  elements.dialogMessage.textContent = `${previous}确认将这条评论标记为“${LABELS[label].name}”。`;
  elements.dialogPreview.textContent = item.text;
  elements.dialogConfirm.textContent = "确认标注";
  elements.dialog.showModal();
}

function requestClear() {
  const item = currentItem();
  if (!item?.label || state.busy) return;
  state.pendingAction = { type: "clear" };
  elements.dialogTitle.textContent = "确认撤销";
  elements.dialogMessage.textContent = `移除本条评论当前的“${LABELS[item.label].name}”标注。`;
  elements.dialogPreview.textContent = item.text;
  elements.dialogConfirm.textContent = "确认撤销";
  elements.dialog.showModal();
}

async function confirmPendingAction() {
  const action = state.pendingAction;
  const item = currentItem();
  if (!action || !item || state.busy) return;

  state.busy = true;
  elements.dialogConfirm.disabled = true;
  render();
  try {
    if (action.type === "label") {
      const updated = await api("/api/annotations", {
        method: "POST",
        body: JSON.stringify({ id: item.id, label: action.label }),
      });
      replaceItem(updated);
      state.exportMessage = "";
      state.recentIds = [
        updated.id,
        ...state.recentIds.filter((id) => id !== updated.id),
      ];
      if (state.filter === "unlabeled") {
        const next = nextUnlabeledAfterCurrent();
        state.currentId = next?.id || null;
      }
      showToast(`已记录为“${LABELS[action.label].name}”`);
    } else {
      const updated = await api(`/api/annotations/${encodeURIComponent(item.id)}`, {
        method: "DELETE",
      });
      replaceItem(updated);
      state.exportMessage = "";
      state.recentIds = state.recentIds.filter((id) => id !== updated.id);
      showToast("已撤销本条标注");
    }
    elements.dialog.close();
  } catch (error) {
    showToast(error.message, true);
  } finally {
    state.busy = false;
    state.pendingAction = null;
    elements.dialogConfirm.disabled = false;
    render();
  }
}

function replaceItem(updated) {
  const index = state.session.items.findIndex((item) => item.id === updated.id);
  if (index >= 0) state.session.items[index] = updated;
}

async function exportCsv() {
  if (state.busy) return;
  state.busy = true;
  render();
  try {
    const result = await api("/api/export/csv", { method: "POST" });
    state.exportMessage = `已导出 ${result.annotated} 条`;
    showToast(`已写入 ${result.path}`);
  } catch (error) {
    showToast(error.message, true);
  } finally {
    state.busy = false;
    render();
  }
}

function showToast(message, isError = false) {
  window.clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.toggle("is-error", isError);
  elements.toast.classList.add("is-visible");
  toastTimer = window.setTimeout(() => {
    elements.toast.classList.remove("is-visible");
  }, 2400);
}

function bindEvents() {
  elements.labelButtons.forEach((button) => {
    button.addEventListener("click", () => requestLabel(button.dataset.label));
  });

  elements.filterButtons.forEach((button) => {
    button.addEventListener("click", () => {
      state.filter = button.dataset.filter;
      updateFilterButtons();
      render();
    });
  });

  elements.prevButton.addEventListener("click", () => move(-1));
  elements.nextButton.addEventListener("click", () => move(1));
  elements.clearButton.addEventListener("click", requestClear);
  elements.exportButton.addEventListener("click", exportCsv);
  elements.dialogCancel.addEventListener("click", () => elements.dialog.close());
  elements.dialogClose.addEventListener("click", () => elements.dialog.close());
  elements.dialogConfirm.addEventListener("click", confirmPendingAction);

  elements.dialog.addEventListener("cancel", () => {
    state.pendingAction = null;
  });
  elements.dialog.addEventListener("click", (event) => {
    if (event.target === elements.dialog) elements.dialog.close();
  });
  elements.dialog.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && state.pendingAction && !state.busy) {
      event.preventDefault();
      confirmPendingAction();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (elements.dialog.open || state.busy) return;
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    if (event.key === "1") requestLabel("neutral");
    if (event.key === "2") requestLabel("positive");
    if (event.key === "3") requestLabel("negative");
    if (event.key === "ArrowLeft") move(-1);
    if (event.key === "ArrowRight") move(1);
  });
}

async function init() {
  bindEvents();
  try {
    state.session = await api("/api/session");
    const annotated = state.session.items
      .filter((item) => item.label)
      .sort((a, b) => String(b.updated_at || "").localeCompare(String(a.updated_at || "")));
    state.recentIds = annotated.slice(0, 5).map((item) => item.id);
    state.currentId = state.session.items.find((item) => !item.label)?.id
      || state.session.items[0]?.id
      || null;
    updateFilterButtons();
    render();
  } catch (error) {
    elements.reviewPanel.classList.add("is-empty");
    elements.reviewText.textContent = `加载失败：${error.message}`;
    showToast(error.message, true);
  }
}

init();
