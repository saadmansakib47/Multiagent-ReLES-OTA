const form = document.querySelector("#runForm");
const resetBtn = document.querySelector("#resetBtn");
const deviceSelect = document.querySelector("#deviceSelect");
const recommendation = document.querySelector("#recommendation");
const runState = document.querySelector("#runState");
const runDot = document.querySelector("#runDot");
const progressBar = document.querySelector("#progressBar");
const seedProgressBar = document.querySelector("#seedProgressBar");
const seedProgressText = document.querySelector("#seedProgressText");
const stepProgressText = document.querySelector("#stepProgressText");
const metricsBody = document.querySelector("#metricsTable tbody");
const leaderboard = document.querySelector("#leaderboard");
const charts = document.querySelector("#charts");
const logBox = document.querySelector("#logBox");
const copyOutputBtn = document.querySelector("#copyOutputBtn");
const chartModal = document.querySelector("#chartModal");
const chartModalTitle = document.querySelector("#chartModalTitle");
const chartModalImage = document.querySelector("#chartModalImage");

let cudaAvailable = false;
let totalTimesteps = Number(field("timesteps").value);
let autoScrollEnabled = false;

function field(name) {
  return form.elements.namedItem(name);
}

function readPayload() {
  const data = new FormData(form);
  return {
    device: data.get("device"),
    n_envs: Number(data.get("n_envs")),
    n_steps: Number(data.get("n_steps")),
    timesteps: Number(data.get("timesteps")),
    seeds: Number(data.get("seeds")),
    n_agents: Number(data.get("n_agents")),
    n_blocks: Number(data.get("n_blocks")),
    algorithm: data.get("algorithm"),
    compare_algorithm: data.get("compare_algorithm"),
    mode: data.get("mode"),
    batch_size: Number(data.get("batch_size")),
    safety: field("safety").checked,
    constrained_network_mode: field("constrained_network_mode").checked,
    death_masking: field("death_masking").checked,
    type_conditioning: field("type_conditioning").checked,
    coupled_channel: field("coupled_channel").checked,
    gateway_bw_mbps: Number(field("gateway_bw_mbps").value) || 50.0,
  };
}

function updateRecommendation() {
  const payload = readPayload();
  const product = payload.n_envs * payload.n_steps;
  totalTimesteps = payload.timesteps;
  let text = `Rollout product: ${payload.n_steps} x ${payload.n_envs} = ${product.toLocaleString()}. `;
  if (product >= 20480 && cudaAvailable) {
    text += "GPU recommended for this rollout size.";
  } else if (product >= 20480) {
    text += "GPU would be recommended, but CUDA is not detected.";
  } else {
    text += "CPU is fine for this quick or small run.";
  }
  recommendation.textContent = text;
}

function setDeviceOptions(device) {
  cudaAvailable = Boolean(device.cuda_available);
  const current = device.current === "cuda" ? `CUDA${device.gpu_name ? ` (${device.gpu_name})` : ""}` : "CPU";
  document.querySelector("#deviceText").textContent = current;
  document.querySelector("#deviceHint").textContent = device.torch_version ? `PyTorch ${device.torch_version}` : "PyTorch status unavailable";

  const selected = deviceSelect.value || "auto";
  deviceSelect.innerHTML = "";
  [
    ["auto", "Auto"],
    ["cpu", "CPU"],
    ...(cudaAvailable ? [["cuda", "CUDA / GPU"]] : []),
  ].forEach(([value, label]) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    deviceSelect.appendChild(option);
  });
  deviceSelect.value = [...deviceSelect.options].some((option) => option.value === selected) ? selected : "auto";
  updateRecommendation();
}

function formatElapsedValue(value) {
  if (value === null || value === undefined || value === "") return "-";
  const text = String(value).trim();
  if (!text || text === "-") return "-";
  const numeric = Number(text);
  if (Number.isFinite(numeric)) return `${numeric.toLocaleString()} s`;
  return `${text} s`;
}

function renderMetrics(metrics = {}) {
  metricsBody.innerHTML = "";
  const entries = Object.entries(metrics);
  if (!entries.length) {
    metricsBody.innerHTML = `<tr><td colspan="2">No live metrics yet.</td></tr>`;
    return;
  }
  entries.forEach(([key, value]) => {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${key}</td><td>${value}</td>`;
    metricsBody.appendChild(row);
  });

  document.querySelector("#fps").textContent = metrics["time / fps"] || metrics.fps || "-";
  document.querySelector("#iterations").textContent = metrics["time / iterations"] || metrics.iterations || "-";
  document.querySelector("#elapsed").textContent = formatElapsedValue(metrics["time / time_elapsed"] || metrics.time_elapsed || "-");
}

function renderLeaderboard(rows = []) {
  const thead = leaderboard.querySelector("thead");
  const tbody = leaderboard.querySelector("tbody");
  thead.innerHTML = "";
  tbody.innerHTML = "";
  if (!rows.length) {
    tbody.innerHTML = `<tr><td>No leaderboard rows yet.</td></tr>`;
    return;
  }
  const columns = Object.keys(rows[0]);
  thead.innerHTML = `<tr>${columns.map((column) => `<th>${column}</th>`).join("")}</tr>`;
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = columns.map((column) => `<td>${row[column] ?? ""}</td>`).join("");
    tbody.appendChild(tr);
  });
}

function openChartModal(item) {
  chartModalTitle.textContent = item.name;
  chartModalImage.src = `${item.url}&v=${item.modified}`;
  chartModalImage.alt = item.name;
  chartModal.classList.add("open");
  chartModal.setAttribute("aria-hidden", "false");
  document.body.classList.add("modal-open");
}

function closeChartModal() {
  chartModal.classList.remove("open");
  chartModal.setAttribute("aria-hidden", "true");
  chartModalImage.removeAttribute("src");
  chartModalImage.alt = "";
  document.body.classList.remove("modal-open");
}

function renderCharts(items = []) {
  charts.innerHTML = "";
  if (!items.length) {
    charts.innerHTML = `<div class="empty">No generated charts yet. Run or compare experiments to fill this area.</div>`;
    return;
  }
  items.forEach((item) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "chart-card";
    card.dataset.name = item.name;
    card.dataset.chartUrl = item.url;
    card.dataset.modified = item.modified;
    card.innerHTML = `<strong>${item.name}</strong><img src="${item.url}&v=${item.modified}" alt="${item.name}">`;
    charts.appendChild(card);
  });
}

function syncTerminalScroll(state) {
  const running = Boolean(state.running);
  autoScrollEnabled = running;
  if (!running) return;
  requestAnimationFrame(() => {
    logBox.scrollTop = logBox.scrollHeight;
  });
}

async function copyLastOutputLines() {
  const rawText = logBox.textContent || "";
  const lines = rawText.split(/\r?\n/).slice(-10).filter((line) => line.length > 0);
  const text = lines.join("\n");
  if (!text) {
    copyOutputBtn.textContent = "No output yet";
    setTimeout(() => {
      copyOutputBtn.textContent = "Copy last 10 lines";
    }, 1200);
    return;
  }

  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      const temp = document.createElement("textarea");
      temp.value = text;
      temp.setAttribute("readonly", "");
      temp.style.position = "fixed";
      temp.style.left = "-9999px";
      document.body.appendChild(temp);
      temp.select();
      document.execCommand("copy");
      document.body.removeChild(temp);
    }
    copyOutputBtn.textContent = "Copied";
  } catch (error) {
    copyOutputBtn.textContent = "Copy failed";
  }

  setTimeout(() => {
    copyOutputBtn.textContent = "Copy last 10 lines";
  }, 1200);
}

function renderState(state) {
  setDeviceOptions(state.device || {});
  const running = Boolean(state.running);
  const status = running ? "running" : state.returncode === 0 ? "done" : state.error ? "error" : "idle";
  runState.textContent = status === "running" ? "Running" : status === "done" ? "Done" : status === "error" ? "Error" : "Idle";
  runDot.classList.remove("running", "done", "error");
  if (status !== "idle") {
    runDot.classList.add(status);
  }
  form.querySelector(".primary").disabled = running;
  resetBtn.disabled = running;
  document.querySelector("#seedText").textContent = state.current_seed ? `Seed ${state.current_seed}` : "Seed -";

  const totalSeeds = Math.max(1, Number(field("seeds").value) || 1);
  const seedMatch = state.current_seed?.match(/(\d+)\s*\/\s*(\d+)/);
  const currentSeedIndex = seedMatch ? Number(seedMatch[1]) : 0;
  const seenSeedCount = seedMatch && totalSeeds > 0
    ? Math.min(totalSeeds, running ? Math.max(0, currentSeedIndex - 1) : currentSeedIndex)
    : 0;
  const seedPercent = totalSeeds > 0 ? Math.min(100, Math.round((seenSeedCount / totalSeeds) * 100)) : 0;
  seedProgressText.textContent = `${seenSeedCount}/${totalSeeds}`;
  seedProgressBar.style.width = `${seedPercent}%`;

  const parsedProgress = Number(state.progress || 0);
  const stepPercent = parsedProgress > 100
    ? Math.min(100, Math.round((parsedProgress / Math.max(totalTimesteps, parsedProgress)) * 100))
    : Math.min(100, Number.isFinite(parsedProgress) ? parsedProgress : 0);
  progressBar.style.width = `${stepPercent}%`;
  stepProgressText.textContent = `${Math.round(stepPercent)}%`;

  renderMetrics(state.metrics);
  renderLeaderboard(state.leaderboard);
  renderCharts(state.charts);
  logBox.textContent = (state.logs || []).join("\n");
  syncTerminalScroll(state);
}

form.addEventListener("input", updateRecommendation);

const coupledChannelInput = field("coupled_channel");
const gatewayBwGroup = document.querySelector("#gatewayBwGroup");
function syncGatewayBwVisibility() {
  if (coupledChannelInput && gatewayBwGroup) {
    gatewayBwGroup.style.display = coupledChannelInput.checked ? "grid" : "none";
  }
}
if (coupledChannelInput) {
  coupledChannelInput.addEventListener("change", syncGatewayBwVisibility);
  syncGatewayBwVisibility();
}

copyOutputBtn.addEventListener("click", copyLastOutputLines);

charts.addEventListener("click", (event) => {
  const card = event.target.closest(".chart-card");
  if (!card) return;
  openChartModal({
    name: card.dataset.name,
    url: card.dataset.chartUrl,
    modified: card.dataset.modified,
  });
});

charts.addEventListener("keydown", (event) => {
  const card = event.target.closest(".chart-card");
  if (!card || !(event.key === "Enter" || event.key === " ")) {
    return;
  }
  event.preventDefault();
  openChartModal({
    name: card.dataset.name,
    url: card.dataset.chartUrl,
    modified: card.dataset.modified,
  });
});

chartModal.addEventListener("click", (event) => {
  if (event.target.matches("[data-close='true']") || event.target.matches(".chart-modal__close")) {
    closeChartModal();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && chartModal.classList.contains("open")) {
    closeChartModal();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const response = await fetch("/api/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(readPayload()),
  });
  const data = await response.json();
  if (!data.ok) alert(data.message);
});

resetBtn.addEventListener("click", async () => {
  await fetch("/api/reset", { method: "POST" });
});

async function boot() {
  try {
    const response = await fetch("/api/status");
    renderState(await response.json());
    const events = new EventSource("/events");
    events.onmessage = (event) => renderState(JSON.parse(event.data));
    events.onerror = () => {
      runState.textContent = "Reconnecting";
    };
  } catch (error) {
    runState.textContent = "UI Error";
    logBox.textContent = `Interface initialization failed: ${error.message}`;
  }
}

boot();
