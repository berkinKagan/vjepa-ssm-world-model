const videoInput = document.querySelector("#videoInput");
const queryInput = document.querySelector("#queryInput");
const searchButton = document.querySelector("#searchButton");
const progressBar = document.querySelector("#progressBar");
const statusText = document.querySelector("#statusText");
const results = document.querySelector("#results");
const resultCount = document.querySelector("#resultCount");
const summary = document.querySelector("#summary");
const durationText = document.querySelector("#durationText");
const segmentsText = document.querySelector("#segmentsText");
const coverageText = document.querySelector("#coverageText");

let polling = null;

async function startSearch() {
  clearResults();
  const file = videoInput.files[0];
  const query = queryInput.value.trim();
  if (!file) {
    setStatus("Choose a video first.", 0);
    return;
  }
  if (!query) {
    setStatus("Type a scene query first.", 0);
    return;
  }
  searchButton.disabled = true;
  setStatus("Uploading video", 0.02);
  try {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("query", query);
    formData.append("top_k", "5");
    const response = await fetch("/demo/search-video", { method: "POST", body: formData });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Search request failed");
    }
    const data = await response.json();
    pollStatus(data.job_id);
  } catch (error) {
    searchButton.disabled = false;
    setStatus(error.message, 0);
  }
}

function pollStatus(jobId) {
  if (polling) {
    clearInterval(polling);
  }
  polling = setInterval(async () => {
    try {
      const response = await fetch(`/demo/status/${jobId}`);
      const data = await response.json();
      setStatus(data.error || data.message, data.progress);
      if (data.status === "completed") {
        clearInterval(polling);
        searchButton.disabled = false;
        await loadResults(jobId);
      }
      if (data.status === "failed") {
        clearInterval(polling);
        searchButton.disabled = false;
      }
    } catch (error) {
      clearInterval(polling);
      searchButton.disabled = false;
      setStatus(error.message, 0);
    }
  }, 1000);
}

async function loadResults(jobId) {
  const response = await fetch(`/demo/results/${jobId}`);
  const data = await response.json();
  const diagnostics = data.diagnostics || {};
  renderDiagnostics(diagnostics);
  renderResults(data.results || []);
  setStatus(diagnostics.warning || "Done", 1);
}

function renderDiagnostics(diagnostics) {
  const duration = Number(diagnostics.video_duration || 0);
  const coverage = Number(diagnostics.coverage_ratio || 0);
  durationText.textContent = duration ? `${duration.toFixed(2)}s` : "-";
  segmentsText.textContent = diagnostics.segment_count ?? "-";
  coverageText.textContent = coverage ? `${Math.round(coverage * 100)}%` : "-";
  summary.hidden = false;
}

function renderResults(items) {
  resultCount.textContent = `${items.length} ${items.length === 1 ? "result" : "results"}`;
  results.innerHTML = "";
  for (const item of items) {
    const card = document.createElement("article");
    card.className = "resultCard";
    const image = document.createElement("img");
    image.alt = "";
    image.src = item.frame_urls && item.frame_urls[0] ? item.frame_urls[0] : "";
    const body = document.createElement("div");
    body.className = "resultBody";
    const title = document.createElement("div");
    title.className = "resultTitle";
    title.textContent = `#${item.rank} · score ${Number(item.score).toFixed(3)} · ${formatTime(item.start_time)}-${formatTime(item.end_time)}`;
    const caption = document.createElement("p");
    caption.className = "caption";
    caption.textContent = item.caption;
    body.append(title, caption);
    card.append(image, body);
    results.appendChild(card);
  }
}

function setStatus(message, progress) {
  statusText.textContent = message;
  progressBar.style.width = `${Math.round((progress || 0) * 100)}%`;
}

function clearResults() {
  results.innerHTML = "";
  resultCount.textContent = "0 results";
  progressBar.style.width = "0%";
  summary.hidden = true;
  durationText.textContent = "-";
  segmentsText.textContent = "-";
  coverageText.textContent = "-";
}

function formatTime(value) {
  return `${Number(value).toFixed(2)}s`;
}

searchButton.addEventListener("click", startSearch);
