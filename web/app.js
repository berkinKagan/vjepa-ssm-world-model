const videoInput = document.querySelector("#videoInput");
const queryInput = document.querySelector("#queryInput");
const searchButton = document.querySelector("#searchButton");
const progressBar = document.querySelector("#progressBar");
const statusText = document.querySelector("#statusText");
const results = document.querySelector("#results");
const resultCount = document.querySelector("#resultCount");
const diagnosticsPanel = document.querySelector("#diagnostics");
const durationText = document.querySelector("#durationText");
const segmentsText = document.querySelector("#segmentsText");
const coverageText = document.querySelector("#coverageText");
const retrievalModeText = document.querySelector("#retrievalModeText");
const alignerStatusText = document.querySelector("#alignerStatusText");
const rerankingStatusText = document.querySelector("#rerankingStatusText");

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
  renderDiagnostics(data);
  renderResults(data.results || []);
  setStatus((data.diagnostics || {}).warning || "Done", 1);
}

function renderDiagnostics(data) {
  const diagnostics = data.diagnostics || {};
  const videoDuration = data.video_duration ?? diagnostics.video_duration;
  const segmentCount = data.segment_count ?? diagnostics.segment_count;
  const coverageRatio = data.coverage_ratio ?? diagnostics.coverage_ratio;
  const coverage = Number(coverageRatio || 0);
  durationText.textContent = videoDuration ? `${Number(videoDuration).toFixed(2)}s` : "-";
  segmentsText.textContent = segmentCount ?? "-";
  coverageText.textContent = coverage ? `${Math.round(coverage * 100)}%` : "-";
  retrievalModeText.textContent = data.retrieval_mode || "-";
  alignerStatusText.textContent = data.aligner_status || "-";
  rerankingStatusText.textContent = data.reranking_status || "-";
  diagnosticsPanel.hidden = false;
}

function renderResults(items) {
  resultCount.textContent = `${items.length} ${items.length === 1 ? "result" : "results"}`;
  results.innerHTML = "";
  for (const item of items) {
    const card = document.createElement("article");
    card.className = "resultCard";
    const preview = makePreview(item);
    const body = document.createElement("div");
    body.className = "resultBody";
    const title = document.createElement("div");
    title.className = "resultTitle";
    const score = item.final_score ?? item.score;
    title.textContent = `#${item.rank} · score ${Number(score).toFixed(3)} · ${formatTime(item.start_time)}-${formatTime(item.end_time)}`;
    const caption = document.createElement("p");
    caption.className = "caption";
    caption.textContent = item.searchable_summary || item.caption;
    body.append(title, caption);
    card.append(preview, body);
    results.appendChild(card);
  }
}

function makePreview(item) {
  const url = item.frame_urls && item.frame_urls[0] ? item.frame_urls[0] : "";
  if (url) {
    const image = document.createElement("img");
    image.alt = "";
    image.src = url;
    return image;
  }
  const placeholder = document.createElement("div");
  placeholder.className = "framePlaceholder";
  placeholder.textContent = "No frame preview";
  return placeholder;
}

function setStatus(message, progress) {
  statusText.textContent = message;
  progressBar.style.width = `${Math.round((progress || 0) * 100)}%`;
}

function clearResults() {
  results.innerHTML = "";
  resultCount.textContent = "0 results";
  progressBar.style.width = "0%";
  diagnosticsPanel.hidden = true;
  durationText.textContent = "-";
  segmentsText.textContent = "-";
  coverageText.textContent = "-";
  retrievalModeText.textContent = "-";
  alignerStatusText.textContent = "-";
  rerankingStatusText.textContent = "-";
}

function formatTime(value) {
  return `${Number(value).toFixed(2)}s`;
}

searchButton.addEventListener("click", startSearch);
