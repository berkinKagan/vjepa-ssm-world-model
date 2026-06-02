const statusBadge = document.querySelector("#statusBadge");
const pipelineTab = document.querySelector("#pipelineTab");
const benchmarkTab = document.querySelector("#benchmarkTab");
const pipelineView = document.querySelector("#pipelineView");
const benchmarkView = document.querySelector("#benchmarkView");
const fileInput = document.querySelector("#fileInput");
const videoSelect = document.querySelector("#videoSelect");
const pipelineEmbeddingSelect = document.querySelector("#pipelineEmbeddingSelect");
const runPipelineButton = document.querySelector("#runPipelineButton");
const pipelineProgressBar = document.querySelector("#pipelineProgressBar");
const pipelineStatusText = document.querySelector("#pipelineStatusText");
const embeddingPath = document.querySelector("#embeddingPath");
const checkpointPath = document.querySelector("#checkpointPath");
const predictionPath = document.querySelector("#predictionPath");
const predictionShape = document.querySelector("#predictionShape");
const pipelineOutput = document.querySelector("#pipelineOutput");
const pipelineModelText = document.querySelector("#pipelineModelText");
const benchmarkEmbeddingSelect = document.querySelector("#benchmarkEmbeddingSelect");
const benchmarkStepsInput = document.querySelector("#benchmarkStepsInput");
const runBenchmarkButton = document.querySelector("#runBenchmarkButton");
const benchmarkProgressBar = document.querySelector("#benchmarkProgressBar");
const benchmarkStatusText = document.querySelector("#benchmarkStatusText");
const benchmarkEmbeddingPath = document.querySelector("#benchmarkEmbeddingPath");
const benchmarkOutputPath = document.querySelector("#benchmarkOutputPath");
const benchmarkOutput = document.querySelector("#benchmarkOutput");
const benchmarkModelText = document.querySelector("#benchmarkModelText");

let pipelinePolling = null;
let benchmarkPolling = null;

async function loadVideos() {
  try {
    const response = await fetch("/videos");
    const data = await response.json();
    fillSelect(videoSelect, data.videos, "No sample videos found", "Select sample video");
  } catch (error) {
    videoSelect.innerHTML = '<option value="">Could not load videos</option>';
  }
}

async function loadEmbeddings() {
  try {
    const response = await fetch("/embeddings");
    const data = await response.json();
    fillSelect(pipelineEmbeddingSelect, data.files, "No embeddings found", "Select existing latents");
    fillSelect(benchmarkEmbeddingSelect, data.files, "No embeddings found", "Select embeddings");
  } catch (error) {
    pipelineEmbeddingSelect.innerHTML = '<option value="">Could not load embeddings</option>';
    benchmarkEmbeddingSelect.innerHTML = '<option value="">Could not load embeddings</option>';
  }
}

function fillSelect(select, values, emptyText, promptText) {
  select.innerHTML = "";
  const empty = document.createElement("option");
  empty.value = "";
  empty.textContent = values.length ? promptText : emptyText;
  select.appendChild(empty);
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  }
}

async function startPipeline() {
  resetPipeline();
  const formData = new FormData();
  if (pipelineEmbeddingSelect.value) {
    formData.append("embedding", pipelineEmbeddingSelect.value);
  } else if (fileInput.files.length > 0) {
    formData.append("file", fileInput.files[0]);
  } else if (videoSelect.value) {
    formData.append("video_path", videoSelect.value);
  } else {
    setPipelineStatus("idle", "Choose latents or a video.", 0);
    return;
  }
  runPipelineButton.disabled = true;
  setPipelineStatus("queued", "Submitting pipeline job.", 0);
  try {
    const response = await fetch("/pipeline/run", { method: "POST", body: formData });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Pipeline request failed");
    }
    const data = await response.json();
    pollPipeline(data.job_id);
  } catch (error) {
    runPipelineButton.disabled = false;
    setPipelineStatus("failed", error.message, 0);
  }
}

function pollPipeline(jobId) {
  if (pipelinePolling) {
    clearInterval(pipelinePolling);
  }
  pipelinePolling = setInterval(async () => {
    try {
      const response = await fetch(`/pipeline/status/${jobId}`);
      const data = await response.json();
      renderPipelineStatus(data);
      if (data.status === "completed") {
        clearInterval(pipelinePolling);
        runPipelineButton.disabled = false;
        await loadPipelineResults(jobId);
        await loadEmbeddings();
      }
      if (data.status === "failed") {
        clearInterval(pipelinePolling);
        runPipelineButton.disabled = false;
      }
    } catch (error) {
      clearInterval(pipelinePolling);
      runPipelineButton.disabled = false;
      setPipelineStatus("failed", error.message, 0);
    }
  }, 1000);
}

async function loadPipelineResults(jobId) {
  const response = await fetch(`/pipeline/results/${jobId}`);
  const result = await response.json();
  pipelineOutput.textContent = JSON.stringify(result, null, 2);
  pipelineModelText.textContent = `model: ${result.model_type || "-"}`;
}

function renderPipelineStatus(data) {
  setPipelineStatus(data.status, data.error || data.message, data.progress);
  embeddingPath.textContent = data.embedding_path || "-";
  checkpointPath.textContent = data.checkpoint_path || "-";
  predictionPath.textContent = data.prediction_path || "-";
  predictionShape.textContent = data.prediction_shape ? JSON.stringify(data.prediction_shape) : "-";
}

function setPipelineStatus(status, message, progress) {
  statusBadge.textContent = status;
  pipelineStatusText.textContent = message;
  pipelineProgressBar.style.width = `${Math.round((progress || 0) * 100)}%`;
}

function resetPipeline() {
  embeddingPath.textContent = "-";
  checkpointPath.textContent = "-";
  predictionPath.textContent = "-";
  predictionShape.textContent = "-";
  pipelineOutput.textContent = "{}";
  pipelineModelText.textContent = "model: -";
}

async function startBenchmark() {
  if (!benchmarkEmbeddingSelect.value) {
    setBenchmarkStatus("idle", "Choose embeddings first.", 0);
    return;
  }
  resetBenchmark();
  runBenchmarkButton.disabled = true;
  setBenchmarkStatus("queued", "Submitting benchmark job.", 0);
  try {
    const formData = new FormData();
    formData.append("embedding", benchmarkEmbeddingSelect.value);
    formData.append("steps", benchmarkStepsInput.value || "5");
    const response = await fetch("/benchmark/run", { method: "POST", body: formData });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Benchmark request failed");
    }
    const data = await response.json();
    pollBenchmark(data.job_id);
  } catch (error) {
    runBenchmarkButton.disabled = false;
    setBenchmarkStatus("failed", error.message, 0);
  }
}

function pollBenchmark(jobId) {
  if (benchmarkPolling) {
    clearInterval(benchmarkPolling);
  }
  benchmarkPolling = setInterval(async () => {
    try {
      const response = await fetch(`/benchmark/status/${jobId}`);
      const data = await response.json();
      renderBenchmarkStatus(data);
      if (data.status === "completed") {
        clearInterval(benchmarkPolling);
        runBenchmarkButton.disabled = false;
        await loadBenchmarkResults(jobId);
      }
      if (data.status === "failed") {
        clearInterval(benchmarkPolling);
        runBenchmarkButton.disabled = false;
      }
    } catch (error) {
      clearInterval(benchmarkPolling);
      runBenchmarkButton.disabled = false;
      setBenchmarkStatus("failed", error.message, 0);
    }
  }, 1000);
}

async function loadBenchmarkResults(jobId) {
  const response = await fetch(`/benchmark/results/${jobId}`);
  const result = await response.json();
  benchmarkOutput.textContent = JSON.stringify(result, null, 2);
  benchmarkModelText.textContent = `model: ${result.actual_model_type || "-"}`;
  benchmarkOutputPath.textContent = result.benchmark_file || "-";
}

function renderBenchmarkStatus(data) {
  setBenchmarkStatus(data.status, data.error || data.message, data.progress);
  benchmarkEmbeddingPath.textContent = data.embedding_path || "-";
}

function setBenchmarkStatus(status, message, progress) {
  statusBadge.textContent = status;
  benchmarkStatusText.textContent = message;
  benchmarkProgressBar.style.width = `${Math.round((progress || 0) * 100)}%`;
}

function resetBenchmark() {
  benchmarkEmbeddingPath.textContent = "-";
  benchmarkOutputPath.textContent = "-";
  benchmarkOutput.textContent = "{}";
  benchmarkModelText.textContent = "model: -";
}

function showTab(name) {
  const benchmark = name === "benchmark";
  pipelineView.hidden = benchmark;
  benchmarkView.hidden = !benchmark;
  pipelineTab.classList.toggle("active", !benchmark);
  benchmarkTab.classList.toggle("active", benchmark);
}

pipelineTab.addEventListener("click", () => showTab("pipeline"));
benchmarkTab.addEventListener("click", () => showTab("benchmark"));
runPipelineButton.addEventListener("click", startPipeline);
runBenchmarkButton.addEventListener("click", startBenchmark);
loadVideos();
loadEmbeddings();
