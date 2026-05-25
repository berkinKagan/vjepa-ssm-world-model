const fileInput = document.querySelector("#fileInput");
const videoSelect = document.querySelector("#videoSelect");
const extractButton = document.querySelector("#extractButton");
const statusBadge = document.querySelector("#statusBadge");
const statusText = document.querySelector("#statusText");
const progressBar = document.querySelector("#progressBar");
const embeddingPath = document.querySelector("#embeddingPath");
const metadataPath = document.querySelector("#metadataPath");
const metadataOutput = document.querySelector("#metadataOutput");
const shapeText = document.querySelector("#shapeText");

let polling = null;

async function loadVideos() {
  try {
    const response = await fetch("/videos");
    const data = await response.json();
    videoSelect.innerHTML = "";
    const empty = document.createElement("option");
    empty.value = "";
    empty.textContent = data.videos.length ? "Select a sample video" : "No sample videos found";
    videoSelect.appendChild(empty);
    for (const video of data.videos) {
      const option = document.createElement("option");
      option.value = video;
      option.textContent = video;
      videoSelect.appendChild(option);
    }
  } catch (error) {
    videoSelect.innerHTML = '<option value="">Could not load videos</option>';
  }
}

async function startExtraction() {
  clearStatus();
  const formData = new FormData();
  if (fileInput.files.length > 0) {
    formData.append("file", fileInput.files[0]);
  } else if (videoSelect.value) {
    formData.append("video_path", videoSelect.value);
  } else {
    setStatus("idle", "Choose a video first.", 0);
    return;
  }
  extractButton.disabled = true;
  setStatus("queued", "Submitting extraction job.", 0);
  try {
    const response = await fetch("/extract", { method: "POST", body: formData });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Extraction request failed");
    }
    const data = await response.json();
    pollStatus(data.job_id);
  } catch (error) {
    extractButton.disabled = false;
    setStatus("failed", error.message, 0);
  }
}

function pollStatus(jobId) {
  if (polling) {
    clearInterval(polling);
  }
  polling = setInterval(async () => {
    try {
      const response = await fetch(`/status/${jobId}`);
      const data = await response.json();
      renderStatus(data);
      if (data.status === "completed") {
        clearInterval(polling);
        extractButton.disabled = false;
        await loadMetadata(jobId);
      }
      if (data.status === "failed") {
        clearInterval(polling);
        extractButton.disabled = false;
      }
    } catch (error) {
      clearInterval(polling);
      extractButton.disabled = false;
      setStatus("failed", error.message, 0);
    }
  }, 1000);
}

async function loadMetadata(jobId) {
  const response = await fetch(`/metadata/${jobId}`);
  const metadata = await response.json();
  metadataOutput.textContent = JSON.stringify(metadata, null, 2);
  shapeText.textContent = `shape: ${JSON.stringify(metadata.embedding_shape || "-")}`;
}

function renderStatus(data) {
  setStatus(data.status, data.error || data.message, data.progress);
  embeddingPath.textContent = data.embedding_path || "-";
  metadataPath.textContent = data.metadata_path || "-";
}

function setStatus(status, message, progress) {
  statusBadge.textContent = status;
  statusText.textContent = message;
  progressBar.style.width = `${Math.round((progress || 0) * 100)}%`;
}

function clearStatus() {
  embeddingPath.textContent = "-";
  metadataPath.textContent = "-";
  metadataOutput.textContent = "{}";
  shapeText.textContent = "shape: -";
}

extractButton.addEventListener("click", startExtraction);
loadVideos();
