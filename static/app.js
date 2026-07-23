const statusEl = document.querySelector("#status");
const uploadForm = document.querySelector("#uploadForm");
const askForm = document.querySelector("#askForm");
const filesInput = document.querySelector("#files");
const fileList = document.querySelector("#fileList");
const questionInput = document.querySelector("#question");
const answerBox = document.querySelector("#answerBox");
const sourcesEl = document.querySelector("#sources");
const rebuildButton = document.querySelector("#rebuildButton");

function setStatus(message, kind = "") {
  statusEl.textContent = message;
  statusEl.className = `status ${kind}`.trim();
}

function setBusy(isBusy) {
  document.querySelectorAll("button").forEach((button) => {
    button.disabled = isBusy;
  });
}

function renderFiles(files) {
  fileList.innerHTML = "";
  files.forEach((file) => {
    const item = document.createElement("li");
    item.textContent = file;
    fileList.appendChild(item);
  });
}

function renderSources(sources) {
  sourcesEl.innerHTML = "";
  sources.forEach((source) => {
    const card = document.createElement("div");
    card.className = "source";
    const title = document.createElement("strong");
    title.textContent = source.source;
    const preview = document.createElement("p");
    preview.textContent = source.preview;
    card.append(title, preview);
    sourcesEl.appendChild(card);
  });
}

async function requestJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed.");
  }
  return data;
}

filesInput.addEventListener("change", () => {
  renderFiles(Array.from(filesInput.files).map((file) => file.name));
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!filesInput.files.length) {
    setStatus("Choose files first", "error");
    return;
  }

  const formData = new FormData();
  Array.from(filesInput.files).forEach((file) => formData.append("files", file));

  try {
    setBusy(true);
    setStatus("Uploading and indexing...");
    const data = await requestJson("/api/upload", {
      method: "POST",
      body: formData,
    });
    renderFiles(data.files || []);
    setStatus(`Indexed ${data.chunks} chunks`, "ok");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
});

rebuildButton.addEventListener("click", async () => {
  try {
    setBusy(true);
    setStatus("Rebuilding index...");
    const data = await requestJson("/api/ingest", { method: "POST" });
    setStatus(`Indexed ${data.chunks} chunks`, "ok");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
});

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) {
    setStatus("Type a question first", "error");
    questionInput.focus();
    return;
  }

  try {
    setBusy(true);
    setStatus("Thinking...");
    answerBox.textContent = "Searching your documents and asking Ollama...";
    sourcesEl.innerHTML = "";
    const data = await requestJson("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    answerBox.textContent = data.answer;
    renderSources(data.sources || []);
    setStatus("Answered", "ok");
  } catch (error) {
    answerBox.innerHTML = '<p class="muted">Your answer will appear here.</p>';
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
});
