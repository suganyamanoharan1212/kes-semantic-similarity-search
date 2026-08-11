const statusPill = document.getElementById("status-pill");
const buildForm = document.getElementById("build-form");
const fileInput = document.getElementById("file-input");
const fileDrop = document.getElementById("file-drop");
const fileDropLabel = document.getElementById("file-drop-label");
const buildBtn = document.getElementById("build-btn");
const buildLog = document.getElementById("build-log");

const searchForm = document.getElementById("search-form");
const queryInput = document.getElementById("query-input");
const topkInput = document.getElementById("topk-input");
const thresholdInput = document.getElementById("threshold-input");
const searchBtn = document.getElementById("search-btn");
const tokensLine = document.getElementById("tokens-line");
const resultsEl = document.getElementById("results");

async function refreshStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (data.ready) {
      statusPill.textContent = `index ready \u00b7 ${data.num_documents} docs \u00b7 vocab ${data.vocabulary_size}`;
      statusPill.className = "masthead-status ready";
    } else {
      statusPill.textContent = "no index yet \u2014 build one below";
      statusPill.className = "masthead-status empty";
    }
  } catch (e) {
    statusPill.textContent = "status unavailable";
  }
}

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) {
    fileDropLabel.textContent = fileInput.files[0].name;
    fileDrop.classList.add("has-file");
  } else {
    fileDropLabel.textContent = "Choose spreadsheet\u2026";
    fileDrop.classList.remove("has-file");
  }
});

buildForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!fileInput.files.length) {
    showBuildLog("Pick a .xlsx file first.", true);
    return;
  }

  const fd = new FormData();
  fd.append("file", fileInput.files[0]);

  buildBtn.disabled = true;
  buildBtn.textContent = "Building\u2026";
  showBuildLog("Uploading and training \u2014 this can take a minute or two for larger datasets.", false);

  try {
    const res = await fetch("/api/build", { method: "POST", body: fd });
    const data = await res.json();

    if (!res.ok) {
      showBuildLog(`Error: ${data.error || "build failed"}`, true);
    } else {
      const info = data.info || {};
      showBuildLog(
        `Index built \u2014 ${info.num_documents} documents, vocabulary ${info.vocabulary_size}, vector size ${info.vector_size}.`,
        false
      );
      refreshStatus();
    }
  } catch (err) {
    showBuildLog(`Error: ${err}`, true);
  } finally {
    buildBtn.disabled = false;
    buildBtn.textContent = "Build index";
  }
});

function showBuildLog(msg, isError) {
  buildLog.hidden = false;
  buildLog.textContent = msg;
  buildLog.className = "build-log" + (isError ? " error" : "");
}

searchForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;

  searchBtn.disabled = true;
  resultsEl.innerHTML = '<div class="empty-note">Searching\u2026</div>';
  tokensLine.hidden = true;

  try {
    const res = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: query,
        top_k: parseInt(topkInput.value || "5", 10),
        threshold: (parseFloat(thresholdInput.value || "0")) / 100,
      }),
    });
    const data = await res.json();

    if (!res.ok) {
      resultsEl.innerHTML = `<div class="error-note">${data.error || "Search failed."}</div>`;
      return;
    }

    renderTokens(data.tokens || []);
    renderResults(data.results || [], data.message);
  } catch (err) {
    resultsEl.innerHTML = `<div class="error-note">${err}</div>`;
  } finally {
    searchBtn.disabled = false;
  }
});

function renderTokens(tokens) {
  if (!tokens.length) {
    tokensLine.hidden = true;
    return;
  }
  tokensLine.hidden = false;
  tokensLine.innerHTML =
    "recognized: " + tokens.map((t) => `<span class="tok">${escapeHtml(t)}</span>`).join("");
}

function renderResults(results, message) {
  if (!results.length) {
    resultsEl.innerHTML = `<div class="empty-note">${message || "No matching documents above the similarity threshold."}</div>`;
    return;
  }

  resultsEl.innerHTML = results
    .map(
      (r, i) => `
    <div class="result-card">
      <div class="result-rank">RANK ${i + 1} \u00b7 DOC ${escapeHtml(String(r.document_id))}</div>
      <div class="result-title">${escapeHtml(r.title || "(untitled)")}</div>
      <div class="result-meta">category: ${escapeHtml(r.category || "\u2014")} \u00b7 score ${r.similarity_score}</div>
      <div class="stamp">
        <span class="pct">${r.similarity_percentage}%</span>
        <span class="pct-label">MATCH</span>
      </div>
    </div>`
    )
    .join("");
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

refreshStatus();
