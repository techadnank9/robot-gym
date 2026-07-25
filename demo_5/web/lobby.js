const launcherElement = document.querySelector("#match-launcher");
const broadcastElement = document.querySelector("#match-broadcast");
const launcherStatus = document.querySelector("#launcher-status");
const launcherError = document.querySelector("#launcher-error");
const modeList = document.querySelector("#mode-list");
const modeButtons = Array.from(document.querySelectorAll("[data-mode]"));
const seatPanel = document.querySelector("#seat-panel");
const enterP1 = document.querySelector("#enter-p1");
const copyP2 = document.querySelector("#copy-p2");
const copyStatus = document.querySelector("#copy-status");
const rematchButton = document.querySelector("#rematch-button");
const rematchNote = document.querySelector("#rematch-note");
const seatFromUrl = new URLSearchParams(location.search).get("player");

let launcherAvailable = false;
let selectedMode = sessionStorage.getItem("demo5-match-mode");
let starting = false;

function playerUrl(playerId) {
  const url = new URL(location.href);
  url.searchParams.set("player", playerId);
  return url.toString();
}

function setBusy(busy, label = "STARTING MATCH") {
  starting = busy;
  modeButtons.forEach((button) => {
    button.disabled = busy;
  });
  launcherStatus.textContent = busy ? label : "ARENA READY";
}

function showBroadcast() {
  broadcastElement.hidden = false;
  launcherElement.classList.add("is-leaving");
  window.setTimeout(() => {
    launcherElement.hidden = true;
  }, 430);
}

function showLauncher() {
  launcherElement.hidden = false;
  launcherElement.classList.remove("is-leaving");
  broadcastElement.hidden = true;
}

function resetResult() {
  document.querySelector("#result")?.classList.remove("show");
  if (rematchNote) rematchNote.textContent = "";
}

async function launcherRequest(path, options = {}) {
  const response = await fetch(path, {
    cache: "no-store",
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const value = await response.json();
  if (!response.ok) {
    throw new Error(value.error || `Launcher request failed (${response.status})`);
  }
  return value;
}

async function startMatch(mode, { rematch = false } = {}) {
  if (starting) return;
  launcherError.textContent = "";
  setBusy(true, rematch ? "RESETTING ARENA" : "STARTING MATCH");
  try {
    const state = await launcherRequest("/api/matches", {
      method: "POST",
      body: JSON.stringify({ mode }),
    });
    selectedMode = state.mode;
    sessionStorage.setItem("demo5-match-mode", selectedMode);
    resetResult();
    if (selectedMode === "human-vs-human" && !seatFromUrl) {
      modeList.hidden = true;
      seatPanel.hidden = false;
      launcherStatus.textContent = "WAITING FOR BOTH PLAYERS";
      return;
    }
    showBroadcast();
  } catch (error) {
    launcherError.textContent = error.message;
    launcherStatus.textContent = "LAUNCH FAILED";
  } finally {
    setBusy(false);
    if (!seatPanel.hidden) {
      launcherStatus.textContent = "WAITING FOR BOTH PLAYERS";
    }
  }
}

modeButtons.forEach((button) => {
  button.addEventListener("click", () => startMatch(button.dataset.mode));
});

enterP1?.addEventListener("click", () => {
  location.href = playerUrl("p1");
});

copyP2?.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(playerUrl("p2"));
    copyStatus.textContent = "P2 LINK COPIED";
  } catch {
    copyStatus.textContent = playerUrl("p2");
  }
});

rematchButton?.addEventListener("click", () => {
  if (!selectedMode) return;
  startMatch(selectedMode, { rematch: true });
});

window.addEventListener("robotgym:match-state", (event) => {
  const state = event.detail?.state;
  if (!state) return;
  selectedMode = selectedMode || state.matchMode;
  if (selectedMode) sessionStorage.setItem("demo5-match-mode", selectedMode);
  if (!launcherAvailable || seatFromUrl || selectedMode !== "human-vs-human") {
    showBroadcast();
  }
  if (state.winner && selectedMode === "human-vs-human" && seatFromUrl === "p2") {
    rematchButton.hidden = true;
    rematchNote.textContent = "P1 CONTROLS THE REMATCH";
  } else if (state.winner) {
    rematchButton.hidden = false;
  }
});

async function initializeLauncher() {
  try {
    const state = await launcherRequest("/api/launcher");
    launcherAvailable = true;
    selectedMode = state.mode || selectedMode;
    if (selectedMode) sessionStorage.setItem("demo5-match-mode", selectedMode);
    if (state.status === "running") {
      if (state.mode === "human-vs-human" && !seatFromUrl) {
        modeList.hidden = true;
        seatPanel.hidden = false;
        launcherStatus.textContent = "MATCH ACTIVE // CHOOSE SEAT";
      } else {
        showBroadcast();
      }
    } else {
      showLauncher();
      if (state.status === "error") {
        launcherStatus.textContent = "MATCH PROCESS ERROR";
        launcherError.textContent = state.error || "The match process exited.";
      }
    }
  } catch {
    launcherAvailable = false;
    launcherStatus.textContent = "CONNECTING TO MATCH";
  }
}

initializeLauncher();
