const qs = new URLSearchParams(location.search);
const demo5Host = location.port === "8085" ||
  /-8085\.proxy\.runpod\.net$/.test(location.hostname);
const socketPort = qs.get("wsPort") || (demo5Host ? "8765" : "8763");
let socketHost = qs.get("wsHost") || location.hostname || "127.0.0.1";
const scheme = location.protocol === "https:" ? "wss" : "ws";
const runpodProxy = !qs.has("wsHost") &&
  /^.+-\d+\.proxy\.runpod\.net$/.test(socketHost);
if (runpodProxy) {
  socketHost = socketHost.replace(
    /-\d+\.proxy\.runpod\.net$/,
    `-${socketPort}.proxy.runpod.net`,
  );
}
const socketUrl = runpodProxy
  ? `${scheme}://${socketHost}`
  : `${scheme}://${socketHost}:${socketPort}`;

const elements = {
  connectionLight: document.querySelector("#connection-light"),
  connectionLabel: document.querySelector("#connection-label"),
  matchId: document.querySelector("#match-id"),
  phase: document.querySelector("#phase"),
  clock: document.querySelector("#clock"),
  result: document.querySelector("#result"),
  winner: document.querySelector("#winner"),
  keyboardHelp: document.querySelector("#keyboard-help"),
  keyboardPlayer: document.querySelector("#keyboard-player"),
};

let activeSocket = null;
let keyboardPlayer = null;
let sequence = 0;
let selectedSkill = "wait";
let handClose = 0;
const heldKeys = new Set();
const controlCodes = new Set([
  "KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE",
  "KeyG", "KeyC", "KeyR", "KeyU",
]);

function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60).toString().padStart(2, "0");
  const rest = (seconds % 60).toFixed(3).padStart(6, "0");
  return `${minutes}:${rest}`;
}

function playerState(player) {
  if (player.disqualified) return "DISQUALIFIED";
  if (!player.connected) return "DAMPED STOP";
  if (player.delivered) return "DELIVERED";
  if (player.fallen) return "RECOVER";
  if (player.carrying) return "CARRYING";
  return "ACTIVE";
}

function updatePlayer(id, player) {
  document.querySelector(`#${id}-name`).textContent = player.display_name;
  document.querySelector(`#${id}-model`).textContent =
    `${player.mode.toUpperCase()} // ${player.model_name}`;
  document.querySelector(`#${id}-rationale`).textContent =
    player.rationale || "Awaiting the next grounded action.";
  document.querySelector(`#${id}-progress`).style.width = `${player.progress * 100}%`;
  document.querySelector(`#${id}-skill`).textContent = player.current_skill.replaceAll("_", " ");
  document.querySelector(`#${id}-state`).textContent = playerState(player);
}

function setFrame(id, data) {
  if (!data) return;
  const image = document.querySelector(`#${id}`);
  image.src = `data:image/jpeg;base64,${data}`;
  image.classList.add("ready");
}

function update(message) {
  const state = message.state;
  elements.matchId.textContent = state.matchId;
  elements.phase.textContent = state.phase.toUpperCase();
  elements.clock.textContent = formatTime(state.elapsedTime);
  updatePlayer("p1", state.players.p1);
  updatePlayer("p2", state.players.p2);
  const browserPlayer = ["p1", "p2"].find((id) =>
    state.players[id].mode === "human" &&
    state.players[id].model_name === "Browser keyboard"
  );
  keyboardPlayer = browserPlayer || null;
  elements.keyboardHelp.hidden = !keyboardPlayer;
  if (keyboardPlayer) {
    elements.keyboardPlayer.textContent = `${keyboardPlayer.toUpperCase()} KEYBOARD`;
  }
  setFrame("broadcast-feed", message.frames.broadcast);
  setFrame("p1-feed", message.frames.p1);
  setFrame("p2-feed", message.frames.p2);
  if (state.winner) {
    elements.winner.textContent = state.result || `${state.winner.toUpperCase()} WON`;
    elements.result.classList.add("show");
  }
}

function connect() {
  const socket = new WebSocket(socketUrl);
  activeSocket = socket;
  socket.addEventListener("open", () => {
    elements.connectionLight.classList.add("live");
    elements.connectionLabel.textContent = "LIVE";
  });
  socket.addEventListener("message", (event) => {
    const value = JSON.parse(event.data);
    if (value.type === "match_state") update(value);
  });
  socket.addEventListener("close", () => {
    if (activeSocket === socket) activeSocket = null;
    elements.connectionLight.classList.remove("live");
    elements.connectionLabel.textContent = "RECONNECTING";
    setTimeout(connect, 1200);
  });
}

function axis(positive, negative) {
  return (heldKeys.has(positive) ? 1 : 0) - (heldKeys.has(negative) ? 1 : 0);
}

function sendTeleop() {
  if (!keyboardPlayer || !activeSocket || activeSocket.readyState !== WebSocket.OPEN) return;
  const moveX = axis("KeyD", "KeyA");
  const moveY = axis("KeyW", "KeyS");
  const yaw = axis("KeyE", "KeyQ");
  activeSocket.send(JSON.stringify({
    type: "teleop",
    frame: {
      protocol_version: "3.0",
      player_id: keyboardPlayer,
      sequence: ++sequence,
      timestamp_s: performance.now() / 1000,
      connected: true,
      deadman: moveX !== 0 || moveY !== 0 || yaw !== 0,
      move_x: moveX,
      move_y: moveY,
      yaw,
      skill: selectedSkill,
      hand_close: handClose,
    },
  }));
}

window.addEventListener("keydown", (event) => {
  if (!controlCodes.has(event.code)) return;
  event.preventDefault();
  heldKeys.add(event.code);
  if (event.code === "KeyG") {
    selectedSkill = "grasp";
    handClose = 1;
  } else if (event.code === "KeyC") {
    selectedSkill = "navigate_goal";
    handClose = 1;
  } else if (event.code === "KeyR") {
    selectedSkill = "release";
    handClose = 0;
  } else if (event.code === "KeyU") {
    selectedSkill = "recover";
  }
  sendTeleop();
});

window.addEventListener("keyup", (event) => {
  if (!controlCodes.has(event.code)) return;
  event.preventDefault();
  heldKeys.delete(event.code);
  sendTeleop();
});

window.addEventListener("blur", () => {
  heldKeys.clear();
  sendTeleop();
});

setInterval(sendTeleop, 20);
connect();
