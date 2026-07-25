const qs = new URLSearchParams(location.search);
const requestedPlayer = ["p1", "p2"].includes(qs.get("player"))
  ? qs.get("player")
  : null;
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
  inputHelp: document.querySelector("#input-help"),
};

let activeSocket = null;
let keyboardPlayer = null;
let sequence = Date.now() * 1000;
let selectedSkill = "wait";
let handClose = 0;
let previousGamepadButtons = [];
const heldKeys = new Set();
const controlCodes = new Set([
  "KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE",
  "KeyG", "KeyC", "KeyR", "KeyU", "Space",
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
  const browserPlayers = ["p1", "p2"].filter((id) =>
    state.players[id].mode === "human" &&
    state.players[id].model_name === "Browser keyboard"
  );
  keyboardPlayer = requestedPlayer && browserPlayers.includes(requestedPlayer)
    ? requestedPlayer
    : (browserPlayers.length === 1 ? browserPlayers[0] : null);
  const seatSelectionRequired = browserPlayers.length > 1 && !keyboardPlayer;
  elements.keyboardHelp.hidden = !keyboardPlayer && !seatSelectionRequired;
  if (keyboardPlayer) {
    updateInputHelp(connectedGamepad());
  } else if (seatSelectionRequired) {
    elements.keyboardPlayer.textContent = "CHOOSE PLAYER";
    if (elements.inputHelp) {
      elements.inputHelp.textContent =
        "OPEN THIS PAGE WITH ?player=p1 OR ?player=p2";
    }
  }
  setFrame("broadcast-feed", message.frames.broadcast);
  setFrame("p1-feed", message.frames.p1);
  setFrame("p2-feed", message.frames.p2);
  if (state.winner) {
    elements.winner.textContent = state.result || `${state.winner.toUpperCase()} WON`;
    elements.result.classList.add("show");
  } else {
    elements.result.classList.remove("show");
  }
  window.dispatchEvent(new CustomEvent("robotgym:match-state", {
    detail: { state, frames: message.frames },
  }));
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

function connectedGamepad() {
  if (!navigator.getGamepads) return null;
  return Array.from(navigator.getGamepads()).find((gamepad) =>
    gamepad && gamepad.connected
  ) || null;
}

function gamepadAxis(gamepad, index) {
  const value = Number(gamepad.axes[index] || 0);
  const deadzone = 0.14;
  if (Math.abs(value) <= deadzone) return 0;
  return Math.sign(value) *
    Math.min(1, (Math.abs(value) - deadzone) / (1 - deadzone));
}

function readGamepad(gamepad) {
  const buttons = gamepad.buttons.map((button) => Boolean(button.pressed));
  const pressed = (index) =>
    Boolean(buttons[index] && !previousGamepadButtons[index]);

  if (pressed(0)) {
    selectedSkill = "grasp";
    handClose = 1;
  } else if (pressed(2)) {
    selectedSkill = "navigate_goal";
    handClose = 1;
  } else if (pressed(1)) {
    selectedSkill = "release";
    handClose = 0;
  } else if (pressed(3)) {
    selectedSkill = "recover";
  }
  if (pressed(9) && activeSocket?.readyState === WebSocket.OPEN) {
    activeSocket.send(JSON.stringify({
      type: "reset_payload",
      player_id: keyboardPlayer,
    }));
  }
  const cameraReset = pressed(8);
  previousGamepadButtons = buttons;

  return {
    moveX: gamepadAxis(gamepad, 0),
    moveY: -gamepadAxis(gamepad, 1),
    yaw: gamepadAxis(gamepad, 2),
    cameraReset,
  };
}

function updateInputHelp(gamepad) {
  if (!keyboardPlayer) return;
  if (gamepad) {
    elements.keyboardPlayer.textContent = `${keyboardPlayer.toUpperCase()} GAMEPAD`;
    if (elements.inputHelp) {
      elements.inputHelp.textContent =
        "LEFT STICK MOVE · RIGHT STICK TURN · A/✕ GRASP · X/□ CARRY · B/○ RELEASE · Y/△ RECOVER · BACK CAMERA · START RESET";
    }
    return;
  }
  elements.keyboardPlayer.textContent = `${keyboardPlayer.toUpperCase()} KEYBOARD`;
  if (elements.inputHelp) {
    elements.inputHelp.textContent = demo5Host
      ? "ARROWS MOVE · Q/E TURN · G EASY GRASP · C CARRY · R RELEASE · U RECOVER · X RESET"
      : "WASD MOVE · Q/E TURN · G GRASP · C CARRY · R RELEASE · U RECOVER";
  }
}

function sendTeleop() {
  if (!keyboardPlayer || !activeSocket || activeSocket.readyState !== WebSocket.OPEN) return;
  const gamepad = connectedGamepad();
  let moveX;
  let moveY;
  let yaw;
  let cameraReset = false;
  if (gamepad) {
    ({ moveX, moveY, yaw, cameraReset } = readGamepad(gamepad));
  } else {
    previousGamepadButtons = [];
    moveX = axis("KeyD", "KeyA");
    moveY = axis("KeyW", "KeyS");
    yaw = axis("KeyE", "KeyQ");
  }
  updateInputHelp(gamepad);
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
      camera_reset: cameraReset,
    },
  }));
}

window.addEventListener("keydown", (event) => {
  if (!controlCodes.has(event.code)) return;
  event.preventDefault();
  if (event.code === "Space") {
    heldKeys.clear();
  } else {
    heldKeys.add(event.code);
  }
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

window.addEventListener("gamepadconnected", () => {
  previousGamepadButtons = [];
  updateInputHelp(connectedGamepad());
  sendTeleop();
});

window.addEventListener("gamepaddisconnected", () => {
  previousGamepadButtons = [];
  updateInputHelp(null);
  sendTeleop();
});

setInterval(sendTeleop, 20);
connect();
