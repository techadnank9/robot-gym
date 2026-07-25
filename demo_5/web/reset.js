window.addEventListener("keydown", (event) => {
  if (event.code !== "KeyX" || event.repeat || !keyboardPlayer) return;
  event.preventDefault();
  if (!activeSocket || activeSocket.readyState !== WebSocket.OPEN) return;
  activeSocket.send(JSON.stringify({
    type: "reset_payload",
    player_id: keyboardPlayer,
  }));
});
