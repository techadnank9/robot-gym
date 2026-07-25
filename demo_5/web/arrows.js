const arrowToMovement = {
  ArrowUp: "KeyW",
  ArrowDown: "KeyS",
  ArrowLeft: "KeyA",
  ArrowRight: "KeyD",
};

function forwardArrow(event) {
  const code = arrowToMovement[event.code];
  if (!code || event.__demo5Forwarded) return;
  event.preventDefault();
  window.dispatchEvent(new KeyboardEvent(event.type, {
    code,
    key: code.slice(-1).toLowerCase(),
    bubbles: false,
  }));
}

window.addEventListener("keydown", forwardArrow);
window.addEventListener("keyup", forwardArrow);
