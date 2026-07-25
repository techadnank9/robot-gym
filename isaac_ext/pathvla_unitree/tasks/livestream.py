from __future__ import annotations

import os

from pathvla.errors import LivestreamError
from pathvla.schemas import RunMode
from isaac_ext.pathvla_unitree.tasks.room_nav_env_cfg import LivestreamConfigModel


def configure_livestream(mode: RunMode, livestream_cfg: LivestreamConfigModel, logger) -> dict[str, str]:
    if mode == RunMode.NONE:
        return {"mode": "none", "instructions": "Headless recorded mode selected."}

    if mode == RunMode.REMOTE_DESKTOP:
        instructions = (
            "Remote desktop mode selected. Provision NICE DCV, VNC, or RDP on the GPU VM. "
            "This path is documented but is not used as a fallback for broken WebRTC."
        )
        logger.info(instructions)
        return {"mode": "remote_desktop", "instructions": instructions}

    try:
        import carb
        import omni.kit.app
    except ImportError as exc:
        raise LivestreamError("Isaac livestream configuration requires carb and omni.kit.app imports.") from exc

    app = omni.kit.app.get_app()
    ext_manager = app.get_extension_manager()
    missing_extensions = []
    for extension_name in livestream_cfg.livestream.required_extensions:
        enabled = ext_manager.is_extension_enabled(extension_name)
        if not enabled:
            enabled = ext_manager.set_extension_enabled_immediate(extension_name, True)
        if not enabled:
            missing_extensions.append(extension_name)
    if missing_extensions:
        raise LivestreamError(f"Livestream extensions could not be enabled: {missing_extensions}")

    settings = carb.settings.get_settings()
    settings.set("/app/livestream/enabled", True)
    settings.set("/app/window/drawMouse", True)

    host = os.getenv("BREV_PUBLIC_HOST")
    if not host:
        raise LivestreamError("BREV_PUBLIC_HOST is required for --live webrtc.")

    instructions = (
        f"Open the Isaac Sim WebRTC streaming client or supported browser/client and connect to "
        f"{host}:{livestream_cfg.livestream.signaling_port}. Also expose HTTP port "
        f"{livestream_cfg.livestream.http_port} and UDP ports {livestream_cfg.livestream.udp_port_range}. "
        f"{livestream_cfg.livestream.browser_path_hint}"
    )
    logger.info(instructions)
    return {"mode": "webrtc", "instructions": instructions}
