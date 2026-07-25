class PathVLAError(RuntimeError):
    """Base project error."""


class ConfigurationError(PathVLAError):
    """Raised when required configuration is missing or invalid."""


class IsaacRuntimeError(PathVLAError):
    """Raised when Isaac runtime requirements are not met."""


class RobotAssetError(PathVLAError):
    """Raised when the required robot asset or controller is unavailable."""


class VLAEndpointError(PathVLAError):
    """Raised when VLA endpoint configuration or response is invalid."""


class LivestreamError(PathVLAError):
    """Raised when livestream mode is requested but unavailable."""


class PlanningError(PathVLAError):
    """Raised when no valid plan or path can be created."""


class RecordingError(PathVLAError):
    """Raised when required recording output cannot be produced."""


class RunStopped(PathVLAError):
    """Raised when a remote stop is acknowledged at a safe action boundary."""
