"""Typed failures for the real-G1 demo."""


class Demo2Error(RuntimeError):
    """Base class for demo_2 failures."""


class Demo2ConfigurationError(Demo2Error):
    """Raised for invalid configuration or unavailable dependencies."""


class HardwareSafetyError(Demo2Error):
    """Raised when a physical-motion safety requirement is not satisfied."""


class Sdk2Error(Demo2Error):
    """Raised when Unitree SDK2 initialization or an RPC fails."""


class UnsupportedCapabilityError(Demo2Error):
    """Raised when a simulator-only capability is requested on hardware."""
