"""Control module for the 2-DOF arm."""

from control.controllers import PositionTargetController
from control.action_space import ActionSpace

__all__ = ["PositionTargetController", "ActionSpace"]
