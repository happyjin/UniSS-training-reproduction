"""Immutable trajectory data model and builders for the E2E experiment."""

from .schema import E2ETrajectory, TrajectoryEvent, validate_trajectory

__all__ = ["E2ETrajectory", "TrajectoryEvent", "validate_trajectory"]
