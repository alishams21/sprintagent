"""Base class for project planning agents.

This module defines the shared interface and minimal common state for
agents that operate on project-level planning (project → modules → tasks).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.prompts import prompt_registry


class BaseProjectAgent(ABC):
    """Base class with shared functionality for project planning agents.

    A new ProjectAgent instance is typically created for each project brief.
    """

    def __init__(self, cfg: DictConfig, logger: Any):
        """Initialize base project agent.

        Args:
            cfg: Configuration for project planning.
            logger: Experiment / run logger with at least an ``output_dir`` attribute.
        """
        self.cfg = cfg
        self.logger = logger

        # Prompt registry for agent prompts.
        self.prompt_registry = prompt_registry

        # Prompt describing the project to plan (set by subclasses).
        self.project_prompt: str = ""

    @abstractmethod
    async def generate_project_plan(self, prompt: str, output_dir: Path) -> str:
        """Generate a hierarchical project plan.

        This is the main entry point for project planning. Implementations should
        orchestrate designer / critic / planner agents and return the final plan.

        Args:
            prompt: High-level project brief.
            output_dir: Directory to save any artifacts (e.g., final plan file).

        Returns:
            Final hierarchical project plan as a string.
        """

