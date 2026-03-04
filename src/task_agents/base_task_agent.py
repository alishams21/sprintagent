"""Base class for task planning agents.

This module defines the shared interface and minimal common state for
agents that operate on task-level planning (single task with optional subtasks).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.prompts import prompt_registry


class BaseTaskAgent(ABC):
    """Base class with shared functionality for task planning agents.

    A new TaskAgent instance is typically created for each task brief
    (e.g. derived from module_plan.md).
    """

    def __init__(self, cfg: DictConfig, logger: Any):
        """Initialize base task agent.

        Args:
            cfg: Configuration for task planning.
            logger: Experiment / run logger with at least an ``output_dir`` attribute.
        """
        self.cfg = cfg
        self.logger = logger

        # Prompt registry for agent prompts.
        self.prompt_registry = prompt_registry

        # Prompt describing the task to plan (set by subclasses).
        self.task_prompt: str = ""

    @abstractmethod
    async def generate_task_plan(self, prompt: str, output_dir: Path) -> str:
        """Generate a task plan.

        This is the main entry point for task planning. Implementations should
        orchestrate designer / critic / planner agents and return the final plan.

        Args:
            prompt: Task brief (e.g. content from module_plan.md or a task section).
            output_dir: Directory to save any artifacts (e.g., final plan file).

        Returns:
            Final task plan as a string.
        """
