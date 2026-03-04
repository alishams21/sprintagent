"""Base class for module planning agents.

This module defines the shared interface and minimal common state for
agents that operate on module-level planning (module structure + tasks).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.prompts import prompt_registry


class BaseModuleAgent(ABC):
    """Base class with shared functionality for module planning agents.

    A new ModuleAgent instance is typically created for each module brief
    (e.g. derived from project_plan.md).
    """

    def __init__(self, cfg: DictConfig, logger: Any):
        """Initialize base module agent.

        Args:
            cfg: Configuration for module planning.
            logger: Experiment / run logger with at least an ``output_dir`` attribute.
        """
        self.cfg = cfg
        self.logger = logger

        # Prompt registry for agent prompts.
        self.prompt_registry = prompt_registry

        # Prompt describing the module to plan (set by subclasses).
        self.module_prompt: str = ""

    @abstractmethod
    async def generate_module_plan(self, prompt: str, output_dir: Path) -> str:
        """Generate a module plan.

        This is the main entry point for module planning. Implementations should
        orchestrate designer / critic / planner agents and return the final plan.

        Args:
            prompt: Module brief (e.g. content from project_plan.md or a module section).
            output_dir: Directory to save any artifacts (e.g., final plan file).

        Returns:
            Final module plan as a string.
        """
