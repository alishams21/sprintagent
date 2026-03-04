"""Base class for visualization agents that draw hierarchy diagrams from plans."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.prompts import prompt_registry


class BaseVisualizationAgent(ABC):
    """Base class for agents that produce a hierarchy diagram from plan markdown files.

    Reads project_plan.md, module_plan.md, and task_plan.md from a directory and
    produces a diagram (e.g. PNG/PDF) showing project → modules → tasks.
    """

    def __init__(self, cfg: DictConfig, logger: Any):
        """Initialize base visualization agent.

        Args:
            cfg: Configuration for the visualization agent.
            logger: Experiment / run logger with at least an ``output_dir`` attribute.
        """
        self.cfg = cfg
        self.logger = logger
        self.prompt_registry = prompt_registry

    @abstractmethod
    async def generate_hierarchy_diagram(self, output_dir: Path) -> Path | str:
        """Generate a hierarchy diagram from plan files in output_dir.

        Reads project_plan.md, module_plan.md, task_plan.md from output_dir and
        produces a diagram file (e.g. hierarchy_diagram.png).

        Args:
            output_dir: Directory containing the three plan .md files and where
                the diagram output will be written.

        Returns:
            Path to the generated diagram file, or a string identifier.
        """
