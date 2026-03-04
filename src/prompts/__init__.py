"""Prompt management for plan-generation agents (project / module / task)."""

from pathlib import Path

from .manager import PromptManager
from .registry import (
    ModuleAgentPrompts,
    ProjectAgentPrompts,
    PromptRegistry,
    TaskAgentPrompts,
    VisualizationAgentPrompts,
)

# Data directory: agent YAMLs under project_agent/, module_agent/, task_agent/
PROMPTS_DATA_DIR = Path(__file__).parent / "data"

prompt_manager = PromptManager(prompts_dir=PROMPTS_DATA_DIR)
prompt_registry = PromptRegistry(prompt_manager)

__all__ = [
    "prompt_manager",
    "prompt_registry",
    "PROMPTS_DATA_DIR",
    "ProjectAgentPrompts",
    "ModuleAgentPrompts",
    "TaskAgentPrompts",
    "VisualizationAgentPrompts",
]
