from enum import Enum, EnumMeta, nonmember
from typing import Any, Dict, Type, Union

from .manager import PromptManager


class PromptEnumMeta(EnumMeta):
    """Metaclass for PromptEnum that handles _BASE_PATH validation."""

    def __new__(
        metacls: Type["PromptEnumMeta"],
        cls: str,
        bases: tuple[Type, ...],
        classdict: Dict[str, Any],
        **kwds: Any,
    ) -> Type["PromptEnum"]:
        """Create enum class with automatic _BASE_PATH validation."""
        base_path_value = classdict.get("_BASE_PATH")
        enum_class = super().__new__(metacls, cls, bases, classdict, **kwds)

        if cls != "PromptEnum" and base_path_value is None:
            raise AttributeError(
                f"{cls} must define a _BASE_PATH class attribute. "
                f"Example: _BASE_PATH = nonmember('agent_folder')"
            )

        return enum_class


class PromptEnum(str, Enum, metaclass=PromptEnumMeta):
    """Base class for all prompt enums.

    All prompt enums should inherit from this class and define a _BASE_PATH.
    The enum values are automatically combined with _BASE_PATH to create full paths.

    Example:
        class ProjectAgentPrompts(PromptEnum):
            _BASE_PATH = nonmember("project_agent")
            PLANNER_AGENT = "planner_agent"  # Full path: "project_agent/planner_agent"

    Note:
        _BASE_PATH must be wrapped with nonmember() to prevent it from becoming
        an enum member in Python 3.11+.
    """

    def __new__(cls, value):
        """Create enum member with _BASE_PATH prefix."""
        base_path = getattr(cls, "_BASE_PATH", "")

        if base_path:
            full_value = f"{base_path}/{value}"
        else:
            full_value = value

        obj = str.__new__(cls, full_value)
        obj._value_ = full_value
        return obj


class ProjectAgentPrompts(PromptEnum):
    """Registry of project agent prompts (data/project_agent/)."""

    _BASE_PATH = nonmember("project_agent")

    PLANNER_AGENT = "planner_agent"
    PLANNER_RUNNER_INSTRUCTION = "planner_runner_instruction"
    DESIGNER_AGENT = "designer_agent"
    DESIGNER_INITIAL_INSTRUCTION = "designer_initial_instruction"
    DESIGNER_CRITIQUE_INSTRUCTION = "designer_critique_instruction"
    CRITIC_AGENT = "critic_agent"
    CRITIC_RUNNER_INSTRUCTION = "critic_runner_instruction"


class ModuleAgentPrompts(PromptEnum):
    """Registry of module agent prompts (data/module_agent/)."""

    _BASE_PATH = nonmember("module_agent")

    PLANNER_AGENT = "planner_agent"
    PLANNER_RUNNER_INSTRUCTION = "planner_runner_instruction"
    DESIGNER_AGENT = "designer_agent"
    DESIGNER_INITIAL_INSTRUCTION = "designer_initial_instruction"
    DESIGNER_CRITIQUE_INSTRUCTION = "designer_critique_instruction"
    CRITIC_AGENT = "critic_agent"
    CRITIC_RUNNER_INSTRUCTION = "critic_runner_instruction"


class TaskAgentPrompts(PromptEnum):
    """Registry of task agent prompts (data/task_agent/)."""

    _BASE_PATH = nonmember("task_agent")

    PLANNER_AGENT = "planner_agent"
    PLANNER_RUNNER_INSTRUCTION = "planner_runner_instruction"
    DESIGNER_AGENT = "designer_agent"
    DESIGNER_INITIAL_INSTRUCTION = "designer_initial_instruction"
    DESIGNER_CRITIQUE_INSTRUCTION = "designer_critique_instruction"
    CRITIC_AGENT = "critic_agent"
    CRITIC_RUNNER_INSTRUCTION = "critic_runner_instruction"


class VisualizationAgentPrompts(PromptEnum):
    """Registry of visualization agent prompts (data/visualization_agent/)."""

    _BASE_PATH = nonmember("visualization_agent")

    PLANNER_AGENT = "planner_agent"
    PLANNER_RUNNER_INSTRUCTION = "planner_runner_instruction"
    DESIGNER_AGENT = "designer_agent"
    DESIGNER_INITIAL_INSTRUCTION = "designer_initial_instruction"
    DESIGNER_CRITIQUE_INSTRUCTION = "designer_critique_instruction"
    CRITIC_AGENT = "critic_agent"
    CRITIC_RUNNER_INSTRUCTION = "critic_runner_instruction"


class PromptRegistry:
    """Central registry for all available prompts."""

    def __init__(self, prompt_manager: PromptManager):
        """Initialize registry with a prompt manager."""
        self.prompt_manager = prompt_manager

    def get_prompt(self, prompt_enum: Union[PromptEnum, str], **kwargs) -> str:
        """
        Get a prompt by its enum value or path string.

        Args:
            prompt_enum: Prompt enum member (e.g. ProjectAgentPrompts.PLANNER_AGENT)
                or path string (e.g. "project_agent/planner_agent").
            **kwargs: Template variables for rendering

        Returns:
            Rendered prompt string

        Raises:
            ValueError: If template variables don't match YAML declaration exactly
        """
        path = prompt_enum.value if isinstance(prompt_enum, PromptEnum) else prompt_enum
        metadata = self.prompt_manager.get_prompt_metadata(path)
        required_vars = set(metadata.get("template_variables", []))
        provided_vars = set(kwargs.keys())

        missing_vars = required_vars - provided_vars
        if missing_vars:
            raise ValueError(
                f"Missing required template variables for {path}: "
                f"{', '.join(sorted(missing_vars))}. "
                f"Required: {sorted(required_vars)}, Provided: {sorted(provided_vars)}"
            )

        extra_vars = provided_vars - required_vars
        if extra_vars:
            raise ValueError(
                f"Unexpected template variables for {path}: "
                f"{', '.join(sorted(extra_vars))}. "
                f"Required: {sorted(required_vars)}, Provided: {sorted(provided_vars)}"
            )

        return self.prompt_manager.get_prompt(path, **kwargs)

    def validate_prompt_args(
        self, prompt_enum: Union[PromptEnum, str], **kwargs
    ) -> bool:
        """
        Validate that all required variables are provided for a prompt.

        Args:
            prompt_enum: Prompt enum value or path string
            **kwargs: Provided template variables

        Returns:
            True if all required variables match exactly
        """
        try:
            path = (
                prompt_enum.value if isinstance(prompt_enum, PromptEnum) else prompt_enum
            )
            metadata = self.prompt_manager.get_prompt_metadata(path)
            required_vars = set(metadata.get("template_variables", []))
            provided_vars = set(kwargs.keys())
            return required_vars == provided_vars
        except Exception:
            return False

class SessionMemoryPrompts(PromptEnum):
    """Registry of session memory management prompts."""

    _BASE_PATH = nonmember("session_memory")

    TURN_SUMMARIZATION = "turn_summarization"